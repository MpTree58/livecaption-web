"""Manual smoke test: web Session start/stop lifecycle edge cases (real models).

Drives webapp.Session through its public start/stop API with a file-fed stand-in for the
microphone (macOS `say` speech -- no real mic is opened) and a throwaway meeting store, so
nothing touches ~/.cache/livecaption/meetings.

1. Stop right after the first final: every line of the saved meeting must still get its
   translation (the translator lags the ASR by a sentence or two when Stop is clicked).
2. Stop while Start is still loading the models: the start backs out -- it must never go
   on to listen (a pipeline nobody can stop), and it creates no meeting.
3. Start while Stop is still finishing: refused, instead of the old run's teardown
   closing the new run's meeting.
4. A start that fails half-way ("both", the mic already capturing when audiotee dies --
   simulated with /usr/bin/false as the audiotee binary) tears down what did start and
   leaves no empty meeting behind.
    uv run python scripts/smoke_web_session.py
"""

import queue
import subprocess
import tempfile
import threading
import time
from pathlib import Path

from livecaption import config, webapp
from livecaption.audio import FileSource
from livecaption.meetings import MeetingStore
from livecaption.websink import WebSink

# three short sentences with pauses long enough for rule2 to cut each into its own final
SPEECH = (
    "First, the budget is approved. [[slnc 900]] Second, hiring starts in March. "
    "[[slnc 900]] Third, we ship the new version in May."
)
TIMEOUT = 300


def synthesize(text: str, out_wav: Path) -> None:
    """say -> aiff -> afconvert to 16k mono s16 wav (all macOS built-in tools)."""
    aiff = out_wav.with_suffix(".aiff")
    subprocess.run(["say", "-v", "Samantha", "-o", str(aiff), text], check=True)
    subprocess.run(
        ["afconvert", "-f", "WAVE", "-d", f"LEI16@{config.SAMPLE_RATE}", "-c", "1",
         str(aiff), str(out_wav)],
        check=True,
    )


def wait_for(events: queue.Queue, pred, what: str) -> dict:  # noqa: ANN001
    deadline = time.monotonic() + TIMEOUT
    while time.monotonic() < deadline:
        try:
            ev = events.get(timeout=1.0)
        except queue.Empty:
            continue
        if ev.get("type") == "log" and ev.get("level") == "warn":
            print(f"  [warn] {ev['message']}")
        if pred(ev):
            return ev
    raise TimeoutError(f"timed out waiting for {what}")


with tempfile.TemporaryDirectory() as td:
    wav = Path(td) / "speech.wav"
    synthesize(SPEECH, wav)
    webapp.MicSource = lambda label="me": FileSource(str(wav), label=label)

    sink = WebSink()
    events = sink.subscribe()
    store = MeetingStore(Path(td) / "meetings")
    session = webapp.Session(
        sink,
        store,
        presets=webapp._resolve_presets(),
        default_lang="en",
        diarize=False,
        translate=True,
        mt_model_id=config.DEFAULT_MT_MODEL,
        context=config.MT_CONTEXT_SENTENCES,
        audiotee_path=None,
    )

    # warm-up run so the timed run below doesn't race the translation model's first load
    assert session.start("mic", "en")["ok"]
    wait_for(events, lambda e: e.get("type") == "translation", "warm-up translation")
    session.stop()
    wait_for(events, lambda e: e.get("type") == "status" and e["state"] == "idle", "idle")

    # --- 1. Stop right after the first final: no translation may be lost ---
    assert session.start("mic", "en")["ok"]
    ev = wait_for(events, lambda e: e.get("type") == "status" and e["state"] == "listening",
                  "listening")
    mid = ev["meeting"]
    wait_for(events, lambda e: e.get("type") == "final", "first final")
    session.stop()  # the translator is still busy with that first line
    wait_for(events, lambda e: e.get("type") == "status" and e["state"] == "idle", "idle")
    saved = store.load(mid)
    lines = saved["utterances"]
    missing = [u["en"][0]["text"] for u in lines if not u.get("zh")]
    for u in lines:
        zh = " ".join(s["zh"] for s in u["zh"]) if u.get("zh") else "(no translation)"
        print(f"  {u['en'][0]['text']}  ->  {zh}")
    assert lines, "the stopped meeting saved no lines"
    assert not missing, f"translations lost on Stop for: {missing}"
    print("stop keeps every translation: OK")

    # --- 2. Stop while Start is still loading the models ---
    session._unload_idle_models()  # cold start, so "starting" lasts a few seconds
    before = {m["id"] for m in store.list()}
    assert session.start("mic", "en")["ok"]
    wait_for(events, lambda e: e.get("type") == "status" and e["state"] == "starting",
             "starting")
    assert session.stop()["ok"], "Stop during Start was refused"
    wait_for(events, lambda e: e.get("type") == "status" and e["state"] == "idle", "idle")
    time.sleep(8)  # long enough for an un-cancelled start to finish loading and go live
    late = []
    while not events.empty():
        ev = events.get_nowait()
        if ev.get("type") == "status":
            late.append(ev["state"])
    assert "listening" not in late, f"the cancelled start went live anyway: {late}"
    assert not session.state()["running"], "session still marked running"
    assert {m["id"] for m in store.list()} == before, "the cancelled start created a meeting"
    print("stop during start backs out: OK")

    # --- 3. Start while Stop is still finishing ---
    assert session.start("mic", "en")["ok"], "start after a cancelled start was refused"
    wait_for(events, lambda e: e.get("type") == "status" and e["state"] == "listening",
             "listening")
    wait_for(events, lambda e: e.get("type") == "final", "final")
    session.stop()
    again = session.start("mic", "en")  # the stop above is still draining / finalizing
    wait_for(events, lambda e: e.get("type") == "status" and e["state"] == "idle", "idle")
    assert not again["ok"], "Start was accepted while the previous run was still stopping"
    print("start during stop is refused: OK")

    # --- 4. A half-way failed start tears down what did start ---
    session._audiotee_path = "/usr/bin/false"  # resolves as a binary, exits at once
    before = {m["id"] for m in store.list()}
    assert session.start("both", "en")["ok"]
    wait_for(events, lambda e: e.get("type") == "log" and "Failed to start" in e["message"],
             "the start failure")
    wait_for(events, lambda e: e.get("type") == "status" and e["state"] == "idle", "idle")
    alive = [t.name for t in threading.enumerate() if t.name.startswith(("asr-", "file-"))]
    assert not alive, f"capture/ASR threads left running after the failed start: {alive}"
    assert {m["id"] for m in store.list()} == before, "the failed start left a meeting behind"
    print("failed start cleans up: OK")

    session.shutdown()

print("web session smoke test PASSED")
