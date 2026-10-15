"""Manual smoke test: the web Session's model memory across language switches (real models).

Drives webapp.Session through its public start/stop API with a file-fed stand-in for the
microphone (macOS `say` speech -- no real mic is opened) and a throwaway meeting store, so
nothing touches ~/.cache/livecaption/meetings. For each preset in the sequence it waits for
a translated final, stops, and records the process's MLX active memory and physical
footprint. A model left resident after a switch shows up as a step that never comes back
down; returning to the first preset must land back near the first measurement.

Also checks that each run's translation comes out in that preset's target language (the
translator is shared across presets and switches direction in place), and that an idle
session frees every model (idle unload shortened to IDLE_UNLOAD_MIN here) and reloads them
on the next Start.
    uv run --extra qwen python scripts/smoke_model_swap.py
"""

import ctypes
import os
import queue
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import mlx.core as mx

from livecaption import config, webapp
from livecaption.audio import FileSource
from livecaption.meetings import MeetingStore
from livecaption.websink import WebSink

# preset -> (voice, text spoken into the fake mic)
CLIPS = {
    "en": ("Samantha", "Let's review the quarterly budget and the hiring plan before Friday."),
    "zh": ("Tingting", "我们下周一开会讨论新产品的上线计划。"),
    "mixed": ("Tingting", "我们今天先 review 一下 budget，然后讨论 hiring plan。"),
}
SEQUENCE = ["en", "mixed", "zh", "en"]
TIMEOUT = 600  # first loads can include a cold disk read of several GB
IDLE_UNLOAD_MIN = 0.15  # 9 s -- far longer than the gap between the runs below


class _RUsageV2(ctypes.Structure):
    _fields_ = [("ri_uuid", ctypes.c_uint8 * 16)] + [
        (name, ctypes.c_uint64)
        for name in (
            "ri_user_time", "ri_system_time", "ri_pkg_idle_wkups", "ri_interrupt_wkups",
            "ri_pageins", "ri_wired_size", "ri_resident_size", "ri_phys_footprint",
            "ri_proc_start_abstime", "ri_proc_exit_abstime", "ri_child_user_time",
            "ri_child_system_time", "ri_child_pkg_idle_wkups", "ri_child_interrupt_wkups",
            "ri_child_pageins", "ri_child_elapsed_abstime", "ri_diskio_bytesread",
            "ri_diskio_byteswritten",
        )
    ]


_LIBPROC = ctypes.CDLL("/usr/lib/libproc.dylib")


def footprint_gb() -> float:
    """Physical footprint (what Activity Monitor shows as Memory), incl. Metal buffers."""
    ru = _RUsageV2()
    _LIBPROC.proc_pid_rusage(os.getpid(), 2, ctypes.byref(ru))  # 2 = RUSAGE_INFO_V2
    return ru.ri_phys_footprint / 1e9


def synthesize(voice: str, text: str, out_wav: Path) -> None:
    """say -> aiff -> afconvert to 16k mono s16 wav (all macOS built-in tools)."""
    aiff = out_wav.with_suffix(".aiff")
    subprocess.run(["say", "-v", voice, "-o", str(aiff), text], check=True)
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


def has_cjk(s: str) -> bool:
    return any("一" <= c <= "鿿" for c in s)


with tempfile.TemporaryDirectory() as td:
    clips = {}
    for lang, (voice, text) in CLIPS.items():
        clips[lang] = Path(td) / f"{lang}.wav"
        synthesize(voice, text, clips[lang])

    # the fake mic: each Start feeds the clip of the preset being started
    current = {"lang": "en"}
    webapp.MicSource = lambda label="me": FileSource(str(clips[current["lang"]]), label=label)

    sink = WebSink()
    events = sink.subscribe()
    session = webapp.Session(
        sink,
        MeetingStore(Path(td) / "meetings"),
        presets=webapp._resolve_presets(),
        default_lang="en",
        diarize=True,
        translate=True,
        mt_model_id=config.DEFAULT_MT_MODEL,
        context=config.MT_CONTEXT_SENTENCES,
        audiotee_path=None,
        idle_unload_min=IDLE_UNLOAD_MIN,
    )

    rows = []
    for lang in SEQUENCE:
        current["lang"] = lang
        target = session._presets[lang]["target_prompt"]
        t0 = time.monotonic()
        assert session.start("mic", lang)["ok"], f"start {lang} refused"
        wait_for(events, lambda e: e.get("type") == "status" and e["state"] == "listening",
                 f"{lang} listening")
        ready_sec = time.monotonic() - t0
        tr = wait_for(events, lambda e: e.get("type") == "translation", f"{lang} translation")
        zh = " ".join(s["zh"] for s in tr["segments"])
        session.stop()
        wait_for(events, lambda e: e.get("type") == "status" and e["state"] == "idle",
                 f"{lang} idle")
        mx.clear_cache()
        rows.append((lang, ready_sec, mx.get_active_memory() / 1e9, footprint_gb()))
        print(f"{lang:>5} → {target}: ready {ready_sec:5.1f}s  "
              f"MLX active {rows[-1][2]:.2f} GB  footprint {rows[-1][3]:.2f} GB  | {zh}")
        # the shared translator must have switched direction for this preset
        if target == "English":
            assert not has_cjk(zh), f"{lang}: expected an English translation, got {zh!r}"
        else:
            assert has_cjk(zh), f"{lang}: expected a Chinese translation, got {zh!r}"

    first_en, last_en = rows[0], rows[-1]
    peak = max(r[2] for r in rows)
    print(f"en first {first_en[2]:.2f} GB, en again {last_en[2]:.2f} GB, peak {peak:.2f} GB")
    # back on the first preset, the other presets' models must be gone (allow buffer noise)
    leaked = last_en[2] - first_en[2]
    if leaked > 0.3:
        print(f"model swap smoke test FAILED: {leaked:.2f} GB still resident after "
              "switching back to the first preset")
        sys.exit(1)

    # idle: every model is freed once the (shortened) idle timeout passes -- MLX's buffer
    # cache emptied too, and the pages actually handed back to the OS (which takes a moment)
    deadline = time.monotonic() + IDLE_UNLOAD_MIN * 60 + 20
    while (mx.get_active_memory() + mx.get_cache_memory()) / 1e9 > 0.3 \
            and time.monotonic() < deadline:
        time.sleep(0.5)
    time.sleep(3)
    idle_gb, idle_fp = mx.get_active_memory() / 1e9, footprint_gb()
    print(f"idle after {IDLE_UNLOAD_MIN * 60:.0f}s: MLX active {idle_gb:.2f} GB  "
          f"cache {mx.get_cache_memory() / 1e9:.2f} GB  footprint {idle_fp:.2f} GB")
    if idle_gb > 0.3 or idle_fp > 1.0:
        print("model swap smoke test FAILED: models still held after the idle timeout")
        sys.exit(1)
    # ... and the next Start reloads them (translator included)
    current["lang"] = "en"
    t0 = time.monotonic()
    assert session.start("mic", "en")["ok"], "start after idle unload refused"
    tr = wait_for(events, lambda e: e.get("type") == "translation", "translation after reload")
    print(f"restart after unload: first translation in {time.monotonic() - t0:.1f}s | "
          f"{' '.join(s['zh'] for s in tr['segments'])}")
    session.stop()
    wait_for(events, lambda e: e.get("type") == "status" and e["state"] == "idle", "idle")

    session.shutdown()

print("model swap smoke test PASSED")
