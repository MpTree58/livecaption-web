"""Manual smoke test: SIGTERM mid-recording closes the meeting gracefully (real models).

The menu-bar app's Quit / Restart end the web engine with SIGTERM. This runs the engine
through scripts/demo_web.py (a `say` clip plays as the microphone; throwaway meeting store),
starts a recording, waits for the first saved line, sends SIGTERM, and checks the process
exits on its own with the meeting closed (ended_at set) and every line translated.
    uv run python scripts/smoke_web_sigterm.py
"""

import json
import signal
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

from livecaption.config import SAMPLE_RATE, WEB_STOP_DRAIN_SEC

SPEECH = "Good morning everyone. [[slnc 1500]] The release is on track for Friday."
DEMO = Path(__file__).with_name("demo_web.py")


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def wait_until(pred, what: str, timeout: float) -> None:  # noqa: ANN001
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if pred():
            return
        time.sleep(0.5)
    raise TimeoutError(f"timed out waiting for {what}")


def meeting_files(root: Path) -> list[Path]:
    return list(root.glob("*/meeting.json"))


def lines_saved(root: Path) -> bool:
    files = meeting_files(root)
    return bool(files) and bool(json.loads(files[0].read_text())["utterances"])


with tempfile.TemporaryDirectory() as td:
    aiff, wav = Path(td) / "speech.aiff", Path(td) / "speech.wav"
    subprocess.run(["say", "-v", "Samantha", "-o", str(aiff), SPEECH], check=True)
    subprocess.run(["afconvert", "-f", "WAVE", "-d", f"LEI16@{SAMPLE_RATE}", "-c", "1",
                    str(aiff), str(wav)], check=True)
    meetings = Path(td) / "meetings"
    port = free_port()
    base = f"http://127.0.0.1:{port}"
    engine = subprocess.Popen(
        [sys.executable, str(DEMO), str(wav), "--meetings", str(meetings),
         "--port", str(port), "--no-open"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    try:
        def up() -> bool:
            try:
                with urllib.request.urlopen(f"{base}/state", timeout=1):
                    return True
            except OSError:
                return False

        wait_until(up, "the engine to serve", 60)
        req = urllib.request.Request(
            f"{base}/start", data=json.dumps({"source": "mic", "lang": "en"}).encode(),
            headers={"Content-Type": "application/json"}, method="POST",
        )
        with urllib.request.urlopen(req, timeout=5) as r:
            assert json.loads(r.read())["ok"], "start refused"
        wait_until(lambda: lines_saved(meetings), "a saved line", 180)

        t0 = time.monotonic()
        engine.send_signal(signal.SIGTERM)
        code = engine.wait(timeout=WEB_STOP_DRAIN_SEC + 30)
        print(f"engine exited with {code} after {time.monotonic() - t0:.1f}s")
    finally:
        if engine.poll() is None:
            engine.kill()

    data = json.loads(meeting_files(meetings)[0].read_text())
    for u in data["utterances"]:
        zh = " ".join(s["zh"] for s in u["zh"]) if u.get("zh") else "(no translation)"
        print(f"  {u['en'][0]['text']}  ->  {zh}")
    assert data["ended_at"], "SIGTERM left the meeting unclosed (ended_at is null)"
    assert all(u.get("zh") for u in data["utterances"]), "lines lost their translation"

print("SIGTERM smoke test PASSED")
