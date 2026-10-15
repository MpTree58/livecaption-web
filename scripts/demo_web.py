"""Dev harness: the real web UI + models, with audio files standing in for the capture devices.

Runs livecaption.webapp unchanged except for its inputs: the microphone ("me") plays MIC_FILE
and system audio ("them") plays --them FILE (default: MIC_FILE), both paced to real time and
looped with a short pause, so the page can be exercised end to end -- partials, finals,
diarization, translation, meeting records -- without speaking, granting mic / system-audio
permission, or building audiotee. Meetings go to a throwaway directory unless --meetings DIR
is given, never ~/.cache/livecaption. Other options are passed through to the web app.
    uv run --extra qwen python scripts/demo_web.py speech.wav --port 8765 --no-open
    uv run --extra qwen python scripts/demo_web.py en.wav --them zh.wav --meetings /tmp/m
"""

import sys
import tempfile
import time
from pathlib import Path

import numpy as np

from livecaption import webapp
from livecaption.audio import FileSource
from livecaption.config import SAMPLE_RATE

LOOP_GAP_SEC = 3.0  # silence between repetitions, long enough for an endpoint


class LiveFileSource(FileSource):
    """A file played like a live device: real-time pace, never blocks (drops like the mic
    does), loops forever -- a capture device doesn't end, so the run stays "listening"."""

    def _feed(self) -> None:
        from mlx_audio.stt.utils import load_audio

        clip = np.array(load_audio(self.path, SAMPLE_RATE), dtype=np.float32)
        audio = np.concatenate([clip, np.zeros(int(LOOP_GAP_SEC * SAMPLE_RATE), np.float32)])
        t0, sent = time.monotonic(), 0
        while not self._stop.is_set():
            for i in range(0, len(audio), self.blocksize):
                if self._stop.is_set():
                    break
                chunk = audio[i : i + self.blocksize]
                self._offer(chunk)
                sent += len(chunk)
                time.sleep(max(0.0, t0 + sent / SAMPLE_RATE - time.monotonic()))
        self._put_sentinel()


class LiveSystemSource(LiveFileSource):
    """Stands in for SystemAudioSource (same constructor shape; the audiotee path is unused)."""

    path_for_them = ""

    def __init__(self, _audiotee_bin: str, label: str = "them") -> None:
        super().__init__(self.path_for_them, label=label)


def _pop_option(args: list[str], name: str) -> str | None:
    if name not in args:
        return None
    i = args.index(name)
    value = args[i + 1]
    del args[i : i + 2]
    return value


if __name__ == "__main__":
    args = sys.argv[1:]
    if not args or args[0].startswith("-"):
        sys.exit(__doc__)
    mic_file = str(Path(args.pop(0)).resolve())
    them_file = str(Path(_pop_option(args, "--them") or mic_file).resolve())
    meetings = _pop_option(args, "--meetings") or tempfile.mkdtemp(prefix="livecaption-demo-")

    webapp.MicSource = lambda label="me": LiveFileSource(mic_file, label=label)
    LiveSystemSource.path_for_them = them_file
    webapp.SystemAudioSource = LiveSystemSource  # a class: webapp also isinstance()-checks it
    webapp.resolve_audiotee = lambda _explicit: "audiotee (demo: not used)"
    webapp._MEETINGS_DIR = Path(meetings)
    print(f"demo: mic <- {mic_file}\ndemo: system <- {them_file}\ndemo: meetings -> {meetings}")

    sys.argv = [sys.argv[0], *args]
    webapp.run()
