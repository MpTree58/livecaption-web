"""Manual smoke test for the qwen3 ASR backend: synthesize Chinese speech with macOS `say`,
verify the QwenOnlineStream streaming path emits a sensible final.

Checks four things end to end:
1. build_qwen_recognizer loads Qwen3-ASR (mlx-qwen3-asr) + Silero VAD (auto-downloaded)
2. QwenOnlineStream fed in 100ms real-time chunks emits partial / final (VAD endpointing)
3. The final text overlaps the spoken sentence (path is wired correctly, Chinese decodes)
4. Decoding (incl. the two-pass re-decode) resolves no Hugging Face repo at all -- it must
   use the loaded model's own files. Only meaningful for a model other than the package
   default Qwen/Qwen3-ASR-0.6B (the two-pass once fetched that whole repo for a tokenizer),
   e.g. the web UI's Six666/mlx-qwen3-asr-1.7b-8bit.

Needs the optional [qwen] extra. Run with either:
    uv run --extra qwen python scripts/smoke_qwen.py
    uv run --with mlx-qwen3-asr python scripts/smoke_qwen.py
Pass a model id to override (default uses the smaller 0.6B for speed):
    uv run --with mlx-qwen3-asr python scripts/smoke_qwen.py Six666/mlx-qwen3-asr-1.7b-8bit
"""

import subprocess
import sys
import tempfile
import wave
from pathlib import Path

import huggingface_hub
import numpy as np

from livecaption.asr_qwen import build_qwen_recognizer
from livecaption.config import SAMPLE_RATE

# mlx-qwen3-asr resolves repo ids through huggingface_hub.snapshot_download (imported at call
# time), so wrapping it records every repo the decode path touches
resolved_repos: list[str] = []
_snapshot_download = huggingface_hub.snapshot_download


def _recording_snapshot_download(*args, **kwargs):  # noqa: ANN002, ANN003
    resolved_repos.append(kwargs.get("repo_id") or args[0])
    return _snapshot_download(*args, **kwargs)


huggingface_hub.snapshot_download = _recording_snapshot_download

TEXT = "今天天气很好，我们一起去公园散步吧。"
MODEL = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen3-ASR-0.6B"


def synthesize(text: str, out_wav: Path) -> None:
    """say -> aiff -> afconvert to 16k mono s16 wav (all macOS built-in tools)."""
    aiff = out_wav.with_suffix(".aiff")
    # Tingting is the bundled zh_CN voice; fall back to the system default if absent
    try:
        subprocess.run(["say", "-v", "Tingting", "-o", str(aiff), text], check=True)
    except subprocess.CalledProcessError:
        subprocess.run(["say", "-o", str(aiff), text], check=True)
    subprocess.run(
        ["afconvert", "-f", "WAVE", "-d", f"LEI16@{SAMPLE_RATE}", "-c", "1",
         str(aiff), str(out_wav)],
        check=True,
    )


def read_wav(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as wf:
        assert wf.getframerate() == SAMPLE_RATE, wf.getframerate()
        raw = wf.readframes(wf.getnframes())
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0


def char_overlap(a: str, b: str) -> float:
    """CJK has no word boundaries; compare on the set of non-punctuation characters."""
    drop = set(" ，。！？、,.!?")
    sa = {c for c in a if c not in drop}
    sb = {c for c in b if c not in drop}
    return len(sa & sb) / max(len(sa), 1)


def _seg_text(segments: list) -> str:  # final events carry [(speaker, text, diff), ...]
    return " ".join(t for _spk, t, _d in segments)


with tempfile.TemporaryDirectory() as td:
    wav = Path(td) / "test.wav"
    synthesize(TEXT, wav)
    samples = read_wav(wav)
    print(f"synthesized {len(samples) / SAMPLE_RATE:.1f}s of Chinese audio")

    rec = build_qwen_recognizer(MODEL, language="Chinese", log=print)
    print(f"recognizer built OK ({MODEL} + silero-vad)")
    resolved_repos.clear()  # loading may resolve repos; decoding below must not

    # Simulate real-time: 100ms per chunk + 3s trailing silence so rule2 flushes the last
    # sentence (mirrors the live MicSource block size).
    stream = rec.create_stream()
    feed = np.concatenate([samples, np.zeros(SAMPLE_RATE * 3, dtype=np.float32)])
    step = int(0.1 * SAMPLE_RATE)
    partials = 0
    finals: list[str] = []
    for i in range(0, len(feed), step):
        for ev in stream.accept_waveform(feed[i : i + step]):
            if ev[0] == "partial":
                partials += 1
            elif ev[0] == "final":
                finals.append(_seg_text(ev[1]))
    finals += [_seg_text(ev[1]) for ev in stream.flush() if ev[0] == "final"]
    print(f"streamed: {' | '.join(finals) if finals else '(no final)'}  ({partials} partials)")

    assert finals, "streaming produced no final text"
    streamed = " ".join(finals)
    ratio = char_overlap(TEXT, streamed)
    print(f"char overlap vs spoken text: {ratio:.0%}")
    assert ratio >= 0.5, f"streamed text diverges from spoken text ({ratio:.0%})"
    assert not resolved_repos, f"decoding resolved Hugging Face repos: {resolved_repos}"

print("qwen3 ASR smoke test PASSED")
