"""Offline ASR accuracy comparison on a labelled speech set (real models; minutes).

Decodes every clip of DATA_DIR with several settings of one engine and prints the mixed error
rate per language subset: edit distance over tokens (each CJK character, each Latin word) after
lower-casing and dropping punctuation / [UNK], divided by the reference length. Meant for
comparing settings of the SAME engine on the same clips, not as an absolute benchmark.

DATA_DIR holds audio files plus manifest.json: [{"file": ..., "text" or "transcription": ...,
"language": "zh" | "en" | "mixed", "duration": seconds}, ...] (e.g. the CAiRE/ASCEND test split
or hf-internal-testing/librispeech_asr_dummy unpacked from their parquet files).
    uv run --extra qwen python scripts/eval_asr.py DATA_DIR --engine qwen
    uv run python scripts/eval_asr.py DATA_DIR --engine nemotron --langs en
"""

import argparse
import json
import random
import re
import sys
import time
from pathlib import Path

from livecaption import config

_TOKEN_RE = re.compile(r"[一-鿿]|[a-z0-9']+")


def tokens(text: str) -> list[str]:
    text = re.sub(r"\[unk\]", " ", text.lower())
    return _TOKEN_RE.findall(text)


def edit_distance(a: list[str], b: list[str]) -> int:
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, y in enumerate(b, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y))
        prev = cur
    return prev[-1]


def pick(manifest: list[dict], langs: list[str], per_lang: int, min_dur: float) -> list[dict]:
    out = []
    for lang in langs:
        # seeded per language: a subset is the same clips whichever --langs / --engine ran
        rng = random.Random(f"eval-{lang}")
        xs = [x for x in manifest
              if x.get("language", "en") == lang and x.get("duration", min_dur) >= min_dur]
        out += rng.sample(xs, min(per_lang, len(xs)))
    return out


def qwen_settings(model_id: str, languages: list[str]) -> dict:
    from mlx_qwen3_asr import Session, load_model

    model, _ = load_model(model_id)
    session = Session(model)

    def decode(language: str | None):  # noqa: ANN202
        return lambda audio: session.transcribe(audio, language=language).text

    # the web UI's Chinese / mixed presets force "Chinese" (asr_lang "zh"); "auto" = detect
    return {
        (f"forced {lang}" if lang != "auto" else "auto language"):
            decode(None if lang == "auto" else lang)
        for lang in languages
    }


def nemotron_settings(model_id: str) -> dict:
    import mlx.core as mx
    from mlx_audio.stt import load as load_stt

    model = load_stt(model_id)

    def decode(att: list[int]):  # noqa: ANN202
        return lambda audio: model.generate(
            mx.array(audio), language="en-US", att_context_size=att
        ).text

    return {
        # offline decode at the streaming look-ahead == what the live (one-pass) path emits
        f"streaming {config.ASR_ATT_CONTEXT}": decode(list(config.ASR_ATT_CONTEXT)),
        f"two-pass {config.ASR_FINAL_ATT_CONTEXT}": decode(list(config.ASR_FINAL_ATT_CONTEXT)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("data_dir", type=Path)
    ap.add_argument("--engine", choices=["qwen", "nemotron"], required=True)
    ap.add_argument("--model", help="HF repo id (default: the web UI's model for the engine)")
    ap.add_argument("--langs", default="zh,mixed,en")
    ap.add_argument("--per-lang", type=int, default=100)
    ap.add_argument("--min-dur", type=float, default=2.0)
    ap.add_argument("--qwen-langs", default="Chinese,auto",
                    help="qwen: languages to force, comma-separated ('auto' = detect)")
    args = ap.parse_args()

    from mlx_audio.stt.utils import load_audio

    manifest = json.loads((args.data_dir / "manifest.json").read_text())
    clips = pick(manifest, args.langs.split(","), args.per_lang, args.min_dur)
    if args.engine == "qwen":
        model_id = args.model or config.ASR_MODELS["qwen3-1.7b-8bit"][1]
        settings = qwen_settings(model_id, args.qwen_langs.split(","))
    else:
        model_id = args.model or config.ASR_MODELS["nemotron-0.6b"][1]
        settings = nemotron_settings(model_id)
    print(f"{model_id}: {len(clips)} clips, settings: {', '.join(settings)}", flush=True)

    errs = {(s, lang): [0, 0] for s in settings for lang in args.langs.split(",")}
    diffs = []
    t0 = time.monotonic()
    for i, clip in enumerate(clips):
        audio = load_audio(str(args.data_dir / clip["file"]), config.SAMPLE_RATE)
        ref = tokens(clip.get("text") or clip.get("transcription") or "")
        if not ref:
            continue
        hyps = {s: fn(audio) for s, fn in settings.items()}
        for s, hyp in hyps.items():
            e = errs[(s, clip.get("language", "en"))]
            e[0] += edit_distance(ref, tokens(hyp))
            e[1] += len(ref)
        if len({" ".join(tokens(h)) for h in hyps.values()}) > 1:
            diffs.append((clip.get("text") or clip.get("transcription"), hyps))
        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(clips)}  {time.monotonic() - t0:.0f}s", flush=True)

    print(f"\nmixed error rate (lower is better), {args.engine}:")
    for lang in args.langs.split(","):
        cells = []
        for s in settings:
            e, n = errs[(s, lang)]
            cells.append(f"{s}: {e / n:6.1%}" if n else f"{s}: -")
        print(f"  {lang:>5}  " + "   ".join(cells))
    print(f"\n{len(diffs)} clips decoded differently; first 12:")
    for ref, hyps in diffs[:12]:
        print(f"  ref: {ref}")
        for s, h in hyps.items():
            print(f"    {s:>24}: {h}")


if __name__ == "__main__":
    sys.exit(main())
