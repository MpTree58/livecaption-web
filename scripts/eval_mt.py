"""Offline translation comparison: Hy-MT2 prompt / decoding settings on a parallel test set.

Translates SRC with each setting and writes one hypothesis file per setting next to OUT_PREFIX,
for scoring with sacrebleu (not a project dependency), e.g.:
    uv run python scripts/eval_mt.py wmt20.en-zh.src --target zh-cn --out /tmp/mt/enzh
    uvx sacrebleu wmt20.en-zh.ref -i /tmp/mt/enzh.*.txt -m bleu chrf --tokenize zh
Sentences are sampled (fixed seed) among subtitle-length ones: --max-len words for a Latin
source, twice that many characters for a CJK one. Settings:
  en-prompt   config.TRANSLATE_PROMPT (Hy-MT2's official English template), official sampling
  zh-prompt   config.TRANSLATE_PROMPT_ZH (the official Chinese template), official sampling
  greedy      English template, temperature 0
  en-prompt.seed1  the default setting again with another seed: the run-to-run noise floor
With --context N, the background templates instead (en-ctx / zh-ctx / en-ctx.seed1), given the
N preceding source lines as the live translator gives its last N sentences. Add --paired-bs
to the sacrebleu call to see which differences are beyond chance.
"""

import argparse
import random
import re
import time
from pathlib import Path

import mlx.core as mx
from mlx_lm import load, stream_generate
from mlx_lm.sample_utils import make_logits_processors, make_sampler

from livecaption import config
from livecaption.languages import normalize_target_language

# the README's Chinese templates take the target language's Chinese name
ZH_NAMES = {"Simplified Chinese": "简体中文", "English": "英语"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("src", type=Path)
    ap.add_argument("--target", required=True, help="e.g. zh-cn / en")
    ap.add_argument("--out", type=Path, required=True, help="output path prefix")
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--max-len", type=int, default=30)
    ap.add_argument("--model", default=config.DEFAULT_MT_MODEL)
    ap.add_argument("--context", type=int, default=0,
                    help="compare the background templates instead, with the N preceding source "
                    "lines (same document in WMT) as background, as the live translator does")
    args = ap.parse_args()

    target = normalize_target_language(args.target).prompt_name
    lines = args.src.read_text(encoding="utf-8").splitlines()

    def short(s: str) -> bool:
        cjk = re.search(r"[一-鿿]", s)
        return 0 < (len(s) if cjk else len(s.split())) <= args.max_len * (2 if cjk else 1)

    idx = sorted(random.Random(0).sample([i for i, s in enumerate(lines) if short(s)], args.n))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    (args.out.parent / f"{args.out.name}.idx").write_text("\n".join(map(str, idx)))

    model, tok = load(args.model)
    sampled = make_sampler(temp=config.MT_TEMPERATURE, top_p=config.MT_TOP_P, top_k=config.MT_TOP_K)
    greedy = make_sampler(temp=0.0)
    procs = make_logits_processors(repetition_penalty=config.MT_REPETITION_PENALTY)

    def en_prompt(text: str) -> str:
        return config.TRANSLATE_PROMPT.format(target_lang=target, text=text)

    def zh_prompt(text: str) -> str:
        return config.TRANSLATE_PROMPT_ZH.format(target_lang=ZH_NAMES[target], text=text)

    def background(i: int) -> str:
        return " ".join(lines[max(0, i - args.context) : i])

    def en_ctx(i: int) -> str:
        return config.TRANSLATE_PROMPT_WITH_CONTEXT.format(
            context=background(i), target_lang=target, text=lines[i])

    def zh_ctx(i: int) -> str:
        return config.TRANSLATE_PROMPT_WITH_CONTEXT_ZH.format(
            context=background(i), target_lang=ZH_NAMES[target], text=lines[i])

    if args.context > 0:
        settings = {
            "en-ctx": (en_ctx, sampled, 0),
            "zh-ctx": (zh_ctx, sampled, 0),
            "en-ctx.seed1": (en_ctx, sampled, 1),
        }
    else:
        settings = {
            "en-prompt": (lambda i: en_prompt(lines[i]), sampled, 0),
            "zh-prompt": (lambda i: zh_prompt(lines[i]), sampled, 0),
            "greedy": (lambda i: en_prompt(lines[i]), greedy, 0),
            "en-prompt.seed1": (lambda i: en_prompt(lines[i]), sampled, 1),
        }
    for name, (build, sampler, seed) in settings.items():
        mx.random.seed(seed)
        t0, outs = time.monotonic(), []
        for i in idx:
            prompt = tok.apply_chat_template(
                [{"role": "user", "content": build(i)}], add_generation_prompt=True
            )
            text = "".join(r.text for r in stream_generate(
                model, tok, prompt=prompt, max_tokens=config.MT_MAX_TOKENS, sampler=sampler,
                logits_processors=procs,
            ))
            outs.append(" ".join(text.split()))
        (args.out.parent / f"{args.out.name}.{name}.txt").write_text("\n".join(outs) + "\n")
        print(f"{name}: {len(outs)} lines in {time.monotonic() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
