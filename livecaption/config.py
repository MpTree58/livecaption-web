"""Default configuration constants.

All tunable parameters live here so they can be adjusted after real-world testing.
"""

from __future__ import annotations

# ---- Model registries (the closed --asr-model / --mt-model choice lists) ----
# Alias -> (ASR engine, HF repo id). The CLI exposes these aliases as a strict choice list and
# derives the engine from the chosen alias -- there is no separate --asr-backend (see cli.py).
#   nemotron: mlx-audio streaming transducer (English-strong, live word partials + diarization)
#   qwen3:    Qwen3-ASR via the optional `mlx-qwen3-asr` package (strong Chinese; install [qwen])
# The qwen3 *-8bit entries are our own conversions; the mlx-community/Qwen3-ASR-*-8bit repos use
# mlx-audio's layer naming and do NOT load in mlx-qwen3-asr.
ASR_MODELS: dict[str, tuple[str, str]] = {
    "nemotron-0.6b":      ("nemotron", "mlx-community/nemotron-3.5-asr-streaming-0.6b"),
    "nemotron-0.6b-8bit": ("nemotron", "mlx-community/nemotron-3.5-asr-streaming-0.6b-8bit"),
    "qwen3-1.7b":         ("qwen3", "Qwen/Qwen3-ASR-1.7B"),
    "qwen3-1.7b-8bit":    ("qwen3", "Six666/mlx-qwen3-asr-1.7b-8bit"),
    "qwen3-0.6b":         ("qwen3", "Qwen/Qwen3-ASR-0.6B"),
    "qwen3-0.6b-8bit":    ("qwen3", "Six666/mlx-qwen3-asr-0.6b-8bit"),
}
DEFAULT_ASR_MODEL_ALIAS = "nemotron-0.6b"

# Translation: Hy-MT2 (Tencent Hunyuan MT, 3rd gen). 1.8B-8bit ~2GB (default); the 7B builds
# (~4.2GB) translate more accurately. Alias -> HF repo id.
MT_MODELS: dict[str, str] = {
    "hy-mt2-1.8b-8bit": "mlx-community/Hy-MT2-1.8B-8bit",
    "hy-mt2-1.8b-4bit": "mlx-community/Hy-MT2-1.8B-4bit",
    "hy-mt2-7b-4bit":   "mlx-community/Hy-MT2-7B-4bit",
    "hy-mt2-7b-8bit":   "mlx-community/Hy-MT2-7B-8bit",
}
DEFAULT_MT_MODEL_ALIAS = "hy-mt2-1.8b-8bit"

# Kept for scripts that import them directly (smoke_asr / smoke_mel / smoke_translate /
# spike_qwen3), bypassing the CLI enums. Derived so the registries stay the single source.
DEFAULT_ASR_MODEL = ASR_MODELS[DEFAULT_ASR_MODEL_ALIAS][1]
DEFAULT_MT_MODEL = MT_MODELS[DEFAULT_MT_MODEL_ALIAS]

# ---- Audio ----
SAMPLE_RATE = 16000  # ASR / VAD / audiotee all use 16k mono
# System audio stall watchdog: a healthy audiotee tap delivers PCM continuously (zeros
# during silence), so receiving NOTHING for this many seconds means the tap died -- the
# observed case is a default-output-device switch (audio keeps playing on the new device
# while the tap stays on the old one, whose IO stops). The source then kills and
# respawns audiotee, which re-taps the current default device, so captions resume.
SYSTEM_AUDIO_STALL_SEC = 5.0
# File/podcast transcription sanity check: warn when the last finalized utterance is this far
# before the end of the audio. A small gap can be outro music or silence; a large one usually
# means the run was interrupted or the recognizer stopped emitting useful speech.
FILE_TRAILING_GAP_WARN_SEC = 300.0

# ---- ASR (mlx-audio nemotron-3.5) ----
# English by default; CLI --asr-lang overrides it. The CLI accepts lowercase tags such as
# "en-us" and English language names such as "English"; the ASR layer normalizes them to the
# exact model prompt key and lists all supported locales when a value is invalid.
# "auto" lets the model detect the language, but with mixed Chinese/English in meetings the
# detected language jumps around, so it's not recommended.
ASR_LANGUAGE = "en-us"
# Streaming look-ahead [left, right]: right+1 downsampled frames = how big a chunk is fed at
# once = partial refresh granularity.
# Settings the model was trained on: [56,0]=80ms  [56,3]=320ms  [56,6]=560ms  [56,13]=1120ms
# (best accuracy)
ASR_ATT_CONTEXT = [56, 6]
# two-pass correction: at finalization, re-decode the whole sentence with the largest
# look-ahead; final (and the translation input) use the re-decoded result -- partial trades
# for speed (each frame sees only 480ms of future), final re-decodes for accuracy (each frame
# sees 1.12s of future).
ASR_TWO_PASS = True
ASR_FINAL_ATT_CONTEXT = [56, 13]

# ---- qwen3 engine notes (selected via the qwen3-* aliases in ASR_MODELS above) ----
# The qwen3 engine's streaming path emits no token timestamps, so partials refresh per chunk
# (~QWEN_CHUNK_SIZE_SEC) rather than per word and carry no live speaker label; diarization is
# utterance-level by default (one speaker per sentence), upgraded to nemotron-style
# mid-sentence split only when QWEN_WORD_DIARIZE is on (see below). fp16 1.7B ~4.7GB; the
# qwen3-*-8bit aliases ~2.4GB / ~1.0GB.
QWEN_CHUNK_SIZE_SEC = 2.0  # rolling-decode chunk size ~= partial refresh granularity
QWEN_MAX_CONTEXT_SEC = 30.0  # rolling context window (trimmed beyond this; linear cost)
QWEN_FINALIZATION_MODE = "accuracy"  # "accuracy" (tail refine) | "latency"
# two-pass: at finalization, re-decode the whole utterance OFFLINE (full context, no chunk
# boundaries) and use that as the final + translation input -- fixes the spurious sentence
# breaks the 2s-chunked rolling decode can insert mid-utterance. Partials still show the
# streaming result; the inline diff highlights what the re-decode changed. Costs one full
# re-decode per sentence (~0.5s at 0.6B, ~1.5s at 1.7B). Set False to skip.
QWEN_TWO_PASS = True
# Word-level diarization for the qwen3 backend (parity with nemotron's mid-sentence speaker
# split). The Qwen3-ASR decoder emits no token timestamps, so we get them from the separate
# Qwen3-ForcedAligner model: at finalization the offline re-decode is run with
# return_timestamps=True, each aligned word is mapped to a Sortformer frame speaker, and the
# text is sliced at speaker-change word boundaries into one final per speaker. On by default
# (the qwen3 backend's diarization is only worth much with it), at the cost of a per-utterance
# forced-alignment pass + an extra ~0.6B model (DEFAULT_QWEN_ALIGNER_MODEL). Disable with
# --no-qwen-word-diarize to fall back to utterance-level (one speaker per sentence) -- the
# alignment pass is pure overhead for mostly single-speaker stretches. Needs --diarize. This
# flag is qwen3-only; passing it with another backend is an error.
QWEN_WORD_DIARIZE = True
DEFAULT_QWEN_ALIGNER_MODEL = "Qwen/Qwen3-ForcedAligner-0.6B"

# ---- Speaker diarization (Sortformer v2.1 streaming; disable with --no-diarize) ----
# At finalization the whole sentence is fed at once (the model's native operating point is 15s
# chunks; feeding small streaming chunks makes speaker identity drift), then text is split and
# attributed to each speaker by RNNT token timestamps, with a separate final + translation per
# segment.
DIAR_MODEL = "mlx-community/diar_streaming_sortformer_4spk-v2.1-fp16"
DIAR_THRESHOLD = 0.5  # frame-level (80ms) speaker-activity probability threshold
# how often to do a read-only probe of the current sentence, giving the partial a tentative
# speaker label
DIAR_PEEK_SEC = 2.0
# Smoothing: a speaker turn shorter than this (seconds) inside one utterance is absorbed into
# the surrounding speaker instead of becoming its own segment. Makes diarization less twitchy --
# a brief probability blip or a one-word backchannel won't split one speaker's sentence into a
# fragment, and since each speaker segment becomes a separate final, it also yields fewer tiny
# finals. Raise it to be stickier; set 0 to honor every momentary speaker change.
DIAR_MIN_SEGMENT_SEC = 1.0

# ---- VAD / endpoint detection (Silero VAD mlx build, one speech probability every 32ms) ----
VAD_MODEL = "mlx-community/silero-vad"
VAD_THRESHOLD = 0.5  # speech-probability threshold; >= counts as a speech frame
# audio padded back in when speech onset is detected, to keep the first word from being clipped
VAD_PRE_ROLL_MS = 320
# Three rules in an OR relationship; any one satisfied marks the end of a sentence (following
# sherpa-onnx semantics). All values are in seconds.
# trailing-silence threshold when there is no decoded text yet: only resets, does not emit a final
RULE1_MIN_TRAILING_SILENCE = 2.4
# rule2 is punctuation-aware. The nemotron RNNT emits . ? ! inline, so we let "is the
# sentence already complete?" decide how long a pause must be before we cut:
#   - text already ends with . ? ! -> a short pause is enough (the boundary lands on a real
#     sentence end instead of mid-word)
#   - no terminal punctuation yet -> wait the longer silence, so a mid-sentence thinking
#     pause doesn't fragment one utterance into pieces
RULE2_PUNCT_SILENCE = 0.6  # trailing silence to cut when the text ends a sentence
RULE2_MIN_TRAILING_SILENCE = 1.2  # trailing silence to cut when it does not
# Soft max for fast continuous speech (podcasts, lively meetings), where pauses rarely
# reach even RULE2_PUNCT_SILENCE and nearly every utterance would otherwise run into the
# rule3 force-cut -- splitting mid-sentence, which wrecks readability and feeds the
# translator dangling fragments. Once the utterance is this many seconds long, it is cut
# at the next decoded sentence-final punctuation WITHOUT waiting for silence: the audio
# is split retroactively at the punctuation token's timestamp and the remainder (decode
# look-ahead that may already hold the next sentence's onset) carries over into the next
# utterance, so nothing is clipped. (A silence-based escalation can't work here: the
# ~560ms decode look-ahead means the punctuation appears in the text only after a short
# pause is already over.)
RULE2_SOFT_MAX_UTTERANCE = 8.0
RULE3_MIN_UTTERANCE_LENGTH = 20.0  # max seconds before a sentence is force-cut

# ---- Translation (Hy-MT2's officially recommended prompt and sampling parameters) ----
TRANSLATE_PROMPT = (
    "Translate the following text into {target_lang}. "
    "Note that you should only output the translated result "
    "without any additional explanation:\n\n{text}"
)
# Translation with context: uses Hy-MT2's official "background information" template verbatim
# (straight from the README, unmodified).
# Preceding sentences serve as background so the model can resolve references / terminology /
# topic.
# number of preceding sentences attached when translating (default 3 for better coherence;
# 0 = sentence-by-sentence with no context)
MT_CONTEXT_SENTENCES = 3
# Sources shorter than this many words are translated WITHOUT the background template:
# a tiny fragment gives the model almost nothing to anchor on, and the output then tends
# to drift into translating the background block itself instead of the source text.
# Words are whitespace-split, so Chinese / mixed sources (few or no spaces) practically never
# get the background -- keep it that way: on WMT20 zh->en the background template scored
# 29.7 -> 27.6 BLEU (scripts/eval_mt.py --context 3).
MT_CONTEXT_MIN_WORDS = 5
# Live translation preview: while a sentence is still being spoken, show a provisional ZH line
# under the EN partial, refreshed once the in-progress text has grown by this many words AND
# contains a sentence-ender (.?!). Debounced so we don't retranslate every chunk; the final
# two-pass-corrected re-translation still replaces it. Set 0 to disable the live preview.
MT_PREVIEW_MIN_WORDS = 6
TRANSLATE_PROMPT_WITH_CONTEXT = (
    "[Background Information]\n"
    "{context}\n\n"
    "Please translate the following text into {target_lang}, "
    "taking the provided background information into consideration.\n\n"
    "[Source Text]\n{text}"
)
# The README gives every template in Chinese too (with Chinese language names), verbatim below.
# Into Chinese they translate measurably better than the English ones, so translate.py picks
# the prompt by the TARGET language (scripts/eval_mt.py, 150 subtitle-length sentences,
# BLEU): WMT20 en->zh 40.5 -> 42.1 without context and 37.6 -> 40.8 with it; IWSLT17 (TED)
# en->zh likewise. Into English the English template stays: the Chinese one scored
# 29.7 -> 25.6 on WMT20 zh->en.
TRANSLATE_PROMPT_ZH = (
    "将以下文本翻译为{target_lang}，"
    "注意只需要输出翻译后的结果，不要额外解释：\n\n{text}"
)
TRANSLATE_PROMPT_WITH_CONTEXT_ZH = (
    "【背景信息】\n"
    "{context}\n\n"
    "请结合背景信息将以下文本翻译为{target_lang}。\n\n"
    "【待翻译文本】\n{text}"
)
# Terminology template (README "Terminology", both versions verbatim), for a sentence that
# contains glossary terms (glossary.py): {terms} is one TERM_LINE per pair. There is no
# official template combining terms with background, so such a sentence goes without it.
TRANSLATE_PROMPT_TERMS = (
    "Reference the following translations:\n{terms}\n\n"
    "Translate the following text into {target_lang}. "
    "Note that you must ONLY output the translated result "
    "without any additional explanation:\n\n{text}"
)
TERM_LINE = "{source} translates to {target}"
TRANSLATE_PROMPT_TERMS_ZH = (
    "参考下面的翻译：\n{terms}\n"
    "将以下文本翻译为{target_lang}，"
    "注意只需要输出翻译后的结果，不要额外解释：\n\n{text}"
)
TERM_LINE_ZH = "{source} 翻译成 {target}"
MT_TEMPERATURE = 0.7
MT_TOP_P = 0.6
MT_TOP_K = 20
MT_REPETITION_PENALTY = 1.05
# plenty for a single subtitle line; just an upper-bound safeguard against the model running away
MT_MAX_TOKENS = 256
DEFAULT_TARGET_LANG = "zh-cn"

# ---- Web UI session (webapp.py) ----
# On Stop, the translator is usually a sentence or two behind the ASR. The meeting is only
# closed once those last lines are translated (the UI shows "Stopping…" meanwhile), waiting at
# most this long -- past it the remaining lines are saved untranslated, with a warning.
WEB_STOP_DRAIN_SEC = 30.0
# Minutes after a Stop before the idle server frees its models (webapp --idle-unload; 0 = keep
# them loaded). The menu-bar app keeps the server running from login, and a warm preset holds
# 3.5-4.7 GB of unified memory; reloading from the disk cache on the next Start takes ~2-4 s.
WEB_IDLE_UNLOAD_MIN = 5.0
