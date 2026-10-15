"""Alternate ASR backend: Qwen3-ASR (Apple GPU / MLX) via the `mlx-qwen3-asr` package.

Chosen for Chinese-heavy meetings, where the nemotron transducer is weak. Qwen3-ASR is an
attention encoder-decoder; `mlx-qwen3-asr` (moona3k) is a ground-up MLX reimplementation
with a real incremental-audio streaming API (`init_streaming` / `feed_audio` /
`finish_streaming`, a rolling decode with a stabilizing prefix). We wrap that behind the
same OnlineStream contract `AsrWorker` consumes, and reuse the existing Silero VAD for
endpointing (rule1/2/3 in config) so silence never enters the decoder.

Speaker diarization (`--diarize`) does a nemotron-style **mid-sentence split by default**
(config.QWEN_WORD_DIARIZE): the qwen3 streaming path emits no token timestamps, so they are
recovered from the separate Qwen3-ForcedAligner model -- the offline re-decode is run with
return_timestamps=True, each aligned word is mapped to a Sortformer frame speaker, and the
text is sliced at speaker-change word boundaries into one final per speaker (see
_attribute_speakers). That costs an extra forced-alignment pass + the ~0.6B aligner model;
`--no-qwen-word-diarize` falls back to **utterance-level** (feed the whole sentence to
Sortformer, label it with its majority speaker -- one S-number per sentence). A speaker change
with a pause is split into separate utterances by rule2 regardless.
It DOES do a two-pass correction (config.QWEN_TWO_PASS): at finalization the whole utterance
is re-decoded OFFLINE (full context, no 2s chunk boundaries, fixing the spurious mid-sentence
breaks the rolling decode can insert), and the inline diff vs the streaming text renders like
nemotron's (CJK-aware, see asr.py:_inline_diff). See asr.py for the nemotron backend.

`mlx-qwen3-asr` is an optional dependency (the `[qwen]` extra); it is imported lazily so a
nemotron-only install never needs it. Model weights load once: we keep the loaded model
object, pass it into every feed/finish call, and run offline re-decodes through a
`Session` bound to it -- so no per-utterance reload, and nothing parked in the package's
process-global model cache (dropping the recognizer really frees the weights).
"""

from __future__ import annotations

import contextlib
from collections import deque
from collections.abc import Callable

import mlx.core as mx
import numpy as np

from . import config
from .asr import _inline_diff
from .runtime import MLX_LOCK as _MLX_LOCK

_VAD_FRAME = 512  # Silero fixed 32ms @ 16k (mirrors asr.py)
_VAD_FRAME_SEC = _VAD_FRAME / config.SAMPLE_RATE
# Sentence-final punctuation, ASCII + the CJK full-width forms Qwen3-ASR emits for Chinese.
# Used (as in nemotron's rule2) to decide whether a short trailing pause is enough to cut.
_SENT_ENDERS = ".?!。？！"

# Map common language tags / names to the names Qwen3-ASR expects. moona3k accepts the
# English language name case-insensitively; we normalize tags (zh-cn) and short codes (zh)
# to it. "auto" / unknown -> None (let the model auto-detect; not recommended for mixed
# zh/en meetings, see config.ASR_LANGUAGE).
_QWEN_LANG_ALIASES = {
    "en": "English", "en-us": "English", "en-gb": "English", "english": "English",
    "zh": "Chinese", "zh-cn": "Chinese", "zh-hans": "Chinese", "cmn": "Chinese",
    "chinese": "Chinese", "mandarin": "Chinese",
    "yue": "Cantonese", "zh-yue": "Cantonese", "cantonese": "Cantonese",
    "ja": "Japanese", "ja-jp": "Japanese", "japanese": "Japanese",
    "ko": "Korean", "ko-kr": "Korean", "korean": "Korean",
    "de": "German", "de-de": "German", "german": "German",
    "fr": "French", "fr-fr": "French", "french": "French",
    "es": "Spanish", "es-es": "Spanish", "spanish": "Spanish",
    "it": "Italian", "italian": "Italian",
    "pt": "Portuguese", "portuguese": "Portuguese",
    "ru": "Russian", "russian": "Russian",
}


def _resolve_qwen_language(language: str | None) -> str | None:
    if not language or language.strip().lower() == "auto":
        return None
    key = language.strip().lower()
    if key in _QWEN_LANG_ALIASES:
        return _QWEN_LANG_ALIASES[key]
    # Pass an unrecognized value through title-cased; the package validates and the spoken
    # language is the user's call (mirrors nemotron letting the model list its locales).
    return language.strip().title()


def _word_offsets(words: list[dict], text: str) -> list[tuple[int, int]]:
    """Locate each aligned word in `text`, returning its (start, end) character offsets.

    The forced aligner's words are a tokenization OF `text` (CJK per char, Latin per word,
    bare tokens with no spacing), so an in-order forward search recovers where each lives in
    the canonical text. A word that can't be found (rare -- e.g. the aligner stripped edge
    punctuation) collapses to a zero-width offset at the current cursor, so it simply joins
    the current speaker group; the search never moves backwards, keeping offsets monotonic."""
    offsets: list[tuple[int, int]] = []
    pos = 0
    low = text.lower()
    for w in words:
        tok = str(w.get("text", "")).strip()
        if not tok:
            offsets.append((pos, pos))
            continue
        found = text.find(tok, pos)
        if found < 0:  # casefold fallback (the re-decode text and tokens should match case)
            found = low.find(tok.lower(), pos)
        if found < 0:
            offsets.append((pos, pos))
        else:
            offsets.append((found, found + len(tok)))
            pos = found + len(tok)
    return offsets


def _segments_from_speaker_seq(
    words: list[dict], spk_seq: list[int | None], text: str
) -> list[tuple[int | None, str]]:
    """Slice `text` into one segment per maximal same-speaker word run, returning
    [(speaker, segment_text), ...].

    `spk_seq` is the per-word speaker label (parallel to `words`). Words are located in `text`
    by an in-order search (_word_offsets); a group boundary is the start offset of each run's
    first word, so the cut keeps `text`'s spacing/punctuation verbatim (the concatenation of
    the returned segment texts is `text` itself, modulo stripped edge whitespace). The first
    group always starts at 0 so any leading text is kept. Falls back to one unlabeled segment
    when there are no words. Trailing punctuation stays with the preceding group because it
    sits before the next group's first word."""
    if not words:
        return [(None, text)]
    offsets = _word_offsets(words, text)
    group_spk: list[int | None] = []
    group_start: list[int] = []
    sentinel = object()
    prev: object = sentinel
    for (start_char, _end), s in zip(offsets, spk_seq, strict=True):
        if s != prev:
            group_spk.append(s)
            group_start.append(start_char)
            prev = s
    if group_start:
        group_start[0] = 0
    segs: list[tuple[int | None, str]] = []
    for i, s in enumerate(group_spk):
        a = group_start[i]
        b = group_start[i + 1] if i + 1 < len(group_start) else len(text)
        seg = text[a:b].strip()
        if seg:
            segs.append((s, seg))
    return segs or [(None, text)]


class QwenRecognizer:
    """Shared model bundle: Qwen3-ASR weights (loaded once) + Silero VAD. No decode state,
    so it can back multiple streams (each create_stream() gets its own session state)."""

    def __init__(
        self,
        asr_model: str,
        language: str = config.ASR_LANGUAGE,
        diarize: bool = False,
        word_diarize: bool = False,
        log: Callable[[str], None] | None = None,
    ):
        from mlx_audio.vad import load as load_vad
        from mlx_qwen3_asr import Session, load_model

        say = log or (lambda _m: None)
        self.model_id = asr_model
        self.language = _resolve_qwen_language(language)
        say(f"Loading ASR model {asr_model} (Qwen3-ASR via mlx-qwen3-asr) …")
        # load_model -> (model, config); keep the model object to pass into feed/finish so
        # the decoder never reloads weights between utterances.
        self.model_obj, self._model_config = load_model(asr_model)
        # Offline re-decodes (two-pass, word timestamps) go through an explicit Session bound
        # to this model: the module-level transcribe(model=<loaded object>) takes its
        # tokenizer from the package default Qwen/Qwen3-ASR-0.6B instead, so the first final
        # downloaded that whole 1.8 GB repo (while holding MLX_LOCK, freezing captions and
        # translation) and, offline, made every two-pass silently fall back to the streaming
        # text. Session reads the tokenizer from this model's own snapshot.
        self.session = Session(self.model_obj)
        say(f"Loading VAD model {config.VAD_MODEL} …")
        self.vad = load_vad(config.VAD_MODEL)
        # Sortformer for diarization (it loads via the VAD loader too).
        self.diar = None
        # Forced aligner for word-level (mid-sentence) speaker split; only loaded when both
        # diarization and word-diarize are on. Costs an extra ~0.6B model + a per-utterance
        # alignment pass, so it stays opt-in -- see config.QWEN_WORD_DIARIZE.
        self.aligner = None
        if diarize:
            say(f"Loading diarization model {config.DIAR_MODEL} …")
            self.diar = load_vad(config.DIAR_MODEL)
            if word_diarize:
                from mlx_qwen3_asr import ForcedAligner

                say(
                    f"Loading forced aligner {config.DEFAULT_QWEN_ALIGNER_MODEL} "
                    "(word-level diarization) …"
                )
                self.aligner = ForcedAligner(config.DEFAULT_QWEN_ALIGNER_MODEL)
        say("Warming up models …")
        self._warmup()

    def _warmup(self) -> None:
        """Run one short decode + one VAD frame on silence to absorb Metal kernel
        compilation now, so the first real sentence's partial doesn't stall."""
        from mlx_qwen3_asr.streaming import feed_audio, finish_streaming, init_streaming

        silence = np.zeros(int(0.5 * config.SAMPLE_RATE), dtype=np.float32)
        with _MLX_LOCK:
            state = init_streaming(
                model=self.model_id,
                chunk_size_sec=config.QWEN_CHUNK_SIZE_SEC,
                max_context_sec=config.QWEN_MAX_CONTEXT_SEC,
                language=self.language,
                finalization_mode=config.QWEN_FINALIZATION_MODE,
            )
            state.forced_language = self.language
            state = feed_audio(silence, state, model=self.model_obj)
            finish_streaming(state, model=self.model_obj)
            self.vad.feed(
                silence[:_VAD_FRAME],
                self.vad.initial_state(sample_rate=config.SAMPLE_RATE),
                sample_rate=config.SAMPLE_RATE,
            )
            if self.diar is not None:
                self.diar.feed(
                    silence,
                    self.diar.init_streaming_state(),
                    sample_rate=config.SAMPLE_RATE,
                    threshold=config.DIAR_THRESHOLD,
                )
            if self.aligner is not None:
                # align a tiny dummy transcript against the silence to compile the aligner's
                # Metal kernels now; failure here is non-fatal (warmup only).
                with contextlib.suppress(Exception):
                    self.aligner.align(silence, "hello", self.language or "English")
        mx.clear_cache()

    def create_stream(self) -> QwenOnlineStream:
        return QwenOnlineStream(self)


def build_qwen_recognizer(
    asr_model: str,
    language: str = config.ASR_LANGUAGE,
    diarize: bool = False,
    word_diarize: bool = False,
    log: Callable[[str], None] | None = None,
) -> QwenRecognizer:
    """Load the Qwen3-ASR + VAD (+ optional Sortformer + forced aligner) models and warm them
    up (first run auto-downloads from HF).

    Raises ImportError with an install hint if the optional `[qwen]` extra is missing."""
    try:
        return QwenRecognizer(asr_model, language, diarize, word_diarize, log)
    except ImportError as e:
        raise ImportError(
            "The qwen3 ASR backend needs the 'mlx-qwen3-asr' package. "
            "Install the optional extra: `uv sync --extra qwen` (or `pip install "
            "mlx-qwen3-asr`)."
        ) from e


class QwenOnlineStream:
    """Single-stream decode state machine: IDLE (accumulating pre-roll, waiting for speech
    onset) <=> ACTIVE (feeding audio into a Qwen3-ASR streaming session). Silero VAD decides
    endpoints (rule1/2/3); the rolling decode produces the partial text, and finalization is
    finish_streaming over the buffered tail.

    accept_waveform / flush return events in the OnlineStream shapes AsrWorker expects:
      ("partial", text, None, utterance_samples)
                                             text = the whole in-progress utterance (no live
                                             speaker label on partials)
      ("final", [(speaker, text, diff), ...], utterance_samples)
                                             one segment per speaker. speaker = Sortformer
                                             number (None if --no-diarize); text is the two-pass
                                             offline re-decode, diff is its CJK-aware inline-diff
                                             spans vs the streaming text (None if unchanged).
                                             Utterance-level diarization yields a single segment;
                                             word-level (config.QWEN_WORD_DIARIZE) can yield
                                             several, split at speaker-change word boundaries.
    """

    def __init__(self, rec: QwenRecognizer):
        from mlx_qwen3_asr.streaming import feed_audio, finish_streaming, init_streaming

        self._rec = rec
        self._init_streaming = init_streaming
        self._feed_audio = feed_audio
        self._finish_streaming = finish_streaming
        self._transcribe = rec.session.transcribe  # see QwenRecognizer.session
        self._aligner = rec.aligner  # None unless word-level diarization is enabled
        self._vad = rec.vad
        self._vad_state = rec.vad.initial_state(sample_rate=config.SAMPLE_RATE)
        self._vad_leftover = np.empty(0, dtype=np.float32)
        n_preroll = max(1, round(config.VAD_PRE_ROLL_MS / 1000 / _VAD_FRAME_SEC))
        self._preroll: deque[np.ndarray] = deque(maxlen=n_preroll)
        # No soft-max re-seeding in this backend (no token timestamps), so the utterance is
        # never seeded with carried-over audio: started_at is just "first partial" wall time.
        self.seed_skew_sec = 0.0
        # Sortformer state persists across utterances so S-numbers stay stable; set only when
        # diarization is on. Utterance-level (_dominant_speaker) or, with a forced aligner,
        # word-level (_attribute_speakers).
        self._diar = rec.diar
        if self._diar is not None:
            self._diar_state = self._diar.init_streaming_state()
        self._reset_utterance()

    def _reset_utterance(self) -> None:
        self._active = False
        self._silence_frames = 0
        self._n_samples = 0
        self._text = ""
        self._utt_audio: list[np.ndarray] = []  # this utterance's audio, for the two-pass re-decode
        self._state = None  # the mlx-qwen3-asr StreamingState, created on speech onset

    # ---- public API ----

    def accept_waveform(self, samples: np.ndarray) -> list[tuple]:
        events: list[tuple] = []
        buf = np.concatenate([self._vad_leftover, np.asarray(samples, dtype=np.float32)])
        n = len(buf) // _VAD_FRAME
        self._vad_leftover = buf[n * _VAD_FRAME :]
        if not n:
            return events
        # All VAD frames of this block under one lock acquisition (the translation thread
        # contends for the same lock token by token).
        flags: list[bool] = []
        with _MLX_LOCK:
            for i in range(n):
                prob, self._vad_state = self._vad.feed(
                    buf[i * _VAD_FRAME : (i + 1) * _VAD_FRAME],
                    self._vad_state,
                    sample_rate=config.SAMPLE_RATE,
                )
                flags.append(float(prob.reshape(-1)[0]) >= config.VAD_THRESHOLD)
        for i, is_speech in enumerate(flags):
            frame = buf[i * _VAD_FRAME : (i + 1) * _VAD_FRAME]
            events += self._on_frame(frame, is_speech)
        return events

    def flush(self) -> list[tuple]:
        """Stream ended: flush the in-progress sentence into a final."""
        if not self._active:
            return []
        return self._finalize()

    # ---- state machine ----

    def _on_frame(self, frame: np.ndarray, is_speech: bool) -> list[tuple]:
        if not self._active:
            self._preroll.append(frame)
            if not is_speech:
                return []
            # Speech onset: open a fresh streaming session and feed the whole pre-roll so the
            # first word isn't clipped.
            self._active = True
            self._silence_frames = 0
            self._state = self._new_state()
            pre = np.concatenate(list(self._preroll))
            self._preroll.clear()
            self._n_samples += len(pre)
            return self._feed(pre)

        self._n_samples += len(frame)
        self._silence_frames = 0 if is_speech else self._silence_frames + 1
        events = self._feed(frame)

        utt_sec = self._n_samples / config.SAMPLE_RATE
        silence_sec = self._silence_frames * _VAD_FRAME_SEC
        # rule2: a shorter pause is enough once the text already ends a sentence (. ? ! / 。？！),
        # otherwise wait longer so a mid-sentence pause doesn't fragment the utterance.
        ends_sentence = self._text.rstrip()[-1:] in _SENT_ENDERS
        rule2_silence = (
            config.RULE2_PUNCT_SILENCE if ends_sentence else config.RULE2_MIN_TRAILING_SILENCE
        )
        if self._text and silence_sec >= rule2_silence:
            return events + self._finalize()
        if not self._text and silence_sec >= config.RULE1_MIN_TRAILING_SILENCE:
            # rule1: silence with nothing decoded yet -> reset only, no final.
            self._reset()
            return events
        if utt_sec >= config.RULE3_MIN_UTTERANCE_LENGTH:
            return events + self._finalize()
        return events

    # ---- decode ----

    def _new_state(self):
        state = self._init_streaming(
            model=self._rec.model_id,
            chunk_size_sec=config.QWEN_CHUNK_SIZE_SEC,
            max_context_sec=config.QWEN_MAX_CONTEXT_SEC,
            language=self._rec.language,
            finalization_mode=config.QWEN_FINALIZATION_MODE,
        )
        state.forced_language = self._rec.language
        return state

    def _feed(self, audio: np.ndarray) -> list[tuple]:
        """Push audio into the streaming session; the rolling decode only advances once its
        internal buffer fills a chunk (~QWEN_CHUNK_SIZE_SEC), so most calls just buffer."""
        self._utt_audio.append(audio)  # retained for the whole-utterance two-pass re-decode
        prev = self._text
        with _MLX_LOCK:
            self._state = self._feed_audio(audio, self._state, model=self._rec.model_obj)
        self._text = (self._state.text or "").strip()
        if self._text and self._text != prev:
            return [("partial", self._text, None, self._n_samples)]
        return []

    def _finalize(self) -> list[tuple]:
        if self._state is None:
            self._reset()
            return []
        with _MLX_LOCK:
            self._state = self._finish_streaming(self._state, model=self._rec.model_obj)
        stream_text = (self._state.text or "").strip()
        audio = np.concatenate(self._utt_audio) if self._utt_audio else None
        utterance_samples = self._n_samples
        have_audio = audio is not None and len(audio) > 0
        # Word-level diarization needs the offline re-decode's word timestamps; running it with
        # return_timestamps=True also gives the corrected text, so it subsumes the two-pass.
        want_words = self._aligner is not None and self._diar is not None and have_audio
        text = stream_text
        words: list[dict] | None = None
        if (config.QWEN_TWO_PASS or want_words) and have_audio:
            redecoded, words = self._second_pass(audio, want_words=want_words)
            if redecoded:
                text = redecoded
        # Speaker attribution: word-level split when we have aligned words, else one
        # utterance-level label, else no label.
        if want_words and words:
            parts = self._attribute_speakers(audio, words, text)
        elif self._diar is not None and have_audio:
            parts = [(self._dominant_speaker(audio), text)]
        else:
            parts = [(None, text)]
        # CJK-aware inline-diff spans vs the streaming text, sliced per speaker segment (None
        # where unchanged); the concatenation of part texts equals `text`, so the slicing is
        # exact (mirrors the nemotron backend's per-segment diff).
        diffs: list = [None] * len(parts)
        if text != stream_text and stream_text:
            diffs = _inline_diff(stream_text, [seg for _s, seg in parts])
        self._reset()
        segments = [
            (spk, seg, d) for (spk, seg), d in zip(parts, diffs, strict=True) if seg
        ]
        if not segments:
            return []
        return [("final", segments, utterance_samples)]

    def _dominant_speaker(self, audio: np.ndarray) -> int | None:
        """Utterance-level speaker label: feed the whole sentence to Sortformer (native working
        point ~15s chunks; state persists across utterances so S-numbers stay stable) and
        return the majority frame label. No mid-sentence split (use --qwen-word-diarize for
        that); a speaker change with a pause is already split into separate utterances by rule2."""
        with _MLX_LOCK:
            out, self._diar_state = self._diar.feed(
                audio,
                self._diar_state,
                sample_rate=config.SAMPLE_RATE,
                threshold=config.DIAR_THRESHOLD,
            )
            probs = np.array(out.speaker_probs.astype(mx.float32))
        probs = probs.reshape(-1, probs.shape[-1])
        labels = [int(p.argmax()) for p in probs if p.max() >= config.DIAR_THRESHOLD]
        if not labels:
            return None
        return max(set(labels), key=labels.count)

    def _attribute_speakers(
        self, audio: np.ndarray, words: list[dict], text: str
    ) -> list[tuple[int | None, str]]:
        """Word-level speaker split (nemotron parity). Feed the whole sentence to Sortformer
        (~its native 15s working point; state persists for stable numbering), map each aligned
        word's start time to a frame speaker, smooth out too-short flips, then slice `text` at
        speaker-change word boundaries. The aligned words are a tokenization OF `text`, so we
        locate each in `text` by an in-order search and cut on character offsets -- this keeps
        the canonical spacing/punctuation verbatim (rebuilding from bare word tokens would lose
        spacing around embedded Latin in CJK text). Returns [(speaker, segment_text), ...]."""
        with _MLX_LOCK:
            out, self._diar_state = self._diar.feed(
                audio,
                self._diar_state,
                sample_rate=config.SAMPLE_RATE,
                threshold=config.DIAR_THRESHOLD,
            )
            probs = np.array(out.speaker_probs.astype(mx.float32))
        probs = probs.reshape(-1, probs.shape[-1])
        n_frames = len(probs)
        if not n_frames:
            return [(None, text)]
        labels = [
            int(p.argmax()) if p.max() >= config.DIAR_THRESHOLD else None for p in probs
        ]
        # Sortformer frames are uniform over the utterance; derive their duration from the
        # actual frame count rather than hard-coding 80ms.
        frame_sec = (len(audio) / config.SAMPLE_RATE) / n_frames

        # word -> speaker: label of the word's start frame; unlabeled (silence/uncertain)
        # carries the previous value
        spk_seq: list[int | None] = []
        last: int | None = None
        for w in words:
            idx = (
                min(int(float(w["start"]) / frame_sec), n_frames - 1)
                if frame_sec > 0
                else 0
            )
            s = labels[idx]
            if s is None:
                s = last
            last = s
            spk_seq.append(s)
        first = next((s for s in spk_seq if s is not None), None)
        if first is None:  # no speaker for the whole sentence -> emit as one
            return [(None, text)]
        spk_seq = [first if s is None else s for s in spk_seq]
        spk_seq = self._smooth_speakers(spk_seq, words)
        return _segments_from_speaker_seq(words, spk_seq, text)

    def _smooth_speakers(
        self, spk_seq: list[int | None], words: list[dict]
    ) -> list[int | None]:
        """Absorb speaker runs shorter than DIAR_MIN_SEGMENT_SEC into the neighbouring speaker
        (mirrors the nemotron backend's _smooth_speakers, keyed on word start/end times instead
        of token timestamps): a brief probability blip or one-word backchannel shouldn't split a
        sentence into a fragment. Genuine turn-taking comes with a pause that rule2 already cuts."""
        min_sec = config.DIAR_MIN_SEGMENT_SEC
        if min_sec <= 0 or len(spk_seq) < 2:
            return spk_seq
        spk_seq = list(spk_seq)
        runs: list[tuple[int, int]] = []  # (lo, hi) of each maximal same-speaker run
        lo = 0
        for i in range(1, len(spk_seq) + 1):
            if i == len(spk_seq) or spk_seq[i] != spk_seq[lo]:
                runs.append((lo, i))
                lo = i
        if len(runs) < 2:
            return spk_seq
        for ri, (lo, hi) in enumerate(runs):
            if float(words[hi - 1]["end"]) - float(words[lo]["start"]) >= min_sec:
                continue
            # previous run is already smoothed (forward pass); fall back to the next run for a
            # leading short run
            repl = spk_seq[runs[ri - 1][0]] if ri > 0 else spk_seq[runs[ri + 1][0]]
            for k in range(lo, hi):
                spk_seq[k] = repl
        return spk_seq

    def _second_pass(
        self, audio: np.ndarray, want_words: bool = False
    ) -> tuple[str, list[dict] | None]:
        """Offline whole-utterance re-decode for the final (config.QWEN_TWO_PASS). When
        want_words, also request word timestamps (return_timestamps=True drives the pre-loaded
        Qwen3-ForcedAligner) and return result.segments ([{"text","start","end"}, ...]) for
        word-level speaker attribution. Returns ("", None) on any failure so finalization falls
        back to the streaming text / utterance-level diarization instead of dropping the
        sentence (a real error still surfaces via the AsrWorker crash path on the next op)."""
        try:
            with _MLX_LOCK:
                if want_words:
                    result = self._transcribe(
                        audio,
                        language=self._rec.language,
                        return_timestamps=True,
                        forced_aligner=self._aligner,
                    )
                else:
                    result = self._transcribe(audio, language=self._rec.language)
            text = (result.text or "").strip()
            words = getattr(result, "segments", None) if want_words else None
            return text, words
        except Exception:  # noqa: BLE001 -- a two-pass hiccup must not kill the worker
            return "", None

    def _reset(self) -> None:
        self._reset_utterance()
        mx.clear_cache()
