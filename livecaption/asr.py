"""Streaming ASR: mlx-audio nemotron-3.5 transducer (Apple GPU / MLX) + Silero VAD endpointing.

mlx-audio's stream_generate is pull-based (you hand it a whole audio segment, it
chunks internally and yields incremental results), so it can't be fed a live mic
stream. Here we reuse its cache-aware streaming kernel (streaming._stream_block) and
rewrite the outer loop into a push-based stepper (_StreamingEncoder): the bookkeeping
mirrors stream_encode line by line, guaranteeing the streaming output matches the
offline generate. Endpointing reimplements sherpa-onnx's rule1/2/3 semantics on top of
Silero VAD's per-frame speech probability (thresholds in config). Silence never enters
the encoder—the GPU only burns while someone is speaking. Speaker diarization
(Sortformer) attributes the whole utterance at finalization: the sentence is split by
token timestamps and each speaker's slice is emitted as a separate final, while during
partials a read-only peek assigns a provisional S label (see the comments on
_attribute_speakers / _diar_peek).

Note: the _stream_block and stream_encode bookkeeping is mlx-audio internal
implementation; pyproject already pins "mlx-audio>=0.4.4,<0.5". Before upgrading, run
scripts/smoke_asr.py to verify the streaming/offline results still match.
"""

from __future__ import annotations

import difflib
import queue
import re
import sys
import threading
import traceback
from collections import deque
from collections.abc import Callable
from datetime import datetime, timedelta

import mlx.core as mx
import numpy as np

from . import config
from .audio import SENTINEL
from .languages import normalize_asr_language

# In both mode the two AsrWorkers share weights (including VAD), and MLX gives no
# guarantee for concurrent multi-threaded evaluation: all ASR-side mlx computation holds
# this process-level lock; the translation thread grabs the same lock per decode step
# (see runtime.py)
from .runtime import MLX_LOCK as _MLX_LOCK

_VAD_FRAME = 512  # Silero fixed 32ms @ 16k
_VAD_FRAME_SEC = _VAD_FRAME / config.SAMPLE_RATE
# Number of mel frames at the buffer tail whose values can still change due to STFT
# center-padding (ceil of n_fft/2 / hop); held back (not fed) until non-final so we wait
# for later audio to arrive, keeping every mel frame consistent with the offline result
_MEL_HOLDBACK = 2
# Extra left-context frames the incremental mel (_mel_grow) pulls in when recomputing the
# tail: must be >= _MEL_HOLDBACK (to cover the center-padding influence range), and must
# keep the retained frames' windows clear of the preemphasis-contaminated first sample of
# the slice; 4 leaves some margin, correctness verified by scripts/smoke_mel.py
_MEL_LCTX = 4
# Sentence end for the soft-max punctuation cut: "." only counts after a word of 3+
# letters, so abbreviations ("U.S.", "Dr.") don't trigger a premature split; "?" / "!"
# always count. Allows trailing closing quotes/brackets.
_SENT_END_RE = re.compile(r"(?:[A-Za-z]{3,}\.|[!?])[\"'”’)\]]*\s*$")
# Audio back-off for the soft-max cut. RNNT token timestamps lag acoustics (emission
# delay, worst for punctuation, which is only decided once the next words are heard), so
# cutting at the punctuation token's own timestamp bleeds the next sentence's first word
# into the head. Anchor on the NEXT token's start instead and back off by this much --
# 4 encoder frames, covering the typical 2-4 frame word-onset emission delay (0.24 was
# measured to still clip short words like "The"). With a real pause the cut lands inside
# it; the head's trailing overlap decodes into words that the post-second-pass
# truncation drops, and the tail re-decodes them.
_SPLIT_BACKOFF_SEC = 0.32


class Recognizer:
    """Shared model bundle: nemotron weights + silero weights. No decode state, can be
    shared across multiple streams."""

    def __init__(
        self,
        asr_model: str,
        language: str = config.ASR_LANGUAGE,
        diarize: bool = False,
        log: Callable[[str], None] | None = None,
    ):
        from mlx_audio.stt import load as load_stt
        from mlx_audio.vad import load as load_vad

        say = log or (lambda _m: None)
        say(f"Loading ASR model {asr_model} …")
        self.model = load_stt(asr_model)
        # The model is conditioned via a language prompt. The CLI accepts tags and English
        # names case-insensitively (for example "en-us" / "English"), then we resolve them
        # to the exact model prompt key here.
        known = getattr(self.model, "prompt_dictionary", None) or {}
        language = normalize_asr_language(language, known if known else None)
        # An unknown key silently falls back to the default language, so we validate
        # explicitly and list every supported locale, avoiding the case where a user typos it
        # and assumes it took effect.
        if known and language not in known:
            raise ValueError(
                f"ASR language '{language}' is not supported by this model.\n"
                f"Available: {', '.join(sorted(known))}"
            )
        self.language = language
        # Override the default [56,13] (a 1.12s refresh is too sluggish); left determines
        # the cache length, right+1 is the feed chunk size
        self.model.default_att_context_size = list(config.ASR_ATT_CONTEXT)
        say(f"Loading VAD model {config.VAD_MODEL} …")
        self.vad = load_vad(config.VAD_MODEL)
        self.diar = None
        if diarize:
            say(f"Loading diarization model {config.DIAR_MODEL} …")
            self.diar = load_vad(config.DIAR_MODEL)
        say("Warming up models …")
        self._warmup()

    def _warmup(self) -> None:
        """Run each model once on empty input to absorb Metal kernel compilation at startup.

        Otherwise the first inference (VAD's first frame / the first sentence's encoder
        step / the first diar peek 2s in) would stall an extra few hundred ms inside
        _MLX_LOCK, making the first sentence's partial noticeably lag.
        """
        silence = np.zeros(int(0.5 * config.SAMPLE_RATE), dtype=np.float32)
        self.model.generate(
            mx.array(silence),
            language=self.language,
            att_context_size=list(config.ASR_FINAL_ATT_CONTEXT),
        )
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
        mx.clear_cache()

    def create_stream(self) -> OnlineStream:
        return OnlineStream(self)


def build_recognizer(
    asr_model: str,
    language: str = config.ASR_LANGUAGE,
    diarize: bool = False,
    log: Callable[[str], None] | None = None,
) -> Recognizer:
    """Load the ASR + VAD (+ optional diarization) models and warm them up (first run
    auto-downloads from HF)."""
    return Recognizer(asr_model, language, diarize, log)


class _StreamingEncoder:
    """Rewrites stream_encode (a pull-based generator) into a push-based stepper.

    State and bookkeeping mirror stream_encode in
    mlx_audio…nemotron_asr/streaming.py line by line; the only differences are that mel
    is fed in chunk by chunk from outside, and is_final is decided by endpointing rather
    than by "the audio is exhausted".
    """

    def __init__(self, model, language: str):  # noqa: ANN001  # model instance dynamically loaded by mlx-audio
        enc = model.encoder
        acs = model.default_att_context_size
        self.model = model
        self.enc = enc
        self.language = language
        self.left_cache = int(acs[0])
        self.right = int(acs[1])
        self.sf = enc.args.subsampling_factor
        self.chunk_mel = (self.right + 1) * self.sf  # how many mel frames consumed per step
        self.conv_left = enc.args.conv_kernel_size - 1
        self.reset()

    def reset(self) -> None:
        n = len(self.enc.layers)
        self._attn_cache = [None] * n
        self._conv_cache = [None] * n
        self._mel_cache = None
        self._emitted = 0
        self._consumed = 0

    def step(self, m: mx.array, is_final: bool):
        """Feed (1, k, F) new mel frames (k <= chunk_mel), return prompted encoder frames
        or None."""
        from mlx_audio.stt.models.nemotron_asr.streaming import (
            _PRE_ENCODE_MEL_CACHE,
            _stream_block,
        )

        enc = self.enc
        cache_len = 0 if self._mel_cache is None else self._mel_cache.shape[1]
        win = m if self._mel_cache is None else mx.concatenate([self._mel_cache, m], axis=1)
        win_len = win.shape[1]
        sub = enc.pre_encode(win, mx.array([win_len], dtype=mx.int32))[0]

        end = self._consumed + m.shape[1]
        base = (self._consumed - cache_len) // self.sf
        lo = self._emitted - base
        hi = sub.shape[1] if is_final else (end // self.sf - base)
        self._consumed = end
        self._mel_cache = win[:, -_PRE_ENCODE_MEL_CACHE:]

        if hi <= lo:
            self._emitted = base + max(lo, hi)
            return None
        self._emitted = base + hi
        h = sub[:, lo:hi]
        for li, block in enumerate(enc.layers):
            h, self._attn_cache[li], self._conv_cache[li] = _stream_block(
                block,
                h,
                enc.pos_enc,
                self._attn_cache[li],
                self._conv_cache[li],
                self.left_cache,
                self.conv_left,
            )
        return self.model.apply_prompt(h, self.language)


# Edge punctuation ignored when comparing words for the inline diff (ASCII plus the
# curly quotes/dashes/ellipsis the model emits); inner apostrophes ("don't") survive
_DIFF_TRIM = "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~“”‘’„‛…—–"


def _diff_key(word: str) -> str:
    """Comparison key: casefold + strip edge punctuation. Two-pass corrections are mostly
    punctuation/casing touch-ups ("America" -> "America,"); rendering each as a del+add
    pair doubles the words and drowns the line in red/green noise, so those compare as
    equal and only true word changes show up as corrections (the final text still carries
    the corrected form -- "same" spans render the new words)."""
    # also strip whitespace: tokens keep a leading space for verbatim span text, but the
    # comparison key must treat " word" and "word" (segment-initial) as equal.
    return word.strip().strip(_DIFF_TRIM).casefold()


# CJK / fullwidth codepoint ranges. Each such character is its own diff token: Chinese has no
# word boundaries, so a word-level split() would make a whole sentence one token and render
# any correction as a single del+add of the entire line. Latin/digit runs stay whole "words".
# Tokens keep their leading whitespace so "".join(tokens) reproduces the text verbatim -- the
# renderer then concatenates diff spans with no separator, keeping the spacing correct for
# Latin words and spaceless CJK alike.
_CJK = "　-〿㐀-䶿一-鿿豈-﫿＀-￯"
_DIFF_TOKEN_RE = re.compile(rf"\s*(?:[{_CJK}]|[^\s{_CJK}]+)")


def _diff_tokenize(text: str) -> list[str]:
    return _DIFF_TOKEN_RE.findall(text)


def _truncate_after_sentence(text: str, tokens: list, old_end: float, cut_sec: float):
    """Drop second-pass tokens decoded from the overlap audio past the soft-max cut.

    Prefer cutting after the sentence-end token nearest in time to the streaming-pass
    punctuation (timestamps shift a frame or two between passes); if the re-decode
    produced no nearby sentence end (it may merge or re-punctuate), fall back to the
    audio boundary itself. The dropped words live in the tail audio and re-decode in
    the next utterance, so nothing is lost or duplicated.
    """
    from mlx_audio.stt.models.nemo.alignment import (
        sentences_to_result,
        tokens_to_sentences,
    )

    best = None
    for i, t in enumerate(tokens):
        if (
            any(c in ".?!" for c in t.text)
            and abs(t.end - old_end) <= 0.6
            and (best is None or abs(t.end - old_end) < abs(tokens[best].end - old_end))
        ):
            best = i
    kept = tokens[: best + 1] if best is not None else [t for t in tokens if t.start < cut_sec]
    if not kept or len(kept) == len(tokens):
        return text, tokens
    new_text = sentences_to_result(tokens_to_sentences(kept)).text.strip()
    return (new_text, kept) if new_text else (text, tokens)


def _inline_diff(old: str, segs: list[str]) -> list[list[tuple[str, str]] | None]:
    """Whole-sentence diff sliced by final segments, for the terminal to render the
    correction effect inline within each final line.

    Tokens are matched via _diff_key and are CJK-aware (each Chinese char is a token, Latin
    runs stay whole words), so punctuation/casing-only corrections don't produce spans. Token
    text keeps its leading whitespace and a span is the verbatim concatenation of its tokens,
    so the renderer joins spans with no separator -- correct for Latin spacing and spaceless
    CJK alike. Returns a list the same length as segs; each item is [(kind, text)] spans
    (kind: same/del/add), or None if that segment has no correction. A deleted token is
    anchored to the segment that owns its position in the new text.
    """
    a = _diff_tokenize(old)
    seg_words = [_diff_tokenize(s) for s in segs]
    b = [w for ws in seg_words for w in ws]
    ops = difflib.SequenceMatcher(
        a=[_diff_key(w) for w in a], b=[_diff_key(w) for w in b], autojunk=False
    ).get_opcodes()
    out: list[list[tuple[str, str]] | None] = []
    lo = 0
    for si, ws in enumerate(seg_words):
        hi = lo + len(ws)
        last = si == len(seg_words) - 1
        spans: list[tuple[str, str]] = []
        for op, i1, i2, j1, j2 in ops:
            if op == "equal":
                jl, jh = max(j1, lo), min(j2, hi)
                if jl < jh:
                    spans.append(("same", "".join(b[jl:jh])))
                continue
            if i1 < i2 and (lo <= j1 < hi or (last and j1 >= hi)):
                spans.append(("del", "".join(a[i1:i2])))
            jl, jh = max(j1, lo), min(j2, hi)
            if jl < jh:
                spans.append(("add", "".join(b[jl:jh])))
        out.append(spans if any(k != "same" for k, _ in spans) else None)
        lo = hi
    return out


class OnlineStream:
    """Decode state machine for a single audio stream: IDLE (accumulating pre-roll, waiting
    for speech onset) <=> ACTIVE (incrementally decoding one sentence).

    accept_waveform returns an event list with two shapes:
      ("partial", text, speaker, utterance_samples)
                                   text = the whole in-progress utterance (non-empty);
                                   speaker = provisional speaker from the diar peek (None if
                                   diarize off / still uncertain); utterance_samples lets file
                                   mode convert the event to audio-time timestamps.
      ("final", segments, utterance_samples)
                                   segments = [(speaker, text, diff), ...] for the WHOLE
                                   utterance in one event. When diarize splits the utterance
                                   across speakers there are multiple segments; otherwise one.
                                   speaker is the Sortformer number (None if diarize off); diff
                                   is that segment's two-pass inline-diff spans (None if no
                                   correction). The renderer shows all segments on a single
                                   line with inline [S1]/[S2] markers rather than one line per
                                   speaker.
    """

    def __init__(self, rec: Recognizer):
        self._model = rec.model
        self._pre = rec.model.preprocessor_config
        self._vad = rec.vad
        self._vad_state = rec.vad.initial_state(sample_rate=config.SAMPLE_RATE)
        self._vad_leftover = np.empty(0, dtype=np.float32)
        self._encoder = _StreamingEncoder(rec.model, rec.language)
        n_preroll = max(1, round(config.VAD_PRE_ROLL_MS / 1000 / _VAD_FRAME_SEC))
        self._preroll: deque[np.ndarray] = deque(maxlen=n_preroll)
        # Diarization: state persists throughout (spkcache keeps speaker numbers stable
        # across sentences), not reset per sentence
        self._diar = rec.diar
        if self._diar is not None:
            self._diar_state = self._diar.init_streaming_state()
        self._frame_sec = (
            rec.model.encoder_config.subsampling_factor
            * self._pre.hop_length
            / self._pre.sample_rate
        )
        self._reset_utterance()

    def _reset_utterance(self) -> None:
        self._active = False
        self._audio: list[np.ndarray] = []
        self._n_samples = 0
        self._mel_consumed = 0
        # cached mel prefix that will no longer change (_mel_grow)
        self._mel_stable: mx.array | None = None
        self._silence_frames = 0
        # RNNT decode state (corresponds to stream_generate's local variables)
        self._last_token = self._model.blank_id
        self._decoder_hidden = None
        self._hypothesis: list = []
        self._global_time = 0
        self._text = ""
        self._live_spk: int | None = None  # provisional speaker used for partials (given by peek)
        self._peeked_samples = 0
        # seconds of carried-over audio this utterance was seeded with (soft-max cut):
        # its speech started this long before the first partial, so AsrWorker backdates
        # started_at by it
        self.seed_skew_sec = 0.0

    # ---- public API ----

    def accept_waveform(self, samples: np.ndarray) -> list[tuple]:
        events: list[tuple] = []
        buf = np.concatenate([self._vad_leftover, np.asarray(samples, dtype=np.float32)])
        n = len(buf) // _VAD_FRAME
        self._vad_leftover = buf[n * _VAD_FRAME :]
        if not n:
            return events
        # Compute all VAD frames of this block in one lock acquisition (VAD state is
        # unaffected by _on_frame side effects), reducing lock acquisitions (the
        # translation thread is contending for the same lock token by token)
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
            # Speech onset: merge the entire pre-roll into this sentence to avoid clipping
            # the first word
            self._active = True
            for f in self._preroll:
                self._audio.append(f)
                self._n_samples += len(f)
            self._preroll.clear()
            return self._drive(final=False)

        self._audio.append(frame)
        self._n_samples += len(frame)
        self._silence_frames = 0 if is_speech else self._silence_frames + 1
        events = self._drive(final=False)

        # Periodically peek (read-only) at the current speaker, so partials have an S label to show
        if (
            self._diar is not None
            and self._text
            and self._n_samples - self._peeked_samples
            >= int(config.DIAR_PEEK_SEC * config.SAMPLE_RATE)
        ):
            self._peeked_samples = self._n_samples
            self._diar_peek()

        utt_sec = self._n_samples / config.SAMPLE_RATE
        # Soft max: once the utterance is this long, cut at the most recent decoded
        # sentence-final punctuation, without waiting for silence -- in fast continuous
        # speech pauses rarely reach RULE2_PUNCT_SILENCE and every utterance would run
        # into the rule3 force-cut mid-sentence. The cut is retroactive at the token
        # timestamp ("text ends with punctuation" can never be observed: a short pause
        # fits inside one decode chunk, so the period and the next sentence's first words
        # arrive together). Audio and tokens past the punctuation carry over into the
        # next utterance inside _finalize, so nothing is clipped.
        if utt_sec >= config.RULE2_SOFT_MAX_UTTERANCE:
            cut = self._last_sentence_end()
            if cut is not None:
                return events + self._finalize(split_token=cut)

        silence_sec = self._silence_frames * _VAD_FRAME_SEC
        # rule2: cut sooner once the text already ends a sentence (. ? !), otherwise wait the
        # longer silence so a mid-sentence pause doesn't fragment the utterance
        ends_sentence = self._text.rstrip()[-1:] in ".?!"
        rule2_silence = (
            config.RULE2_PUNCT_SILENCE
            if ends_sentence
            else config.RULE2_MIN_TRAILING_SILENCE
        )
        if self._text and silence_sec >= rule2_silence:
            events += self._finalize()
        elif not self._text and silence_sec >= config.RULE1_MIN_TRAILING_SILENCE:
            # rule1: reset only, no final emitted (matches sherpa's empty-text endpoint)
            self._reset()
        elif utt_sec >= config.RULE3_MIN_UTTERANCE_LENGTH:
            events += self._finalize()
        return events

    def _last_sentence_end(self) -> int | None:
        """Index of the most recent hypothesis token that ends a sentence, or None.

        A token containing . ? ! only counts when the text up to it passes _SENT_END_RE
        (the 3+-letter-word guard against abbreviations / decimals); a few preceding
        tokens are enough context since they concatenate into the trailing words.
        """
        hyp = self._hypothesis
        for i in range(len(hyp) - 1, -1, -1):
            if not any(c in ".?!" for c in hyp[i].text):
                continue
            prefix = "".join(t.text for t in hyp[max(0, i - 3) : i + 1])
            if _SENT_END_RE.search(prefix):
                return i
        return None

    def _finalize(self, split_token: int | None = None) -> list[tuple]:
        from mlx_audio.stt.models.nemo.alignment import (
            sentences_to_result,
            tokens_to_sentences,
        )

        audio = np.concatenate(self._audio) if self._audio else None
        utterance_samples = len(audio) if audio is not None else self._n_samples
        text = self._text
        tokens = self._hypothesis
        tail: np.ndarray | None = None
        cut_sec: float | None = None
        if split_token is not None and audio is not None and len(tokens) > split_token:
            # Soft-max cut. The audio boundary anchors on the next token's start (see
            # _SPLIT_BACKOFF_SEC: punctuation timestamps lag acoustics too much to cut
            # on); everything past it re-seeds the next utterance below, so the next
            # sentence's first words -- often already inside the decode look-ahead --
            # re-decode there instead of being clipped or orphaned onto this line.
            # No final-flush of the held-back mel: the punctuation is already decoded.
            punct = tokens[split_token]
            # Anchor on the next WORD-BEARING token: a bare separator token right after
            # the punctuation can carry a timestamp even earlier than the punct's. No
            # punct-based floor -- punctuation timestamps lag worst of all (the decoder
            # commits "." only after hearing the next words; measured emitting in the
            # same frame as the next word), so flooring on punct.end cuts INTO the next
            # sentence's first words. Cutting generously early is safe: the head's text
            # comes from the full-buffer second pass truncated at the sentence end.
            nxt = next((t for t in tokens[split_token + 1 :] if t.text.strip()), None)
            anchor = nxt.start if nxt is not None else punct.end
            sec = max(0.0, anchor - _SPLIT_BACKOFF_SEC)
            n_cut = int(sec * config.SAMPLE_RATE)
            if 0 < n_cut < len(audio):
                cut_sec = sec
                tail = audio[n_cut:]
                tokens = tokens[: split_token + 1]
                text = sentences_to_result(tokens_to_sentences(tokens)).text.strip()
        if cut_sec is None:
            self._drive(final=True)  # flush out the held-back tail mel
            audio = np.concatenate(self._audio) if self._audio else None
            text = self._text
            tokens = self._hypothesis
        self._reset()
        if tail is not None and len(tail):
            # the carried-over audio opens the next utterance already ACTIVE (it holds
            # speech, not pre-roll silence); decoding resumes on the next frame. Record
            # its duration so AsrWorker can backdate started_at: the speech in the tail
            # began that long before its first partial will appear.
            self._active = True
            self._audio = [tail]
            self._n_samples = len(tail)
            self.seed_skew_sec = len(tail) / config.SAMPLE_RATE
        if not text:
            return []
        old_text = text
        split_punct_end = tokens[-1].end if cut_sec is not None and tokens else 0.0
        if audio is not None:
            # Second pass over the FULL buffer (head + overlap past the cut): the
            # overlap gives the decoder right-context so the head's last words come out
            # clean; in split mode whatever it decodes beyond the sentence end is
            # truncated away (those words live in the tail and re-decode next utterance)
            text, tokens = self._second_pass(audio, text, tokens)
            if cut_sec is not None:
                text, tokens = _truncate_after_sentence(
                    text, tokens, split_punct_end, cut_sec
                )
        # diarization must consume each audio sample exactly once across finals, so it
        # gets only the head slice; the tail is fed again as part of the next utterance
        head_audio = audio if cut_sec is None else audio[: int(cut_sec * config.SAMPLE_RATE)]
        if self._diar is None or head_audio is None or not tokens:
            parts: list[tuple[int | None, str]] = [(None, text)]
        else:
            # _attribute_speakers returns (text, speaker, t0); the per-segment start time is no
            # longer needed (one timestamp per utterance now), so keep only (speaker, text)
            parts = [
                (spk, seg)
                for seg, spk, _t0 in self._attribute_speakers(head_audio, tokens, text)
            ]
        # two-pass inline-diff spans, sliced per speaker segment (None where no correction)
        diffs: list = [None] * len(parts)
        if old_text != text:
            diffs = _inline_diff(old_text, [seg for _s, seg in parts])
        # One final event for the whole utterance -> the renderer draws a single line with
        # inline [S1]/[S2] markers instead of one line per speaker (avoids a partial flickering
        # into several lines on finalization)
        segments = [(spk, seg, d) for (spk, seg), d in zip(parts, diffs, strict=True)]
        return [("final", segments, utterance_samples)]

    def _second_pass(self, audio: np.ndarray, text: str, tokens: list):
        """two-pass correction: at finalization, re-decode the whole sentence with maximum
        look-ahead.

        Streaming decode sees only ASR_ATT_CONTEXT's 480ms of future per frame; on
        re-decode each frame can see 1.12s, which is acoustically more accurate (the same
        model's official highest-accuracy tier). Partials still show the streaming result;
        final and the translation input are swapped to the re-decoded version. The
        re-decoded tokens likewise count their 80ms timestamps from 0, so speaker
        attribution is reused directly.
        """
        if not config.ASR_TWO_PASS:
            return text, tokens
        with _MLX_LOCK:
            result = self._model.generate(
                mx.array(audio),
                language=self._encoder.language,
                att_context_size=list(config.ASR_FINAL_ATT_CONTEXT),
            )
        new_text = result.text.strip()
        if not new_text:  # re-decode fallback: keep the streaming result on anomaly
            return text, tokens
        new_tokens = [t for s in result.sentences for t in s.tokens]
        return new_text, new_tokens

    # ---- speaker attribution: at finalization, feed the whole sentence to sortformer
    # and split by token timestamps ----

    def _diar_peek(self) -> None:
        """Read-only peek at "who is speaking now": feed the sentence's audio so far once,
        but don't persist the state.

        Gives the partial a provisional S label. Peeking from the authoritative state as a
        starting point keeps numbering consistent with finalization; not persisting the
        state ensures the authoritative feed at finalization won't reconsume the same audio.
        """
        audio = np.concatenate(self._audio)
        with _MLX_LOCK:
            out, _ = self._diar.feed(
                audio,
                self._diar_state,
                sample_rate=config.SAMPLE_RATE,
                threshold=config.DIAR_THRESHOLD,
            )
            probs = np.array(out.speaker_probs.astype(mx.float32))
        probs = probs.reshape(-1, probs.shape[-1])
        labels = [int(p.argmax()) for p in probs if p.max() >= config.DIAR_THRESHOLD]
        if labels:  # mode of the last few labeled frames = current speaker
            tail = labels[-10:]
            self._live_spk = max(set(tail), key=tail.count)

    def _smooth_speakers(self, spk_seq: list[int], tokens: list) -> list[int]:
        """Absorb speaker runs shorter than DIAR_MIN_SEGMENT_SEC into the neighbouring speaker.

        Sortformer occasionally flips a speaker for a frame or two (a probability blip, or a
        listener's one-word backchannel mid-sentence); without smoothing that becomes its own
        token group -> a separate, often mid-word final fragment. We relabel any too-short run
        to the previous run's speaker (or the next run's, for a leading short run), so brief
        flips don't fragment one speaker's sentence. Genuine turn-taking is unaffected: real
        speaker changes come with a pause that rule2 already splits on. Set the config to 0 to
        disable.
        """
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
            if tokens[hi - 1].end - tokens[lo].start >= min_sec:
                continue
            # previous run's label is already smoothed (forward pass); fall back to the next
            # run for a leading short run
            repl = spk_seq[runs[ri - 1][0]] if ri > 0 else spk_seq[runs[ri + 1][0]]
            for k in range(lo, hi):
                spk_seq[k] = repl
        return spk_seq

    def _attribute_speakers(
        self, audio: np.ndarray, tokens: list, text: str
    ) -> list[tuple[str, int | None, float]]:
        """Feed the whole sentence's audio to sortformer at once (~ the model's native 15s
        working point), and attribute by splitting on tokens.

        Feeding small chunks in streaming fashion empirically causes speaker identity to
        drift (extra speakers appear out of nowhere); feeding the whole sentence + carrying
        state across sentences gets both offline-grade quality and stable numbering
        throughout. RNNT token start timestamps and diar frames share the same 80ms
        granularity and the same audio timeline, so they align directly. Cut at speaker
        changes, each segment a separate final. Returns [(text, speaker, the segment's
        first token's in-sentence start in seconds)].
        """
        from mlx_audio.stt.models.nemo.alignment import (
            sentences_to_result,
            tokens_to_sentences,
        )

        with _MLX_LOCK:
            out, self._diar_state = self._diar.feed(
                audio,
                self._diar_state,
                sample_rate=config.SAMPLE_RATE,
                threshold=config.DIAR_THRESHOLD,
            )
            probs = np.array(out.speaker_probs.astype(mx.float32))
        probs = probs.reshape(-1, probs.shape[-1])
        labels = [
            int(p.argmax()) if p.max() >= config.DIAR_THRESHOLD else None
            for p in probs
        ]

        # token -> speaker: take the label of the token's start frame; unlabeled frames
        # (silence/uncertain) carry over the previous value
        spk_seq: list[int | None] = []
        last: int | None = None
        for t in tokens:
            idx = min(int(t.start / self._frame_sec), len(labels) - 1)
            s = labels[idx] if labels else None
            if s is None:
                s = last
            last = s
            spk_seq.append(s)
        first = next((s for s in spk_seq if s is not None), None)
        if first is None:  # no speaker determined for the entire sentence, emit it as one
            return [(text, None, 0.0)]
        spk_seq = [first if s is None else s for s in spk_seq]
        spk_seq = self._smooth_speakers(spk_seq, tokens)

        groups: list[tuple[int, list]] = []
        for t, s in zip(tokens, spk_seq, strict=True):
            # Tokens are subword pieces and a word-start piece carries a leading space
            # (sentence text is plain concatenation of token texts, see nemo alignment).
            # Only open a new speaker group at a word start: a diarization boundary that
            # lands mid-word would otherwise split the word across two finals
            # ("used" -> "u" | "sed").
            starts_word = t.text.startswith(" ")
            if groups and (groups[-1][0] == s or not starts_word):
                groups[-1][1].append(t)
            else:
                groups.append((s, [t]))
        # Boundary punctuation reattachment: sentence-final punctuation ("?" ".") often
        # only gets decoded during the speaker-change gap, so its timestamp lands at the
        # head of the new speaker's group—move it back to the previous group, otherwise the
        # translation input would carry leading punctuation garbage
        for i in range(1, len(groups)):
            prev_g, cur_g = groups[i - 1][1], groups[i][1]
            while cur_g and not any(c.isalnum() for c in cur_g[0].text):
                prev_g.append(cur_g.pop(0))
        merged: list[tuple[int, list]] = []
        for s, g in groups:
            if not g:
                continue
            if merged and merged[-1][0] == s:
                merged[-1][1].extend(g)
            else:
                merged.append((s, g))

        segs: list[tuple[str, int | None, float]] = []
        for s, g in merged:
            seg = sentences_to_result(tokens_to_sentences(g)).text.strip()
            if seg:
                segs.append((seg, s, g[0].start))
        return segs or [(text, None, 0.0)]

    def _reset(self) -> None:
        self._encoder.reset()
        self._reset_utterance()
        mx.clear_cache()

    # ---- incremental encode + decode ----

    def _drive(self, final: bool) -> list[tuple]:
        """Only run an actual step once a full chunk of stable mel frames has accumulated
        (flush everything on final)."""
        hop = self._pre.hop_length
        chunk = self._encoder.chunk_mel
        if not final:
            # Estimate the available stable frame count; if there's not enough for a step,
            # accumulate first, avoiding recomputing the STFT every 32ms
            est = self._n_samples // hop + 1 - _MEL_HOLDBACK
            if est < self._mel_consumed + chunk:
                return []

        prev = self._text
        with _MLX_LOCK:
            mel = self._mel_grow(final)  # (1, T, F)
            avail = mel.shape[1] if final else mel.shape[1] - _MEL_HOLDBACK
            while self._mel_consumed + chunk <= avail:
                out = self._encoder.step(
                    mel[:, self._mel_consumed : self._mel_consumed + chunk], False
                )
                self._mel_consumed += chunk
                if out is not None:
                    self._decode_chunk(out)
            if final and self._mel_consumed < avail:
                out = self._encoder.step(mel[:, self._mel_consumed : avail], True)
                self._mel_consumed = avail
                if out is not None:
                    self._decode_chunk(out)

        if not final and self._text and self._text != prev:
            return [("partial", self._text, self._live_spk, self._n_samples)]
        return []

    def _mel_grow(self, final: bool) -> mx.array:
        """Incrementally maintain the whole-sentence mel: reuse the already-stable prefix
        directly, only run the STFT on the new tail audio.

        mel frame t's window only looks at samples [t*hop - n_fft/2, t*hop + n_fft/2);
        once more than _MEL_HOLDBACK frames from the end, it no longer changes and is
        cached into self._mel_stable, avoiding the per-step recompute getting more
        expensive as the sentence grows (O(n^2), and all on the lock-holding hot path).
        When recomputing the tail, pull in _MEL_LCTX extra frames of audio to the left of
        the stable boundary as context, then discard those frames contaminated by
        center-padding/preemphasis—matching the result of computing the whole sentence at
        once (verified by scripts/smoke_mel.py).
        """
        from mlx_audio.stt.models.nemotron_asr.audio import log_mel_spectrogram

        hop = self._pre.hop_length
        audio = np.concatenate(self._audio)
        stable = 0 if self._mel_stable is None else self._mel_stable.shape[1]
        ctx = min(stable, _MEL_LCTX)
        tail = log_mel_spectrogram(mx.array(audio[(stable - ctx) * hop :]), self._pre)
        tail = tail[:, ctx:]
        mel = (
            tail
            if self._mel_stable is None
            else mx.concatenate([self._mel_stable, tail], axis=1)
        )
        if not final:  # after final the whole sentence resets, so don't update the cache
            n_stable = mel.shape[1] - _MEL_HOLDBACK
            if n_stable > stable:
                self._mel_stable = mel[:, :n_stable]
        return mel

    def _decode_chunk(self, prompted: mx.array) -> None:
        """Greedy RNNT decode of one block of encoder output (ported from stream_generate's
        inner loop)."""
        from mlx_audio.stt.models.nemo.alignment import (
            AlignedToken,
            sentences_to_result,
            tokens_to_sentences,
        )
        from mlx_audio.stt.models.nemotron_asr import tokenizer as tok

        model = self._model
        chunk_len = prompted.shape[1]
        # Defensive: when max_symbols is None the non-blank branch below never advances
        # time -> deadlock inside the lock
        max_symbols = model.max_symbols or 10
        time = 0
        new_symbols = 0
        while time < chunk_len:
            feature = prompted[:, time : time + 1]
            current_token = (
                mx.array([[self._last_token]], dtype=mx.int32)
                if self._last_token != model.blank_id
                else None
            )
            decoder_output, (h, c) = model.decoder(current_token, self._decoder_hidden)
            decoder_output = decoder_output.astype(feature.dtype)
            proposed_hidden = (h.astype(feature.dtype), c.astype(feature.dtype))
            joint_output = model.joint(feature, decoder_output)
            pred_token = int(mx.argmax(joint_output))
            if pred_token != model.blank_id:
                self._last_token = pred_token
                self._decoder_hidden = proposed_hidden
                if not tok.is_special_token(pred_token, model.vocabulary):
                    self._hypothesis.append(
                        AlignedToken(
                            pred_token,
                            start=(self._global_time + time) * self._frame_sec,
                            duration=self._frame_sec,
                            text=tok.decode([pred_token], model.vocabulary),
                        )
                    )
                new_symbols += 1
                if new_symbols >= max_symbols:
                    time += 1
                    new_symbols = 0
            else:
                time += 1
                new_symbols = 0
        self._global_time += chunk_len
        self._text = sentences_to_result(tokens_to_sentences(self._hypothesis)).text.strip()


class AsrWorker(threading.Thread):
    """Consume one audio source's queue, decode chunk by chunk in streaming fashion, and
    emit partial / final events.

    label identifies the source ("me"/"them"); speaker numbers are carried inside the events,
    not in the label (the renderer shows them inline).

    on_partial(label, text, started_at, speaker):  in-progress utterance + its start time +
                 provisional speaker (None if diarize off / uncertain)
    on_final(label, segments, started_at):  segments = [(speaker, text, diff), ...] for the
                 whole utterance + the moment it started
    on_error():  notify when decode or a callback raises (the traceback is already printed
                 to stderr). If not passed, only logging happens—but the caller should pass
                 it: a worker dying silently makes the pipeline look "alive" while no longer
                 producing captions.
    on_done(label, samples_seen):  notify the source-audio duration consumed by the worker.
                 File-mode callers use it for end-of-media completeness checks.
    """

    def __init__(
        self,
        recognizer: Recognizer,
        audio_queue: queue.Queue,
        label: str,
        on_partial: Callable[[str, str, datetime, int | None], None],
        on_final: Callable[[str, list, datetime], None],
        on_error: Callable[[], None] | None = None,
        on_done: Callable[[str, int], None] | None = None,
        use_audio_timestamps: bool = False,
    ):
        super().__init__(daemon=True, name=f"asr-{label}")
        self.recognizer = recognizer
        self.audio_queue = audio_queue
        self.label = label
        self.on_partial = on_partial
        self.on_final = on_final
        self.on_error = on_error
        self.on_done = on_done
        self.use_audio_timestamps = use_audio_timestamps

    def run(self) -> None:
        try:
            self._run()
        except Exception:  # noqa: BLE001
            print(f"\n[error] ASR worker '{self.label}' crashed:", file=sys.stderr)
            traceback.print_exc()
            if self.on_error is not None:
                self.on_error()

    def _audio_timestamp(self, samples_seen: int, utterance_samples: int) -> datetime:
        """Timestamp an utterance by source-audio position instead of wall clock.

        FileSource can decode much faster than real time, so datetime.now() records when the
        transcript was produced, not where the utterance starts in the media. Use a midnight
        anchor so the existing render path still formats it as 00:00:xx.
        """
        start_sample = max(0, samples_seen - max(0, utterance_samples))
        return datetime(2000, 1, 1) + timedelta(
            seconds=start_sample / config.SAMPLE_RATE
        )

    def _run(self) -> None:
        stream = self.recognizer.create_stream()
        last_partial = ""
        started_at: datetime | None = None
        last_final_ts: datetime | None = None
        samples_seen = 0
        while True:
            samples = self.audio_queue.get()
            if samples is SENTINEL:
                events = stream.flush()
            else:
                samples_seen += len(samples)
                events = stream.accept_waveform(samples)
            for ev in events:
                if ev[0] == "final":
                    segments = ev[1]
                    utterance_samples = ev[2] if len(ev) > 2 else 0
                    ts = started_at
                    if ts is None:
                        ts = (
                            self._audio_timestamp(samples_seen, utterance_samples)
                            if self.use_audio_timestamps
                            else datetime.now()
                        )
                    self.on_final(self.label, segments, ts)
                    last_final_ts = ts
                    last_partial = ""
                    started_at = None
                else:  # ("partial", text, speaker)
                    _, text, speaker, *meta = ev
                    utterance_samples = meta[0] if meta else 0
                    if text != last_partial:
                        # the utterance's first non-empty result -> record as the start
                        # time, backdated by any carried-over audio the utterance was
                        # seeded with (soft-max cut), whose speech began before this.
                        # Clamped to the previous final's timestamp: in file mode the
                        # decode runs faster than real time, so the backdating can
                        # overshoot and make timestamps run backwards.
                        if not last_partial:
                            if self.use_audio_timestamps:
                                started_at = self._audio_timestamp(
                                    samples_seen, utterance_samples
                                )
                            else:
                                started_at = datetime.now() - timedelta(
                                    seconds=stream.seed_skew_sec
                                )
                            if last_final_ts is not None:
                                started_at = max(started_at, last_final_ts)
                        last_partial = text
                        self.on_partial(self.label, text, started_at, speaker)
            if samples is SENTINEL:
                break
        if self.on_done is not None:
            self.on_done(self.label, samples_seen)
