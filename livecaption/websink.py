"""Web output layer: a broadcast hub that mirrors the terminal Renderer's event API.

The ASR/translation pipeline calls the same callback contract as the terminal Renderer
(partial / final / translation / preview / flush_pending). Instead of drawing to a
terminal, WebSink serializes each event to a JSON dict and fans it out to every
connected browser over Server-Sent Events (SSE). All the EN/ZH pairing and live-region
bookkeeping the terminal Renderer does is instead done in the browser (keyed by the
event's stable `key`), because the DOM has no scroll-above-live constraint.

WebSink is a single long-lived object for the whole server: browsers connect/disconnect
and start/stop the pipeline, but the hub (and the models behind it) stay warm. It is
thread-safe -- broadcast() is called from the ASR/translator worker threads while
subscribe()/unsubscribe() run on HTTP request threads.
"""

from __future__ import annotations

import queue
import threading
from datetime import datetime

# Per-subscriber (per-browser) event backlog cap. A slow/stuck browser must not grow its
# queue without bound; past this we drop the oldest events for that one client (captions
# are ephemeral -- a lagging client is better off skipping ahead than falling minutes behind).
_QUEUE_MAX = 2000


def _key(label: str, started_at: datetime) -> str:
    """Stable id pairing an utterance's partial/final/translation/preview events. The browser
    uses it as the DOM key: final creates the line, translation fills in its ZH later."""
    return f"{label}|{started_at.isoformat()}"


def _hhmmss(started_at: datetime) -> str:
    return f"{started_at:%H:%M:%S}"


def _segments_out(segments: list) -> list[dict]:
    """Serialize final segments [(speaker, text, diff), ...] for the browser. diff is the
    two-pass inline-diff spans [(kind, words), ...] or None/empty; kind is same/del/add."""
    out = []
    for seg in segments:
        speaker, text, diff = seg[0], seg[1], (seg[2] if len(seg) > 2 else None)
        out.append(
            {
                "speaker": speaker,  # int (0-based) or None when diarize is off
                "text": text,
                "diff": [[kind, words] for kind, words in diff] if diff else None,
            }
        )
    return out


def _zh_segments_out(zh_segments: list) -> list[dict]:
    """Serialize translation segments [(speaker, zh), ...] for the browser."""
    return [{"speaker": spk, "zh": zh} for spk, zh in zh_segments]


class WebSink:
    """Broadcast hub. Renderer-compatible methods build JSON events; subscribers drain them."""

    def __init__(self) -> None:
        self._subs: set[queue.Queue] = set()
        self._lock = threading.Lock()
        # Latest pipeline status, replayed to any browser that connects mid-run so a page
        # refresh re-syncs the Start/Stop button and source without waiting for the next event.
        self._status: dict = {"type": "status", "state": "idle", "source": None,
                              "meeting": None, "lang": None}
        # Latest model-preparation progress, also replayed on connect: a first Start can spend
        # many minutes downloading models, and a page opened mid-download must show it too.
        self._progress: dict | None = None

    # --- subscription management (called from HTTP request threads) ---
    def subscribe(self) -> queue.Queue:
        q: queue.Queue = queue.Queue(maxsize=_QUEUE_MAX)
        with self._lock:
            self._subs.add(q)
            # prime the new connection with current status so the UI reflects a run already
            # in progress
            self._offer(q, self._status)
            if self._progress is not None:
                self._offer(q, self._progress)
        return q

    def unsubscribe(self, q: queue.Queue) -> None:
        with self._lock:
            self._subs.discard(q)

    def _offer(self, q: queue.Queue, event: dict) -> None:
        """Non-blocking enqueue; on overflow drop this client's oldest event to make room."""
        try:
            q.put_nowait(event)
        except queue.Full:
            try:
                q.get_nowait()
                q.put_nowait(event)
            except (queue.Empty, queue.Full):
                pass

    def _broadcast(self, event: dict) -> None:
        with self._lock:
            for q in self._subs:
                self._offer(q, event)

    # --- Renderer-compatible pipeline events (called from worker threads) ---
    def partial(
        self, label: str, text: str, started_at: datetime, speaker: int | None = None
    ) -> None:
        self._broadcast(
            {
                "type": "partial",
                "key": _key(label, started_at),
                "label": label,
                "at": _hhmmss(started_at),
                "speaker": speaker,
                "text": text,
            }
        )

    def final(self, label: str, segments: list, started_at: datetime) -> None:
        self._broadcast(
            {
                "type": "final",
                "key": _key(label, started_at),
                "label": label,
                "at": _hhmmss(started_at),
                "segments": _segments_out(segments),
            }
        )

    def translation(self, label: str, zh_segments: list, started_at: datetime) -> None:
        self._broadcast(
            {
                "type": "translation",
                "key": _key(label, started_at),
                "label": label,
                "at": _hhmmss(started_at),
                "segments": _zh_segments_out(zh_segments),
            }
        )

    def preview(self, label: str, zh_segments: list, started_at: datetime) -> None:
        """Live provisional translation of the in-progress utterance (under the EN partial)."""
        self._broadcast(
            {
                "type": "preview",
                "key": _key(label, started_at),
                "label": label,
                "at": _hhmmss(started_at),
                "segments": _zh_segments_out(zh_segments),
            }
        )

    def flush_pending(self) -> None:
        """Translation disabled at runtime (model failed to load): tell the browser to stop
        waiting for ZH lines and reveal any EN it was pairing."""
        self._broadcast({"type": "flush"})

    # --- status / log (server-driven, not part of the Renderer contract) ---
    def status(self, state: str, source: str | None = None, meeting: str | None = None,
               lang: str | None = None) -> None:
        """Pipeline lifecycle: idle / starting / listening / stopping. Carries the current
        meeting id and language so the page can bind notes / select the live meeting / sync the
        language picker. Latched and replayed to new subscribers so a reconnecting browser
        syncs its controls."""
        ev = {"type": "status", "state": state, "source": source, "meeting": meeting,
              "lang": lang}
        with self._lock:
            self._status = ev
            if state != "starting":  # progress is a starting-phase artifact
                self._progress = None
            for q in self._subs:
                self._offer(q, ev)

    def progress(self, message: str | None, pct: float | None = None) -> None:
        """Model-preparation progress while the pipeline is starting. Unlike log banners
        (which auto-hide), the browser shows this persistently until the run is listening —
        it is the only feedback during a multi-GB first-time model download. message=None
        clears the indicator; pct is 0–100 or None when the total is unknown."""
        ev = {"type": "progress", "message": message, "pct": pct}
        with self._lock:
            self._progress = ev if message is not None else None
            for q in self._subs:
                self._offer(q, ev)

    def log(self, level: str, message: str) -> None:
        """A human-facing notice for the browser banner (info / warn)."""
        self._broadcast({"type": "log", "level": level, "message": message})
