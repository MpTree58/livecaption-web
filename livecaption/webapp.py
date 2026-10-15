"""Local web UI: run the ASR + translation pipeline behind a browser page.

Audio capture and every model (ASR, VAD, diarization, translation) stay in this Python
process exactly as in the CLI -- the browser is a pure display+control surface reached
over a local HTTP server. Server-Sent Events stream captions to the page; small POST
endpoints start/stop the pipeline and pick the source (mic / system / both).

The heavy objects (Recognizer weights, Translator thread) are built on the first Start and
kept warm across Start/Stop, so repeated meeting sessions start instantly. Only the
current language preset's models stay loaded: switching language frees the previous
preset's ASR bundle, and one translator serves every preset. Only the audio sources, ASR
workers, and per-session transcript file are torn down on Stop. This mirrors cli.py's
live-mode wiring; the one swap is WebSink in place of the terminal Renderer (same
partial/final/translation/preview/flush_pending contract).
"""

from __future__ import annotations

import gc
import json
import os
import sys
import threading
import webbrowser
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from queue import Empty

import mlx.core as mx
import typer
from huggingface_hub import HfApi, snapshot_download
from huggingface_hub import constants as hf_constants
from huggingface_hub.constants import HF_HUB_CACHE, HF_XET_CACHE
from rich.console import Console

from . import config
from .asr import AsrWorker, build_recognizer
from .audio import AudioSource, MicSource, SystemAudioSource
from .languages import normalize_target_language
from .meetings import MeetingStore
from .models import resolve_audiotee
from .translate import Translator
from .websink import WebSink, _key

console = Console()

# Make model downloads observable: hf's default xet transfer stages bytes opaquely
# (in-memory reconstruction; files appear only when finished), so the web UI's byte-level
# progress poller (_download_model) would show ~0% for the entire download. The classic
# HTTP path grows blobs/*.incomplete on disk, which the poller can measure. The env-frozen
# constant is re-read per download call (verified in huggingface_hub 1.18
# file_download.py), so overriding the attribute here disables xet for this process only.
hf_constants.HF_HUB_DISABLE_XET = True

_WEBUI = Path(__file__).with_name("webui.html")
_MEETINGS_DIR = Path.home() / ".cache" / "livecaption" / "meetings"

# Language presets offered in the web UI. Each = a spoken-language ASR engine + a translation
# direction. English uses the low-latency streaming nemotron; Chinese and mixed Chinese/English
# use qwen3 (strong Chinese + native code-switching). Translation direction is the sensible
# "other language" default and is trivially changed here (same MT model, different target).
LANG_PRESETS = {
    "en":    {"label": "English → 中文",  "asr_model": "nemotron-0.6b",   "asr_lang": "en-us", "target": "zh-cn"},  # noqa: E501
    "zh":    {"label": "中文 → English",  "asr_model": "qwen3-1.7b-8bit", "asr_lang": "zh",    "target": "en"},     # noqa: E501
    "mixed": {"label": "中英混合 → 中文", "asr_model": "qwen3-1.7b-8bit", "asr_lang": "zh",    "target": "zh-cn"},  # noqa: E501
}


class Session:
    """Owns the live pipeline. The ASR recognizer and the translator are built lazily on first
    use and kept warm, so stopping/starting reuses already-loaded models. Only the current
    language preset's recognizer is kept: each ASR bundle is 1.5-3 GB of unified memory, so
    switching language frees the previous one (reloading from the disk cache takes seconds).
    One translator serves every preset -- the translation direction is only a prompt
    parameter, re-set per run. Each Start spins up fresh audio sources + ASR workers wired to
    the shared WebSink and the translator."""

    def __init__(
        self,
        sink: WebSink,
        store: MeetingStore,
        *,
        presets: dict,
        default_lang: str,
        diarize: bool,
        translate: bool,
        mt_model_id: str,
        context: int,
        audiotee_path: str | None,
        sync_url: str = "",
        sync_token: str = "",
        idle_unload_min: float = config.WEB_IDLE_UNLOAD_MIN,
    ) -> None:
        self.sink = sink
        self.store = store
        self._presets = presets  # lang key -> resolved pipeline config (see main())
        self._default_lang = default_lang
        self._diarize = diarize
        self._translate = translate
        self._mt_model_id = mt_model_id
        self._context = context
        self._audiotee_path = audiotee_path
        # optional central-server sync: push each finished meeting to a shared library
        self._sync_url = sync_url.rstrip("/")
        self._sync_token = sync_token

        # warm models, reused across start/stop: the current preset's recognizer (with the
        # config key it was built for) and the one translator shared by every preset
        self._recognizer: object | None = None
        self._recognizer_key: str | None = None
        self._translator: Translator | None = None
        # ...but only for a while: an idle server (the menu-bar app keeps it running from
        # login) frees them after idle_unload_min minutes (0 = never)
        self._idle_unload_min = idle_unload_min
        self._idle_timer: threading.Timer | None = None

        # serialize start/stop; _running is the user-visible state
        self._op_lock = threading.Lock()
        self._running = False

        # per-session, rebuilt on each start
        self._sources: list[AudioSource] = []
        self._workers: list[AsrWorker] = []
        self._preview_words: dict[str, int] = {}
        # the meeting record being written this run (the store is the source of truth); a
        # stable reference so the active translator's on_translation can persist ZH lines
        self._meeting = None
        self._source_kind: str | None = None
        self._lang: str = default_lang

    # --- public control (called on HTTP request threads; heavy work is offloaded) ---
    def start(self, source_kind: str, lang: str | None = None) -> dict:
        lang = lang or self._default_lang
        with self._op_lock:
            if self._running:
                return {"ok": False, "error": "already running"}
            if source_kind not in ("mic", "system", "both"):
                return {"ok": False, "error": f"bad source {source_kind!r}"}
            if lang not in self._presets:
                return {"ok": False, "error": f"bad lang {lang!r}"}
            self._running = True
            self._lang = lang
            self._cancel_idle_unload()
        self.sink.status("starting", source_kind, lang=lang)
        threading.Thread(
            target=self._start_impl, args=(source_kind, lang), daemon=True, name="session-start"
        ).start()
        return {"ok": True}

    def stop(self) -> dict:
        with self._op_lock:
            if not self._running:
                return {"ok": False, "error": "not running"}
            self._running = False
        self.sink.status("stopping", None)
        threading.Thread(
            target=self._stop_impl, daemon=True, name="session-stop"
        ).start()
        return {"ok": True}

    def state(self) -> dict:
        return {
            "running": self._running,
            "translate": self._translate,
            "meeting": self._meeting.id if self._meeting else None,
            "lang": self._lang,
            "langs": [{"key": k, "label": p["label"]} for k, p in self._presets.items()],
        }

    def _handle_translation(
        self, label: str, zh_segments: list, started_at: datetime
    ) -> None:
        """Shared translator's on_translation (wired once, reused across runs): persist the ZH
        into the current meeting and forward to the browser. Reads self._meeting live so it
        always targets the run in progress."""
        m = self._meeting
        if m is not None:
            m.add_translation(_key(label, started_at), zh_segments)
        self.sink.translation(label, zh_segments, started_at)

    # --- pipeline construction (models stay warm for the current language preset) ---
    def _stage(self, message: str) -> None:
        """A model-preparation stage notice, shown on the persistent progress line (not the
        auto-hiding banner — during a long first start that line is the user's only feedback,
        and mirroring it into the banner just paints the same text twice)."""
        self.sink.progress(message)

    def _prepare_models(self, preset: dict) -> None:
        """Download this run's models into the HF cache before loading, with visible progress.
        Without this, a first Start on a fresh machine downloads several GB behind a bare
        'Starting…'. Warm caches skip straight through (offline-safe); a failed download is
        only warned about here — the loader below retries and its failure ends the start."""
        repos = [(preset["asr_model_id"], "speech model"), (config.VAD_MODEL, "voice detector")]
        if self._diarize:
            repos.append((config.DIAR_MODEL, "speaker model"))
        if self._translate:
            repos.append((self._mt_model_id, "translation model"))
        for i, (repo, what) in enumerate(repos, 1):
            try:
                _download_model(repo, f"Downloading {what} ({i}/{len(repos)})",
                                self.sink.progress)
            except Exception as e:  # noqa: BLE001 -- e.g. offline: let the cache-based load decide
                self.sink.log("warn", f"Could not download {repo}: {e}")

    def _get_recognizer(self, preset: dict) -> object:
        key = "|".join([
            preset["backend"], preset["asr_model_id"], preset["asr_lang"],
            str(self._diarize), str(preset["word_diarize"]),
        ])
        if self._recognizer is not None and self._recognizer_key == key:
            return self._recognizer
        # a different preset: free the previous bundle BEFORE loading the next one, so the
        # two never sit in memory together
        self._release_recognizer()
        self._stage(f"Loading ASR model for {preset['label']} …")
        if preset["backend"] == "qwen3":
            from .asr_qwen import build_qwen_recognizer

            rec = build_qwen_recognizer(
                preset["asr_model_id"], preset["asr_lang"], self._diarize,
                self._diarize and preset["word_diarize"],
                log=self._stage,
            )
        else:
            rec = build_recognizer(
                preset["asr_model_id"], preset["asr_lang"], self._diarize,
                log=self._stage,
            )
        self._recognizer, self._recognizer_key = rec, key
        return rec

    def _release_recognizer(self) -> None:
        """Drop the warm recognizer and hand its memory back (no-op when none is loaded).
        Only called while no run is active, so no ASR worker still references it."""
        if self._recognizer is None:
            return
        self._recognizer = None
        self._recognizer_key = None
        _free_mlx_memory()

    # --- idle unload: don't hold several GB of unified memory between meetings ---
    def _arm_idle_unload(self) -> None:
        if self._idle_unload_min <= 0:
            return
        with self._op_lock:
            self._cancel_idle_unload()
            timer = threading.Timer(self._idle_unload_min * 60, self._unload_idle_models)
            timer.daemon = True  # never hold up process exit
            timer.start()
            self._idle_timer = timer

    def _cancel_idle_unload(self) -> None:
        """Caller holds _op_lock."""
        if self._idle_timer is not None:
            self._idle_timer.cancel()
            self._idle_timer = None

    def _unload_idle_models(self) -> None:
        """Free every model after the configured idle time; the next Start reloads them from
        the disk cache in seconds. Runs under _op_lock, so a concurrent Start either already
        claimed the session (then nothing is unloaded) or waits for the unload to finish."""
        with self._op_lock:
            self._idle_timer = None
            if self._running or (self._recognizer is None and self._translator is None):
                return
            if self._translator is not None:
                # idle means drained: the thread is just waiting on its queue
                self._translator.stop()
                self._translator.join(timeout=10)
                self._translator = None
            self._recognizer = None
            self._recognizer_key = None
            _free_mlx_memory()
        console.print(
            f"[dim]Models unloaded after {self._idle_unload_min:g} idle minutes; the next "
            "Start reloads them.[/dim]"
        )

    def _get_translator(self, target_prompt: str) -> Translator:
        tr = self._translator
        # (re)build on first use, or when the previous one's model failed to load (its
        # thread has exited) -- a fresh Start retries instead of staying silently disabled
        if tr is None or not tr.is_alive():
            self.sink.log("info", "Loading translation model …")
            tr = Translator(
                self._mt_model_id,
                target_prompt,
                on_translation=self._handle_translation,
                on_ready=lambda: self.sink.log("info", "Translation model ready."),
                on_failed=self.sink.flush_pending,
                on_preview=self.sink.preview,
                context_size=self._context,
            )
            tr.start()
            self._translator = tr
        return tr

    def _build_sources(self, source_kind: str) -> list[AudioSource]:
        sources: list[AudioSource] = []
        if source_kind in ("mic", "both"):
            sources.append(MicSource(label="me"))
        if source_kind in ("system", "both"):
            audiotee_bin = resolve_audiotee(self._audiotee_path)
            sources.append(SystemAudioSource(audiotee_bin, label="them"))
        return sources

    def _start_impl(self, source_kind: str, lang: str) -> None:
        try:
            preset = self._presets[lang]
            self._prepare_models(preset)
            recognizer = self._get_recognizer(preset)
            translator = self._get_translator(preset["target_prompt"]) if self._translate else None
            if translator is not None:
                # fresh meeting: don't inherit the previous session's rolling context, and
                # translate in this preset's direction
                translator.reset_context(preset["target_prompt"])
            self._preview_words.clear()

            sources = self._build_sources(source_kind)
            meeting = self.store.create(
                source=source_kind,
                asr_lang=preset["asr_lang"],
                target_lang=preset["target_prompt"] if self._translate else None,
                created_at=datetime.now(),
                lang=lang,
                lang_label=preset["label"],
            )
            self._meeting = meeting

            def handle_partial(
                label: str, text: str, started_at: datetime, speaker: int | None = None
            ) -> None:
                self.sink.partial(label, text, started_at, speaker)
                if translator and config.MT_PREVIEW_MIN_WORDS > 0:
                    nwords = len(text.split())
                    if nwords - self._preview_words.get(label, 0) >= config.MT_PREVIEW_MIN_WORDS \
                            and any(c in text for c in ".?!"):
                        self._preview_words[label] = nwords
                        translator.submit_preview(label, [(speaker, text, None)], started_at)

            def handle_final(label: str, segments: list, started_at: datetime) -> None:
                self._preview_words.pop(label, None)
                # send full segments (incl. two-pass diff spans); the browser can toggle diff
                self.sink.final(label, segments, started_at)
                meeting.add_final(
                    _key(label, started_at), label, f"{started_at:%H:%M:%S}",
                    started_at.isoformat(), segments,
                )
                if translator:
                    translator.submit(label, segments, started_at)

            def on_worker_error() -> None:
                self.sink.log("warn", "ASR worker crashed; stopping.")
                self.stop()

            workers: list[AsrWorker] = []
            for src in sources:
                worker = AsrWorker(
                    recognizer, src.queue, src.label, handle_partial, handle_final,
                    on_error=on_worker_error,
                    use_audio_timestamps=False,
                )
                workers.append(worker)
                worker.start()
                src.start()

            self._sources = sources
            self._workers = workers
            self._source_kind = source_kind

            self.sink.status("listening", source_kind, meeting.id, lang=lang)
            self.sink.log("info", f"Listening: {' + '.join(s.label for s in sources)}")
            if any(isinstance(s, SystemAudioSource) for s in sources):
                self.sink.log(
                    "info",
                    "If system audio stays empty, Terminal likely lacks System Audio "
                    "Recording permission — see README.",
                )
        except Exception as e:  # noqa: BLE001 -- any failure must not leave a half-started run
            self.sink.log("warn", f"Failed to start: {e}")
            self.sink.status("idle", None)
            self._meeting = None
            with self._op_lock:
                self._running = False
            self._arm_idle_unload()  # whatever did load before the failure

    def _stop_impl(self) -> None:
        for src in self._sources:
            src.stop()
        for worker in self._workers:
            worker.join(timeout=15)
            if worker.is_alive():
                self.sink.log("warn", f"ASR worker '{worker.label}' did not finish in time.")
        # the last finals were only just handed to the translator: let it catch up before
        # closing the meeting, or those lines would be saved without their translation
        tr = self._translator
        if self._meeting is not None and tr is not None \
                and not tr.drain(timeout=config.WEB_STOP_DRAIN_SEC) and tr.is_alive():
            self.sink.log("warn", "Translation was still catching up; the last lines were "
                                  "saved without it.")
        if self._meeting is not None:
            self._meeting.finalize(datetime.now().isoformat())
            meta = dict(self._meeting.meta)  # snapshot before clearing, for the central sync
            self._meeting = None
            self._sync_meeting(meta)
        self._sources = []
        self._workers = []
        # idle: the just-ended meeting is now saved; the browser refetches the list
        self.sink.status("idle", None)
        self._arm_idle_unload()

    def _sync_meeting(self, meta: dict) -> None:
        """Push a finished meeting to the central server (best-effort, in the background).

        The local store remains the capture-side source of truth; the server copy is what the
        team browses. Ingest is idempotent (upsert by owner+id), so a retry never duplicates.
        Failures only warn -- nothing is lost locally."""
        if not (self._sync_url and self._sync_token):
            return

        def _push() -> None:
            import urllib.error
            import urllib.request

            body = json.dumps(meta, ensure_ascii=False).encode("utf-8")
            req = urllib.request.Request(
                f"{self._sync_url}/api/ingest", data=body, method="POST",
                headers={"Content-Type": "application/json",
                         "Authorization": f"Bearer {self._sync_token}"},
            )
            try:
                with urllib.request.urlopen(req, timeout=30) as resp:  # noqa: S310 -- our own URL
                    if resp.status == 200:
                        self.sink.log("info", "Synced meeting to the central server.")
                        return
                self.sink.log("warn", f"Central sync failed (HTTP {resp.status}).")
            except urllib.error.HTTPError as e:
                detail = "bad agent token" if e.code == 401 else f"HTTP {e.code}"
                self.sink.log("warn", f"Central sync failed ({detail}). Saved locally.")
            except Exception as e:  # noqa: BLE001 -- network errors must not break stop
                self.sink.log("warn", f"Central sync failed ({e}). Saved locally.")

        threading.Thread(target=_push, daemon=True, name="meeting-sync").start()

    # --- meeting edits: route the live meeting through its in-memory object (which autosaves
    # on every utterance and would otherwise clobber a direct disk write); past ones via the store ---  # noqa: E501
    def rename_meeting(self, mid: str, title: str) -> bool:
        m = self._meeting
        if m is not None and m.id == mid:
            m.set_title(title)
            return True
        return self.store.rename(mid, title)

    def set_meeting_notes(self, mid: str, notes: str) -> bool:
        m = self._meeting
        if m is not None and m.id == mid:
            m.set_notes(notes)
            return True
        return self.store.set_notes(mid, notes)

    def delete_meeting(self, mid: str) -> bool:
        m = self._meeting
        if m is not None and m.id == mid:
            return False  # refuse to delete the meeting currently being recorded
        return self.store.delete(mid)

    def shutdown(self) -> None:
        """Server exit: stop the run (if any) and tear down the warm translator thread."""
        if self._running:
            with self._op_lock:
                self._running = False
            self._stop_impl()
        with self._op_lock:
            self._cancel_idle_unload()
        if self._translator is not None:
            self._translator.stop()
            self._translator.join(timeout=10)


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    # the server owns these (set as attributes on the ThreadingHTTPServer instance)
    @property
    def _sink(self) -> WebSink:
        return self.server.sink  # type: ignore[attr-defined]

    @property
    def _session(self) -> Session:
        return self.server.session  # type: ignore[attr-defined]

    def log_message(self, *_args) -> None:  # noqa: ANN002 -- silence per-request stderr spam
        pass

    def _send_json(self, obj: dict, status: int = 200) -> None:
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    @property
    def _store(self) -> MeetingStore:
        return self._session.store

    def _read_body(self) -> bytes:
        """Read (and thereby drain) the entire request body. On HTTP/1.1 keep-alive the body
        MUST be consumed even when unused, or its bytes stay in the socket and corrupt the next
        request on the connection (seen as a bogus '{}GET' method → 501)."""
        length = int(self.headers.get("Content-Length", 0) or 0)
        return self.rfile.read(length) if length > 0 else b""

    def _body_json(self, raw: bytes) -> dict:
        try:
            return json.loads(raw or b"{}")
        except json.JSONDecodeError:
            return {}

    def do_GET(self) -> None:  # noqa: N802
        path = self.path.split("?", 1)[0]
        if path in ("/", "/index.html"):
            self._serve_html()
        elif path == "/events":
            self._serve_sse()
        elif path == "/state":
            self._send_json(self._session.state())
        elif path == "/meetings":
            self._send_json({"meetings": self._store.list()})
        elif path == "/search":
            from urllib.parse import parse_qs, urlparse

            q = parse_qs(urlparse(self.path).query).get("q", [""])[0]
            self._send_json({"results": self._store.search(q)})
        elif path.startswith("/meetings/") and path.endswith("/export.md"):
            self._serve_export(path.split("/")[2])
        elif path.startswith("/meetings/"):
            data = self._store.load(path.split("/")[2])
            self._send_json(data if data else {"error": "not found"}, 200 if data else 404)
        else:
            self._send_json({"error": "not found"}, status=404)

    def do_POST(self) -> None:  # noqa: N802
        body = self._body_json(self._read_body())  # always drain the body (keep-alive safety)
        path = self.path.split("?", 1)[0]
        if path == "/start":
            self._send_json(self._session.start(body.get("source", "both"), body.get("lang")))
        elif path == "/stop":
            self._send_json(self._session.stop())
        elif path.startswith("/meetings/") and path.endswith("/rename"):
            ok = self._session.rename_meeting(path.split("/")[2], body.get("title", ""))
            self._send_json({"ok": ok}, 200 if ok else 404)
        elif path.startswith("/meetings/") and path.endswith("/notes"):
            ok = self._session.set_meeting_notes(path.split("/")[2], body.get("notes", ""))
            self._send_json({"ok": ok}, 200 if ok else 404)
        else:
            self._send_json({"error": "not found"}, status=404)

    def do_DELETE(self) -> None:  # noqa: N802
        self._read_body()  # drain any body for keep-alive safety
        path = self.path.split("?", 1)[0]
        if path.startswith("/meetings/"):
            ok = self._session.delete_meeting(path.split("/")[2])
            self._send_json({"ok": ok}, 200 if ok else 409)
        else:
            self._send_json({"error": "not found"}, status=404)

    def _serve_html(self) -> None:
        body = _WEBUI.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _serve_export(self, mid: str) -> None:
        md = self._store.export_md(mid)
        if md is None:
            self._send_json({"error": "not found"}, status=404)
            return
        body = md.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/markdown; charset=utf-8")
        self.send_header("Content-Disposition", f'attachment; filename="{mid}.md"')
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _serve_sse(self) -> None:
        q = self._sink.subscribe()
        try:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self.end_headers()
            self.wfile.write(b": connected\n\n")
            self.wfile.flush()
            while True:
                try:
                    ev = q.get(timeout=15)
                except Empty:
                    # heartbeat: keeps the connection alive and surfaces a dead client as a
                    # write error so the thread can exit
                    self.wfile.write(b": hb\n\n")
                    self.wfile.flush()
                    continue
                if ev is None:
                    break
                payload = "data: " + json.dumps(ev, ensure_ascii=False) + "\n\n"
                self.wfile.write(payload.encode("utf-8"))
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, ValueError, OSError):
            pass  # browser navigated away / closed the tab
        finally:
            self._sink.unsubscribe(q)


class _StderrTee:
    """Pass stderr through unchanged, but also forward completed `[warn] …` lines to the
    browser banner. The system-audio silence/permission warning is printed to stderr deep in
    SystemAudioSource; this surfaces it in the web UI without touching that code."""

    def __init__(self, real, sink: WebSink) -> None:  # noqa: ANN001 -- real is a text stream
        self._real = real
        self._sink = sink
        self._buf = ""

    def write(self, s: str) -> int:
        n = self._real.write(s)
        self._buf += s
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            if "[warn]" in line:
                self._sink.log("warn", line.split("[warn]", 1)[1].strip())
        return n

    def flush(self) -> None:
        self._real.flush()

    def __getattr__(self, name):  # noqa: ANN001 -- delegate isatty/encoding/etc.
        return getattr(self._real, name)


def _free_mlx_memory() -> None:
    """Hand the memory of just-dropped models back to the OS: collect the dropped objects
    (module trees and the closures around them can form reference cycles), then empty
    MLX's buffer cache, which otherwise keeps freed Metal buffers for reuse."""
    gc.collect()
    mx.clear_cache()


def _dir_bytes(root: Path) -> int:
    total = 0
    if root.is_dir():
        for f in root.rglob("*"):
            try:
                if f.is_file():
                    total += f.stat().st_size
            except OSError:
                continue
    return total


def _download_bytes_so_far(repo: str, xet_baseline: int) -> int:
    """Bytes of `repo` on disk so far. hf's xet transfer stages chunks in the shared
    HF_XET_CACHE and only materializes the repo blob at the end, so the repo dir alone
    sits near zero for the whole download — count the repo dir (absolute, covers resumed
    plain-HTTP .incomplete blobs) plus the xet cache's growth since this download began."""
    repo_dir = Path(HF_HUB_CACHE) / f"models--{repo.replace('/', '--')}"
    return _dir_bytes(repo_dir) + max(0, _dir_bytes(Path(HF_XET_CACHE)) - xet_baseline)


def _fmt_bytes(n: float) -> str:
    return f"{n / 1e9:.1f} GB" if n >= 1e9 else f"{n / 1e6:.0f} MB"


def _download_model(repo: str, label: str, progress) -> None:  # noqa: ANN001 -- progress = WebSink.progress
    """Ensure `repo` is fully in the local HF cache, reporting byte-level progress via
    `progress(message, pct)`. Fully-cached repos return instantly and offline (no progress
    noise, no network). snapshot_download gives no usable per-byte hook, so a poller thread
    measures the cache directory growing against the total size from the hub's metadata."""
    try:
        snapshot_download(repo, local_files_only=True)
        return
    except Exception:  # noqa: BLE001 -- any miss/corruption just means: download below
        pass

    progress(f"{label}…", None)
    try:
        info = HfApi().model_info(repo, files_metadata=True)
        total = sum(s.size for s in info.siblings if s.size) or None
    except Exception:  # noqa: BLE001 -- metadata is progress-only sugar, never a blocker
        total = None

    stop = threading.Event()
    xet_baseline = _dir_bytes(Path(HF_XET_CACHE))

    def poll() -> None:
        while not stop.wait(1.0):
            done = _download_bytes_so_far(repo, xet_baseline)
            if total:
                pct = min(99.0, done / total * 100)
                progress(f"{label} — {_fmt_bytes(min(done, total))} / {_fmt_bytes(total)}", pct)
            else:
                progress(f"{label} — {_fmt_bytes(done)} so far", None)

    poller = threading.Thread(target=poll, daemon=True, name="model-dl-progress")
    poller.start()
    try:
        snapshot_download(repo)
    finally:
        stop.set()
        poller.join(timeout=2)


def _resolve_asr(asr_model: str) -> tuple[str, str]:
    if asr_model not in config.ASR_MODELS:
        raise typer.BadParameter(
            f"unknown --asr-model {asr_model!r}; choices: {', '.join(config.ASR_MODELS)}"
        )
    return config.ASR_MODELS[asr_model]


def _resolve_mt(mt_model: str) -> str:
    if mt_model not in config.MT_MODELS:
        raise typer.BadParameter(
            f"unknown --mt-model {mt_model!r}; choices: {', '.join(config.MT_MODELS)}"
        )
    return config.MT_MODELS[mt_model]


def _resolve_presets() -> dict:
    """Resolve LANG_PRESETS (UI aliases) to concrete pipeline configs (backend, model id,
    translation prompt name)."""
    out = {}
    for key, p in LANG_PRESETS.items():
        backend, model_id = _resolve_asr(p["asr_model"])
        out[key] = {
            "label": p["label"],
            "backend": backend,
            "asr_model_id": model_id,
            "asr_lang": p["asr_lang"],
            "word_diarize": p.get("word_diarize", False),
            "target_prompt": normalize_target_language(p["target"]).prompt_name,
        }
    return out


def main(
    host: str = typer.Option("127.0.0.1", help="Bind address (default: localhost only)"),
    port: int = typer.Option(8000, help="Port to serve the web UI on"),
    open_browser: bool = typer.Option(
        True, "--open/--no-open", help="Open the page in your browser on startup"
    ),
    default_lang: str = typer.Option(
        "en", "--default-lang", help="Default language preset: en / zh / mixed (pick in the UI)"
    ),
    diarize: bool = typer.Option(True, "--diarize/--no-diarize", help="Speaker diarization"),
    translate: bool = typer.Option(True, "--translate/--no-translate", help="Enable translation"),
    mt_model: str = typer.Option(
        config.DEFAULT_MT_MODEL_ALIAS, "--mt-model", metavar="MODEL",
        help=f"Translation model ({', '.join(config.MT_MODELS)})",
    ),
    context: int = typer.Option(
        config.MT_CONTEXT_SENTENCES, help="Prior sentences as translation context (0=off)"
    ),
    idle_unload: float = typer.Option(
        config.WEB_IDLE_UNLOAD_MIN, "--idle-unload", metavar="MINUTES",
        help="Free the models after this many idle minutes; the next Start reloads them in "
        "seconds (0 = keep them loaded)",
    ),
    audiotee: str | None = typer.Option(None, help="Path to the audiotee binary"),
    sync_url: str | None = typer.Option(
        None, "--sync-url",
        help="Central server URL to push finished meetings to (or set env LC_SYNC_URL)",
    ),
    sync_token: str | None = typer.Option(
        None, "--sync-token",
        help="Your agent API token for the central server (or set env LC_SYNC_TOKEN)",
    ),
) -> None:
    mt_model_id = _resolve_mt(mt_model)
    if default_lang not in LANG_PRESETS:
        console.print(f"[red]--default-lang must be one of: {', '.join(LANG_PRESETS)}[/red]")
        raise typer.Exit(1)
    presets = _resolve_presets()
    # sync config: CLI flag wins, else env var, else off (local-only)
    sync_url = sync_url or os.environ.get("LC_SYNC_URL", "")
    sync_token = sync_token or os.environ.get("LC_SYNC_TOKEN", "")

    sink = WebSink()
    store = MeetingStore(_MEETINGS_DIR)
    session = Session(
        sink,
        store,
        presets=presets,
        default_lang=default_lang,
        diarize=diarize,
        translate=translate,
        mt_model_id=mt_model_id,
        context=context,
        audiotee_path=audiotee,
        sync_url=sync_url,
        sync_token=sync_token,
        idle_unload_min=idle_unload,
    )

    # surface the deep-in-the-stack system-audio silence warning to the browser banner
    sys.stderr = _StderrTee(sys.stderr, sink)

    server = ThreadingHTTPServer((host, port), _Handler)
    server.daemon_threads = True
    server.sink = sink  # type: ignore[attr-defined]
    server.session = session  # type: ignore[attr-defined]

    url = f"http://{host}:{port}"
    console.print(f"[bold green]livecaption web[/bold green] → {url}")
    console.print("[dim]Open the page, pick a source, and click Start. Ctrl-C here to quit.[/dim]")
    if sync_url and sync_token:
        console.print(f"[dim]Central sync ON → finished meetings push to {sync_url}[/dim]")
    elif sync_url or sync_token:
        console.print(
            "[yellow]Central sync needs BOTH --sync-url and --sync-token; sync is OFF.[/yellow]"
        )
    if open_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        console.print("\n[dim]Shutting down …[/dim]")
    finally:
        session.shutdown()
        server.shutdown()
        server.server_close()
    console.print("[dim]Stopped.[/dim]")


def run() -> None:
    typer.run(main)


if __name__ == "__main__":
    run()
