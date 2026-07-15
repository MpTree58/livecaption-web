"""Local web UI: run the ASR + translation pipeline behind a browser page.

Audio capture and every model (ASR, VAD, diarization, translation) stay in this Python
process exactly as in the CLI -- the browser is a pure display+control surface reached
over a local HTTP server. Server-Sent Events stream captions to the page; small POST
endpoints start/stop the pipeline and pick the source (mic / system / both).

The heavy objects (Recognizer weights, Translator thread) are built once on the first
Start and kept warm across Start/Stop, so repeated meeting sessions start instantly.
Only the audio sources, ASR workers, and per-session transcript file are torn down on
Stop. This mirrors cli.py's live-mode wiring; the one swap is WebSink in place of the
terminal Renderer (same partial/final/translation/preview/flush_pending contract).
"""

from __future__ import annotations

import json
import os
import sys
import threading
import webbrowser
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from queue import Empty

import typer
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
    """Owns the live pipeline. Each language preset's ASR recognizer and translator are built
    lazily on first use and kept warm in per-config caches, so switching language (or just
    stopping/starting) reuses already-loaded models. Each Start spins up fresh audio sources +
    ASR workers wired to the shared WebSink and the selected language's translator."""

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

        # warm caches, reused across start/stop and language switches
        self._recognizers: dict[str, object] = {}  # recognizer-config key -> recognizer
        self._translators: dict[str, Translator] = {}  # target prompt name -> translator

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

    # --- pipeline construction (cached per config so switching language stays warm) ---
    def _get_recognizer(self, preset: dict) -> object:
        key = "|".join([
            preset["backend"], preset["asr_model_id"], preset["asr_lang"],
            str(self._diarize), str(preset["word_diarize"]),
        ])
        rec = self._recognizers.get(key)
        if rec is None:
            self.sink.log("info", f"Loading ASR model for {preset['label']} …")
            if preset["backend"] == "qwen3":
                from .asr_qwen import build_qwen_recognizer

                rec = build_qwen_recognizer(
                    preset["asr_model_id"], preset["asr_lang"], self._diarize,
                    self._diarize and preset["word_diarize"],
                    log=lambda m: self.sink.log("info", m),
                )
            else:
                rec = build_recognizer(
                    preset["asr_model_id"], preset["asr_lang"], self._diarize,
                    log=lambda m: self.sink.log("info", m),
                )
            self._recognizers[key] = rec
        return rec

    def _get_translator(self, target_prompt: str) -> Translator:
        tr = self._translators.get(target_prompt)
        if tr is None:
            self.sink.log("info", f"Loading translation model ({target_prompt}) …")
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
            self._translators[target_prompt] = tr
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
            recognizer = self._get_recognizer(preset)
            translator = self._get_translator(preset["target_prompt"]) if self._translate else None
            if translator is not None:
                # fresh meeting: don't inherit the previous session's rolling context
                translator.reset_context()
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

    def _stop_impl(self) -> None:
        for src in self._sources:
            src.stop()
        for worker in self._workers:
            worker.join(timeout=15)
            if worker.is_alive():
                self.sink.log("warn", f"ASR worker '{worker.label}' did not finish in time.")
        if self._meeting is not None:
            self._meeting.finalize(datetime.now().isoformat())
            meta = dict(self._meeting.meta)  # snapshot before clearing, for the central sync
            self._meeting = None
            self._sync_meeting(meta)
        self._sources = []
        self._workers = []
        # idle: the just-ended meeting is now saved; the browser refetches the list
        self.sink.status("idle", None)

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
        """Server exit: stop the run (if any) and tear down every warm translator thread."""
        if self._running:
            with self._op_lock:
                self._running = False
            self._stop_impl()
        for tr in self._translators.values():
            tr.stop()
            tr.join(timeout=10)


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
