"""Pure-HTTP smoke test (no models, seconds): the web API refuses other websites.

Serves webapp._Handler on a free port with a throwaway meeting store, then sends requests the
way a browser would from (a) the page itself, (b) another website, (c) a DNS-rebinding page
(its own domain re-pointed at 127.0.0.1), plus non-browser clients that send no Origin.
Session.start is replaced by a recorder, so even against a vulnerable build a forged /start
can never load a model or open the microphone -- it is only counted.
    uv run python scripts/smoke_web_guard.py
"""

import http.client
import json
import tempfile
import threading
from http.server import ThreadingHTTPServer
from pathlib import Path

from livecaption import config, webapp
from livecaption.meetings import MeetingStore
from livecaption.websink import WebSink

with tempfile.TemporaryDirectory() as td:
    sink = WebSink()
    session = webapp.Session(
        sink, MeetingStore(Path(td)), presets=webapp._resolve_presets(), default_lang="en",
        diarize=False, translate=False, mt_model_id=config.DEFAULT_MT_MODEL, context=0,
        audiotee_path=None,
    )
    started: list[tuple] = []
    session.start = lambda source, lang=None: started.append((source, lang)) or {"ok": True}
    server = ThreadingHTTPServer(("127.0.0.1", 0), webapp._Handler)
    server.daemon_threads = True
    server.sink, server.session = sink, session
    port = server.server_address[1]
    threading.Thread(target=server.serve_forever, daemon=True).start()
    page = f"http://127.0.0.1:{port}"

    def request(method: str, path: str, headers: dict, body: dict | None = None) -> int:
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
        data = json.dumps(body).encode() if body is not None else None
        conn.request(method, path, body=data, headers=headers)
        resp = conn.getresponse()
        if not (path == "/events" and resp.status == 200):  # an accepted SSE stream never ends
            resp.read()  # drain: closing with unread bytes resets the server's connection
        conn.close()
        return resp.status

    own = {"Host": f"127.0.0.1:{port}"}
    cases = [
        # (what, method, path, headers, body, expected status)
        ("page loads", "GET", "/", own, None, 200),
        ("page reads meetings", "GET", "/meetings", own, None, 200),
        ("localhost name works", "GET", "/state", {"Host": f"localhost:{port}"}, None, 200),
        ("page's own POST", "POST", "/stop", {**own, "Origin": page}, {}, 200),
        ("curl / menu bar (no Origin)", "POST", "/stop", own, {}, 200),
        ("rebinding read", "GET", "/meetings", {"Host": f"evil.example:{port}"}, None, 403),
        ("rebinding SSE", "GET", "/events", {"Host": f"evil.example:{port}"}, None, 403),
        ("other site starts the mic", "POST", "/start",
         {**own, "Origin": "https://evil.example", "Content-Type": "text/plain"},
         {"source": "mic", "lang": "en"}, 403),
        ("other site rewrites notes", "POST", "/meetings/20260101-000000/notes",
         {**own, "Origin": "https://evil.example"}, {"notes": "x"}, 403),
        ("other site deletes", "DELETE", "/meetings/20260101-000000",
         {**own, "Origin": "https://evil.example"}, None, 403),
        ("sandboxed iframe (Origin: null)", "POST", "/stop", {**own, "Origin": "null"}, {}, 403),
        ("path traversal id", "GET", "/meetings/..", own, None, 404),
    ]
    failed = []
    for what, method, path, headers, body, want in cases:
        got = request(method, path, headers, body)
        print(f"{'ok ' if got == want else 'BAD'} {got} (want {want})  {what}")
        if got != want:
            failed.append(what)
    assert not started, f"a forged request reached Session.start: {started}"
    server.shutdown()
    assert not failed, f"unexpected responses: {failed}"

print("web guard smoke test PASSED")
