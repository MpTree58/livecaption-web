"""Menu-bar app: a no-terminal client for non-technical users.

It lives in the macOS menu bar (🎙) and, in the background, hosts the local livecaption
recorder (the same `livecaption-web` server, which uses this Mac's GPU). Users never open a
terminal: they click the menu-bar icon → "Open livecaption" to record / browse in the
browser. Meant to be launched at login (see the LaunchAgent installer), so the recorder is
always ready.

The heavy engine runs as a child process (`python -m livecaption.webapp`), so the menu-bar
process stays light and a crash in the engine can't take down the menu bar -- the supervisor
just restarts it. EngineSupervisor has no GUI dependency and is unit-testable on its own; the
rumps UI is a thin shell over it.
"""

from __future__ import annotations

import atexit
import socket
import subprocess
import sys
import urllib.error
import urllib.request
import webbrowser

from .config import WEB_STOP_DRAIN_SEC

DEFAULT_PORT = 8000
# How long Quit / Restart wait for the engine to exit on SIGTERM before killing it. An idle
# engine exits at once; a recording is stopped like the Stop button (last sentence flushed,
# pending translations drained for up to WEB_STOP_DRAIN_SEC, meeting closed).
_ENGINE_STOP_TIMEOUT = WEB_STOP_DRAIN_SEC + 20


def _pick_port(preferred: int) -> int:
    """Use the preferred port if free, otherwise let the OS hand out a free one. Keeps the app
    working even if something else (a stray recorder, another dev tool) already holds 8000."""
    for candidate in (preferred, 0):
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind(("127.0.0.1", candidate))
            port = s.getsockname()[1]
            s.close()
            return port
        except OSError:
            continue
    return preferred


class EngineSupervisor:
    """Owns the background recorder-engine child process (start / stop / health)."""

    def __init__(self, port: int = DEFAULT_PORT) -> None:
        self.port = port
        self._proc: subprocess.Popen | None = None

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def running(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def start(self) -> None:
        if self.running():
            return
        self.port = _pick_port(self.port)  # avoid a busy 8000
        # run the engine in this same interpreter/venv, no browser (the menu opens it)
        self._proc = subprocess.Popen(
            [sys.executable, "-m", "livecaption.webapp", "--no-open",
             "--host", "127.0.0.1", "--port", str(self.port)],
        )

    def stop(self) -> None:
        if self._proc is not None and self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=_ENGINE_STOP_TIMEOUT)
            except subprocess.TimeoutExpired:
                self._proc.kill()
        self._proc = None

    def restart(self) -> None:
        self.stop()
        self.start()

    def responding(self, timeout: float = 1.0) -> bool:
        """True once the engine's HTTP server is actually accepting requests."""
        try:
            with urllib.request.urlopen(self.url + "/state", timeout=timeout) as r:
                return r.status == 200
        except (urllib.error.URLError, OSError):
            return False


def run() -> None:
    import rumps

    supervisor = EngineSupervisor()
    supervisor.start()
    atexit.register(supervisor.stop)

    app = rumps.App("livecaption", title="🎙", quit_button=None)

    def open_ui(_: object) -> None:
        supervisor.start()  # ensure it's up (e.g. after a crash) before opening
        webbrowser.open(supervisor.url)

    def restart(_: object) -> None:
        supervisor.restart()
        rumps.notification("livecaption", "", "Recorder engine restarted.")

    def quit_app(_: object) -> None:
        supervisor.stop()
        rumps.quit_application()

    app.menu = [
        rumps.MenuItem("Open livecaption", callback=open_ui),
        None,
        rumps.MenuItem("Restart recorder engine", callback=restart),
        rumps.MenuItem("Quit", callback=quit_app),
    ]

    # keep the menu-bar title reflecting engine health
    @rumps.timer(5)
    def _health(_: object) -> None:
        app.title = "🎙" if supervisor.responding() else "🎙…"

    app.run()


if __name__ == "__main__":
    run()
