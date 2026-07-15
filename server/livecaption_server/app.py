"""FastAPI app: login sessions, agent ingest, and the multi-user meeting library API.

Two auth paths:
- Browser users log in (POST /api/login) and get a signed session cookie; the web UI calls
  the /api/* endpoints with that cookie.
- Mac capture agents push meetings to POST /api/ingest with an ``Authorization: Bearer
  <api_token>`` header (the token an admin handed them).

Owners can edit / delete / share their meetings; shared or public meetings are read-only to
everyone else. No audio or models here -- this serves stored records only.
"""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import Body, Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from starlette.middleware.sessions import SessionMiddleware

from . import db

_STATIC = Path(__file__).parent / "static"

# initialize storage + load the persistent session key at import so the middleware stack is
# built before the first request (adding middleware in a startup hook is too late)
db.init_db()
app = FastAPI(title="livecaption central server")
app.add_middleware(SessionMiddleware, secret_key=db.get_secret_key(), max_age=14 * 24 * 3600)


# ---- auth helpers ----
def current_user(request: Request) -> dict:
    """Browser-session auth: resolve the logged-in user or 401."""
    uid = request.session.get("uid")
    user = db.get_user_by_id(uid) if uid else None
    if not user:
        raise HTTPException(status_code=401, detail="not logged in")
    return user


def agent_user(authorization: str = Header(default="")) -> dict:
    """Agent auth: resolve the user owning the bearer API token or 401."""
    token = authorization[7:].strip() if authorization.lower().startswith("bearer ") else ""
    user = db.get_user_by_token(token)
    if not user:
        raise HTTPException(status_code=401, detail="bad or missing agent token")
    return user


# ---- pages ----
@app.get("/")
def index() -> FileResponse:
    return FileResponse(_STATIC / "index.html")


# ---- session ----
@app.post("/api/login")
def login(request: Request, body: dict = Body(...)) -> JSONResponse:
    user = db.check_login(body.get("username", ""), body.get("password", ""))
    if not user:
        raise HTTPException(status_code=401, detail="wrong username or password")
    request.session["uid"] = user["id"]
    return JSONResponse({"ok": True, "username": user["username"]})


@app.post("/api/logout")
def logout(request: Request) -> JSONResponse:
    request.session.clear()
    return JSONResponse({"ok": True})


@app.get("/api/me")
def me(user: dict = Depends(current_user)) -> dict:
    return {"username": user["username"], "is_admin": bool(user["is_admin"])}


@app.post("/api/change-password")
def change_password(body: dict = Body(...), user: dict = Depends(current_user)) -> JSONResponse:
    if len(body.get("new_password", "")) < 4:
        raise HTTPException(status_code=400, detail="new password too short (min 4)")
    if not db.change_password(user["id"], body.get("old_password", ""), body["new_password"]):
        raise HTTPException(status_code=400, detail="current password is wrong")
    return JSONResponse({"ok": True})


# ---- agent ingest ----
@app.post("/api/ingest")
def ingest(meeting: dict = Body(...), user: dict = Depends(agent_user)) -> dict:
    if not isinstance(meeting, dict) or "id" not in meeting:
        raise HTTPException(status_code=400, detail="expected a meeting object with an 'id'")
    sid = db.upsert_meeting(user["id"], meeting)
    return {"ok": True, "id": sid}


# ---- library (browser) ----
@app.get("/api/meetings")
def meetings(user: dict = Depends(current_user)) -> dict:
    return {"meetings": db.list_meetings(user["id"])}


@app.get("/api/search")
def search(q: str = "", user: dict = Depends(current_user)) -> dict:
    return {"results": db.search(user["id"], q)}


@app.get("/api/meetings/{mid}")
def get_meeting(mid: str, user: dict = Depends(current_user)) -> dict:
    m = db.get_meeting(mid, user["id"])
    if m is None:
        raise HTTPException(status_code=404, detail="not found")
    return m


@app.get("/api/meetings/{mid}/export.md")
def export_md(mid: str, user: dict = Depends(current_user)) -> PlainTextResponse:
    md = db.export_md(mid, user["id"])
    if md is None:
        raise HTTPException(status_code=404, detail="not found")
    return PlainTextResponse(
        md, media_type="text/markdown; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{mid}.md"'},
    )


@app.post("/api/meetings/{mid}/notes")
def set_notes(mid: str, body: dict = Body(...), user: dict = Depends(current_user)) -> dict:
    return {"ok": db.update_notes(mid, user["id"], body.get("notes", ""))}


@app.post("/api/meetings/{mid}/rename")
def rename(mid: str, body: dict = Body(...), user: dict = Depends(current_user)) -> dict:
    return {"ok": db.update_title(mid, user["id"], body.get("title", ""))}


@app.post("/api/meetings/{mid}/visibility")
def visibility(mid: str, body: dict = Body(...), user: dict = Depends(current_user)) -> dict:
    return {"ok": db.set_visibility(mid, user["id"], body.get("visibility", "private"))}


@app.post("/api/meetings/{mid}/share")
def share(mid: str, body: dict = Body(...), user: dict = Depends(current_user)) -> dict:
    return {"ok": db.share_meeting(mid, user["id"], body.get("username", ""))}


@app.post("/api/meetings/{mid}/unshare")
def unshare(mid: str, body: dict = Body(...), user: dict = Depends(current_user)) -> dict:
    return {"ok": db.unshare_meeting(mid, user["id"], body.get("username", ""))}


@app.delete("/api/meetings/{mid}")
def delete(mid: str, user: dict = Depends(current_user)) -> dict:
    return {"ok": db.delete_meeting(mid, user["id"])}


def run() -> None:
    import uvicorn

    host = os.environ.get("LC_SERVER_HOST", "0.0.0.0")  # noqa: S104 -- LAN service by design
    port = int(os.environ.get("LC_SERVER_PORT", "8080"))
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    run()
