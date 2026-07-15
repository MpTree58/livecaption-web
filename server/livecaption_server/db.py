"""SQLite data layer: users, meetings, and per-meeting shares.

Access model: a meeting is owned by the user whose agent uploaded it. By default it is
``private`` (only the owner sees it). The owner can either flip it to ``public`` (visible
to the whole team) or share it with specific users. Only the owner can edit / delete a
meeting; everyone else with access is read-only.

The DB file and the session-signing key live under the data dir (env ``LC_SERVER_DATA``,
default ``server/data``), which is git-ignored -- meeting content / company info never
touches the public repo.
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path

from .auth import hash_password, new_api_token, new_secret_key, verify_password

DATA_DIR = Path(os.environ.get("LC_SERVER_DATA", Path(__file__).resolve().parents[1] / "data"))
DB_PATH = DATA_DIR / "livecaption.db"
_SECRET_PATH = DATA_DIR / "secret_key"

# SQLite serializes writes itself, but a module lock keeps our read-modify-write helpers
# (e.g. upsert) atomic across FastAPI's threadpool workers.
_LOCK = threading.RLock()

_SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    username     TEXT UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    api_token    TEXT UNIQUE NOT NULL,
    is_admin     INTEGER NOT NULL DEFAULT 0,
    created_at   TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS meetings (
    id          TEXT PRIMARY KEY,              -- server id: "<owner_id>_<client_id>"
    owner_id    INTEGER NOT NULL,
    client_id   TEXT NOT NULL,                 -- the Mac agent's meeting id
    title       TEXT NOT NULL,
    created_at  TEXT,
    ended_at    TEXT,
    source      TEXT,
    lang        TEXT,
    lang_label  TEXT,
    notes       TEXT NOT NULL DEFAULT '',
    visibility  TEXT NOT NULL DEFAULT 'private',   -- 'private' | 'public'
    data_json   TEXT NOT NULL,                 -- utterances array as JSON
    updated_at  TEXT,
    FOREIGN KEY (owner_id) REFERENCES users(id)
);
CREATE TABLE IF NOT EXISTS shares (
    meeting_id  TEXT NOT NULL,
    user_id     INTEGER NOT NULL,
    PRIMARY KEY (meeting_id, user_id)
);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db() -> None:
    DATA_DIR.mkdir(mode=0o700, parents=True, exist_ok=True)
    with _LOCK, _connect() as conn:
        conn.executescript(_SCHEMA)


def get_secret_key() -> str:
    """Load (or first-time create) the persistent session-signing key."""
    DATA_DIR.mkdir(mode=0o700, parents=True, exist_ok=True)
    if _SECRET_PATH.exists():
        return _SECRET_PATH.read_text().strip()
    key = new_secret_key()
    _SECRET_PATH.write_text(key)
    _SECRET_PATH.chmod(0o600)
    return key


# ---- users ----
def create_user(username: str, password: str, is_admin: bool = False) -> dict:
    username = username.strip()
    if not username or not password:
        raise ValueError("username and password are required")
    with _LOCK, _connect() as conn:
        try:
            cur = conn.execute(
                "INSERT INTO users (username, password_hash, api_token, is_admin, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (username, hash_password(password), new_api_token(), int(is_admin), _now()),
            )
        except sqlite3.IntegrityError as e:
            raise ValueError(f"user {username!r} already exists") from e
        return dict(conn.execute("SELECT * FROM users WHERE id=?", (cur.lastrowid,)).fetchone())


def list_users() -> list[dict]:
    with _connect() as conn:
        return [dict(r) for r in conn.execute("SELECT * FROM users ORDER BY username")]


def get_user_by_username(username: str) -> dict | None:
    with _connect() as conn:
        r = conn.execute("SELECT * FROM users WHERE username=?", (username,)).fetchone()
        return dict(r) if r else None


def get_user_by_id(user_id: int) -> dict | None:
    with _connect() as conn:
        r = conn.execute("SELECT * FROM users WHERE id=?", (user_id,)).fetchone()
        return dict(r) if r else None


def get_user_by_token(token: str) -> dict | None:
    if not token:
        return None
    with _connect() as conn:
        r = conn.execute("SELECT * FROM users WHERE api_token=?", (token,)).fetchone()
        return dict(r) if r else None


def check_login(username: str, password: str) -> dict | None:
    u = get_user_by_username(username)
    if u and verify_password(password, u["password_hash"]):
        return u
    return None


def set_password(user_id: int, password: str) -> None:
    with _LOCK, _connect() as conn:
        conn.execute("UPDATE users SET password_hash=? WHERE id=?",
                     (hash_password(password), user_id))


def regen_token(user_id: int) -> str:
    token = new_api_token()
    with _LOCK, _connect() as conn:
        conn.execute("UPDATE users SET api_token=? WHERE id=?", (token, user_id))
    return token


def change_password(user_id: int, old: str, new: str) -> bool:
    """Self-service password change: only succeeds if the current password checks out."""
    u = get_user_by_id(user_id)
    if not u or not verify_password(old, u["password_hash"]) or not new:
        return False
    set_password(user_id, new)
    return True


def count_owned_meetings(user_id: int) -> int:
    with _connect() as conn:
        return conn.execute(
            "SELECT COUNT(*) c FROM meetings WHERE owner_id=?", (user_id,)
        ).fetchone()["c"]


def transfer_meetings(from_user_id: int, to_user_id: int) -> int:
    """Reassign ownership of all of a user's meetings to another user (used before deleting a
    departing member so their meetings survive). The meeting id stays the same (it is just an
    opaque unique key); only owner_id changes."""
    with _LOCK, _connect() as conn:
        cur = conn.execute("UPDATE meetings SET owner_id=? WHERE owner_id=?",
                           (to_user_id, from_user_id))
        return cur.rowcount


def delete_user(user_id: int) -> bool:
    """Remove a user and everything tied to them: their meetings (and those meetings' shares)
    and any meetings shared TO them. Reassign first with transfer_meetings() to keep the
    meetings."""
    with _LOCK, _connect() as conn:
        if not conn.execute("SELECT 1 FROM users WHERE id=?", (user_id,)).fetchone():
            return False
        conn.execute(
            "DELETE FROM shares WHERE meeting_id IN (SELECT id FROM meetings WHERE owner_id=?)",
            (user_id,))
        conn.execute("DELETE FROM meetings WHERE owner_id=?", (user_id,))
        conn.execute("DELETE FROM shares WHERE user_id=?", (user_id,))
        conn.execute("DELETE FROM users WHERE id=?", (user_id,))
        return True


# ---- meetings ----
def upsert_meeting(owner_id: int, m: dict) -> str:
    """Insert or update a meeting pushed by an agent. Idempotent per (owner, client id), so a
    re-upload of the same meeting overwrites in place rather than duplicating."""
    client_id = str(m.get("id") or m.get("client_id") or _now())
    sid = f"{owner_id}_{client_id}"
    row = (
        sid, owner_id, client_id,
        m.get("title") or client_id,
        m.get("created_at"), m.get("ended_at"), m.get("source"),
        m.get("lang"), m.get("lang_label"),
        m.get("notes", ""),
        json.dumps(m.get("utterances", []), ensure_ascii=False),
        _now(),
    )
    with _LOCK, _connect() as conn:
        existing = conn.execute("SELECT id FROM meetings WHERE id=?", (sid,)).fetchone()
        if existing:
            # keep server-side edits to title/notes/visibility; refresh transcript + timestamps
            conn.execute(
                "UPDATE meetings SET created_at=?, ended_at=?, source=?, lang=?, lang_label=?, "
                "data_json=?, updated_at=? WHERE id=?",
                (m.get("created_at"), m.get("ended_at"), m.get("source"), m.get("lang"),
                 m.get("lang_label"), json.dumps(m.get("utterances", []), ensure_ascii=False),
                 _now(), sid),
            )
        else:
            conn.execute(
                "INSERT INTO meetings (id, owner_id, client_id, title, created_at, ended_at, "
                "source, lang, lang_label, notes, data_json, updated_at) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                row,
            )
    return sid


def _accessible_ids_clause() -> str:
    # owner OR public OR explicitly shared with the user
    return ("(m.owner_id = :uid OR m.visibility = 'public' "
            "OR EXISTS (SELECT 1 FROM shares s WHERE s.meeting_id = m.id AND s.user_id = :uid))")


def _summary(row: sqlite3.Row, user_id: int) -> dict:
    data = json.loads(row["data_json"] or "[]")
    return {
        "id": row["id"],
        "title": row["title"],
        "created_at": row["created_at"],
        "ended_at": row["ended_at"],
        "source": row["source"],
        "lang_label": row["lang_label"],
        "utterances": len(data),
        "has_notes": bool((row["notes"] or "").strip()),
        "visibility": row["visibility"],
        "owner_id": row["owner_id"],
        "owned": row["owner_id"] == user_id,
    }


def list_meetings(user_id: int) -> list[dict]:
    with _connect() as conn:
        rows = conn.execute(
            f"SELECT m.* FROM meetings m WHERE {_accessible_ids_clause()} "
            "ORDER BY m.created_at DESC",
            {"uid": user_id},
        ).fetchall()
        return [_summary(r, user_id) for r in rows]


def _get_row(conn: sqlite3.Connection, meeting_id: str, user_id: int) -> sqlite3.Row | None:
    return conn.execute(
        f"SELECT m.* FROM meetings m WHERE m.id = :mid AND {_accessible_ids_clause()}",
        {"mid": meeting_id, "uid": user_id},
    ).fetchone()


def get_meeting(meeting_id: str, user_id: int) -> dict | None:
    with _connect() as conn:
        row = _get_row(conn, meeting_id, user_id)
        if not row:
            return None
        owner = conn.execute("SELECT username FROM users WHERE id=?", (row["owner_id"],)).fetchone()
        shared = [r["username"] for r in conn.execute(
            "SELECT u.username FROM shares s JOIN users u ON u.id=s.user_id WHERE s.meeting_id=?",
            (meeting_id,))]
        return {
            "id": row["id"], "title": row["title"], "created_at": row["created_at"],
            "ended_at": row["ended_at"], "source": row["source"], "lang_label": row["lang_label"],
            "notes": row["notes"], "visibility": row["visibility"],
            "owner": owner["username"] if owner else None,
            "owned": row["owner_id"] == user_id,
            "shared_with": shared,
            "utterances": json.loads(row["data_json"] or "[]"),
        }


def _owns(conn: sqlite3.Connection, meeting_id: str, user_id: int) -> bool:
    r = conn.execute("SELECT owner_id FROM meetings WHERE id=?", (meeting_id,)).fetchone()
    return bool(r and r["owner_id"] == user_id)


def update_notes(meeting_id: str, user_id: int, notes: str) -> bool:
    with _LOCK, _connect() as conn:
        if not _owns(conn, meeting_id, user_id):
            return False
        conn.execute("UPDATE meetings SET notes=?, updated_at=? WHERE id=?",
                     (notes, _now(), meeting_id))
        return True


def update_title(meeting_id: str, user_id: int, title: str) -> bool:
    with _LOCK, _connect() as conn:
        if not _owns(conn, meeting_id, user_id):
            return False
        conn.execute("UPDATE meetings SET title=?, updated_at=? WHERE id=?",
                     (title.strip() or meeting_id, _now(), meeting_id))
        return True


def set_visibility(meeting_id: str, user_id: int, visibility: str) -> bool:
    if visibility not in ("private", "public"):
        return False
    with _LOCK, _connect() as conn:
        if not _owns(conn, meeting_id, user_id):
            return False
        conn.execute("UPDATE meetings SET visibility=? WHERE id=?", (visibility, meeting_id))
        return True


def share_meeting(meeting_id: str, user_id: int, target_username: str) -> bool:
    with _LOCK, _connect() as conn:
        if not _owns(conn, meeting_id, user_id):
            return False
        target = conn.execute("SELECT id FROM users WHERE username=?",
                              (target_username.strip(),)).fetchone()
        if not target:
            return False
        conn.execute("INSERT OR IGNORE INTO shares (meeting_id, user_id) VALUES (?, ?)",
                     (meeting_id, target["id"]))
        return True


def unshare_meeting(meeting_id: str, user_id: int, target_username: str) -> bool:
    with _LOCK, _connect() as conn:
        if not _owns(conn, meeting_id, user_id):
            return False
        target = conn.execute("SELECT id FROM users WHERE username=?",
                              (target_username.strip(),)).fetchone()
        if not target:
            return False
        conn.execute("DELETE FROM shares WHERE meeting_id=? AND user_id=?",
                     (meeting_id, target["id"]))
        return True


def delete_meeting(meeting_id: str, user_id: int) -> bool:
    with _LOCK, _connect() as conn:
        if not _owns(conn, meeting_id, user_id):
            return False
        conn.execute("DELETE FROM shares WHERE meeting_id=?", (meeting_id,))
        conn.execute("DELETE FROM meetings WHERE id=?", (meeting_id,))
        return True


def search(user_id: int, query: str) -> list[dict]:
    q = query.strip().lower()
    if not q:
        return []
    results = []
    with _connect() as conn:
        rows = conn.execute(
            f"SELECT m.* FROM meetings m WHERE {_accessible_ids_clause()} "
            "ORDER BY m.created_at DESC",
            {"uid": user_id},
        ).fetchall()
    for row in rows:
        hay = (row["title"] or "") + " " + (row["notes"] or "")
        snippet = None
        if q in hay.lower():
            snippet = row["title"]
        if snippet is None:
            for u in json.loads(row["data_json"] or "[]"):
                text = " ".join(seg.get("text", "") for seg in u.get("en", []))
                text += " " + " ".join(seg.get("zh", "") for seg in (u.get("zh") or []))
                if q in text.lower():
                    snippet = _snippet(text, q)
                    break
        if snippet is not None:
            s = _summary(row, user_id)
            s["snippet"] = snippet
            results.append(s)
    return results


def _snippet(text: str, q: str, width: int = 60) -> str:
    i = text.lower().find(q)
    if i < 0:
        return text[:width]
    start = max(0, i - width // 2)
    end = min(len(text), i + len(q) + width // 2)
    return ("…" if start > 0 else "") + text[start:end].strip() + ("…" if end < len(text) else "")


_SOURCE_LABELS = {"me": "现场", "them": "线上"}


def export_md(meeting_id: str, user_id: int) -> str | None:
    m = get_meeting(meeting_id, user_id)
    if m is None:
        return None
    lines = [f"# {m['title']}", ""]
    meta = (m.get("created_at") or "").replace("T", " ")[:16]
    if m.get("source"):
        meta += f" · source: {m['source']}"
    if m.get("lang_label"):
        meta += f" · {m['lang_label']}"
    lines += [f"*{meta}*", "", "## Transcript", ""]
    def _seg(segs: list, field: str) -> str:
        parts = []
        for s in segs:
            spk = f"S{s['speaker'] + 1} " if s.get("speaker") is not None else ""
            parts.append(spk + s.get(field, ""))
        return "  ".join(parts)

    multi = m.get("source") == "both"
    for u in m.get("utterances", []):
        badge = (_SOURCE_LABELS.get(u.get("label", ""), "") + " ") if multi else ""
        lines.append(f"**[{u.get('at', '')}] {badge}**{_seg(u.get('en', []), 'text')}")
        if u.get("zh"):
            lines.append(f"> {_seg(u['zh'], 'zh')}")
        lines.append("")
    if (m.get("notes") or "").strip():
        lines += ["## Notes", "", m["notes"], ""]
    return "\n".join(lines)
