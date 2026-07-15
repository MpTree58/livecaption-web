"""Meeting record store: persist each recording session as a browsable, searchable note.

A meeting is one Start→Stop session. It is stored as a single JSON file under
`<root>/<id>/meeting.json` holding metadata, free-form notes, and the full list of
utterances (each with speaker / source-track / English / Chinese, in the same serialized
shape WebSink sends to the browser, so the page can re-render a saved meeting with the
same code path as the live stream).

Two roles:
- `Meeting` is the live, in-memory record the Session appends to during a run (finals and
  their translations, paired by `key`); it autosaves on every change.
- `MeetingStore` manages the on-disk collection: list / load / rename / notes / delete /
  export-Markdown / full-text search.
"""

from __future__ import annotations

import contextlib
import json
import os
import threading
from datetime import datetime
from pathlib import Path

from .websink import _segments_out, _zh_segments_out

_SOURCE_LABELS = {"me": "现场", "them": "线上"}


def _default_title(created_at: datetime) -> str:
    return f"Meeting {created_at:%m-%d %H:%M}"


def _atomic_write(path: Path, data: dict) -> None:
    """Write JSON via a temp file + rename so a crash mid-write can't corrupt the record."""
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(tmp, path)


class Meeting:
    """One live recording's growing record. Thread-safe: finals/translations arrive from the
    translator/ASR worker threads. Utterances are keyed by `key` (label|started_at) so a
    translation can find and fill in the Chinese for the English line added earlier."""

    def __init__(self, path: Path, meta: dict) -> None:
        self.path = path
        self.meta = meta  # full record dict, incl. "utterances" and "notes"
        self._index: dict[str, dict] = {u["key"]: u for u in meta.get("utterances", [])}
        self._lock = threading.Lock()

    @property
    def id(self) -> str:
        return self.meta["id"]

    def add_final(self, key: str, label: str, at: str, started_at: str, segments: list) -> None:
        with self._lock:
            u = self._index.get(key)
            en = _segments_out(segments)
            if u is None:
                u = {"key": key, "label": label, "at": at, "started_at": started_at,
                     "en": en, "zh": None}
                self.meta["utterances"].append(u)
                self._index[key] = u
            else:
                u["en"] = en  # a re-emitted final (diar re-split) overwrites in place
            self._save_locked()

    def add_translation(self, key: str, zh_segments: list) -> None:
        with self._lock:
            u = self._index.get(key)
            if u is None:
                return
            u["zh"] = _zh_segments_out(zh_segments)
            self._save_locked()

    def set_notes(self, notes: str) -> None:
        with self._lock:
            self.meta["notes"] = notes
            self._save_locked()

    def set_title(self, title: str) -> None:
        with self._lock:
            self.meta["title"] = title.strip() or _default_title(
                datetime.fromisoformat(self.meta["created_at"])
            )
            self._save_locked()

    def finalize(self, ended_at: str) -> None:
        with self._lock:
            self.meta["ended_at"] = ended_at
            self._save_locked()

    def _save_locked(self) -> None:
        _atomic_write(self.path, self.meta)


class MeetingStore:
    def __init__(self, root: Path) -> None:
        self.root = root
        # meeting transcripts may be sensitive: owner-only directory
        root.mkdir(mode=0o700, parents=True, exist_ok=True)

    def _path(self, mid: str) -> Path:
        return self.root / mid / "meeting.json"

    def create(self, *, source: str, asr_lang: str, target_lang: str | None,
               created_at: datetime, lang: str = "en", lang_label: str = "") -> Meeting:
        # id = creation timestamp; disambiguate the rare same-second collision with a suffix
        base = f"{created_at:%Y%m%d-%H%M%S}"
        mid = base
        n = 1
        while (self.root / mid).exists():
            n += 1
            mid = f"{base}-{n}"
        path = self._path(mid)
        path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        meta = {
            "id": mid,
            "title": _default_title(created_at),
            "created_at": created_at.isoformat(),
            "ended_at": None,
            "source": source,
            "asr_lang": asr_lang,
            "target_lang": target_lang,
            "lang": lang,
            "lang_label": lang_label,
            "notes": "",
            "utterances": [],
        }
        m = Meeting(path, meta)
        _atomic_write(path, meta)
        return m

    def _load_raw(self, mid: str) -> dict | None:
        path = self._path(mid)
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None

    def load(self, mid: str) -> dict | None:
        return self._load_raw(mid)

    @staticmethod
    def _summary(data: dict) -> dict:
        return {
            "id": data["id"],
            "title": data.get("title", data["id"]),
            "created_at": data.get("created_at"),
            "ended_at": data.get("ended_at"),
            "source": data.get("source"),
            "lang_label": data.get("lang_label", ""),
            "utterances": len(data.get("utterances", [])),
            "has_notes": bool(data.get("notes", "").strip()),
        }

    def list(self) -> list[dict]:
        out = []
        if not self.root.exists():
            return out
        for d in self.root.iterdir():
            if not d.is_dir():
                continue
            data = self._load_raw(d.name)
            if data:
                out.append(self._summary(data))
        # newest first
        out.sort(key=lambda s: s.get("created_at") or "", reverse=True)
        return out

    def rename(self, mid: str, title: str) -> bool:
        data = self._load_raw(mid)
        if data is None:
            return False
        data["title"] = title.strip() or _default_title(
            datetime.fromisoformat(data["created_at"])
        )
        _atomic_write(self._path(mid), data)
        return True

    def set_notes(self, mid: str, notes: str) -> bool:
        data = self._load_raw(mid)
        if data is None:
            return False
        data["notes"] = notes
        _atomic_write(self._path(mid), data)
        return True

    def delete(self, mid: str) -> bool:
        d = self.root / mid
        p = self._path(mid)
        if not p.exists():
            return False
        p.unlink()
        # remove the (now empty) meeting dir; leave it if anything unexpected remains
        with contextlib.suppress(OSError):
            d.rmdir()
        return True

    # --- text helpers (export + search) ---
    @staticmethod
    def _utterance_text(u: dict) -> str:
        en = " ".join(seg.get("text", "") for seg in u.get("en", []))
        zh = " ".join(seg.get("zh", "") for seg in (u.get("zh") or []))
        return f"{en} {zh}"

    def export_md(self, mid: str) -> str | None:
        data = self._load_raw(mid)
        if data is None:
            return None
        lines = [f"# {data.get('title', mid)}", ""]
        created = data.get("created_at", "")
        meta_line = created.replace("T", " ")[:16] if created else ""
        src = data.get("source")
        if src:
            meta_line += f" · source: {src}"
        lines += [f"*{meta_line}*", "", "## Transcript", ""]
        multi = data.get("source") == "both"
        for u in data.get("utterances", []):
            badge = ""
            if multi:
                badge = _SOURCE_LABELS.get(u.get("label", ""), "") + " "
            en = "  ".join(
                (f"S{seg['speaker'] + 1} " if seg.get("speaker") is not None else "")
                + seg.get("text", "")
                for seg in u.get("en", [])
            )
            lines.append(f"**[{u.get('at', '')}] {badge}**{en}")
            if u.get("zh"):
                zh = "  ".join(
                    (f"S{seg['speaker'] + 1} " if seg.get("speaker") is not None else "")
                    + seg.get("zh", "")
                    for seg in u["zh"]
                )
                lines.append(f"> {zh}")
            lines.append("")
        notes = data.get("notes", "").strip()
        if notes:
            lines += ["## Notes", "", notes, ""]
        return "\n".join(lines)

    def search(self, query: str) -> list[dict]:
        """Case-insensitive substring search across title, transcript, and notes. Returns
        summaries plus a short snippet of the first match."""
        q = query.strip().lower()
        if not q:
            return []
        results = []
        for d in (self.root.iterdir() if self.root.exists() else []):
            if not d.is_dir():
                continue
            data = self._load_raw(d.name)
            if not data:
                continue
            snippet = None
            if q in data.get("title", "").lower():
                snippet = data["title"]
            if snippet is None and q in data.get("notes", "").lower():
                snippet = self._snippet(data["notes"], q)
            if snippet is None:
                for u in data.get("utterances", []):
                    text = self._utterance_text(u)
                    if q in text.lower():
                        snippet = self._snippet(text, q)
                        break
            if snippet is not None:
                summ = self._summary(data)
                summ["snippet"] = snippet
                results.append(summ)
        results.sort(key=lambda s: s.get("created_at") or "", reverse=True)
        return results

    @staticmethod
    def _snippet(text: str, q: str, width: int = 60) -> str:
        i = text.lower().find(q)
        if i < 0:
            return text[:width]
        start = max(0, i - width // 2)
        end = min(len(text), i + len(q) + width // 2)
        s = text[start:end].strip()
        return ("…" if start > 0 else "") + s + ("…" if end < len(text) else "")
