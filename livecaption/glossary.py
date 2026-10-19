"""Glossary: names and jargon the models would otherwise mishear or mistranslate.

One entry per line; `#` starts a comment:
    Flander                        a term: transcribed as written, kept as is in a translation
    联调 = integration testing      a term and its translation, used in either direction

Speech recognition (qwen3 engines only -- nemotron has no biasing input) gets every term, both
sides of each entry, as Qwen3-ASR's `context`: its documented way to bias decoding toward
domain vocabulary (space-separated terms in the system prompt). Measured on CAiRE/ASCEND
conversations it recovered misheard terms without hurting other sentences; on our own jargon
it turned "Flender 再连调" into "Flander 再联调" and "MPTS ... Superbase" into "MPPS ... Supabase".

Translation gets, per sentence, the entries occurring in it, through Hy-MT2's official
terminology template ("A translates to B"). The side of an entry written in the target
language is the translation, so one entry serves both directions; a Latin-script term without
a translation is pinned as itself when translating into Chinese (names stay "Andy", not "安迪").
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_CJK_RE = re.compile(r"[㐀-鿿豈-﫿]")


@dataclass(frozen=True)
class Entry:
    term: str
    translation: str | None = None


def parse(text: str) -> list[Entry]:
    """Read glossary text (see the module docstring); blank lines, comments and duplicate
    terms are dropped."""
    entries: list[Entry] = []
    seen: set[str] = set()
    for line in text.splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        term, _, translation = (part.strip() for part in line.partition("="))
        if term and term.casefold() not in seen:
            seen.add(term.casefold())
            entries.append(Entry(term, translation or None))
    return entries


def asr_context(entries: list[Entry]) -> str:
    """The recognizer's biasing context: every term and translation, space-separated (either
    side may be what is actually spoken)."""
    words: list[str] = []
    for e in entries:
        for w in (e.term, e.translation):
            if w and w not in words:
                words.append(w)
    return " ".join(words)


def _is_cjk(s: str) -> bool:
    return bool(_CJK_RE.search(s))


def _occurs(needle: str, text: str) -> bool:
    # CJK has no word boundaries: substring. Latin: whole words, case-insensitive ("API" must
    # not match inside "rapid")
    if _is_cjk(needle):
        return needle in text
    return re.search(rf"(?<![\w]){re.escape(needle)}(?![\w])", text, re.IGNORECASE) is not None


def translation_pairs(entries: list[Entry], text: str, into_chinese: bool) -> list[tuple[str, str]]:
    """The (source, translation) pairs to give the translator for `text`: entries occurring in
    it, oriented so the translation side is in the target language. A source word covered by
    two entries keeps the first one (two conflicting instructions would only confuse)."""
    pairs: list[tuple[str, str]] = []
    used: set[str] = set()

    def add(src: str, out: str) -> None:
        if src.casefold() not in used:
            used.add(src.casefold())
            pairs.append((src, out))

    for e in entries:
        if e.translation:
            for src, out in ((e.term, e.translation), (e.translation, e.term)):
                if _is_cjk(out) == into_chinese and _is_cjk(src) != into_chinese \
                        and _occurs(src, text):
                    add(src, out)
                    break
        elif into_chinese and not _is_cjk(e.term) and _occurs(e.term, text):
            add(e.term, e.term)
    return pairs
