"""Chords and tabs downloaded from sources of your own choosing, the source of truth for the ones we generate.

    karaokifex-tabs <song folder>... [--sources FILE] [--force]

Asks each source in the local sources file (never checked in) for the song's chords or tab, writes what the first one that
has it says to `tab-source.json` in the song folder -- its chords in order, and its sections with the chords over the words
or its raw tab -- and, where the folder has the chords we made ourselves (`chords.json`, the music plugin's), `tab-check.json`:
which chords the two agree on and which only one has. A downloaded tab is right: the check says what to change in ours.

The module knows no site. Each source in the file says where to ask and how to read the answer:

    [[source]]
    name = "mine"                                  # for the log and the files
    url = "https://example.com/{artist_slug}/{song_slug}.txt"
    kind = "chords-over-lyrics"                    # or "chordpro" (a [Am] in the words) or "ascii-tab" (e|--0--1--|)
    instrument = "guitar"                          # (informational)
    headers = { "User-Agent" = "…" }               # optional

in `./tab-sources.toml`, `$KARAOKIFEX_TAB_SOURCES`, or `~/.config/karaokifex/tab-sources.toml`, the first that exists. In the url,
{artist} and {song} are the names as written, the _q forms URL-quoted and the _slug forms lower case with dashes. The repo
has only tab-sources.example.toml, with made-up addresses: the best sources are for whoever runs the pipeline to find.
"""

from __future__ import annotations

import json
import logging
import os
import re
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import quote

import click

from karaokifex.console import setup_logging

log = logging.getLogger("karaokifex")

KINDS = ("chords-over-lyrics", "chordpro", "ascii-tab")
CHORD = re.compile(r"^[A-G][#b]?(?:maj|min|dim|aug|sus|add|m|M|\+|°|ø)?\d*(?:(?:sus|add|maj|b|#)\d+)*(?:/[A-G][#b]?)?$")
INLINE = re.compile(r"\[([^\]\s]+)\]")
TAB_LINE = re.compile(r"^\s*([A-Ga-g][#b]?)?\s*[|:]?[-0-9hpbrx/\\~^()|:\s.*]{6,}$")
TAB_STRING = re.compile(r"^\s*([A-Ga-g][#b]?)\s*\|")


def is_chord(token: str) -> bool:
    return bool(CHORD.match(token))


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


@dataclass
class Source:
    name: str
    url: str
    kind: str = "chords-over-lyrics"
    instrument: str = "guitar"
    headers: dict[str, str] = field(default_factory=dict)

    def address(self, artist: str, song: str) -> str:
        return self.url.format(artist=artist, song=song, artist_q=quote(artist), song_q=quote(song),
                               artist_slug=slug(artist), song_slug=slug(song))


def sources_file(explicit: Path | None = None) -> Path | None:
    candidates = [explicit, Path(os.environ["KARAOKIFEX_TAB_SOURCES"]) if os.environ.get("KARAOKIFEX_TAB_SOURCES") else None,
                  Path("tab-sources.toml"), Path.home() / ".config" / "karaokifex" / "tab-sources.toml"]
    return next((c for c in candidates if c and c.is_file()), None)


def load_sources(path: Path) -> list[Source]:
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    out = []
    for entry in data.get("source", []):
        if entry.get("kind", "chords-over-lyrics") not in KINDS:
            raise ValueError(f"{path}: source {entry.get('name')!r}: kind must be one of {', '.join(KINDS)}")
        out.append(Source(name=entry["name"], url=entry["url"], kind=entry.get("kind", "chords-over-lyrics"),
                          instrument=entry.get("instrument", "guitar"), headers=dict(entry.get("headers", {}))))
    return out


# ---------------------------------------------------------------- reading what a source says
def parse_chordpro(text: str) -> list[dict]:
    """[Am] in the words: each line's chords with the position in its words where each falls."""
    lines = []
    for raw in text.splitlines():
        if not raw.strip() or raw.lstrip().startswith(("#", "{")):
            continue
        chords, words, last = [], "", 0
        for m in INLINE.finditer(raw):
            words += raw[last:m.start()]
            chords.append({"name": m.group(1), "at": len(words)})
            last = m.end()
        words += raw[last:]
        lines.append({"chords": chords, "text": words.strip() if not chords else words})
    return lines


def parse_chords_over_lyrics(text: str) -> list[dict]:
    """A line of chords with the words under it; a line that is only chords stands alone."""
    raw = text.splitlines()
    lines, i = [], 0
    while i < len(raw):
        line = raw[i]
        tokens = line.split()
        if tokens and all(is_chord(t) for t in tokens):
            chords = [{"name": m.group(0), "at": m.start()} for m in re.finditer(r"\S+", line)]
            nxt = raw[i + 1] if i + 1 < len(raw) else ""
            under = nxt.split()
            if under and not all(is_chord(t) for t in under):
                lines.append({"chords": chords, "text": nxt})
                i += 2
                continue
            lines.append({"chords": chords, "text": ""})
        elif line.strip():
            lines.append({"chords": [], "text": line})
        i += 1
    return lines


def parse_ascii_tab(text: str) -> list[dict]:
    """Blocks of string lines (e|--0--1--|) kept as written, one block per run of them."""
    blocks, run = [], []
    for line in text.splitlines():
        if TAB_STRING.match(line) and TAB_LINE.match(line):
            run.append(line.rstrip())
            continue
        if len(run) >= 3:
            blocks.append({"tab": run})
        run = []
    if len(run) >= 3:
        blocks.append({"tab": run})
    return blocks


PARSERS = {"chordpro": parse_chordpro, "chords-over-lyrics": parse_chords_over_lyrics, "ascii-tab": parse_ascii_tab}
SECTION = re.compile(r"^\s*[\[(]?\s*(intro|verse|chorus|bridge|pre-?chorus|outro|solo|interlude|refrain)\b[^\]):]*[\])]?\s*:?\s*\d*\s*$", re.I)


def parse(text: str, kind: str) -> dict:
    """The sections (split at "[Chorus]"-style headings) and the chords in the order they first come."""
    sections, current = [], {"name": None, "lines": []}
    body = text.replace("\r\n", "\n")
    for raw in body.split("\n"):
        if kind != "ascii-tab" and SECTION.match(raw) and not INLINE.search(raw.replace("[" + raw.strip("[] ") + "]", "")):
            if current["lines"]:
                sections.append(current)
            current = {"name": raw.strip(" []():").strip(), "lines": []}
            continue
        current["lines"].append(raw)
    if current["lines"]:
        sections.append(current)
    out = [{"name": s["name"], "lines": PARSERS[kind]("\n".join(s["lines"]))} for s in sections]
    order: list[str] = []
    for s in out:
        for line in s["lines"]:
            for c in line.get("chords", []):
                if c["name"] not in order:
                    order.append(c["name"])
    return {"sections": out, "chords": order}


# ---------------------------------------------------------------- the check against ours
def own_chords(folder: Path) -> list[str] | None:
    """The chords we made ourselves (chords.json: [[start, length, name], ...]), in order, 'N' left out."""
    path = folder / "chords.json"
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except ValueError:
        return None
    names: list[str] = []
    for c in data.get("chords", []):
        name = c[2] if isinstance(c, list) and len(c) > 2 else c.get("name") if isinstance(c, dict) else None
        if name and name != "N" and name not in names:
            names.append(name)
    return names


def check(ours: list[str], theirs: list[str]) -> dict:
    """Downloaded is right: what we have that it lacks is wrong, what it has that we lack is missing."""
    return {"agree": [c for c in theirs if c in ours], "wrong": [c for c in ours if c not in theirs],
            "missing": [c for c in theirs if c not in ours]}


# ---------------------------------------------------------------- asking
def fetch(url: str, headers: dict[str, str]) -> str | None:
    import requests

    try:
        r = requests.get(url, headers=headers, timeout=20)
    except requests.RequestException as e:
        log.warning("%s: %s", url, e)
        return None
    return r.text if r.ok and r.text.strip() else None


def download(folder: Path, sources: list[Source], get=fetch, force: bool = False) -> dict | None:
    """The first source that has the song: written to tab-source.json (and tab-check.json beside ours)."""
    target = folder / "tab-source.json"
    if target.exists() and not force:
        log.info("%s: downloaded already", folder.name)
        return json.loads(target.read_text(encoding="utf-8"))
    artist, _, song = folder.name.partition(" - ")
    if not song:
        log.warning("%s: not named “Artist - Song”, skipped", folder.name)
        return None
    for source in sources:
        url = source.address(artist, song)
        text = get(url, source.headers)
        if not text:
            log.info("%s: nothing at %s", folder.name, source.name)
            continue
        found = parse(text, source.kind)
        if not found["chords"] and not any("tab" in line for s in found["sections"] for line in s["lines"]):
            log.info("%s: %s answered, but no chords or tab in it", folder.name, source.name)
            continue
        doc = {"version": 1, "source": source.name, "kind": source.kind, "instrument": source.instrument, "url": url, **found}
        partial = target.with_name(target.name + ".partial")
        partial.write_text(json.dumps(doc, ensure_ascii=False, indent=1), encoding="utf-8")
        partial.replace(target)
        ours = own_chords(folder)
        if ours is not None:
            report = {"source": source.name, **check(ours, found["chords"])}
            (folder / "tab-check.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
            log.info("%s: %s: %d chords agree, %d of ours wrong, %d missing", folder.name, source.name,
                     len(report["agree"]), len(report["wrong"]), len(report["missing"]))
        else:
            log.info("%s: %s: %d chords", folder.name, source.name, len(found["chords"]))
        return doc
    log.warning("%s: no source has it", folder.name)
    return None


@click.command(context_settings={"help_option_names": ["-h", "--help"], "max_content_width": 100})
@click.argument("folders", nargs=-1, required=True, type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--sources", "sources_path", type=click.Path(exists=True, dir_okay=False, path_type=Path), help="The sources file (else tab-sources.toml, $KARAOKIFEX_TAB_SOURCES, ~/.config/karaokifex/).")
@click.option("--force", is_flag=True, help="Ask again for songs already downloaded.")
@click.option("-v", "--verbose", is_flag=True, help="Show debug output.")
def main(folders: tuple[Path, ...], sources_path: Path | None, force: bool, verbose: bool) -> None:
    """Write tab-source.json into each song folder: the chords or tab a source of yours has for it."""
    setup_logging(verbose)
    path = sources_file(sources_path)
    if not path:
        raise click.ClickException("no sources file: copy tab-sources.example.toml to tab-sources.toml and set your sources")
    sources = load_sources(path)
    if not sources:
        raise click.ClickException(f"{path} has no [[source]]")
    for folder in folders:
        download(folder, sources, force=force)
