"""Known notes for a song's lead melody, from sources of your own choosing: the source of truth for the notes we detect.

    karaokifex-notes <song folder>... [--sources FILE] [--force] [--only-local] [--dry-run]
    karaokifex-notes --report <song folder>... | --library <root>

Asks the song for its notes where it has them -- first in `<song folder>/notes/` (files you dropped there: `*.txt` an
UltraStar chart, `*.mid`, `*.midi` or `*.kar` a MIDI file), then each source in the local sources file (never checked
in), in order -- and checks every answer against the lead melody we detected (melody.json, made by the music plugin):
which transposition and which time warp put the file's notes over our pitch line, and how well they then agree
(music/align.py). A chart or MIDI is usually an octave or more off the recording, starts earlier or later, and a MIDI
may run at another tempo; a MIDI file has many tracks (the melody is one) and a collection of them has many versions of a song:
the best-agreeing track and file of a source is its answer. The first answer that agrees well enough is kept; the others
are in `note-check.json`, which the song folder has for every answer, accepted or not, so the figure of how right our notes
are (--report) does not depend on where the line for accepting was drawn.

The module knows no site. Each source in the file says where to ask and how to read the answer:

    [[source]]
    name = "mine"                                  # for the log and the files
    kind = "auto"                                  # "ultrastar" (a .txt), "midi" (a .mid or .kar), or by the file's extension
    url = "https://example.com/{artist_slug}/{song_slug}.mid"      # asked with GET; the body is the file (bytes)
    # or, instead of url, a file on this machine (a glob; "*" and "?" as in a shell):
    path = "C:/Karaokino-notes/{artist} - {song}.*"
    headers = { "User-Agent" = "…" }               # optional (url)

in `./note-sources.toml`, `$KARAOKIFEX_NOTE_SOURCES`, or `~/.config/karaokifex/note-sources.toml`, the first that exists. In
the url and the path, {artist} and {song} are the names as written, the _q forms URL-quoted and the _slug forms lower case
with dashes. A path that names no file is looked for again by the names folded (accents, case and punctuation ignored,
"The " dropped, "feat." and what follows it ignored, "&" and "and" ignored; a trailing version number such as "Title.2.mid"
ignored), in the template's own folders: "<root>/{artist}/{song}*.mid" finds "Beatles, The/Yesterday.3.mid" and its
sisters ("Yesterday.1.mid", ...), all of which are tried. A path source whose folder is not there is skipped without a
word, so one can be listed before the files are. `--only-local` asks only the path sources and the dropped-in files;
`--dry-run` says what would be asked and where, and asks nothing; `--force` asks again for songs already checked.
The repo has only note-sources.example.toml, with made-up addresses: the best sources are for whoever runs the pipeline
to find.

Reading what a source answers (no library beyond numpy): an UltraStar chart (#BPM, #GAP, #RELATIVE, `:` and `*` notes
(`F` freestyle, `R` rap and `G` golden rap have no pitch and are left out), `-` line breaks, `B` tempo changes, `P1`/`P2`
duets each their own line), time = GAP/1000 + beat * 15/BPM, MIDI note = 60 + pitch; and a standard MIDI file (format 0 or
1, running status, the tempo map from any track), each channel of each track but the drums' a candidate melody line when it
is mostly one note at a time, ranked by what a melody looks like (its name, lyric events as a karaoke file has them, its
register, its length) and the best few tried.

What the song folder gets:
  note-check.json   for every answer: {version, source, kind, file, candidate (the track or player), accepted, reason (when
                    not), same_song (the agreement is above what songs that are not the same come to), transpose (semitones
                    added to the file's notes), offset and warp {scale, offset, max_dev, span, aligned} (the recording's
                    time of the file's 0, and the file's time on the recording's clock as the straight line that fits it),
                    notes_known, notes_ours, matched, missing, extra, pitch_exact, pitch_within_1, onset_median_ms,
                    onset_p90_ms, covered_known, covered_ours, agree_known (see music/align.py), wrong (a sample of
                    [time, ours, known]), others (the checks of the other answers, each as this one)}
  notes-source.json when an answer is accepted: {version: 1, source, kind, url or path, transpose, notes: [[start s, length
                    s, MIDI note on the song's grid, syllable or null], ...] on the recording's clock, offset, the
                    check's headline numbers}. The file and its `source` are what tag a song as having known notes, as
                    tab-source.json's do a song with a downloaded tab.
  melody.json       by apply(), the `known` step of python -m karaokifex.music: our notes that a known note lies under take
                    its note (conf at least 0.9), known notes we have no note for are added where the pitch line is voiced
                    under them (conf 0.6), our notes with nothing known under them stay (conf at most 0.5); notes_detected
                    keeps the notes as detected so that it can be made again, known says from where.

Accepting: at least ACCEPT_AGREE of the file's time lies within a semitone of our pitch line, at least ACCEPT_OURS of our
pitch has a known note under it, at least ACCEPT_ALIGNED of the file's time falls on the recording. (Set against the
library: the melodies of other songs reach 0.40 of the file's time at most and a line of bass roots or pads 0.30,
while a file made from our own notes, warped and thinned and moved a semitone here and there, comes to 0.80 or more.)
"""

from __future__ import annotations

import bisect
import fnmatch
import glob
import json
import logging
import os
import re
import struct
import tomllib
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import quote

import click
import numpy as np

from karaokifex.console import setup_logging
from karaokifex.music import align

log = logging.getLogger("karaokifex")

KINDS = ("ultrastar", "midi", "auto")
EXTENSIONS = {".txt": "ultrastar", ".mid": "midi", ".midi": "midi", ".kar": "midi"}
ACCEPT_AGREE = 0.55         # of the file's time within a semitone of our pitch line
ACCEPT_OURS = 0.50          # of our pitched time with a known note under it
ACCEPT_ALIGNED = 0.60       # of the file's time on the recording at all
SAME_SONG = 0.40            # agree_known under this is another song's (the most the unrelated reach: see the header)
LINES_PER_FILE = 6          # MIDI lines tried a file, the likeliest first
FILES_PER_SOURCE = 8        # the most files of one source tried (a collection may have several versions of a song)
VERSION = 1


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def fold(text: str) -> str:
    """A name as the matching sees it: accents, case and punctuation gone, "feat." and what follows, bracketed words,
    "and", a leading "the" and a trailing ", The" dropped."""
    t = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode().lower()
    t = re.sub(r"\([^)]*\)|\[[^\]]*\]", " ", t)
    t = re.sub(r"\b(?:feat|ft|featuring)\b\.?.*$", "", t)
    t = re.sub(r",\s*the\s*$", "", t)
    t = re.sub(r"[^a-z0-9]+", " ", t)
    t = re.sub(r"\band\b", " ", t)
    t = re.sub(r"^\s*the\s+", "", t)
    return " ".join(t.split())


def strip_version(stem: str) -> str:
    """"Title.2", "Title-2", "Title (2)", "Title_2" and "Title 2" as "Title" (a file's number among its song's versions)."""
    return re.sub(r"[\s._-]*\(?\d{1,2}\)?$", "", stem)


# ---------------------------------------------------------------- the sources
@dataclass
class Source:
    name: str
    kind: str = "auto"
    url: str | None = None
    path: str | None = None
    headers: dict[str, str] = field(default_factory=dict)

    def names(self, artist: str, song: str, escape: bool = False) -> dict[str, str]:
        a, s = (glob.escape(artist), glob.escape(song)) if escape else (artist, song)
        return {"artist": a, "song": s, "artist_q": quote(artist), "song_q": quote(song), "artist_slug": slug(artist),
                "song_slug": slug(song)}

    def address(self, artist: str, song: str) -> str:
        assert self.url
        return self.url.format(**self.names(artist, song))

    def pattern(self, artist: str, song: str) -> str:
        assert self.path
        return self.path.format(**self.names(artist, song, escape=True))

    def split(self) -> tuple[Path | None, list[str]]:
        """A path source's template as (the folder it starts in: what it has before its first placeholder or wildcard, the
        components of the rest); (None, []) for a url source."""
        if not self.path:
            return None, []
        parts = self.path.replace("\\", "/").split("/")
        n = 0
        while n < len(parts) - 1 and not ("{" in parts[n] or any(c in parts[n] for c in "*?[")):
            n += 1
        head = "/".join(parts[:n]) or "."
        return Path(head + "/" if head.endswith(":") else head), parts[n:]

    @property
    def root(self) -> Path | None:
        return self.split()[0]


def sources_file(explicit: Path | None = None) -> Path | None:
    env = os.environ.get("KARAOKIFEX_NOTE_SOURCES")
    candidates = [explicit, Path(env) if env else None, Path("note-sources.toml"),
                  Path.home() / ".config" / "karaokifex" / "note-sources.toml"]
    return next((c for c in candidates if c and c.is_file()), None)


def load_sources(path: Path) -> list[Source]:
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    out = []
    for entry in data.get("source", []):
        kind = entry.get("kind", "auto")
        if kind not in KINDS:
            raise ValueError(f"{path}: source {entry.get('name')!r}: kind must be one of {', '.join(KINDS)}")
        if bool(entry.get("url")) == bool(entry.get("path")):
            raise ValueError(f"{path}: source {entry.get('name')!r}: it needs a url or a path, and not both")
        out.append(Source(name=entry["name"], kind=kind, url=entry.get("url"), path=entry.get("path"),
                          headers=dict(entry.get("headers", {}))))
    return out


# ---------------------------------------------------------------- finding the files of a path source
def _matches(component: str, name: str, artist: str, song: str, last: bool) -> bool:
    """Whether a file's (or folder's) `name` is what the path template's `component` names for the song: by the
    template's own glob, else by the folded names."""
    names = {"artist": glob.escape(artist), "song": glob.escape(song), "artist_q": "", "song_q": "", "artist_slug": "", "song_slug": ""}
    if fnmatch.fnmatchcase(name.lower(), component.format(**names).lower()):
        return True
    stem_pattern, ext_pattern = os.path.splitext(component) if last else (component, "")
    stem, ext = os.path.splitext(name) if last else (name, "")
    if last and ext_pattern and not fnmatch.fnmatchcase(ext.lower(), ext_pattern.lower()):
        return False
    has_artist, has_song = "{artist}" in stem_pattern, "{song}" in stem_pattern
    if has_artist and has_song:
        # the artist's part and the song's part of the name, each folded on its own ("feat." ends the artist, not the name)
        rx = re.escape(stem_pattern).replace(re.escape("{artist}"), "(?P<artist>.+?)").replace(re.escape("{song}"), "(?P<song>.+?)")
        m = re.fullmatch(rx.replace(r"\*", ".*").replace(r"\?", "."), stem, re.I)
        return bool(m) and bool(fold(artist)) and fold(m["artist"]) == fold(artist) \
            and fold(song) in {fold(m["song"]), fold(strip_version(m["song"]))}
    if has_artist:
        return fold(stem) == fold(artist) != ""
    if has_song:
        return fold(song) != "" and fold(song) in {fold(stem), fold(strip_version(stem))}
    return False


def find_files(source: Source, artist: str, song: str) -> list[Path]:
    """The files a path source has for the song, in name order: those its template names, else those whose names
    fold to the song's, down the template's own folders. Only files that read as notes (EXTENSIONS)."""
    exact = sorted(p for p in glob.glob(source.pattern(artist, song)) if os.path.isfile(p))
    exact = [p for p in exact if os.path.splitext(p)[1].lower() in EXTENSIONS]
    if exact:
        return [Path(p) for p in exact]
    root, parts = source.split()
    if root is None or not root.is_dir():
        return []

    def walk(folder: Path, rest: list[str]) -> list[Path]:
        found: list[Path] = []
        try:
            entries = sorted(os.scandir(folder), key=lambda e: e.name.lower())
        except OSError:
            return found
        for entry in entries:
            if len(rest) > 1:
                if entry.is_dir() and _matches(rest[0], entry.name, artist, song, False):
                    found += walk(Path(entry.path), rest[1:])
            elif entry.is_file() and os.path.splitext(entry.name)[1].lower() in EXTENSIONS and _matches(rest[0], entry.name, artist, song, True):
                found.append(Path(entry.path))
        return found

    return walk(root, parts)


def dropped_in(folder: Path) -> list[Path]:
    """The files in `<folder>/notes/` that read as notes, in name order."""
    notes = folder / "notes"
    if not notes.is_dir():
        return []
    return sorted(p for p in notes.iterdir() if p.is_file() and p.suffix.lower() in EXTENSIONS)


def looks_like(kind: str, data: bytes) -> bool:
    """Whether `data` could be a file of `kind` (a page of HTML an address answered with is not)."""
    if kind == "midi":
        return b"MThd" in data[:64]
    return b"#BPM" in data[:20000].upper() and b"<html" not in data[:500].lower()


def kind_of(name: str, data: bytes, declared: str = "auto") -> str | None:
    """"ultrastar" or "midi" for a file: as declared, else by its extension, else by what it starts with."""
    if declared != "auto":
        return declared
    ext = os.path.splitext(name.split("?")[0])[1].lower()
    if ext in EXTENSIONS:
        return EXTENSIONS[ext]
    if data[:4] == b"MThd" or (data[:4] == b"RIFF" and b"MThd" in data[:64]):
        return "midi"
    return "ultrastar" if b"#BPM" in data[:4000].upper() else None


# ---------------------------------------------------------------- reading what a source answers: an UltraStar chart
def _text(data: bytes) -> str:
    """A chart's text: UTF-8, else what its #ENCODING says, else Windows-1252."""
    for enc in ("utf-8-sig",):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            pass
    declared = re.search(rb"#ENCODING\s*:\s*([\w-]+)", data, re.I)
    for enc in ([declared.group(1).decode()] if declared else []) + ["cp1252", "latin-1"]:
        try:
            return data.decode(enc)
        except (UnicodeDecodeError, LookupError):
            pass
    return data.decode("utf-8", "replace")


def _number(text: str) -> float:
    return float(text.strip().replace(",", "."))


@dataclass
class Line:
    """One line of known notes: [(start s, length s, MIDI note, syllable or None), ...] on the file's clock, and what it
    is called ("P1", "track 3 (Vocals)") and how like a melody it looks (the higher the likelier)."""
    notes: list[tuple]
    label: str
    prior: float = 0.0


def parse_ultrastar(data: bytes | str) -> list[Line]:
    """The pitched notes of an UltraStar chart, a line for each player (one unless it is a duet). Time is
    GAP/1000 + beat * 15/BPM seconds (the chart's BPM is a quarter of the real one: a beat is 1/16 of a note), MIDI note
    60 + the pitch; in #RELATIVE mode a note's beat counts from the line's start, which each `-` line moves on by its
    second number. `B beat bpm` changes the tempo from that beat on."""
    text = data if isinstance(data, str) else _text(data)
    header: dict[str, str] = {}
    body: list[str] = []
    for raw in text.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        raw = raw.strip("\ufeff")
        if raw.startswith("#") and ":" in raw and not body:
            key, _, value = raw[1:].partition(":")
            header[key.strip().upper()] = value.strip()
        elif raw.strip():
            body.append(raw)
    try:
        bpm = _number(header["BPM"])
        gap = _number(header.get("GAP", "0") or "0") / 1000.0
    except (KeyError, ValueError):
        return []
    if bpm <= 0:
        return []
    relative = header.get("RELATIVE", "").upper() in ("YES", "TRUE", "1")
    tempo = [(0.0, bpm)]            # (beat, bpm) in the order the chart gives them
    notes: dict[str, list[tuple[int, int, int, str]]] = {"P1": []}
    player, offset = "P1", 0.0
    for raw in body:
        kind, _, rest = raw.partition(" ")
        kind = kind.strip()
        if kind == "E":
            break
        if re.fullmatch(r"P\s*\d", raw.strip()):
            player = "P" + re.sub(r"\D", "", raw)
            notes.setdefault(player, [])
            offset = 0.0
            continue
        fields = rest.split(" ", 3)
        try:
            if kind in (":", "*"):
                start, length, pitch = int(float(fields[0])), int(float(fields[1])), int(float(fields[2]))
                syllable = fields[3] if len(fields) > 3 else ""
                if length > 0:
                    notes[player].append((int(offset) + start, length, 60 + pitch, syllable))
            elif kind == "-" and relative:
                numbers = rest.split()
                offset += _number(numbers[1]) if len(numbers) > 1 else _number(numbers[0])
            elif kind == "B":
                tempo.append((_number(fields[0]) + (offset if relative else 0.0), _number(fields[1])))
        except (ValueError, IndexError):
            continue
    tempo.sort()

    def seconds(beat: float) -> float:
        total, last_beat, last_bpm = 0.0, 0.0, tempo[0][1]
        for b, v in tempo[1:]:
            if b >= beat:
                break
            total += (b - last_beat) * 15.0 / last_bpm
            last_beat, last_bpm = b, v
        return gap + total + (beat - last_beat) * 15.0 / last_bpm

    out = []
    for name, items in notes.items():
        lines = [(seconds(s), seconds(s + l) - seconds(s), m, t.strip() or None) for s, l, m, t in items]
        if lines:
            out.append(Line(sorted(lines), name, 1.0))
    return out


# ---------------------------------------------------------------- reading what a source answers: a MIDI file
def _vlq(data: bytes, i: int) -> tuple[int, int]:
    value = 0
    while True:
        byte = data[i]
        i += 1
        value = (value << 7) | (byte & 0x7F)
        if not byte & 0x80:
            return value, i


def parse_smf(data: bytes) -> tuple[int, list[list[tuple]]]:
    """(ticks per quarter note or, for a SMPTE file, the negative of its ticks a second, the tracks) of a standard MIDI file,
    each track a list of events in time order: (tick, "note", channel, pitch, velocity), (tick, "program", channel, n),
    (tick, "tempo", microseconds per quarter), (tick, "text", type, bytes). A damaged track is read as far as it goes."""
    at = data.find(b"MThd")
    if at < 0:
        raise ValueError("not a MIDI file")
    division = struct.unpack(">H", data[at + 12:at + 14])[0]
    pos = at + 8 + struct.unpack(">I", data[at + 4:at + 8])[0]
    if division & 0x8000:
        division = -((256 - ((division >> 8) & 0xFF)) * (division & 0xFF))
    tracks: list[list[tuple]] = []
    while pos + 8 <= len(data):
        kind, length = data[pos:pos + 4], struct.unpack(">I", data[pos + 4:pos + 8])[0]
        body = data[pos + 8:pos + 8 + length]
        pos += 8 + length
        if kind != b"MTrk":
            continue
        events: list[tuple] = []
        i, tick, status = 0, 0, None
        try:
            while i < len(body):
                delta, i = _vlq(body, i)
                tick += delta
                byte = body[i]
                if byte == 0xFF:
                    typ = body[i + 1]
                    size, j = _vlq(body, i + 2)
                    payload = body[j:j + size]
                    i = j + size
                    status = None
                    if typ == 0x51 and size == 3:
                        events.append((tick, "tempo", int.from_bytes(payload, "big")))
                    elif typ in (0x01, 0x03, 0x04, 0x05):
                        events.append((tick, "text", typ, payload))
                    elif typ == 0x2F:
                        break
                    continue
                if byte in (0xF0, 0xF7):
                    size, j = _vlq(body, i + 1)
                    i = j + size
                    status = None
                    continue
                if byte & 0x80:
                    status = byte
                    i += 1
                elif status is None:
                    break
                high, channel = status & 0xF0, status & 0x0F
                if high in (0xC0, 0xD0):
                    if high == 0xC0:
                        events.append((tick, "program", channel, body[i]))
                    i += 1
                else:
                    first, second = body[i], body[i + 1]
                    i += 2
                    if high == 0x90 and second > 0:
                        events.append((tick, "note", channel, first, second))
                    elif high == 0x80 or high == 0x90:
                        events.append((tick, "note", channel, first, 0))
        except IndexError:
            pass
        tracks.append(events)
    return division, tracks


def _seconds(ticks: np.ndarray, division: int, tempos: list[tuple[int, int]]) -> np.ndarray:
    """Seconds of ticks, over a tempo map [(tick, microseconds per quarter), ...]."""
    ticks = np.asarray(ticks, dtype=float)
    if division < 0:
        return ticks / -division
    marks = sorted({0: 500000, **{t: v for t, v in tempos}}.items())
    at = np.array([m[0] for m in marks], dtype=float)
    per_tick = np.array([m[1] for m in marks], dtype=float) / 1e6 / division
    start = np.concatenate([[0.0], np.cumsum(np.diff(at) * per_tick[:-1])])
    k = np.clip(np.searchsorted(at, ticks, side="right") - 1, 0, len(at) - 1)
    return start[k] + (ticks - at[k]) * per_tick[k]


def _polyphony(starts: np.ndarray, ends: np.ndarray) -> float:
    """The share of the time something sounds that two or more notes do."""
    if not len(starts):
        return 0.0
    moments = np.concatenate([np.stack([starts, np.ones(len(starts))], 1), np.stack([ends, -np.ones(len(ends))], 1)])
    moments = moments[np.lexsort((moments[:, 1], moments[:, 0]))]
    depth = np.cumsum(moments[:, 1])
    spans = np.diff(moments[:, 0])
    return float(spans[depth[:-1] >= 2].sum() / max(spans[depth[:-1] >= 1].sum(), 1e-9))


MELODY_NAME = re.compile(r"vocal|voice|vox|lead|melod|sing|words|lyric|solo|choir|karaoke", re.I)


def midi_lines(data: bytes) -> list[Line]:
    """The candidate melody lines of a MIDI file, likeliest first: each channel of each track, the drums' (channel 10)
    left out, that is mostly one note at a time and has notes enough, with how like a melody it looks."""
    division, tracks = parse_smf(data)
    tempos = [(e[0], e[2]) for t in tracks for e in t if e[1] == "tempo"]
    lines = []
    for n, events in enumerate(tracks):
        name = next((e[3].decode("latin-1").strip() for e in events if e[1] == "text" and e[2] == 0x03), "")
        syllables = [(e[0], e[3].decode("latin-1")) for e in events if e[1] == "text" and e[2] in (0x01, 0x05)
                     and e[3] and not e[3].startswith(b"@")]
        programs = {e[2]: e[3] for e in events if e[1] == "program"}
        held: dict[tuple[int, int], list[int]] = {}
        made: dict[int, list[tuple[int, int, int]]] = {}
        for e in events:
            if e[1] != "note" or e[2] == 9:
                continue
            channel, pitch, velocity = e[2], e[3], e[4]
            if velocity > 0:
                held.setdefault((channel, pitch), []).append(e[0])
            elif held.get((channel, pitch)):
                made.setdefault(channel, []).append((held[(channel, pitch)].pop(0), e[0], pitch))
        for channel, items in made.items():
            if len(items) < 20:
                continue
            ticks = np.array([[a, b] for a, b, _ in items], dtype=float)
            secs = _seconds(ticks.ravel(), division, tempos).reshape(-1, 2)
            if _polyphony(secs[:, 0], secs[:, 1]) > 0.3:
                continue
            notes = [(float(secs[k, 0]), float(max(secs[k, 1] - secs[k, 0], 0.02)), p, None) for k, (_, _, p) in enumerate(items)]
            median = float(np.median([p for _, _, p in items]))
            label = f"track {n + 1}" + (f" ({name})" if name else "") + (f" channel {channel + 1}" if len({c for c in made}) > 1 else "")
            prior = (2.0 * bool(MELODY_NAME.search(name)) + 2.0 * (len(syllables) >= 20) + 1.0 * (52 <= median <= 79)
                     - 3.0 * (median < 45) - 2.0 * (median > 90) + 1.0 * (40 <= len(items) <= 1500) - 1.0 * (len(items) > 2500)
                     + 1.0 * (programs.get(channel, 0) in (52, 53, 54, 85)))
            lines.append(Line(align.monophonic(notes), label, prior))
    return sorted(lines, key=lambda line: (-line.prior, -len(line.notes)))


# ---------------------------------------------------------------- our side
def read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def ours_of(folder: Path) -> align.Ours | None:
    """The lead we detected, from the folder's melody.json: its pitch line on the song's grid (a frame is ours only where
    the contour is sure of it and, where there are timings, sung) and its notes as detected."""
    melody = read_json(folder / "melody.json")
    if not melody or not (melody.get("contour") or {}).get("pitch"):
        return None
    line = melody["contour"]
    step = float(line.get("step") or align.STEP)
    pitch = np.array([np.nan if v is None else v / 10.0 for v in line["pitch"]], dtype=float) - float(melody.get("tuning") or 0.0) / 100.0
    if line.get("conf") is not None and len(line["conf"]) == len(pitch):
        pitch[np.array(line["conf"]) < align.CONF_MIN * 100] = np.nan
    else:
        from karaokifex.music import melody as melody_mod
        words = melody_mod.read_words(folder)
        if words:
            pitch[~melody_mod.sung_mask(len(pitch), step, melody_mod.sung_windows(words))] = np.nan
    if abs(step - align.STEP) > 1e-6:
        pitch = np.interp(np.arange(0, len(pitch) * step, align.STEP) / step, np.arange(len(pitch)), pitch)
    return align.Ours(pitch, melody.get("notes_detected") or melody.get("notes") or [])


# ---------------------------------------------------------------- checking an answer
@dataclass
class Answer:
    source: str
    kind: str
    file: str                   # the path, or the url
    data: bytes


def verdict(result: dict | None, lines: int) -> tuple[bool, str | None]:
    """Whether an alignment's measures are good enough to accept the notes, and why not."""
    if lines == 0:
        return False, "no melody-like line in it"
    if result is None:
        return False, "too little to compare: the file's line or our pitch line is too short, or the recording too long"
    scale = result["warp"].line()[0]
    if not align.SLOPE[0] <= scale <= align.SLOPE[1]:
        return False, f"the file's clock runs x{scale:.2f} of the recording's, outside {align.SLOPE[0]} to {align.SLOPE[1]}"
    if result["aligned"] < ACCEPT_ALIGNED:
        return False, f"only {result['aligned']:.0%} of the file lies on the recording"
    if result["agree_known"] < ACCEPT_AGREE:
        return False, f"{result['agree_known']:.0%} of the file lies within a semitone of our pitch line, under {ACCEPT_AGREE:.0%}"
    if result["covered_ours"] < ACCEPT_OURS:
        return False, f"only {result['covered_ours']:.0%} of our pitch has a known note under it, under {ACCEPT_OURS:.0%}"
    return True, None


def check_answer(answer: Answer, ours: align.Ours) -> dict:
    """note-check.json's content for one answer: its best line (the best-agreeing track or player), measured against ours."""
    try:
        lines = parse_ultrastar(answer.data) if answer.kind == "ultrastar" else midi_lines(answer.data)
    except (ValueError, struct.error) as e:
        return {"version": VERSION, "source": answer.source, "kind": answer.kind, "file": answer.file, "accepted": False,
                "reason": f"not readable as {answer.kind}: {e}", "same_song": False, "agree_known": 0.0}
    best, best_line = None, None
    for line in lines[:LINES_PER_FILE]:
        result = align.align(line.notes, ours)
        if result is not None and (best is None or result["agree_known"] > best["agree_known"]):
            best, best_line = result, line
    accepted, reason = verdict(best, len(lines))
    out = {"version": VERSION, "source": answer.source, "kind": answer.kind, "file": answer.file, "accepted": accepted}
    if reason:
        out["reason"] = reason
    if best is None:
        out.update({"same_song": False, "agree_known": 0.0})
        return out
    scale, offset, dev = best["warp"].line()
    span = best["warp"].span
    aligned = [n for n in best["known"] if n[4]]
    out.update({
        "candidate": best_line.label, "same_song": best["agree_known"] >= SAME_SONG, "transpose": best["transpose"],
        "offset": round(offset, 2),
        "warp": {"scale": round(scale, 4), "offset": round(offset, 2), "max_dev": round(dev, 2),
                 "span": [round(span[0], 2), round(span[1], 2)], "aligned": round(best["aligned"], 3)},
        **{k: (None if best[k] is None else round(best[k], 3) if isinstance(best[k], float) else best[k])
           for k in ("notes_known", "notes_ours", "matched", "missing", "extra", "pitch_exact", "pitch_within_1",
                     "onset_median_ms", "onset_p90_ms", "covered_known", "covered_ours", "agree_known")},
        "wrong": best["wrong"]})
    out["_notes"] = [[round(n[0], 3), round(n[1], 3), n[2] + best["transpose"], n[3]] for n in aligned]
    return out


def asked(folder: Path, sources: list[Source], only_local: bool) -> list[tuple[str, str]]:
    """What would be asked, in order, as (source, where): the dropped-in files, then each source's url or the files its path names."""
    artist, _, song = folder.name.partition(" - ")
    out = [("dropped-in", str(p)) for p in dropped_in(folder)]
    for source in sources:
        if source.url:
            if not only_local:
                out.append((source.name, source.address(artist, song)))
        elif source.root and source.root.is_dir():
            out += [(source.name, str(p)) for p in find_files(source, artist, song)]
    return out


def get(url: str, headers: dict[str, str]) -> bytes | None:
    import requests

    try:
        r = requests.get(url, headers=headers, timeout=60)
    except requests.RequestException as e:
        log.warning("%s: %s", url, e)
        return None
    return r.content if r.ok and r.content else None


def answers(folder: Path, sources: list[Source], only_local: bool = False, fetch=get):
    """The answers for the song, in the order they are to be tried: each source's as (source, Answer list), the dropped-in files
    first. A source with no file, a url that answers nothing or something that is not a chart or MIDI gives none."""
    artist, _, song = folder.name.partition(" - ")
    dropped = dropped_in(folder)
    if dropped:
        batch = [Answer("dropped-in", kind_of(p.name, b"") or "auto", str(p), p.read_bytes()) for p in dropped]
        batch = [a for a in batch if looks_like(a.kind, a.data)]
        if batch:
            yield "dropped-in", batch
    for source in sources:
        if source.url:
            if only_local:
                continue
            body = fetch(source.address(artist, song), source.headers)
            kind = kind_of(source.url, body, source.kind) if body else None
            if kind and looks_like(kind, body):
                yield source.name, [Answer(source.name, kind, source.address(artist, song), body)]
            elif body:
                log.info("%s: %s answered, but it is not a chart or a MIDI file", folder.name, source.name)
            continue
        if not source.root or not source.root.is_dir():
            continue
        files = find_files(source, artist, song)
        if files:
            batch = []
            for p in files:
                data = p.read_bytes()
                kind = kind_of(p.name, data, source.kind)
                if kind and looks_like(kind, data):
                    batch.append(Answer(source.name, kind, str(p), data))
            if batch:
                yield source.name, batch


def write_json(path: Path, doc: dict) -> None:
    partial = path.with_name(path.stem + ".partial" + path.suffix)
    partial.write_text(json.dumps(doc, ensure_ascii=False, indent=1), encoding="utf-8")
    partial.replace(path)


def make(folder: Path, sources: list[Source], *, force: bool = False, only_local: bool = False, fetch=get) -> dict | None:
    """Asks for the song's notes and checks what comes against our melody: the first answer that is accepted written to
    notes-source.json (and applied to melody.json), every answer's check to note-check.json (the accepted one's, else the
    best of the rest). Returns note-check.json's content, None where there was no answer."""
    target, check_file = folder / "notes-source.json", folder / "note-check.json"
    if (target.exists() or check_file.exists()) and not force:
        log.info("%s: checked already", folder.name)
        return read_json(check_file)
    if " - " not in folder.name:
        log.warning("%s: not named “Artist - Song”, skipped", folder.name)
        return None
    ours = ours_of(folder)
    if ours is None:
        log.warning("%s: no melody.json to check against: python -m karaokifex.music first", folder.name)
        return None
    checks: list[tuple[dict, Answer | None]] = []
    chosen = None
    for source, batch in answers(folder, sources, only_local, fetch):
        best = None
        for answer in batch[:FILES_PER_SOURCE]:
            result = check_answer(answer, ours)
            if best is None or (result["accepted"], result["agree_known"]) > (best[0]["accepted"], best[0]["agree_known"]):
                best = (result, answer)
        checks.append(best)
        if best[0]["accepted"]:
            chosen = best
            break
    if not checks:
        log.warning("%s: no source has it", folder.name)
        return None
    if chosen is None:
        chosen = max(checks, key=lambda c: c[0]["agree_known"])
    doc, answer = chosen
    notes = doc.pop("_notes", None)
    for c, _ in checks:
        c.pop("_notes", None)
    doc["others"] = [c for c, _ in checks if c is not doc]
    write_json(check_file, doc)
    if doc["accepted"] and notes is not None:
        source_doc = {"version": VERSION, "source": doc["source"], "kind": doc["kind"],
                      ("url" if answer.file.startswith("http") else "path"): answer.file,
                      "transpose": doc["transpose"], "offset": doc["offset"], "candidate": doc["candidate"], "notes": notes,
                      **{k: doc[k] for k in ("agree_known", "pitch_exact", "pitch_within_1", "onset_median_ms", "notes_known", "matched")}}
        write_json(target, source_doc)
        log.info("%s: %s: %d notes, transpose %+d, %.0f%% of the file within a semitone of ours, our notes %.0f%% exact",
                 folder.name, doc["source"], len(notes), doc["transpose"], doc["agree_known"] * 100, doc["pitch_exact"] * 100)
        apply(folder)
    else:
        log.info("%s: %s: not accepted: %s", folder.name, doc["source"], doc.get("reason"))
    return doc


# ---------------------------------------------------------------- using them: melody.json with the known notes
def apply(folder: Path) -> bool:
    """melody.json made again from the notes as detected and notes-source.json: each detected note with a known note
    under it takes the known note (conf at least 0.9, the cents against the note it was detected at kept as they lie
    against the known one), a known note with no detected note under it is added where the pitch line is voiced under
    it (conf 0.6; its cents from that pitch), the detected notes with nothing known under them keep their note with
    a conf of at most 0.5. The notes as detected are kept as notes_detected, and `known` says where the others came from; made
    again from notes_detected each time, so running it twice changes nothing. Without a notes-source.json the notes as
    detected are put back. True where melody.json was changed."""
    melody_file = folder / "melody.json"
    melody = read_json(melody_file)
    if not melody:
        return False
    source = read_json(folder / "notes-source.json")
    detected = melody.get("notes_detected") or melody.get("notes") or []
    if not source or not source.get("notes"):
        if "notes_detected" not in melody:
            return False
        melody["notes"] = melody.pop("notes_detected")
        melody.pop("known", None)
        _rewrite(melody_file, melody)
        return True
    known = sorted((float(a), float(a) + float(b), int(m)) for a, b, m, *_ in source["notes"])
    k_start, k_end = np.array([k[0] for k in known]), np.array([k[1] for k in known])
    tuning = float(melody.get("tuning") or 0.0) / 100.0
    pitch = np.array([np.nan if v is None else v / 10.0 for v in melody["contour"]["pitch"]], dtype=float) - tuning
    step = float(melody["contour"].get("step") or align.STEP)
    out, taken, matched = [], np.zeros(len(known), dtype=bool), 0
    covered: list[tuple[float, float]] = []
    for n in detected:
        start, end = float(n[0]), float(n[0]) + float(n[1])
        lo, hi = int(np.searchsorted(k_end, start, side="right")), int(np.searchsorted(k_start, end, side="left"))
        over = [(min(end, known[j][1]) - max(start, known[j][0]), j) for j in range(lo, hi)]
        over = [(o, j) for o, j in over if o > 0]
        note = list(n) + [1.0] * (6 - len(n))
        if over and max(over)[0] >= 0.4 * min(end - start, known[max(over)[1]][1] - known[max(over)[1]][0]):
            j = max(over)[1]
            actual = float(n[2]) + float(n[3]) / 100.0
            note[2], note[3] = known[j][2], int(round((actual - known[j][2]) * 100))
            note[5] = round(max(float(note[5]), 0.9), 2)
            taken[j] = True
            matched += 1
        else:
            note[5] = round(min(float(note[5]), 0.5), 2)
        out.append(note)
        covered.append((start, end))
    added = 0
    covered.sort()
    covered_starts, covered_ends = [c[0] for c in covered], [c[1] for c in covered]
    levels = [float(n[4]) for n in detected] or [0.5]
    for j, (a, b, m) in enumerate(known):
        if taken[j]:
            continue
        # the stretch of it no detected note lies on, if there is a pitch there to take its cents from
        free = [(a, b)]
        for s_, e_ in covered[max(0, bisect.bisect_left(covered_ends, a)):bisect.bisect_right(covered_starts, b)]:
            free = [piece for lo_, hi_ in free for piece in ((lo_, min(hi_, s_)), (max(lo_, e_), hi_)) if piece[1] - piece[0] > 1e-6]
        for lo_, hi_ in free:
            if hi_ - lo_ < 0.06:
                continue
            frames = pitch[max(0, int(lo_ / step)):int(hi_ / step) + 1]
            frames = frames[~np.isnan(frames)]
            if len(frames) >= max(2, 0.3 * (hi_ - lo_) / step):
                out.append([round(lo_, 3), round(hi_ - lo_, 3), m, int(round((float(np.median(frames)) - m) * 100)),
                            round(float(np.median(levels)), 2), 0.6])
                added += 1
    out.sort(key=lambda n: n[0])
    melody["notes_detected"] = detected
    melody["notes"] = out
    melody["known"] = {"source": source["source"], "kind": source.get("kind"), "matched": matched, "added": added}
    pitches = [n[2] for n in out]
    melody["range"] = [min(pitches), max(pitches)] if pitches else None
    _rewrite(melody_file, melody)
    return True


def _rewrite(path: Path, melody: dict) -> None:
    from karaokifex.music.melody import write
    write(path.parent, path.name, melody)


def known_stale(folder: Path) -> bool:
    """Whether melody.json needs apply(): there is a notes-source.json and the file does not carry its notes (made again
    since, or never applied), or there is none and the file still carries known ones."""
    melody = read_json(folder / "melody.json")
    source = read_json(folder / "notes-source.json")
    if not melody:
        return False
    if source and source.get("notes"):
        return (melody.get("known") or {}).get("source") != source.get("source") or "notes_detected" not in melody \
            or (folder / "notes-source.json").stat().st_mtime > (folder / "melody.json").stat().st_mtime
    return "notes_detected" in melody


# ---------------------------------------------------------------- the figure: how right our notes are
def percentile(values: list[float], q: float) -> float | None:
    return float(np.percentile(values, q)) if values else None


def answers_of(checks: list[dict]) -> list[dict]:
    """Every answer's check: those of note-check.json (the one kept) and the others kept in them."""
    return [a for c in checks for a in [{k: v for k, v in c.items() if k != "others"}, *c.get("others", [])]]


def summary(checks: list[dict]) -> dict:
    """The gold set's figures over note-check.json contents: how many songs have an answer, accepted or not, how many
    answers there are in all and how many of them are obviously another song's (same_song false), and the median, p10 and
    p90 of pitch_exact, pitch_within_1, agree_known and the onset error over the accepted answers and over every answer
    that is not obviously another song's."""
    def spread(rows: list[dict], key: str) -> dict:
        values = [r[key] for r in rows if r.get(key) is not None]
        return {"n": len(values), "median": percentile(values, 50), "p10": percentile(values, 10), "p90": percentile(values, 90)}

    every = answers_of(checks)
    accepted = [a for a in every if a.get("accepted")]
    same = [a for a in every if a.get("same_song")]
    keys = ("pitch_exact", "pitch_within_1", "onset_median_ms", "agree_known")
    return {"songs": len(checks), "accepted": sum(bool(c.get("accepted")) for c in checks), "rejected": sum(not c.get("accepted") for c in checks),
            "answers": len(every), "same_song": len(same), "obviously_another": len(every) - len(same),
            "of_accepted": {k: spread(accepted, k) for k in keys}, "of_all_same_song": {k: spread(same, k) for k in keys}}


def collect(folders: list[Path]) -> tuple[list[dict], int]:
    """(the note-check.json contents of the folders that have one, how many of the folders have a notes-source.json)."""
    checks, known = [], 0
    for folder in folders:
        doc = read_json(folder / "note-check.json")
        if doc:
            checks.append(doc)
        known += (folder / "notes-source.json").exists()
    return checks, known


def format_report(checks: list[dict], known: int) -> str:
    s = summary(checks)
    fmt = lambda v, pct=True: "-" if v is None else (f"{v * 100:.0f}%" if pct else f"{v:.0f}")
    lines = [f"{s['songs']} songs have an answer from a source ({s['answers']} answers in all): {s['accepted']} songs accepted "
             f"({known} carry notes-source.json), {s['rejected']} rejected. {s['obviously_another']} answers are obviously another "
             f"song's (under {SAME_SONG:.0%} of the file near our pitch line) and are left out of the second figure."]
    for title, key in (("accepted", "of_accepted"), ("every answer that is not obviously another song", "of_all_same_song")):
        lines.append(f"\nHow right our notes are, over the {title}:")
        for label, k, pct in (("same semitone (pitch_exact)", "pitch_exact", True), ("within a semitone (pitch_within_1)", "pitch_within_1", True),
                              ("file's time near our pitch line (agree_known)", "agree_known", True), ("start of a note, median ms", "onset_median_ms", False)):
            d = s[key][k]
            lines.append(f"  {label:48s} n={d['n']:<4d} median {fmt(d['median'], pct):>5s}   p10 {fmt(d['p10'], pct):>5s}   p90 {fmt(d['p90'], pct):>5s}")
    return "\n".join(lines)


# ---------------------------------------------------------------- the command
def library_folders(root: Path) -> list[Path]:
    return sorted(p for p in root.iterdir() if p.is_dir())


@click.command(context_settings={"help_option_names": ["-h", "--help"], "max_content_width": 110})
@click.argument("folders", nargs=-1, type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--library", "library", type=click.Path(exists=True, file_okay=False, path_type=Path), help="Every song folder under this root.")
@click.option("--sources", "sources_path", type=click.Path(exists=True, dir_okay=False, path_type=Path), help="The sources file (else note-sources.toml, $KARAOKIFEX_NOTE_SOURCES, ~/.config/karaokifex/).")
@click.option("--force", is_flag=True, help="Ask again for songs already checked.")
@click.option("--only-local", is_flag=True, help="Only the path sources and the files dropped in <folder>/notes/: no request goes out.")
@click.option("--dry-run", is_flag=True, help="Say what would be asked and where, and ask nothing.")
@click.option("--report", is_flag=True, help="Instead of asking: how right our notes are, over the note-check.json files of the folders.")
@click.option("-v", "--verbose", is_flag=True, help="Show debug output.")
def main(folders: tuple[Path, ...], library: Path | None, sources_path: Path | None, force: bool, only_local: bool, dry_run: bool,
         report: bool, verbose: bool) -> None:
    """Write notes-source.json and note-check.json into each song folder: the notes a source of yours has for it, and how our
    detected notes agree with them."""
    setup_logging(verbose)
    todo = list(folders) + (library_folders(library) if library else [])
    if not todo:
        raise click.UsageError("name song folders, or --library <root>")
    if report:
        checks, known = collect(todo)
        click.echo(format_report(checks, known))
        return
    path = sources_file(sources_path)
    sources = load_sources(path) if path else []
    if not path and not any(dropped_in(f) for f in todo):
        raise click.ClickException("no sources file: copy note-sources.example.toml to note-sources.toml and set your sources")
    for folder in todo:
        if dry_run:
            for source, where in asked(folder, sources, only_local):
                click.echo(f"{folder.name}: {source}: {where}")
            continue
        make(folder, sources, force=force, only_local=only_local)


if __name__ == "__main__":
    main()
