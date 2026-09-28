"""A guitar tab of our own: tab.json, from chords.json and melody.json, in standard tuning.

The chords as shapes a guitarist would reach for: the open chords where there is a common one (C, Am, G7, Dsus4
...), else a barre shape moved up the neck, its root on the low E string or the A string, whichever sits lower
(E-shape and A-shape barres; the kinds without a good E-shape take the A-shape). The melody as single notes, an
octave or two up or down as a whole where the voice sits out of the guitar's reach, each note on the string and
fret that keeps the hand's moves smallest over the whole tune (a shortest path through every place each note can
be played, frets 0-15, higher frets costing a little more).

Which finger where: a chord's fingers by its frets, lowest first and low string first -- the standard C, G, D,
Am, E -- with the index laid across as a barre where the shape needs one (its lowest fret on two strings or
more, and more than four notes to hold); the melody's by where the hand sits, the index at the fret its
position starts on, the hand moving only when a note is out of its four frets.

tab.json (version 2):
  version, source ("own"), tuning (["E2", "A2", "D3", "G3", "B3", "E4"], low to high)
  shift     octaves the melody was moved by to sit on the guitar (0, -1, 1 ...)
  chords    [[start s, length s, name, shape, fingers], ...]; shape: a fret a string, low E to high e ("x"
            muted), "x02210" for Am, frets past 9 as "(10)"; fingers the same way, "x02310" (0 open, 1 the index
            .. 4 the little finger), a barre the one finger on several strings
  notes     [[start s, length s, string (1 high e .. 6 low E), fret, finger], ...]
"""

from __future__ import annotations

import numpy as np

VERSION = 2
TUNING = [40, 45, 50, 55, 59, 64]           # MIDI, low E to high e
TUNING_NAMES = ["E2", "A2", "D3", "G3", "B3", "E4"]
TOP_FRET = 15
X = -1

# the open chords guitarists use, low E to high e
OPEN = {
    "C": [X, 3, 2, 0, 1, 0], "Cmaj7": [X, 3, 2, 0, 0, 0], "C7": [X, 3, 2, 3, 1, 0], "Csus2": [X, 3, 0, 0, 1, 3],
    "D": [X, X, 0, 2, 3, 2], "Dm": [X, X, 0, 2, 3, 1], "D7": [X, X, 0, 2, 1, 2], "Dmaj7": [X, X, 0, 2, 2, 2],
    "Dm7": [X, X, 0, 2, 1, 1], "Dsus2": [X, X, 0, 2, 3, 0], "Dsus4": [X, X, 0, 2, 3, 3], "D6": [X, X, 0, 2, 0, 2],
    "E": [0, 2, 2, 1, 0, 0], "Em": [0, 2, 2, 0, 0, 0], "E7": [0, 2, 0, 1, 0, 0], "Em7": [0, 2, 0, 0, 0, 0],
    "Emaj7": [0, 2, 1, 1, 0, 0], "Esus4": [0, 2, 2, 2, 0, 0], "Em6": [0, 2, 2, 0, 2, 0],
    "G": [3, 2, 0, 0, 0, 3], "G7": [3, 2, 0, 0, 0, 1], "Gmaj7": [3, 2, 0, 0, 0, 2], "G6": [3, 2, 0, 0, 0, 0],
    "A": [X, 0, 2, 2, 2, 0], "Am": [X, 0, 2, 2, 1, 0], "A7": [X, 0, 2, 0, 2, 0], "Am7": [X, 0, 2, 0, 1, 0],
    "Amaj7": [X, 0, 2, 1, 2, 0], "Asus2": [X, 0, 2, 2, 0, 0], "Asus4": [X, 0, 2, 2, 3, 0], "A6": [X, 0, 2, 2, 2, 2],
    "Am6": [X, 0, 2, 2, 1, 2], "Adim": [X, 0, 1, 2, 1, X], "B7": [X, 2, 1, 2, 0, 2], "Fmaj7": [X, X, 3, 2, 1, 0],
}
# movable shapes, frets over the barre at 0: the root on the low E string ...
E_SHAPE = {"maj": [0, 2, 2, 1, 0, 0], "min": [0, 2, 2, 0, 0, 0], "7": [0, 2, 0, 1, 0, 0], "min7": [0, 2, 0, 0, 0, 0],
           "maj7": [0, X, 1, 1, 0, X], "sus4": [0, 2, 2, 2, 0, 0], "min6": [0, 2, 2, 0, 2, 0]}
# ... or on the A string
A_SHAPE = {"maj": [X, 0, 2, 2, 2, 0], "min": [X, 0, 2, 2, 1, 0], "7": [X, 0, 2, 0, 2, 0], "min7": [X, 0, 2, 0, 1, 0],
           "maj7": [X, 0, 2, 1, 2, 0], "sus2": [X, 0, 2, 2, 0, 0], "sus4": [X, 0, 2, 2, 3, 0], "dim": [X, 0, 1, 2, 1, X],
           "dim7": [X, 0, 1, 2, 1, 2], "hdim7": [X, 0, 1, 0, 1, X], "aug": [X, 0, 3, 2, 2, X], "maj6": [X, 0, 2, 2, 2, 2],
           "min6": [X, 0, 2, 2, 1, 2], "minmaj7": [X, 0, 2, 1, 1, 0]}
NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
FLAT_NAMES = {"Db": 1, "Eb": 3, "Gb": 6, "Ab": 8, "Bb": 10}
WRITTEN = {"": "maj", "m": "min", "dim": "dim", "aug": "aug", "m6": "min6", "6": "maj6", "m7": "min7",
           "m(maj7)": "minmaj7", "maj7": "maj7", "7": "7", "dim7": "dim7", "m7b5": "hdim7", "sus2": "sus2", "sus4": "sus4"}


def parse(name: str) -> tuple[int, str] | None:
    """A chord's name as chords.py writes it ("Bbm7", "F#", "Am") as (root 0-11, kind), None for N and X."""
    for root_len in (2, 1):
        root, rest = name[:root_len], name[root_len:]
        pc = NAMES.index(root) if root in NAMES else FLAT_NAMES.get(root)
        if pc is not None and rest in WRITTEN:
            return pc, WRITTEN[rest]
    return None


def shape(name: str) -> list[int] | None:
    """Frets a string, low E to high e (-1 muted), for the chord called `name`; None for no chord."""
    parsed = parse(name)
    if parsed is None:
        return None
    root, kind = parsed
    sharp = NAMES[root] + next(w for w, k in WRITTEN.items() if k == kind)
    for spelled in (name, sharp):
        if spelled in OPEN:
            return list(OPEN[spelled])
    options = []
    if kind in E_SHAPE:
        fret = (root - 4) % 12 or 12
        options.append((fret, [f if f == X else f + fret for f in E_SHAPE[kind]]))
    if kind in A_SHAPE:
        fret = (root - 9) % 12 or 12
        options.append((fret, [f if f == X else f + fret for f in A_SHAPE[kind]]))
    return min(options)[1] if options else None


def fingers(frets: list[int]) -> list[int]:
    """A finger for each string of a chord shape, low E to high e: -1 muted, 0 open, 1 (the index) to 4."""
    held = sorted(((f, s) for s, f in enumerate(frets) if f > 0))
    out = [X if f == X else 0 for f in frets]
    if not held:
        return out
    low = held[0][0]
    on_low = [s for f, s in held if f == low]
    barre = len(on_low) >= 2 and len(held) > 4
    finger = 1
    if barre:
        for s in on_low:
            out[s] = 1
        held = [(f, s) for f, s in held if f != low]
        finger = 2
    for f, s in held:
        out[s] = min(4, finger)
        finger += 1
    return out


def hand(places: list[tuple[int, int] | None]) -> list[int | None]:
    """A finger for each melody note (0 an open string), from where the hand sits: the index at its position's
    first fret, moving only when a note is out of its four frets."""
    out: list[int | None] = []
    position = None
    for place in places:
        if not place:
            out.append(None)
            continue
        fret = place[1]
        if fret == 0:
            out.append(0)
            continue
        if position is None or fret < position:
            position = fret
        elif fret > position + 3:
            position = fret - 3
        out.append(fret - position + 1)
    return out


def written(frets: list[int]) -> str:
    return "".join("x" if f == X else str(f) if f < 10 else f"({f})" for f in frets)


def places(midi: int) -> list[tuple[int, int]]:
    """Every (string 1-6, fret) a note can be played at, string 1 the high e."""
    return [(6 - s, midi - open_) for s, open_ in enumerate(TUNING) if 0 <= midi - open_ <= TOP_FRET]


def octave_shift(pitches: list[int]) -> int:
    """Octaves to move the melody by so the most of it sits within the guitar's reach (0 first when it's a tie)."""
    lo, hi = TUNING[0], TUNING[-1] + TOP_FRET
    return max((0, -1, 1, -2, 2), key=lambda k: (sum(lo <= p + 12 * k <= hi for p in pitches), -abs(k)))


def _move(a: tuple[int, int], b: tuple[int, int]) -> float:
    """What going from place `a` to place `b` costs the hand: the frets it slides (an open string costs none) and,
    less, the strings it crosses."""
    return (0.0 if a[1] == 0 or b[1] == 0 else abs(a[1] - b[1])) + 0.5 * abs(a[0] - b[0])


def _path(options: list[list[tuple[int, int]]]) -> list[tuple[int, int]]:
    """The cheapest place for each note of a run where every note can be played (Viterbi)."""
    cost = [0.08 * o[1] for o in options[0]]
    back: list[list[int]] = [[-1] * len(options[0])]
    for t in range(1, len(options)):
        prev, row, how = options[t - 1], [], []
        for o in options[t]:
            c = [cost[j] + _move(prev[j], o) for j in range(len(prev))]
            j = int(np.argmin(c))
            row.append(c[j] + 0.08 * o[1])
            how.append(j)
        cost = row
        back.append(how)
    j = int(np.argmin(cost))
    path: list[tuple[int, int]] = [options[-1][j]] * len(options)
    for t in range(len(options) - 1, -1, -1):
        path[t] = options[t][j]
        j = back[t][j]
    return path


def finger(pitches: list[int]) -> list[tuple[int, int] | None]:
    """A string and fret for each note, the hand's moves smallest over each run of playable notes; None for a note
    out of reach."""
    options = [places(p) for p in pitches]
    out: list[tuple[int, int] | None] = [None] * len(pitches)
    t = 0
    while t < len(pitches):
        if not options[t]:
            t += 1
            continue
        end = t
        while end < len(pitches) and options[end]:
            end += 1
        out[t:end] = _path(options[t:end])
        t = end
    return out


def make(chords: dict | None, melody: dict | None) -> dict:
    """tab.json's content from chords.json's and melody.json's."""
    out = {"version": VERSION, "source": "own", "tuning": TUNING_NAMES, "shift": 0, "chords": [], "notes": []}
    for start, length, name, *_ in (chords or {}).get("chords", []):
        frets = shape(name)
        if frets:
            out["chords"].append([start, length, name, written(frets), written(fingers(frets))])
    notes = (melody or {}).get("notes", [])
    if notes:
        shift = octave_shift([n[2] for n in notes])
        out["shift"] = shift
        places = finger([n[2] + 12 * shift for n in notes])
        for (start, length, pitch, *_), place, digit in zip(notes, places, hand(places)):
            if place:
                out["notes"].append([start, length, place[0], place[1], digit])
    return out
