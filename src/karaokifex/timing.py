"""Time lyric lines (lrclib) word by word — pure functions, no I/O.

The lyrics decide *what* is displayed and how it is split into lines; the audio
decides *when* each word is sung. Several sources of word times are combined,
best first:

- `lrc-tag`: enhanced-LRC word tags, mapped onto the video's timeline;
- `forced`: wav2vec2 forced alignment of the known lyric line inside its window;
- `whisper`: words whisperx heard, paired with lyric words by a DP sequence
  alignment (only inside the line's window, so a repeated chorus is matched to
  the right occurrence);
- `lrc-line` / `interpolated`: everything else, packed into the voiced parts
  of the gap it falls in.

`plan_alignment` first places synced lines on what whisper heard (`_place_lines`:
the best order-keeping path through every place each line was heard, where
keeping lrclib's spacing is free and a jump costs a few words), so a video that
leaves out a block of identical lines is followed through the cut; with too few
lines heard it maps lrclib's timeline onto the video's (mapping.py) instead. It
derives a search window per line and finds lines that were cut from the video.
The lead-vocal activity (activity.py), when given, vetoes timings: heard words in
silence are dropped, words don't start in silence, and held notes last until
the singing stops.
"""

from __future__ import annotations

import bisect
import difflib
import functools
import math
import re
import statistics
import unicodedata
from dataclasses import dataclass, field, replace
from itertools import groupby
from typing import TYPE_CHECKING, Any, Sequence

import jellyfish
import numpy as np

from karaokifex.mapping import MIN_OFFSET_SAMPLES, TimeMap, fit_time_map, xcorr_offset
from karaokifex.models import LyricLine, TimedLine, TimedWord

if TYPE_CHECKING:
    from karaokifex.activity import Activity

MATCH_WINDOW = 5.0  # seconds a matched word may lie outside its line window
ANCHOR_SLACK = 15.0  # the same while collecting anchors for the time map (the map isn't known yet)
FUZZY_RATIO = 0.6  # minimum spelling similarity to pair two non-identical words
PHONETIC = 0.85  # similarity of words that sound alike ("there"/"their"); English only
MERGE_MIN = 0.8  # similarity needed to pair one word with two ("alright" / "all right")
LAST_LINE_WINDOW = 15.0  # assumed length of the final lrclib line
WINDOW_PAD = 1.0  # seconds added around a line's window for forced alignment
SECONDS_PER_CHAR = 0.08  # natural word duration estimate, clamped to the range below
MIN_WORD, MAX_WORD = 0.2, 0.8
MIN_DURATION = 0.05
FORCED_LINE_MIN = 0.25  # mean forced-alignment score above which a line trusts forced alignment
FORCED_WORD_MIN = 0.1  # forced words below this lose to a whisper match or get interpolated
MIN_HEARD_VOICED = 0.2  # heard words with less of their span voiced are hallucinations
CUT_FIT = 0.4  # unheard lines need this share of their natural duration as voiced time, or they were cut
MAX_CUT_SHARE = 0.35  # more cut lines than this means the time map is wrong: cut nothing
ONSET_SNAP = 0.3  # seconds a line's first word may move to a phrase onset
MAX_HOLD = 4.0  # seconds a line's last word may be stretched to the end of a held note
VOICED_PACK_MIN = 0.5  # interpolate into voiced time only if there is at least this share of the natural duration
VOTE_AGREE = 0.3  # seconds within which the lead-stem and full-mix transcriptions agree
# fill_unlyricked: a gap between timed lines at least this long (s), kept this far from its lines (s),
# holding at least this many heard words, with at least this share of it voiced, is sung without lyrics
FILL_MIN, FILL_PAD, FILL_WORDS, FILL_VOICED = 4.0, 0.3, 6, 0.4
FILL_SCORE = 0.3  # heard words scored below this aren't shown
BACKING = "backing"  # the source of a word heard in the backing vocals
BACKING_FOR_LEAD = 0.3  # what such a word counts for a lyric word outside parentheses
# placing lines on what was heard (_place_lines): a place for a line needs this share of its words heard...
PLACE_MIN = 0.5
# ...two placed lines keep lrclib's spacing within this many seconds (plus this share of their distance)...
DRIFT_TOL, DRIFT_RATE = 1.0, 0.06
# ...and a jump between them (a cut in the video, or lrclib's clock off) costs this many heard words
JUMP_COST = 3.0
# merge_heard: a phrase of the unprompted pass ends at a pause this long (s); the prompted pass's words over
# it (at least DISAGREE_MIN of them) are dropped when less than AGREE_SHARE of them are in the phrase
PHRASE_PAUSE, DISAGREE_MIN, AGREE_SHARE = 0.6, 3, 0.5
PLACE_PAUSE = 2.5  # seconds between two heard words of a place that end it
LINE_BREAK = 3.0  # seconds between a line's heard words that split them into groups (_cohere)...
STRAY_SCORE = 0.3  # ...of which a smaller one with a word scored under this loses its times
MIN_PLACED = 3  # fewer placed lines than this: the time map's anchors decide alone

_APOSTROPHES = str.maketrans({"’": "'", "‘": "'", "`": "'", "´": "'"})
# Spellings that differ between lyric sites and whisper; compared with apostrophes and spaces removed.
_EQUIVALENTS = {
    "wanna": "wantto", "gonna": "goingto", "gotta": "gotto", "cause": "because", "cuz": "because",
    "coz": "because", "alright": "allright", "kinda": "kindof", "outta": "outof", "lemme": "letme",
    "gimme": "giveme", "ok": "okay",
}

Window = tuple[float, float]
Times = tuple[float, float, float | None]  # start, end, score


@dataclass(frozen=True)
class Alignment:
    lines: list[TimedLine]
    matched: int  # lyric words whisperx heard
    total: int
    forced: int = 0  # lyric words timed by forced alignment
    forced_score: float | None = None  # mean forced score over the displayed words
    cut: int = 0  # lyric lines left out because the video doesn't contain them

    @property
    def match_ratio(self) -> float:
        return self.matched / self.total if self.total else 0.0

    @property
    def quality(self) -> float:
        """How well these lyrics fit the audio (0–1), for choosing between lyrics versions."""
        if self.forced_score is None:
            return self.match_ratio
        return 0.5 * self.match_ratio + 0.5 * self.forced_score


@dataclass(frozen=True)
class AlignmentPlan:
    """Where each lyric line is searched for in the video (indices: lines with words, see `kept_lines`)."""

    time_map: TimeMap = field(default_factory=TimeMap)
    starts: tuple[float | None, ...] = ()  # lrclib line starts mapped onto the video
    windows: tuple[Window | None, ...] = ()
    cut: frozenset[int] = frozenset()

    def to_dict(self) -> dict[str, Any]:
        return {"time_map": self.time_map.to_dict(), "starts": list(self.starts),
                "windows": [None if w is None else list(w) for w in self.windows], "cut": sorted(self.cut)}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AlignmentPlan:
        return cls(TimeMap.from_dict(data["time_map"]), tuple(data["starts"]),
                   tuple(None if w is None else (w[0], w[1]) for w in data["windows"]), frozenset(data["cut"]))


@dataclass(frozen=True)
class _Token:
    line: int
    position: int  # word index within its line
    text: str
    norm: str
    backing: bool = False  # inside parentheses: backing vocals

    @property
    def natural(self) -> float:
        """Rough time it takes to sing this word."""
        return min(max(len(self.norm) * SECONDS_PER_CHAR, MIN_WORD), MAX_WORD)


@dataclass(frozen=True)
class _Match:
    """Lyric tokens [i, i + ni) sung as heard words [j, j + nj); ni, nj ∈ {1, 2}, not both 2."""

    i: int
    ni: int
    j: int
    nj: int


def normalize(word: str) -> str:
    """Spelling used for comparing words: "Don’t!" -> "don't", "Café" -> "cafe"."""
    decomposed = unicodedata.normalize("NFKD", word.translate(_APOSTROPHES))
    stripped = "".join(c for c in decomposed if not unicodedata.combining(c))
    return re.sub(r"[^\w']|_", "", stripped.casefold()).strip("'")


def split_words(text: str) -> list[str]:
    """Split a lyric line into words, gluing punctuation-only tokens ("-", "...") onto a neighbour."""
    words: list[str] = []
    leading = ""
    for token in text.split():
        if normalize(token):
            words.append(leading + token)
            leading = ""
        elif words:
            words[-1] += " " + token
        else:
            leading += token + " "
    return words


def kept_lines(lines: Sequence[LyricLine]) -> list[tuple[LyricLine, list[str]]]:
    """The lines that have words, with their words — what every per-line index here refers to."""
    return [(line, words) for line in lines if (words := split_words(line.text))]


def filter_heard(heard: Sequence[TimedWord], activity: Activity | None = None) -> list[TimedWord]:
    """Heard words worth matching: real words, and (with activity) not in silence — whisper hallucinates there.

    Words from the backing vocals (`backing_heard`) were filtered by their own activity and pass as they are.
    """
    words = [word for word in heard if normalize(word.text)]
    if activity is None:
        return words
    return [w for w in words if w.source == BACKING
            or activity.voiced_fraction(w.start, max(w.end, w.start + 0.02)) >= MIN_HEARD_VOICED]


def backing_heard(heard: Sequence[TimedWord], activity: Activity | None = None,
                  lead: Sequence[TimedWord] = ()) -> list[TimedWord]:
    """Words heard in the backing vocals, ready to go with the lead's: filtered by the backing's own activity and
    marked `backing`: they count in full for lyric words in parentheses (backing lines, or a line's backing part)
    and at BACKING_FOR_LEAD for the others. A word the lead was heard singing too (`lead`: alike, starting within
    0.15 s) is the backing doubling it, or the lead bleeding into the stem, and adds nothing: left out."""
    starts = sorted((w.start, normalize(w.text)) for w in lead)
    keys = [t for t, _ in starts]

    def doubled(word: TimedWord) -> bool:
        low, high = bisect.bisect_left(keys, word.start - 0.15), bisect.bisect_right(keys, word.start + 0.15)
        return any(_similarity(normalize(word.text), n) for _, n in starts[low:high])

    return [replace(w, source=BACKING) for w in filter_heard(heard, activity) if not doubled(w)]


def merge_heard(primary: Sequence[TimedWord], extra: Sequence[TimedWord], margin: float = 0.15) -> list[TimedWord]:
    """The words of both transcriptions: `primary` prompted with the lyrics, `extra` without.

    Prompted, whisper hears the lyrics' words better, but skips repeats (a one-word line sung four times) and, where
    the singing is unclear, writes lyric lines that aren't sung there; unprompted, it hears what is sung. So:
    over each phrase `extra` heard, `primary`'s words are dropped when most of them aren't in the phrase (it
    wrote other words there); a word `extra` heard that `primary` has too (spelt alike, starting within
    `margin`) is kept once; the rest of both stay, and the matching with the lyrics takes what fits.
    """
    phrases: list[list[TimedWord]] = []
    for word in sorted(extra, key=lambda w: w.start):
        if phrases and word.start - phrases[-1][-1].end <= PHRASE_PAUSE:
            phrases[-1].append(word)
        else:
            phrases.append([word])
    dropped: set[int] = set()
    for phrase in phrases:
        sung = [normalize(w.text) for w in phrase]
        over = [k for k, w in enumerate(primary)
                if phrase[0].start - margin <= w.start <= phrase[-1].end + margin and normalize(w.text)]
        if len(over) < DISAGREE_MIN:
            continue
        alike = sum(any(_similarity(normalize(primary[k].text), h) for h in sung) for k in over)
        if alike < AGREE_SHARE * len(over):
            dropped.update(over)
    kept = [w for k, w in enumerate(primary) if k not in dropped]
    by_start = sorted(kept, key=lambda w: w.start)
    starts = [w.start for w in by_start]
    added = []
    for word in extra:
        norm = normalize(word.text)
        low, high = bisect.bisect_left(starts, word.start - margin), bisect.bisect_right(starts, word.start + margin)
        if not any(_similarity(norm, normalize(other.text)) for other in by_start[low:high]):
            added.append(word)
    return sorted([*kept, *added], key=lambda w: w.start)


def is_backing(text: str) -> bool:
    """A line wholly in parentheses: lyric sites' mark for backing vocals, which the lead stem doesn't hold."""
    text = text.strip()
    return text.startswith("(") and text.endswith(")") and text.count("(") == 1


def _tokens(kept: Sequence[tuple[LyricLine, list[str]]]) -> list[_Token]:
    return [_Token(index, position, word, normalize(word), inside)
            for index, (_, words) in enumerate(kept) for position, (word, inside) in enumerate(_in_parentheses(words))]


def _in_parentheses(words: Sequence[str]) -> list[tuple[str, bool]]:
    """Each word, and whether it is inside parentheses."""
    out, depth = [], 0
    for word in words:
        opened = depth > 0 or word.lstrip().startswith("(")
        depth = max(0, depth + word.count("(") - word.count(")"))
        out.append((word, opened))
    return out


# --- planning: time map, windows, cut lines -------------------------------------------


def plan_alignment(lines: Sequence[LyricLine], heard: Sequence[TimedWord], activity: Activity | None = None, *,
                   language: str | None = None, window: float = MATCH_WINDOW,
                   backing_activity: Activity | None = None) -> AlignmentPlan:
    kept = kept_lines(lines)
    tokens = _tokens(kept)
    heard = filter_heard(heard, activity)
    starts = [line.start for line, _ in kept]
    synced = any(start is not None for start in starts)
    time_map, supported = TimeMap(), False
    if synced:
        # Candidate maps: from anchors found without any timing constraint, and from anchors found near the
        # onset cross-correlation's offset. The correlation can lock onto a wrong peak (repetitive songs), so
        # the map under which more lyric words match what whisper heard wins; ties go to the first.
        prior = xcorr_offset(starts, activity.onset_envelope(), activity.hop) if activity is not None else None
        searches = [([None] * len(kept), window)]
        if prior is not None:
            searches.append((_line_windows([None if s is None else s + prior for s in starts]), ANCHOR_SLACK))
        candidates = []
        for search, slack in searches:
            anchors = _anchors(kept, tokens, heard, _match_pairs(tokens, heard, search, slack, language))
            candidate = fit_time_map(anchors, prior)
            mapped = [None if s is None else candidate.map(s) for s in starts]
            matched = len(_match_pairs(tokens, heard, _line_windows(mapped), window, language))
            candidates.append((matched, candidate, anchors))
        _, time_map, anchors = max(candidates, key=lambda c: c[0])
        supported = time_map.support(anchors) >= MIN_OFFSET_SAMPLES
    mapped = tuple(None if s is None else time_map.map(s) for s in starts)
    by_line = [[k for k in range(len(tokens)) if tokens[k].line == index] for index in range(len(kept))]
    natural = [sum(tokens[k].natural for k in indices) for indices in by_line]
    backing = frozenset(index for index, (line, _) in enumerate(kept) if is_backing(line.text))

    # Where whisper heard the synced lines decides over the map: the map is one line with shifted sections,
    # which can't follow a video that leaves out one of several identical blocks.
    jumped: set[int] = set()
    if synced and (placed := _place_lines(tokens, by_line, heard, starts, language)) and len(placed) >= MIN_PLACED:
        mapped, jumped, time_map = _placed_plan(placed, tokens, by_line, starts, natural, backing, heard,
                                                   activity, language, backing_activity)
        supported = True
    skip = frozenset(jumped)
    times = _matched_times(tokens, heard, _match_pairs(tokens, heard, _line_windows(mapped), window, language,
                                                       skip=skip))
    _drop_non_monotonic(times)
    _cohere(tokens, times)

    heard_lines = [any(times[k] is not None for k in indices) for indices in by_line]
    ends_before, starts_after = _neighbour_times(by_line, times)

    windows: list[Window | None] = []
    for index in range(len(kept)):
        windows.append(None if index in jumped else
                       _line_search_window(index, mapped, by_line[index], times, natural[index],
                                           ends_before[index], starts_after[index],
                                           None if index in backing else activity))  # the lead's voice: not theirs
    cut = _cut_lines(mapped, heard_lines, natural, ends_before, starts_after, activity, backing,
                     backing_activity) - jumped
    # Only trust "the video doesn't contain this line" when heard words back the time map, and never for
    # a large part of the song: that means the map is wrong, not that the video cut half of it. Lines left
    # out where the heard lines jump are cut on that evidence alone.
    if (synced and not supported) or len(cut) > MAX_CUT_SHARE * len(kept):
        cut = set()
    return AlignmentPlan(time_map, mapped, tuple(windows), frozenset(cut | jumped))


@dataclass(frozen=True)
class _Place:
    """Line `line` heard as heard words [first, last], its words matched as `matches` (token, heard index)."""

    line: int
    first: int
    last: int
    score: float
    start: float  # where the line starts in the video, by its first heard word
    matches: tuple[tuple[int, int], ...]


def _line_places(tokens: Sequence[_Token], indices: Sequence[int], heard: Sequence[TimedWord],
                 heard_norm: Sequence[str], heard_starts: Sequence[float], weights: Sequence[float],
                 phonetic: bool) -> list[_Place]:
    """Every stretch of heard words that sings at least PLACE_MIN of the line's words (its lead's), in order."""
    line_tokens = [tokens[k] for k in indices]
    count = len(line_tokens)
    first, second = line_tokens[0].norm, line_tokens[1].norm if count > 1 else None
    # the bar counts the lead's words: a part in parentheses is backing, which whisper on the lead stem rarely hears
    lead = sum(not t.backing for t in line_tokens) or count
    places = []
    for j in range(len(heard)):
        # A place opens on the line's first word, or on its second when the first wasn't heard just before
        if not _similarity(first, heard_norm[j], phonetic) and not (
                second and _similarity(second, heard_norm[j], phonetic)
                and (j == 0 or not _similarity(first, heard_norm[j - 1], phonetic))):
            continue
        # the heard words within the line's likely span (two transcriptions' words may interleave)
        span = heard[j].start + 1.5 * sum(t.natural for t in line_tokens) + 1.5
        stretch = heard[j: min(j + 3 * count + 4, bisect.bisect_right(heard_starts, span))]
        local = [_Token(0, t.position, t.text, t.norm, t.backing) for t in line_tokens]
        opener = 0 if _similarity(first, heard_norm[j], phonetic) else 1  # the place starts on heard word j
        rest = _match_pairs(local[opener + 1:], stretch[1:], [None], 0.0, "en" if phonetic else None)
        matches = [_Match(opener, 1, 0, 1), *(_Match(m.i + opener + 1, m.ni, m.j + 1, m.nj) for m in rest)]
        for n in range(1, len(matches)):  # a line is sung in one go: end the place at a long pause
            if stretch[matches[n].j].start - stretch[matches[n - 1].j + matches[n - 1].nj - 1].end > PLACE_PAUSE:
                matches = matches[:n]
                break
        found = [_similarity(local[m.i].norm, "".join(heard_norm[j + m.j: j + m.j + m.nj]), phonetic)
                 * weights[j + m.j] * m.ni for m in matches]
        # whether the line was heard here counts the backing's words in full; how much the path gains by it, the
        # lead's more (where both sang it, the lead's place wins)
        if sum(found) < max(PLACE_MIN * lead, 0.6):
            continue
        score = sum(f * (BACKING_FOR_LEAD if heard[j + m.j].source == BACKING and not local[m.i].backing else 1.0)
                    for f, m in zip(found, matches))
        before = sum(t.natural for t in line_tokens[:matches[0].i])  # unheard words ahead of the first heard one
        pairs = tuple((indices[m.i + d], j + m.j) for m in matches for d in range(m.ni))
        places.append(_Place(line_tokens[0].line, j, j + matches[-1].j + matches[-1].nj - 1, score,
                             heard[j].start - before, pairs))
    return places


def _place_lines(tokens: Sequence[_Token], by_line: Sequence[list[int]], heard: Sequence[TimedWord],
                 starts: Sequence[float | None], language: str | None) -> list[_Place]:
    """The best order-keeping choice of places for the lines (some lines left without one).

    Every place a line was heard is a candidate; a path takes lines in order, each on heard words after the
    previous one's. It gains each place's score (heard words, weighted), and between two places pays nothing
    when they keep lrclib's spacing (within DRIFT_TOL, DRIFT_RATE) and JUMP_COST when they don't: a repeated
    block is placed where the timing around it agrees, and a block the video leaves out is jumped over.
    """
    phonetic = language == "en"
    heard_norm = [normalize(word.text) for word in heard]
    weights = [_weight(word) for word in heard]
    heard_starts = [word.start for word in heard]
    places = [place for indices in by_line if indices
              for place in _line_places(tokens, indices, heard, heard_norm, heard_starts, weights, phonetic)]
    if not places:
        return []
    best: list[float] = []
    back: list[int] = []
    for n, place in enumerate(places):
        best.append(place.score)
        back.append(-1)
        for m in range(n):
            earlier = places[m]
            if earlier.line >= place.line or earlier.last >= place.first:
                continue
            gain = best[m] + place.score - _step_cost(earlier, place, starts)
            if gain > best[n]:
                best[n], back[n] = gain, m
    n = max(range(len(places)), key=best.__getitem__)
    path = []
    while n >= 0:
        path.append(places[n])
        n = back[n]
    return path[::-1]


def _step_cost(earlier: _Place, later: _Place, starts: Sequence[float | None]) -> float:
    a, b = starts[earlier.line], starts[later.line]
    if a is None or b is None:
        return 0.0
    drift = abs((later.start - b) - (earlier.start - a))
    tolerance = DRIFT_TOL + DRIFT_RATE * abs(b - a)
    if drift <= tolerance:
        return 0.5 * drift / tolerance
    return JUMP_COST + 0.01 * drift  # of two jumps, the smaller


def _placed_plan(placed: Sequence[_Place], tokens: Sequence[_Token], by_line: Sequence[list[int]],
                 starts: Sequence[float | None], natural: Sequence[float], backing: frozenset[int],
                 heard: Sequence[TimedWord], activity: Activity | None, language: str | None,
                 backing_activity: Activity | None = None) -> tuple[tuple[float | None, ...], set[int], TimeMap]:
    """Video starts for every line from the placed ones, the lines left out, and the map they make.

    Between two placed lines that keep lrclib's spacing the others follow their shift. Where the placed lines
    jump, and before the first and after the last, the lines in the gap are looked for among the words heard
    in it; the rest were cut, unless the gap has voiced time for them, which they then share (backing lines,
    which the lead stem doesn't hold, only time).
    """
    mapped: list[float | None] = list(starts)
    cut: set[int] = set()
    shift = {p.line: p.start - starts[p.line] for p in placed if starts[p.line] is not None}  # type: ignore[operator]
    for p in placed:
        mapped[p.line] = p.start
    end_of_song = activity.duration if activity is not None else max(w.end for w in heard) + LAST_LINE_WINDOW
    gaps = [(range(0, placed[0].line), 0.0, placed[0].start)]
    sections = [placed[0]]
    for earlier, later in zip(placed, placed[1:]):
        between = range(earlier.line + 1, later.line)
        if _step_cost(earlier, later, starts) < JUMP_COST:
            a, b = shift.get(earlier.line), shift.get(later.line)
            for index in between:
                if (s := starts[index]) is not None and a is not None and b is not None:
                    fraction = (s - starts[earlier.line]) / max(starts[later.line] - starts[earlier.line], 1e-6)
                    mapped[index] = s + a + (b - a) * fraction
            continue
        sections.append(later)
        gaps.append((between, heard[earlier.last].end, later.start))
    gaps.append((range(placed[-1].line + 1, len(starts)), heard[placed[-1].last].end, end_of_song))
    for lines, low, high in gaps:
        if lines:
            cut |= _resolve_gap(list(lines), low, high, mapped, tokens, by_line, starts, natural, backing, heard,
                                activity, language, backing_activity)
    base = shift.get(placed[0].line, 0.0)
    corrections = tuple((starts[p.line], shift[p.line] - base) for p in sections[1:]
                        if p.line in shift and starts[p.line] is not None)
    return tuple(mapped), cut, TimeMap(1.0, base, corrections)  # type: ignore[arg-type]


def _resolve_gap(lines: list[int], low: float, high: float, mapped: list[float | None], tokens: Sequence[_Token],
                 by_line: Sequence[list[int]], starts: Sequence[float | None], natural: Sequence[float],
                 backing: frozenset[int], heard: Sequence[TimedWord], activity: Activity | None,
                 language: str | None, backing_activity: Activity | None = None) -> set[int]:
    """Place the gap's lines heard in it; share its voiced time among the others, or cut them if too little.

    Voiced time where words were heard that none of the gap's lines has (a verse the lyrics leave out) is
    no room for them.
    """
    inside = [w for w in heard if low <= w.start < high]
    gap_lines = [by_line[index] if index in lines else [] for index in range(len(by_line))]
    placed = _place_lines(tokens, gap_lines, inside, starts, language)
    anchors = {p.line: p.start for p in placed}
    used = {j for p in placed for _, j in p.matches}
    others = [w for j, w in enumerate(inside) if j not in used and w.source != BACKING]

    def room_in(start: float, end: float) -> float:
        if end <= start:
            return 0.0
        if activity is None:
            return end - start
        taken = [(max(start, w.start - 0.1), min(end, w.end + 0.1)) for w in others if w.end > start and w.start < end]
        return max(0.0, sum(b - a for a, b in activity.voiced_intervals(start, end)) - _covered(taken))

    cut: set[int] = set()
    run: list[int] = []
    for index in [*lines, None]:
        if index is not None and index not in anchors:
            run.append(index)
            continue
        if run:  # unheard lines between the gap's edges or heard lines
            before = max((anchors[i] for i in anchors if i < run[0]), default=low)
            after = anchors[index] if index is not None else high
            need = sum(natural[i] for i in run)
            lead = sum(natural[i] for i in run if i not in backing)
            if lead and activity is not None:
                room = room_in(before, after)
            elif backing_activity is not None:  # backing lines alone, judged by the backing vocals' voice
                room = sum(b - a for a, b in backing_activity.voiced_intervals(before, after)) if after > before else 0.0
                lead = need
            else:  # backing lines alone, the backing unknown: the lead's silence says nothing about them, only time
                room, lead = max(0.0, after - before), need
            if room < CUT_FIT * lead:
                cut.update(run)
                for i in run:
                    mapped[i] = before
            else:
                cursor = before
                for i in run:
                    mapped[i] = cursor
                    cursor += (after - before) * natural[i] / need
            run = []
        if index is not None:
            mapped[index] = anchors[index]
    return cut


def forced_requests(lines: Sequence[LyricLine], plan: AlignmentPlan, *, use_word_tags: bool = True,
                    backing: bool | None = None) -> list[tuple[int, tuple[int, ...], list[str], Window]]:
    """(line index, word positions, normalised words, window) for every line forced alignment should time.

    `backing` True: the words in parentheses (backing lines, and a line's backing part: "one two (three four)"),
    to be aligned on the backing vocals; False: the words outside them, on the lead; None: the lead's words of every
    line and the backing lines whole, all on the lead (no backing stem). A line's backing part is never aligned on
    the lead: the lead doesn't sing it, and the aligner crushed it into whatever gap the lead left.
    """
    requests = []
    for index, (line, words) in enumerate(kept_lines(lines)):
        if index in plan.cut or index >= len(plan.windows) or (window := plan.windows[index]) is None:
            continue
        if use_word_tags and line.word_starts:
            continue
        inside = [flag for _, flag in _in_parentheses(words)]
        if all(inside):
            if backing is False:
                continue
            positions = tuple(range(len(words)))
        elif backing:
            positions = tuple(k for k, flag in enumerate(inside) if flag)
        else:
            positions = tuple(k for k, flag in enumerate(inside) if not flag)
        if positions:
            requests.append((index, positions, [normalize(words[k]) for k in positions], window))
    return requests


def _anchors(kept: Sequence[tuple[LyricLine, list[str]]], tokens: Sequence[_Token], heard: Sequence[TimedWord],
             matches: Sequence[_Match]) -> list[tuple[float, float]]:
    """(lrclib time, video time) pairs: heard first words of synced lines, and heard tagged words."""
    anchors = []
    for match in matches:
        token = tokens[match.i]
        line = kept[token.line][0]
        if token.position == 0 and line.start is not None:
            anchors.append((line.start, heard[match.j].start))
        elif line.word_starts and (tag := line.word_starts[token.position]) is not None:
            anchors.append((tag, heard[match.j].start))
    return anchors


def _line_windows(starts: Sequence[float | None]) -> list[Window | None]:
    """(start, end) of each lyric line according to lrclib, or None when unsynced."""
    windows: list[Window | None] = []
    for index, start in enumerate(starts):
        if start is None:
            windows.append(None)
            continue
        following = next((s for s in starts[index + 1:] if s is not None), None)
        windows.append((start, following if following is not None else start + LAST_LINE_WINDOW))
    return windows


def _neighbour_times(by_line: Sequence[list[int]], times: Sequence[Times | None]
                     ) -> tuple[list[float | None], list[float | None]]:
    """Per line: end of the last heard word before it, start of the first heard word after it."""
    ends_before: list[float | None] = []
    last: float | None = None
    for indices in by_line:
        ends_before.append(last)
        for k in indices:
            if (t := times[k]) is not None:
                last = t[1]
    starts_after: list[float | None] = [None] * len(by_line)
    first: float | None = None
    for index in range(len(by_line) - 1, -1, -1):
        starts_after[index] = first
        for k in reversed(by_line[index]):
            if (t := times[k]) is not None:
                first = t[0]
    return ends_before, starts_after


def _line_search_window(index: int, mapped: Sequence[float | None], indices: Sequence[int],
                        times: Sequence[Times | None], natural: float, end_before: float | None,
                        start_after: float | None, activity: Activity | None) -> Window | None:
    heard_times = [t for k in indices if (t := times[k]) is not None]
    if (start := mapped[index]) is not None:
        following = next((s for s in mapped[index + 1:] if s is not None), start + LAST_LINE_WINDOW)
        low, high = start - WINDOW_PAD, following + WINDOW_PAD
    elif heard_times:
        low, high = heard_times[0][0] - WINDOW_PAD, heard_times[-1][1] + WINDOW_PAD
    elif end_before is not None or start_after is not None:
        low = end_before if end_before is not None else max(0.0, start_after - 2 * natural)  # type: ignore[operator]
        high = start_after if start_after is not None else low + 2 * natural
    else:
        return None
    # Don't reach into the neighbours' heard words (unless that leaves too little room).
    clamped_low = max(low, end_before - 0.2) if end_before is not None else low
    clamped_high = min(high, start_after + 0.2) if start_after is not None else high
    if heard_times:
        clamped_low, clamped_high = min(clamped_low, heard_times[0][0]), max(clamped_high, heard_times[-1][1])
    if clamped_high - clamped_low >= natural:
        low, high = clamped_low, clamped_high
    if activity is not None and (voiced := activity.voiced_intervals(low, high)):
        low, high = max(low, voiced[0][0] - 0.3), min(high, voiced[-1][1] + 0.3)
    low = max(0.0, low)
    return (low, max(high, low + natural))


def _covered(spans: Sequence[tuple[float, float]]) -> float:
    """Seconds covered by possibly overlapping spans."""
    total, reach = 0.0, -math.inf
    for a, b in sorted(spans):
        if b > reach:
            total += b - max(a, reach)
            reach = b
    return total


def _cut_lines(mapped: Sequence[float | None], heard_lines: Sequence[bool], natural: Sequence[float],
               ends_before: Sequence[float | None], starts_after: Sequence[float | None],
               activity: Activity | None, backing: frozenset[int] = frozenset(),
               backing_activity: Activity | None = None) -> set[int]:
    """Unheard lines the video can't contain: silence where they belong, or no room between heard neighbours.

    Backing lines are never cut for the lead's silence (the lead stem doesn't hold them); with the backing vocals'
    activity, for theirs.
    """
    if activity is None:
        return set()

    def voiced_seconds(start: float, end: float, of: Activity = activity) -> float:
        return sum(b - a for a, b in of.voiced_intervals(start, end))

    cut: set[int] = set()
    for index, start in enumerate(mapped):
        if start is None or heard_lines[index] or (index in backing and backing_activity is None):
            continue
        following = next((s for s in mapped[index + 1:] if s is not None), start + LAST_LINE_WINDOW)
        voice = backing_activity if index in backing else activity
        if voiced_seconds(start, following, voice) < CUT_FIT * natural[index]:  # type: ignore[arg-type]
            cut.add(index)
    index = 0
    while index < len(heard_lines):  # runs of unheard lines between two heard ones
        if heard_lines[index]:
            index += 1
            continue
        end = index
        while end < len(heard_lines) and not heard_lines[end]:
            end += 1
        left, right = ends_before[index], starts_after[end - 1]
        lead = sum(natural[i] for i in range(index, end) if i not in backing)
        if left is not None and right is not None and lead:
            if voiced_seconds(left, right) < CUT_FIT * lead:
                cut.update(range(index, end))
        index = end
    return cut


# --- matching lyric words with heard words -----------------------------------------------


@functools.lru_cache(maxsize=16384)
def _phonetic_code(word: str) -> str:
    return jellyfish.metaphone(word.replace("'", "")) if word else ""


@functools.lru_cache(maxsize=65536)
def _similarity(lyric: str, heard: str, phonetic: bool = False) -> float:
    """1 for identical words, a bit less for near-misses ("colour"/"color", "there"/"their"), 0 otherwise."""
    lyric, heard = lyric.replace("'", ""), heard.replace("'", "")
    lyric, heard = _EQUIVALENTS.get(lyric, lyric), _EQUIVALENTS.get(heard, heard)
    if lyric == heard:
        return 1.0
    ratio = difflib.SequenceMatcher(None, lyric, heard).ratio()
    similarity = ratio * 0.9 if ratio >= FUZZY_RATIO else 0.0
    if phonetic and similarity < PHONETIC and _phonetic_code(lyric) and _phonetic_code(lyric) == _phonetic_code(heard):
        return PHONETIC
    return similarity


def _weight(word: TimedWord) -> float:
    """Trust in a heard word, from its alignment score."""
    return 1.0 if word.score is None else 0.5 + 0.5 * min(max(word.score, 0.0), 1.0)


def _canon(word: str) -> str:
    """A word as _similarity compares it: apostrophes out, a spelling that differs between lyric sites folded."""
    word = word.replace("'", "")
    return _EQUIVALENTS.get(word, word)


class _Similar:
    """Which heard words (and pairs of neighbouring heard words) a lyric word (or pair) can match, and how well.

    The same scores as _similarity, but asked only of the spellings that can reach the needed ratio or that sound
    alike, and once per spelling rather than once per heard word: a long film has some 17 000 words each side, and a
    few thousand spellings. A ratio is at most 2·(letters in common)/(len + len) (difflib's quick_ratio), which numpy
    works out for every spelling at once from their letter counts.
    """

    def __init__(self, heard_norm: Sequence[str], phonetic: bool):
        self.phonetic = phonetic
        vocab: dict[str, int] = {}
        self.ids = np.fromiter((vocab.setdefault(h, len(vocab)) for h in heard_norm), dtype=np.int64, count=len(heard_norm))
        self.vocab = list(vocab)
        self.words = self._index(self.vocab)
        # each heard word with the one after it, by where the pair starts
        self.pair_at: dict[str, list[int]] = {}
        for k in range(len(heard_norm) - 1):
            if heard_norm[k]:                                 # the pair's first letter is its first word's
                self.pair_at.setdefault(heard_norm[k] + heard_norm[k + 1], []).append(k)
        self.pairs = self._index(list(self.pair_at))
        self._single: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self._merged: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self._split: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    def _index(self, words: Sequence[str]) -> dict[str, Any]:
        """Spellings' letter counts, lengths and first letters, and by sound, as numbers into `words`."""
        canons = [_canon(word) for word in words]
        letters = sorted({c for canon in canons for c in canon})
        at = {c: n for n, c in enumerate(letters)}
        counts = np.zeros((len(words), max(1, len(letters))), dtype=np.int32)
        for n, canon in enumerate(canons):
            for c in canon:
                counts[n, at[c]] += 1
        by_sound: dict[str, list[int]] = {}
        for n, canon in enumerate(canons):
            if self.phonetic and (code := _phonetic_code(canon)):
                by_sound.setdefault(code, []).append(n)
        return {"words": words, "at": at, "counts": counts, "lengths": np.array([len(c) for c in canons]),
                "first": np.array([word[:1] for word in words], dtype=object), "sound": by_sound}

    def _candidates(self, index: dict[str, Any], word: str, least: float, first: bool) -> list[int]:
        """The spellings `word` may reach the ratio `least` with, or that sound like it; only those starting alike if
        `first`."""
        canon = _canon(word)
        query = np.zeros(index["counts"].shape[1], dtype=np.int32)
        for c in canon:
            if (n := index["at"].get(c)) is not None:
                query[n] += 1
        common = np.minimum(index["counts"], query).sum(axis=1)
        total = index["lengths"] + len(canon)
        # 2·common/total, as difflib works it out; both empty is a ratio of 1
        bound = np.where(total > 0, 2.0 * common / np.maximum(total, 1), 1.0)
        ok = bound >= least - 1e-9
        if self.phonetic and (code := _phonetic_code(canon)):
            sounds = index["sound"].get(code, [])
            ok[sounds] = True
        if first:
            ok &= index["first"] == word[:1]
        return np.flatnonzero(ok).tolist()

    def _scored(self, index: dict[str, Any], word: str, ratio: float, keep: float, first: bool
                ) -> tuple[np.ndarray, np.ndarray]:
        """The spellings `word` (a lyric word or two) scores more than 0 and at least `keep` with, and the scores;
        only those whose ratio can reach `ratio` or that sound alike are asked."""
        numbers, scores = [], []
        for n in set(self._candidates(index, word, ratio, first)):
            score = _similarity(word, index["words"][n], self.phonetic)
            if score > 0 and score >= keep:
                numbers.append(n)
                scores.append(score)
        return np.array(numbers, dtype=np.int64), np.array(scores, dtype=np.float64)

    def single(self, lyric: str) -> np.ndarray:
        """Per heard word: how well `lyric` matches it (0 where it can't)."""
        if lyric not in self._single:
            self._single[lyric] = self._scored(self.words, lyric, FUZZY_RATIO, 0.0, False)
        numbers, scores = self._single[lyric]
        per_spelling = np.zeros(len(self.vocab))
        per_spelling[numbers] = scores
        return per_spelling[self.ids]

    def merged(self, lyric: str, size: int) -> np.ndarray:
        """Per column j (heard words j-2 and j-1 together): how well `lyric` matches the two, at least MERGE_MIN."""
        if lyric not in self._merged:
            numbers, scores = self._scored(self.pairs, lyric, MERGE_MIN, MERGE_MIN, True)
            at, of = [], []
            for n, score in zip(numbers.tolist(), scores.tolist()):
                for k in self.pair_at[self.pairs["words"][n]]:
                    at.append(k + 2)
                    of.append(score)
            self._merged[lyric] = (np.array(at, dtype=np.int64), np.array(of, dtype=np.float64))
        at, of = self._merged[lyric]
        out = np.zeros(size)
        out[at] = of
        return out

    def split(self, lyric_pair: str) -> np.ndarray:
        """Per heard word: how well two lyric words together (`lyric_pair`) match it, at least MERGE_MIN."""
        if lyric_pair not in self._split:
            self._split[lyric_pair] = self._scored(self.words, lyric_pair, MERGE_MIN, MERGE_MIN, True)
        numbers, scores = self._split[lyric_pair]
        per_spelling = np.zeros(len(self.vocab))
        per_spelling[numbers] = scores
        return per_spelling[self.ids]


def _match_pairs(tokens: Sequence[_Token], heard: Sequence[TimedWord], windows: Sequence[Window | None],
                 slack: float, language: str | None = None, *, skip: frozenset[int] = frozenset()
                 ) -> list[_Match]:
    """Order-preserving pairing of lyric and heard words maximising total (score-weighted) similarity.

    An LCS-style dynamic programme that also allows one lyric word sung as two heard
    words ("alright" / "all right") and vice versa. A pair is only allowed when the
    heard word lies within its lyric line's window (± slack); the lines in `skip` (cut) pair with nothing; a word
    heard in the backing vocals counts in full for a lyric word in parentheses, at BACKING_FOR_LEAD for another
    (lyric sites leave many a backing part unmarked, and the backing echoes the lead).

    A row at a time with numpy: a cell's pairings read only the rows above it, so a row is its pairings, then a
    running maximum with "skip a heard word". The scores are the same, cell for cell, as working through the cells
    one by one; a row costs a few array operations, and only three rows of scores are kept, with a byte per cell for
    the way back. An 84-minute film (17 000 words each side) takes seconds and some 300 MB, not hours and 30 GB.
    """
    rows, columns = len(tokens) + 1, len(heard) + 1
    if len(tokens) == 0 or len(heard) == 0:
        return []
    heard_norm = [normalize(word.text) for word in heard]
    similar = _Similar(heard_norm, language == "en")
    weights = np.array([_weight(word) for word in heard])
    backing = np.array([word.source == BACKING for word in heard])
    # a heard word's weight for a lyric word in parentheses (as it is) and for one outside (a backing word counts less)
    for_backing = np.concatenate(([0.0], weights))
    for_lead = np.concatenate(([0.0], weights * np.where(backing, BACKING_FOR_LEAD, 1.0)))
    starts = np.concatenate(([np.nan], np.array([word.start for word in heard])))
    first = np.array([""] + [h[:1] for h in heard_norm], dtype=object)
    no = np.full(columns, -np.inf)
    move = np.zeros((rows, columns), dtype=np.uint8)  # 0 skip lyric, 1 skip heard, 2 = 1:1, 3 = 1:2, 4 = 2:1
    before, previous = np.zeros(columns), np.zeros(columns)

    for i in range(1, rows):
        token = tokens[i - 1]
        window = windows[token.line]
        if token.line in skip:
            low, high = math.inf, -math.inf
        else:
            low, high = (window[0] - slack, window[1] + slack) if window else (-math.inf, math.inf)
        inside = (starts >= low) & (starts <= high)       # column j: heard word j-1 in the window (never column 0)
        w1 = for_backing if token.backing else for_lead
        candidates = []
        if inside.any():
            one = np.concatenate(([0.0], similar.single(token.norm)))
            c2 = no.copy()
            c2[1:] = previous[:-1] + one[1:] * w1[1:]
            candidates.append((np.where(inside & (one > 0), c2, -np.inf), 2))
            two = similar.merged(token.norm, columns)       # heard words j-2 and j-1 as this one lyric word
            if two.any():
                c3 = no.copy()
                c3[2:] = previous[:-2] + two[2:] * (w1[1:-1] + w1[2:]) / 2
                ok = np.zeros(columns, dtype=bool)
                ok[2:] = starts[1:-1] >= low
                candidates.append((np.where(inside & ok & (two >= MERGE_MIN), c3, -np.inf), 3))
            if i >= 2 and tokens[i - 2].line == token.line:
                w0 = for_backing if tokens[i - 2].backing else for_lead
                pair = np.concatenate(([0.0], similar.split(tokens[i - 2].norm + token.norm)))
                if pair.any():
                    c4 = no.copy()
                    c4[1:] = before[:-1] + pair[1:] * np.minimum(w0[1:], w1[1:]) * 1.5
                    starts_alike = first == tokens[i - 2].norm[:1]
                    candidates.append((np.where(inside & starts_alike & (pair >= MERGE_MIN), c4, -np.inf), 4))
        best = previous.copy()
        for value, _ in candidates:
            best = np.maximum(best, value)
        best[0] = 0.0
        row = np.maximum.accumulate(best)
        # how each cell was reached, ties broken as the cell-by-cell programme does: a skip first, then 2, 3, 4
        left = np.concatenate(([0.0], row[:-1]))
        so_far = np.maximum(previous, left)
        how = np.where(previous >= left, 0, 1).astype(np.uint8)
        for value, kind in candidates:
            better = value > so_far
            so_far = np.where(better, value, so_far)
            how[better] = kind
        how[0] = 0
        move[i] = how
        before, previous = previous, row

    matches: list[_Match] = []
    i, j = rows - 1, columns - 1
    while i > 0 and j > 0:
        how = move[i, j]
        if how == 0:
            i -= 1
        elif how == 1:
            j -= 1
        elif how == 2:
            matches.append(_Match(i - 1, 1, j - 1, 1))
            i, j = i - 1, j - 1
        elif how == 3:
            matches.append(_Match(i - 1, 1, j - 2, 2))
            i, j = i - 1, j - 2
        else:
            matches.append(_Match(i - 2, 2, j - 1, 1))
            i, j = i - 2, j - 1
    return matches[::-1]


def _matched_times(tokens: Sequence[_Token], heard: Sequence[TimedWord], matches: Sequence[_Match]
                   ) -> list[Times | None]:
    times: list[Times | None] = [None] * len(tokens)
    for m in matches:
        first, last = heard[m.j], heard[m.j + m.nj - 1]
        scores = [w.score for w in heard[m.j: m.j + m.nj] if w.score is not None]
        score = statistics.fmean(scores) if scores else None
        if m.ni == 1:
            times[m.i] = (first.start, last.end, score)
        else:  # one heard word covers two lyric words: split it by their natural lengths
            a, b = tokens[m.i].natural, tokens[m.i + 1].natural
            middle = first.start + (first.end - first.start) * a / (a + b)
            times[m.i], times[m.i + 1] = (first.start, middle, score), (middle, first.end, score)
    return times


def _cohere(tokens: Sequence[_Token], times: list[Times | None]) -> None:
    """A line is sung in one go: where its heard words fall into groups apart by more than LINE_BREAK, a smaller
    group holding a word whisper was unsure of (scored under STRAY_SCORE: squeezed in where it skipped, or taken
    from a neighbouring line) loses its times, to be timed again with the rest of the line. A group of sure words
    stays: a breath in the middle of a line, or a held note before the rest."""
    for _, group in groupby(range(len(tokens)), key=lambda k: tokens[k].line):
        timed = [k for k in group if times[k] is not None]
        runs: list[list[int]] = []
        for k in timed:
            # start to start, less a word's longest natural length: a squeezed word's end is often stretched
            if runs and times[k][0] - times[runs[-1][-1]][0] > LINE_BREAK + MAX_WORD:  # type: ignore[index]
                runs.append([k])
            elif runs:
                runs[-1].append(k)
            else:
                runs.append([k])
        if len(runs) > 1:
            keep = max(runs, key=len)
            for run in runs:
                unsure = any(times[k][2] is not None and times[k][2] < STRAY_SCORE for k in run)  # type: ignore[index]
                if run is not keep and unsure:
                    for k in run:
                        times[k] = None


def _drop_non_monotonic(times: list[Times | None], apart: Sequence[int | None] | None = None) -> None:
    """Safety net in case whisperx reports words out of order. Words `apart` (a line's backing part, _backing_apart)
    keep an order of their own: the backing may answer over the lead's last words."""
    last = {False: -math.inf, True: -math.inf}
    for index, value in enumerate(times):
        if value is None:
            continue
        own = apart is not None and apart[index] is not None
        if value[0] < last[own]:
            times[index] = None
        else:
            last[own] = value[0]


def _backing_apart(tokens: Sequence[_Token]) -> list[int | None]:
    """For each word of a line's backing part (a line with words in and out of parentheses: "one two (three four)")
    the index of its line's first word, None for every other word: those keep an order of their own."""
    mixed = {line for line, group in groupby(tokens, key=lambda t: t.line) if len({t.backing for t in group}) == 2}
    line_first = {tokens[k].line: k for k in range(len(tokens) - 1, -1, -1)}
    return [line_first[t.line] if t.backing and t.line in mixed else None for t in tokens]


# --- merging -------------------------------------------------------------------------------


def align_lyrics(lines: Sequence[LyricLine], heard: Sequence[TimedWord], *, plan: AlignmentPlan | None = None,
                 forced: Sequence[Sequence[TimedWord] | None] | None = None, activity: Activity | None = None,
                 language: str | None = None, use_word_tags: bool = True,
                 mix_heard: Sequence[TimedWord] | None = None, window: float = MATCH_WINDOW,
                 backing_activity: Activity | None = None) -> Alignment:
    """Time every word of `lines`.

    `forced` holds forced-alignment results per kept line (None where it wasn't run),
    `mix_heard` a second transcription (of the full mix) that votes with `heard`. `heard` may hold words heard
    in the backing vocals too (`backing_heard`), and `backing_activity` is when those are sung.
    """
    kept = kept_lines(lines)
    tokens = _tokens(kept)
    if not tokens:
        return Alignment([], 0, 0)
    if plan is None:
        plan = plan_alignment(lines, heard, activity, language=language, window=window,
                              backing_activity=backing_activity)
    heard = filter_heard(heard, activity)
    windows = _line_windows(plan.starts)
    times = _matched_times(tokens, heard, _match_pairs(tokens, heard, windows, window, language, skip=plan.cut))
    if mix_heard is not None:
        mix = filter_heard(mix_heard, activity)
        times = _vote(times, _matched_times(tokens, mix, _match_pairs(tokens, mix, windows, window, language,
                                                                      skip=plan.cut)))
    apart = _backing_apart(tokens)
    _drop_non_monotonic(times, apart)
    _cohere(tokens, times)
    matched = sum(t is not None for t in times)
    sources = ["whisper" if t is not None else "" for t in times]
    forced_scores = _apply_better_sources(kept, tokens, times, sources, plan, forced, use_word_tags)
    _drop_non_monotonic(times, apart)
    sources = [source if t is not None else "" for source, t in zip(sources, times)]

    all_tokens = tokens
    active = [k for k in range(len(tokens)) if tokens[k].line not in plan.cut]
    tokens = [tokens[k] for k in active]
    times = [times[k] for k in active]
    sources = [sources[k] for k in active]
    _label(sources, times, lambda: _time_unheard_synced_lines(tokens, times, plan.starts, activity), "lrc-line")
    _label(sources, times, lambda: _fill_gaps(tokens, times, activity), "interpolated")
    if activity is not None:
        _snap_to_voice(tokens, times, activity, backing_activity)  # type: ignore[arg-type]  # every slot is filled by now
    # (a line's backing part keeps its own order: the backing may answer over the lead's last words)
    final = _make_monotonic(times, _backing_apart(tokens))  # type: ignore[arg-type]

    timed_lines = [
        TimedLine(tuple(TimedWord(tokens[k].text, *final[k], sources[k]) for k, _ in group))
        for _, group in groupby(enumerate(tokens), key=lambda item: item[1].line)
    ]
    shown = [score for k, score in forced_scores.items() if all_tokens[k].line not in plan.cut]
    return Alignment(timed_lines, matched, len(all_tokens), forced=sum(source == "forced" for source in sources),
                     forced_score=statistics.fmean(shown) if shown else None, cut=len(plan.cut))


def fill_unlyricked(lines: Sequence[TimedLine], heard: Sequence[TimedWord], activity: Activity | None = None
                    ) -> list[TimedLine]:
    """Add what's sung where the lyrics have nothing: a rap or a verse the lyric site left out.

    In each gap between the timed lines (and before the first, after the last) at least FILL_MIN long, the
    words heard there (not in silence, scored at least FILL_SCORE) become lines of their own, labelled
    `heard`, if there are at least FILL_WORDS of them and (with activity) the lead sings through at least
    FILL_VOICED of the gap. The timed lines themselves are kept as they are.
    """
    lines = sorted(lines, key=lambda line: line.start)
    words = [w for w in filter_heard(heard, activity) if w.score is None or w.score >= FILL_SCORE]
    bounds = [-math.inf] + [edge for line in lines for edge in (line.start, line.end)] + [math.inf]
    added: list[TimedLine] = []
    for k in range(0, len(bounds), 2):
        start, end = bounds[k] + FILL_PAD, bounds[k + 1] - FILL_PAD
        if end - start < FILL_MIN:
            continue
        inside = [w for w in words if w.start >= start and w.end <= end]
        if len(inside) < FILL_WORDS:
            continue
        # sung through: the stretch from the first heard word to the last mostly voiced
        if activity is not None and activity.voiced_fraction(inside[0].start, inside[-1].end) < FILL_VOICED:
            continue
        added += [TimedLine(tuple(TimedWord(w.text, w.start, w.end, w.score, "heard") for w in line.words))
                  for line in lines_from_words(inside)]
    return sorted([*lines, *added], key=lambda line: line.start)


def lines_from_words(words: Sequence[TimedWord], *, max_words: int = 8, max_pause: float = 0.8) -> list[TimedLine]:
    """Fallback when there are no lyrics: build display lines straight from the transcription."""
    lines: list[TimedLine] = []
    current: list[TimedWord] = []

    def flush() -> None:
        if current:
            lines.append(TimedLine(tuple(current)))
            current.clear()

    for word in words:
        if not normalize(word.text):
            continue
        if current and (word.start - current[-1].end > max_pause or len(current) >= max_words):
            flush()
        current.append(TimedWord(word.text.strip(), word.start, word.end, word.score, "whisper"))
        if len(current) >= 3 and word.text.rstrip().endswith((".", "?", "!")):
            flush()
    flush()
    return lines


def _apply_better_sources(kept: Sequence[tuple[LyricLine, list[str]]], tokens: Sequence[_Token],
                          times: list[Times | None], sources: list[str], plan: AlignmentPlan,
                          forced: Sequence[Sequence[TimedWord] | None] | None, use_word_tags: bool) -> dict[int, float]:
    """Overwrite whisper times with enhanced-LRC tags and good forced alignments; returns forced scores per token."""
    forced_scores: dict[int, float] = {}
    for index, (line, _) in enumerate(kept):
        indices = [k for k in range(len(tokens)) if tokens[k].line == index]
        if use_word_tags and line.word_starts:
            tags = line.word_starts
            for position, k in enumerate(indices):
                if tags[position] is None:
                    continue
                start = plan.time_map.map(tags[position])  # type: ignore[arg-type]
                following = tags[position + 1] if position + 1 < len(tags) else line.end
                end = plan.time_map.map(following) if following is not None else start + tokens[k].natural
                times[k], sources[k] = (start, max(end, start + MIN_DURATION), None), "lrc-tag"
            continue
        line_forced = forced[index] if forced is not None and index < len(forced) else None
        if not line_forced or len(line_forced) != len(indices):
            continue
        # (a word left out of the line's forced alignment -- its backing part, aligned apart or not at all -- is None)
        placed = [(k, word, word.score or 0.0) for k, word in zip(indices, line_forced) if word is not None]
        if not placed:
            continue
        forced_scores.update((k, score) for k, _, score in placed)
        trusted_line = statistics.fmean(score for _, _, score in placed) >= FORCED_LINE_MIN
        for k, word, score in placed:
            if trusted_line:
                use = score >= FORCED_WORD_MIN or times[k] is None
            else:
                use = times[k] is None and score >= FORCED_WORD_MIN
            if use:
                times[k], sources[k] = (word.start, max(word.end, word.start + MIN_DURATION), word.score), "forced"
    return forced_scores


def _vote(primary: Sequence[Times | None], secondary: Sequence[Times | None]) -> list[Times | None]:
    """Combine two transcriptions' matches: agreeing ones are averaged, otherwise the more confident wins."""
    result: list[Times | None] = []
    for a, b in zip(primary, secondary):
        if a is None or b is None:
            result.append(a or b)
            continue
        score_a, score_b = (0.5 if a[2] is None else max(a[2], 1e-3)), (0.5 if b[2] is None else max(b[2], 1e-3))
        if abs(a[0] - b[0]) <= VOTE_AGREE:
            total = score_a + score_b
            result.append(((a[0] * score_a + b[0] * score_b) / total, (a[1] * score_a + b[1] * score_b) / total,
                           max(score_a, score_b)))
        else:
            result.append(a if score_a >= score_b else b)
    return result


def _label(sources: list[str], times: list[Times | None], fill: Any, label: str) -> None:
    """Run `fill` and label the words it timed."""
    untimed = [t is None for t in times]
    fill()
    for k, was_untimed in enumerate(untimed):
        if was_untimed and times[k] is not None:
            sources[k] = label


def _time_unheard_synced_lines(tokens: list[_Token], times: list[Times | None], line_starts: Sequence[float | None],
                               activity: Activity | None) -> None:
    """Lines no source timed but lrclib has a timestamp for start at that (mapped) timestamp."""
    for line, group in groupby(range(len(tokens)), key=lambda k: tokens[k].line):
        indices = list(group)
        start = line_starts[line] if line < len(line_starts) else None
        if start is None or any(times[k] is not None for k in indices):
            continue
        first, last = indices[0], indices[-1]
        previous_end = next((t[1] for k in range(first - 1, -1, -1) if (t := times[k]) is not None), 0.0)
        next_start = next((t[0] for k in range(last + 1, len(times)) if (t := times[k]) is not None), math.inf)
        next_line = next((s for s in line_starts[line + 1:] if s is not None), math.inf)
        natural = sum(tokens[k].natural for k in indices)
        begin = max(start, previous_end)
        if activity is not None and (voiced := activity.next_voiced(begin, min(next_line, next_start))) is not None:
            begin = voiced
        finish = min(next_line - 0.1, next_start, begin + natural * 2.5)
        _pack_voiced(times, indices, [tokens[k].natural for k in indices], begin, max(finish, begin + natural),
                     activity)


def _fill_gaps(tokens: list[_Token], times: list[Times | None], activity: Activity | None) -> None:
    """Interpolate every remaining run of untimed words between its timed neighbours."""
    count, i = len(times), 0
    while i < count:
        if times[i] is not None:
            i += 1
            continue
        j = i
        while j < count and times[j] is None:
            j += 1
        left = times[i - 1][1] if i > 0 else None  # type: ignore[index]
        right = times[j][0] if j < count else None  # type: ignore[index]
        _fill_run(tokens, times, i, j, left, right, activity)
        i = j


def _fill_run(tokens: list[_Token], times: list[Times | None], i: int, j: int,
              left: float | None, right: float | None, activity: Activity | None) -> None:
    run = list(range(i, j))
    natural = sum(tokens[k].natural for k in run)
    if left is None and right is None:
        left, right = 0.0, natural
    elif left is None:
        left = max(0.0, right - natural)  # type: ignore[operator]
    elif right is None:
        right = left + natural
    assert left is not None and right is not None
    right = max(right, left)

    def weights(indices: list[int]) -> list[float]:
        return [tokens[k].natural for k in indices]

    previous_line = tokens[i - 1].line if i > 0 else None
    next_line = tokens[j].line if j < len(tokens) else None
    if right - left <= natural * 1.5 or previous_line == next_line:
        _pack_voiced(times, run, weights(run), left, right, activity)
        return

    # A long gap (instrumental break, unheard lines): keep words close to the line they belong to.
    head = [k for k in run if tokens[k].line == previous_line]
    tail = [k for k in run if tokens[k].line == next_line]
    middle = [k for k in run if k not in head and k not in tail]
    if head:
        need = sum(weights(head))
        left = _pack_voiced(times, head, weights(head), left, right, activity, need=need)
    if tail:
        need = sum(weights(tail))
        right = _pack_voiced(times, tail, weights(tail), left, right, activity, need=need, from_end=True)
    groups = [list(g) for _, g in groupby(middle, key=lambda k: tokens[k].line)]
    if groups:
        slot = (right - left) / len(groups)
        for number, group in enumerate(groups):
            start = left + number * slot
            _pack_voiced(times, group, weights(group), start, start + slot, activity,
                         need=min(slot, sum(weights(group))))


def _pack(times: list[Times | None], indices: list[int], weights: list[float], start: float, end: float) -> None:
    """Spread consecutive words over [start, end], proportionally to their weights."""
    total = sum(weights) or 1.0
    cursor = start
    for index, weight in zip(indices, weights):
        length = (end - start) * weight / total
        times[index] = (cursor, cursor + length, None)
        cursor += length


def _pack_voiced(times: list[Times | None], indices: list[int], weights: list[float], start: float, end: float,
                 activity: Activity | None, *, need: float | None = None, from_end: bool = False) -> float:
    """Like _pack, but only into the voiced parts of [start, end].

    With `need`, only the first (or, `from_end`, the last) `need` seconds of voiced
    time are used. Returns the far edge of the time used, so callers can pack more
    words next to it. Falls back to plain packing when there is too little voice.
    """
    natural = sum(weights)
    intervals = activity.voiced_intervals(start, end) if activity is not None and end > start else []
    voiced = sum(b - a for a, b in intervals)
    if voiced < VOICED_PACK_MIN * natural:
        if need is None:
            _pack(times, indices, weights, start, end)
            return end
        if from_end:
            _pack(times, indices, weights, end - need, end)
            return end - need
        _pack(times, indices, weights, start, start + need)
        return start + need
    use = voiced if need is None else min(need, voiced)
    offset = voiced - use if from_end else 0.0
    total = natural or 1.0
    for index, weight in zip(indices, weights):
        length = use * weight / total
        times[index] = (_voiced_time(intervals, offset, False), _voiced_time(intervals, offset + length, True), None)
        offset += length
    return _voiced_time(intervals, voiced - use, False) if from_end else _voiced_time(intervals, offset, True)


def _voiced_time(intervals: Sequence[tuple[float, float]], offset: float, is_end: bool) -> float:
    """The moment `offset` seconds of voiced time into the intervals."""
    for a, b in intervals:
        span = b - a
        if offset < span or (is_end and offset <= span + 1e-9):
            return a + offset
        offset -= span
    return intervals[-1][1]


def _snap_to_voice(tokens: Sequence[_Token], times: list[Times], activity: Activity,
                   backing_activity: Activity | None = None) -> None:
    """Line starts move to a phrase onset, words don't start in silence, last words last as long as the note.

    Words in parentheses (a backing line, or a line's backing part) follow the backing vocals' activity instead, or
    stay as they are without: the lead's says nothing of them. A line's backing part starts and ends like a line.
    """
    count = len(times)
    lead_activity = activity
    part = lambda k: (tokens[k].line, tokens[k].backing)
    for k in range(count):
        activity = backing_activity if tokens[k].backing else lead_activity  # type: ignore[assignment]
        if activity is None:
            continue
        start, end, score = times[k]
        first_of_line = k == 0 or part(k - 1) != part(k)
        last_of_line = k == count - 1 or part(k + 1) != part(k)
        following = times[k + 1][0] if k + 1 < count else math.inf
        if first_of_line and (onset := activity.onset_near(start, ONSET_SNAP)) is not None and onset < end:
            start = onset
        elif not activity.is_voiced(start):
            if (voiced := activity.next_voiced(start, min(end, following))) is not None:
                start = voiced
        if last_of_line:
            limit = min(following - 0.1, end + MAX_HOLD)
            if limit > end:
                end = max(end, activity.phrase_end(max(start, end - 0.05), limit))
        times[k] = (start, max(end, start + MIN_DURATION), score)


def _make_monotonic(times: list[Times], apart: Sequence[int | None] | None = None) -> list[Times]:
    """Starts never going back, ends never past the next start. Words `apart` (a line's backing part: the index of
    their line's first word) do so among themselves, never starting before their line's first word: the backing may
    sing over the lead, which a single order would crush into the gap after it."""
    if apart is not None and any(a is not None for a in apart):
        result = list(times)
        lead = [k for k in range(len(times)) if apart[k] is None]
        back = [k for k in range(len(times)) if apart[k] is not None]
        for indices, made in ((lead, _make_monotonic([times[k] for k in lead])), (back, _make_monotonic([times[k] for k in back]))):
            for k, t in zip(indices, made):
                result[k] = t
        for k in back:
            first = apart[k]
            if result[k][0] < result[first][0]:
                start, end, score = result[k]
                result[k] = (result[first][0], max(end, result[first][0] + MIN_DURATION), score)
        return result
    result: list[Times] = []
    previous_start = 0.0
    for start, end, score in times:
        start = max(start, previous_start)
        result.append((start, max(end, start + MIN_DURATION), score))
        previous_start = start
    for index in range(len(result) - 1):
        start, end, score = result[index]
        next_start = result[index + 1][0]
        if end > next_start:
            result[index] = (start, max(next_start, start + MIN_DURATION), score)
    return result
