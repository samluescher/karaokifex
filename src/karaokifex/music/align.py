"""Known notes against the lead voice we detected: which transposition and which time warp make the one lie on the
other, and how well they then agree. Pure numpy, no model and no GPU; notes.py (karaokifex-notes) reads and writes the
files around it.

The two sides are the notes of a known source (UltraStar chart, MIDI melody track: [(start s, length s, MIDI note,
syllable), ...] on the file's own clock) and ours (melody.json: the pitch line every STEP s and the notes, on the song's
own grid: A440's less the song's tuning). The file is often a whole octave or more off the recording (charts are written
where the singer is comfortable), starts later or earlier, and a MIDI may run at another tempo; so, in order:

  1. the pitch classes. Both sides are folded to 12 pitch classes a frame of HOP s (a few of ours' STEP frames) and the
     shift of the file's pitch classes that makes the two histograms alike is tried for the best four (a 12-bin
     correlation), each followed by
  2. a coarse alignment: dynamic programming over the matrix of how well file frame i and recording frame j agree
     (the same pitch class +1, a semitone off +0.5, another -0.4; a frame sung on one side only costs MISS or EXTRA),
     the path a free start to a free end (a subsequence: the file may run on past the recording or begin before it,
     and the recording has its intro) in steps of (1,1), (3,4) and (4,3), which is a slope between 0.75 and 1.33
     (SLOPE) at most, the steps weighed by the file frames they cross. The shift with the best score is kept;
  3. the octave: the file moved by 12 k semitones (k -2 .. 1) over that shift, whichever puts most of the file's
     time within a semitone of our pitch line, then
  4. a fine alignment at STEP, the same dynamic programming in a band of BAND s either side of the coarse path, on
     exact pitch (within half a semitone +1, a semitone +0.5, 1.5 semitones 0, farther -0.6), start and end where the
     coarse path's are, the warp smoothed over SMOOTH s.

measure() then says how right our notes are against the warped file (notes.py writes it as note-check.json):
  agree_known     the share of the file's time that lies within a semitone of our pitch line (the file's time not
                  aligned at all counts against it)
  covered_known   the share of the file's time where we have a pitch;  covered_ours  the share of our pitch that has
                  a known note under it
  matched         the file's notes that our notes cover by half or more;  missing  those that they do not;  extra
                  our notes that no known note covers by half
  pitch_exact, pitch_within_1  the share of the overlap of our notes with known notes at the same semitone, and within
                  one (our note on the song's grid, the file's with transpose)
  onset_median_ms, onset_p90_ms  how far our note starts lie from the known ones' (our nearest start within 0.3 s)

The word windows of timings.json do not depend on pitch, so they are no part of the alignment; in_words says how much of the
file's time falls in them.
"""

from __future__ import annotations

import bisect
from dataclasses import dataclass

import numpy as np

STEP = 0.02             # s: a frame of our pitch line and of the fine alignment
HOP = 0.1               # s: a frame of the coarse alignment (more where the song is long)
MAX_CELLS = 25_000_000  # the coarse matrix: file frames times recording frames; HOP grows to fit it
MAX_SECONDS = 1500.0    # a recording longer is not aligned
SLOPE = (0.75, 1.33)    # recording seconds a file second may take, at most and at least
STEPS = ((1, 1), (3, 4), (4, 3))        # coarse steps (file frames, recording frames)
FINE_STEPS = ((1, 1), (1, 2), (2, 1))
BAND = 0.7              # s: how far the fine alignment may stray from the coarse path
SMOOTH = 0.14           # s: the fine warp is averaged over this
CANDIDATES = 4          # pitch-class shifts tried by the coarse alignment
NEAR, FAR = 0.5, -0.4   # coarse agreement of pitch classes a semitone apart and others (the same is 1)
MISS, EXTRA = 0.35, 0.2     # a frame sung only in the file; only in the recording
CONF_MIN = 0.3          # contour frames with a lower conf are not ours
MIN_NOTES, MIN_SECONDS = 15, 8.0        # the file's notes, and our pitched time, needed to say anything
NEG = -1e9


@dataclass
class Ours:
    """The lead we detected: its pitch line (MIDI on the song's grid, NaN where none or unsure) a STEP s, its notes
    [[start, length, MIDI note on the grid, cents, ...], ...] and, where the song has timings, the windows its words are
    sung in ([(start, end), ...], widened as melody.py counts a word sung), None where it has none."""
    pitch: np.ndarray
    notes: list[list]
    windows: list[tuple[float, float]] | None = None

    @property
    def seconds(self) -> float:
        return len(self.pitch) * STEP


@dataclass
class Warp:
    """File time to recording time: piecewise linear over the file frames rows[0] .. rows[-1] (STEP s each), the
    recording's frame (a float) for each in cols."""
    rows: np.ndarray
    cols: np.ndarray

    @property
    def span(self) -> tuple[float, float]:
        return float(self.rows[0] * STEP), float((self.rows[-1] + 1) * STEP)

    def __call__(self, t: np.ndarray | float) -> np.ndarray:
        """The recording's time of file time t (NaN outside the aligned span)."""
        t = np.asarray(t, dtype=float)
        frame = t / STEP
        out = np.interp(frame, self.rows, self.cols) * STEP
        return np.where((frame >= self.rows[0] - 1e-6) & (frame <= self.rows[-1] + 1 + 1e-6), out, np.nan)

    def line(self) -> tuple[float, float, float]:
        """(scale, offset, largest deviation in s) of the straight line recording = scale * file + offset that fits it."""
        t, u = self.rows * STEP, self.cols * STEP
        if len(t) < 3 or t[-1] - t[0] < 1e-6:
            return 1.0, float(u[0] - t[0]), 0.0
        scale, offset = np.polyfit(t, u, 1)
        return float(scale), float(offset), float(np.abs(u - (scale * t + offset)).max())


def monophonic(notes: list[tuple]) -> list[tuple]:
    """The notes in time order, one at a time: a note cut where the next begins (what sounds together at one
    moment: the highest), those left with no length dropped."""
    out: list[list] = []
    for n in sorted(notes, key=lambda n: (round(n[0], 4), -n[2])):
        n = list(n)
        if out and abs(out[-1][0] - n[0]) < 1e-4:
            continue
        if out and out[-1][0] + out[-1][1] > n[0]:
            out[-1][1] = n[0] - out[-1][0]
        out.append(n)
    return [tuple(n) for n in out if n[1] > 1e-3]


def line_of(notes: list[tuple], frames: int, shift: int = 0) -> np.ndarray:
    """The notes' pitch on a STEP grid of `frames` (NaN where none), `shift` semitones added."""
    out = np.full(frames, np.nan)
    for start, length, midi, *_ in notes:
        a = int(round(start / STEP))
        b = max(a + 1, int(round((start + length) / STEP)))
        if a < frames:
            out[max(a, 0):min(b, frames)] = midi + shift
    return out


def chroma(pitch: np.ndarray, per: int) -> np.ndarray:
    """(frames of `per` STEPs, 12): the share of each frame sung at each pitch class."""
    n = -(-len(pitch) // per)
    voiced = np.flatnonzero(~np.isnan(pitch))
    flat = (voiced // per) * 12 + np.mod(np.round(pitch[voiced]).astype(int), 12)
    return (np.bincount(flat, minlength=n * 12).reshape(n, 12) / per).astype(np.float32)


def kernel() -> np.ndarray:
    k = np.full((12, 12), FAR, dtype=np.float32)
    for i in range(12):
        k[i, i] = 1.0
        k[i, (i + 1) % 12] = k[i, (i - 1) % 12] = NEAR
    return k


def coarse_scores(a: np.ndarray, b: np.ndarray, shift: int) -> np.ndarray:
    """(file frames, recording frames): how well each file frame (chroma `a`, its pitch classes moved up by `shift`)
    agrees with each of the recording's (chroma `b`)."""
    a = np.roll(a, shift, axis=1)
    va, vb = a.sum(axis=1, keepdims=True), b.sum(axis=1, keepdims=True).T
    return (a @ kernel() @ b.T - MISS * va * (1 - vb) - EXTRA * vb * (1 - va)).astype(np.float32)


def shifts(a: np.ndarray, b: np.ndarray) -> list[int]:
    """The CANDIDATES pitch-class shifts that make the file's histogram (chroma `a`) most like the recording's `b`."""
    ha, hb = a.sum(axis=0), b.sum(axis=0)
    ha, hb = ha - ha.mean(), hb - hb.mean()
    score = [float(np.dot(np.roll(ha, s), hb)) for s in range(12)]
    return [int(s) for s in np.argsort(score)[::-1][:CANDIDATES]]


def dp(scores: np.ndarray, base: np.ndarray, steps: tuple, *, start_anywhere: bool = True, end_anywhere: bool = True) -> tuple[np.ndarray, np.ndarray, float]:
    """The best monotone path through `scores` (rows x band): row r holds the recording's frames base[r] ..
    base[r] + width - 1. A step (di, dj) goes from the cell di rows up and dj columns back and scores di times the
    new cell; a path starts on any cell (or on row 0) and ends on any cell (or on the last row). Returns the path's
    rows and (absolute) columns and its score."""
    rows, width = scores.shape
    best = np.full((rows, width), NEG, dtype=np.float32)
    how = np.full((rows, width), -1, dtype=np.int8)
    for r in range(rows):
        cur = scores[r] * 1.0 if (start_anywhere or r == 0) else np.full(width, NEG, dtype=np.float32)
        arg = np.full(width, -1, dtype=np.int8)
        for k, (di, dj) in enumerate(steps):
            if r - di < 0:
                continue
            shift = int(base[r] - dj - base[r - di])
            lo, hi = max(0, -shift), min(width, width - shift)
            if lo >= hi:
                continue
            cand = np.full(width, NEG, dtype=np.float32)
            cand[lo:hi] = best[r - di, lo + shift:hi + shift] + di * scores[r, lo:hi]
            better = cand > cur
            cur = np.where(better, cand, cur)
            arg = np.where(better, k, arg).astype(np.int8)
        best[r], how[r] = cur, arg
    if end_anywhere:
        r, d = np.unravel_index(int(np.argmax(best)), best.shape)
    else:
        r, d = rows - 1, int(np.argmax(best[-1]))
    total = float(best[r, d])
    path_r, path_c = [int(r)], [int(base[r] + d)]
    while how[r, d] >= 0:
        di, dj = steps[how[r, d]]
        d = int(d + base[r] - dj - base[r - di])
        r -= di
        path_r.append(int(r))
        path_c.append(int(base[r] + d))
    return np.array(path_r[::-1]), np.array(path_c[::-1]), total


def warp_of(rows: np.ndarray, cols: np.ndarray, smooth: float = 0.0) -> Warp:
    """A warp through the path (rows and cols in STEP frames), one point a row, smoothed over `smooth` s."""
    full = np.arange(rows[0], rows[-1] + 1)
    c = np.interp(full, rows, cols.astype(float))
    n = int(round(smooth / STEP)) | 1
    if n > 1 and len(c) > n:
        padded = np.pad(c, n // 2, mode="edge")
        c = np.convolve(padded, np.ones(n) / n, mode="valid")
    return Warp(full, np.maximum.accumulate(c))


def coarse(known: list[tuple], ours: Ours) -> tuple[Warp, int, float] | None:
    """The coarse warp of the file onto the recording, the pitch-class shift it is on and its score; None where no
    path is found."""
    total = max((n[0] + n[1] for n in known), default=0.0)
    per = max(round(HOP / STEP), int(np.ceil(np.sqrt(total * ours.seconds / MAX_CELLS) / STEP)))
    a = chroma(line_of(known, int(np.ceil(total / STEP)) + 1), per)
    b = chroma(ours.pitch, per)
    best = None
    for shift in shifts(a, b):
        s = coarse_scores(a, b, shift)
        r, c, score = dp(s, np.zeros(len(s), dtype=np.int64), STEPS)
        if best is None or score > best[0]:
            best = (score, shift, r, c)
    if best is None or len(best[2]) < 3:
        return None
    score, shift, r, c = best
    # the path at a coarse frame's middle, in STEP frames
    return warp_of(r * per + per // 2, c * per + per // 2), shift, score


def warped(known: list[tuple], warp: Warp) -> list[tuple]:
    """The file's notes on the recording's clock: (start, length, MIDI note, syllable, aligned) where the notes that
    start inside the warp's span keep their length scaled; (nan, ..., False) for the others."""
    out = []
    t0, t1 = warp.span
    for start, length, midi, *rest in known:
        end = min(start + length, t1)
        if start < t0 or start >= t1 or end <= start:
            out.append((float("nan"), length, midi, rest[0] if rest else None, False))
            continue
        a, b = float(warp(start)), float(warp(end))
        out.append((a, max(b - a, STEP), midi, rest[0] if rest else None, True))
    return out


def measure_line(known_w: list[tuple], ours: Ours, shift: int, total: float, near: float = 1.0) -> dict:
    """The agreement along the pitch line of the warped notes (as warped()) moved `shift` semitones, `total` the file's
    time in all (recording seconds)."""
    ok = [n for n in known_w if n[4]]
    line = line_of([(n[0], n[1], n[2]) for n in ok], len(ours.pitch), shift)
    voiced, sung = ~np.isnan(ours.pitch), ~np.isnan(line)
    both = voiced & sung
    diff = np.abs(ours.pitch - line)
    return {"agree_known": float(((diff <= near) & both).sum() * STEP / max(total, 1e-6)),
            "covered_known": float(both.sum() * STEP / max(total, 1e-6)),
            "covered_ours": float(both.sum() / max(voiced.sum(), 1)),
            "exact_line": float(((np.round(ours.pitch) == line) & both).sum() / max(both.sum(), 1))}


def in_words(known_w: list[tuple], windows: list[tuple[float, float]] | None) -> float | None:
    """The share of the warped file's time (notes as warped() gives them) that lies in the sung windows, None without any."""
    if not windows:
        return None
    starts = [w[0] for w in windows]
    ends = [w[1] for w in windows]
    inside = total = 0.0
    for start, length, *_, ok in known_w:
        if not ok:
            continue
        total += length
        k = bisect.bisect_right(ends, start)
        while k < len(windows) and starts[k] < start + length:
            inside += max(0.0, min(start + length, ends[k]) - max(start, starts[k]))
            k += 1
    return inside / total if total else None


def measure_notes(known_w: list[tuple], ours: Ours, shift: int) -> dict:
    """How our notes agree with the warped file's: matched, missing, extra, pitch_exact, pitch_within_1, the onsets
    and a sample of the wrong ones."""
    known = [(n[0], n[0] + n[1], n[2] + shift) for n in known_w if n[4]]
    mine = [(float(n[0]), float(n[0] + n[1]), int(n[2])) for n in ours.notes]
    starts = np.array([m[0] for m in mine])
    ends = np.array([m[1] for m in mine])
    exact = within = overlap_total = 0.0
    matched = 0
    onsets, wrong = [], []
    covered_by = np.zeros(len(mine))
    for a, b, k in known:
        lo = int(np.searchsorted(ends, a, side="right"))
        hi = int(np.searchsorted(starts, b, side="left"))
        over = [(max(0.0, min(b, mine[j][1]) - max(a, mine[j][0])), j) for j in range(lo, hi)]
        over = [(o, j) for o, j in over if o > 0]
        share = sum(o for o, _ in over)
        for o, j in over:
            covered_by[j] += o
            overlap_total += o
            exact += o * (mine[j][2] == k)
            within += o * (abs(mine[j][2] - k) <= 1)
            if abs(mine[j][2] - k) >= 1:
                wrong.append((a, mine[j][2], k, o))
        if share >= 0.5 * (b - a):
            matched += 1
            near = [abs(mine[j][0] - a) for _, j in over if abs(mine[j][0] - a) <= 0.3]
            near += [abs(starts[j] - a) for j in (lo - 1, hi) if 0 <= j < len(mine) and abs(starts[j] - a) <= 0.3]
            if near:
                onsets.append(min(near))
    extra = int(sum(covered_by[j] < 0.5 * (mine[j][1] - mine[j][0]) for j in range(len(mine))))
    wrong.sort(key=lambda w: w[0])
    keep = wrong[:: max(1, len(wrong) // 10)][:10]
    return {"notes_known": len(known), "notes_ours": len(mine), "matched": matched, "missing": len(known) - matched,
            "extra": extra, "pitch_exact": float(exact / overlap_total) if overlap_total else 0.0,
            "pitch_within_1": float(within / overlap_total) if overlap_total else 0.0,
            "onset_median_ms": float(np.median(onsets) * 1000) if onsets else None,
            "onset_p90_ms": float(np.percentile(onsets, 90) * 1000) if onsets else None,
            "wrong": [[round(t, 2), ours_, known_] for t, ours_, known_, _ in keep]}


def align(known: list[tuple], ours: Ours) -> dict | None:
    """Aligns the file's notes `known` ([(start, length, MIDI, syllable), ...] on its clock) to `ours`: a dict with
    transpose (semitones added to the file's notes), warp (the Warp), the notes warped (see warped()), and the measures
    of measure_line() and measure_notes(); None where there is too little of either side to say anything."""
    known = monophonic(known)
    if len(known) < MIN_NOTES or (~np.isnan(ours.pitch)).sum() * STEP < MIN_SECONDS or ours.seconds > MAX_SECONDS:
        return None
    found = coarse(known, ours)
    if found is None:
        return None
    warp, shift, score = found
    total = sum(n[1] for n in known)
    # the octave: the shift moved up or down by twelves, by what puts most of the file within a semitone of the line
    rough, scale = warped(known, warp), warp.line()[0]
    best = None
    for k in (-2, -1, 0, 1):
        t = shift + 12 * k
        m = measure_line(rough, ours, t, total * scale)
        if best is None or m["agree_known"] > best[0] + 1e-9 or (abs(m["agree_known"] - best[0]) <= 1e-9 and abs(t) < abs(best[1])):
            best = (m["agree_known"], t)
    transpose = best[1]
    warp = refine(known, ours, warp, transpose)
    known_w = warped(known, warp)
    scale = warp.line()[0]
    out = {"transpose": transpose, "warp": warp, "known": known_w, "score": score,
           "aligned": sum(k[1] for k, w in zip(known, known_w) if w[4]) / max(total, 1e-9)}
    out.update(measure_line(known_w, ours, transpose, total * scale))
    out.update(measure_notes(known_w, ours, transpose))
    out["in_words"] = in_words(known_w, ours.windows)
    return out


def refine(known: list[tuple], ours: Ours, warp: Warp, transpose: int) -> Warp:
    """The warp again at STEP, in a band of BAND s round the coarse one, on exact pitch."""
    r0, r1 = int(warp.rows[0]), int(warp.rows[-1])
    frames = len(ours.pitch)
    kline = line_of(known, int(np.ceil(max(n[0] + n[1] for n in known) / STEP)) + 2, transpose)
    rows = np.arange(r0, min(r1 + 1, len(kline)))
    half = int(round(BAND / STEP))
    centre = np.interp(rows, warp.rows, warp.cols)
    base = np.round(centre).astype(np.int64) - half
    cols = base[:, None] + np.arange(2 * half + 1)[None, :]
    inside = (cols >= 0) & (cols < frames)
    o = np.where(inside, ours.pitch[np.clip(cols, 0, frames - 1)], np.nan)
    k = kline[rows][:, None]
    d = np.abs(o - k)
    both = ~np.isnan(k) & ~np.isnan(o)
    s = np.where(both, np.select([d <= 0.5, d <= 1.0, d <= 1.5], [1.0, 0.5, 0.0], -0.6), 0.0)
    s = np.where(~np.isnan(k) & np.isnan(o), -MISS, s)
    s = np.where(np.isnan(k) & ~np.isnan(o), -EXTRA, s)
    s = np.where(inside, s, -50.0).astype(np.float32)
    r, c, _ = dp(s, base, FINE_STEPS, start_anywhere=False, end_anywhere=False)
    return warp_of(rows[r], c, SMOOTH)
