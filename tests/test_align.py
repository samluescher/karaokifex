"""Laying known notes over the melody we detected (music/align.py): the parts, on made-up notes."""

import numpy as np
import pytest

from karaokifex.music import align


def ours_of(notes, seconds=None, noise=0.0, seed=0):
    """Our side for a made-up voice that sings (start, length, MIDI) `notes`: the pitch line a frame each 20 ms and the notes."""
    rng = np.random.default_rng(seed)
    end = seconds or max(s + l for s, l, _ in notes) + 2.0
    pitch = np.full(int(end / align.STEP), np.nan)
    for s, l, m in notes:
        pitch[int(round(s / align.STEP)):int(round((s + l) / align.STEP))] = m + rng.normal(0, noise, int(round((s + l) / align.STEP)) - int(round(s / align.STEP)))
    return align.Ours(pitch, [[s, l, m, 0, 0.5, 0.9] for s, l, m in notes])


def melody(seed=5, count=60):
    rng = np.random.default_rng(seed)
    t, out, pitch = 2.0, [], 62
    for k in range(count):
        pitch = int(np.clip(pitch + rng.choice([-3, -2, -1, 0, 1, 2, 3]), 55, 74))
        length = float(rng.choice([0.2, 0.3, 0.5, 0.8]))
        out.append((round(t, 3), length, pitch))
        t += length + (1.5 if k % 8 == 7 else 0.05)
    return out


def test_notes_are_cut_where_the_next_begins_and_one_at_a_time():
    got = align.monophonic([(1.0, 2.0, 60, "a"), (1.5, 1.0, 64, "b"), (1.5, 1.0, 67, "c"), (4.0, 0.0, 70, "d"), (0.0, 0.5, 50, "e")])
    assert [(n[0], round(n[1], 3), n[2]) for n in got] == [(0.0, 0.5, 50), (1.0, 0.5, 60), (1.5, 1.0, 67)]
    assert align.monophonic([]) == []


def test_the_pitch_line_of_notes_and_their_pitch_classes():
    line = align.line_of([(0.0, 0.1, 60), (0.2, 0.06, 62)], 20, shift=2)
    assert np.isnan(line[5]) and line[0] == 62 and line[4] == 62 and line[10] == 64 and line[12] == 64 and np.isnan(line[13])
    chroma = align.chroma(np.array([60, 60, 60, np.nan, 62, 62, 71, np.nan, np.nan, np.nan]), 5)
    assert chroma.shape == (2, 12)
    assert chroma[0, 0] == pytest.approx(0.6) and chroma[0, 2] == pytest.approx(0.2) and chroma[1, 2] == pytest.approx(0.2) and chroma[1, 11] == pytest.approx(0.2)
    assert chroma.sum() == pytest.approx(1.2)


def test_the_best_pitch_class_shifts_are_found_by_how_alike_the_histograms_are():
    song = melody()
    a = align.chroma(align.line_of([(s, l, m) for s, l, m in song], 3000), 5)
    b = align.chroma(align.line_of([(s, l, m + 3) for s, l, m in song], 3000), 5)
    assert align.shifts(a, b)[0] == 3


def test_a_path_is_found_through_scores_on_a_diagonal_from_anywhere_to_anywhere():
    scores = np.full((40, 60), -0.2, dtype=np.float32)
    for r in range(10, 30):
        scores[r, r + 14:r + 17] = 1.0                   # the file's frames 10 to 29 on the recording's 25 to 44, give or take one
    rows, cols, total = align.dp(scores, np.zeros(40, dtype=np.int64), align.STEPS)
    assert 6 <= rows[0] <= 11 and rows[-1] >= 28 and all(14 <= c - r <= 16 for r, c in zip(rows[1:], cols[1:]))
    assert total > 20


def test_the_steps_allow_a_slope_from_a_quarter_less_to_a_third_more_and_no_more():
    rows = np.arange(0, 60)
    scores = np.full((60, 100), -0.3, dtype=np.float32)
    on = np.round(rows * 4 / 3).astype(int)
    scores[rows, on] = 1.0                               # slope 4/3: a recording frame and a third for each of the file's
    r, c, _ = align.dp(scores, np.zeros(60, dtype=np.int64), align.STEPS)
    assert (c[-1] - c[0]) / (r[-1] - r[0]) == pytest.approx(4 / 3, abs=0.05) and r[-1] - r[0] > 50
    scores = np.full((60, 130), -0.3, dtype=np.float32)
    scores[rows, rows * 2] = 1.0                         # slope 2: out of what the steps do, so it is not followed to the end
    r, c, _ = align.dp(scores, np.zeros(60, dtype=np.int64), align.STEPS)
    assert (c[-1] - c[0]) / max(r[-1] - r[0], 1) <= 1.34


def test_a_banded_path_may_start_and_end_on_the_first_and_last_row_only():
    scores = np.full((30, 9), -0.1, dtype=np.float32)
    base = np.arange(30) - 4                              # the band follows the diagonal
    scores[:, 4] = 1.0
    rows, cols, total = align.dp(scores, base, align.FINE_STEPS, start_anywhere=False, end_anywhere=False)
    assert rows[0] == 0 and rows[-1] == 29 and (cols == rows).all() and total == pytest.approx(30.0)


def test_a_warp_is_a_line_through_the_path_and_has_no_time_outside_it():
    rows = np.arange(100, 200)
    warp = align.warp_of(rows, rows * 1.1 + 40)
    assert warp.span == (pytest.approx(2.0), pytest.approx(4.0))
    assert warp(2.5) == pytest.approx((125 * 1.1 + 40) * 0.02, abs=1e-6)
    assert np.isnan(warp(1.0)) and np.isnan(warp(4.5))
    scale, offset, dev = warp.line()
    assert scale == pytest.approx(1.1, abs=1e-6) and offset == pytest.approx(40 * 0.02, abs=1e-3) and dev < 1e-6


def test_the_share_of_a_file_in_the_sung_windows():
    known = [(1.0, 1.0, 60, None, True), (3.0, 2.0, 62, None, True), (9.0, 1.0, 64, None, True), (float("nan"), 5.0, 65, None, False)]
    assert align.in_words(known, [(0.5, 1.5), (3.5, 4.0), (10.0, 12.0)]) == pytest.approx((0.5 + 0.5 + 0.0) / 4.0)
    assert align.in_words(known, None) is None and align.in_words(known, []) is None
    assert align.in_words([(float("nan"), 1.0, 60, None, False)], [(0.0, 9.0)]) is None


def test_too_little_of_either_side_is_no_alignment():
    song = melody()
    assert align.align(song[:10], ours_of(song)) is None                                  # ten notes
    assert align.align([(s, l, m, None) for s, l, m in song], align.Ours(np.full(500, np.nan), [])) is None       # nothing sung


def test_a_file_lying_on_the_voice_is_found_and_measured():
    song = melody()
    ours = ours_of(song, noise=0.1)
    file = [(s + 5.0, l, m - 12, None) for s, l, m in song]                         # an octave lower and five seconds late
    got = align.align(file, ours)
    assert got["transpose"] == 12
    scale, offset, _ = got["warp"].line()
    assert scale == pytest.approx(1.0, abs=0.01) and offset == pytest.approx(-5.0, abs=0.1)
    assert got["agree_known"] > 0.9 and got["covered_ours"] > 0.9 and got["aligned"] > 0.95
    assert got["matched"] == got["notes_known"] == len(song) and got["missing"] == 0 and got["extra"] <= 2
    assert got["pitch_exact"] > 0.95 and got["onset_median_ms"] < 60
    assert all(n[4] for n in got["known"])


def test_what_is_sung_differently_is_counted_as_wrong_and_what_is_not_there_as_missing_or_extra():
    song = melody()
    ours = ours_of(song)
    ours.notes[3][2] += 2                                      # we have a note two semitones off
    del ours.notes[10]                                         # and miss one
    ours.notes.append([ours.notes[-1][0] + 0.5, 0.3, 60, 0, 0.5, 0.9])      # and have one the file does not
    got = align.align([(s, l, m, None) for s, l, m in song], ours)
    assert got["missing"] >= 1 and got["extra"] >= 1
    assert got["pitch_exact"] < 1.0 and got["pitch_within_1"] <= got["pitch_exact"] + 0.1
    assert any(w[1] - w[2] == 2 for w in got["wrong"])
