"""melody.json version 2 (music/melody.py): the song's tuning, the cleaned notes, the contour's confidence and the
reliability. Made-up tones and pitch lines only; nothing here needs the model."""

import functools
import json
import os
import time
from pathlib import Path

import numpy as np
import pytest

from karaokifex.music import melody, tab
from karaokifex.music.__main__ import newer


@functools.lru_cache(maxsize=8)
def tones(detune_cents: float, seconds: float = 12.0, sr: int = 16000, seed: int = 3) -> np.ndarray:
    """A mix of sustained notes of C major from C3 up to C5, each with its first four harmonics, all `detune_cents`
    off the A440 grid."""
    rng = np.random.default_rng(seed)
    scale = [0, 2, 4, 5, 7, 9, 11]
    t = np.arange(int(seconds * sr)) / sr
    y = np.zeros_like(t)
    for start in np.arange(0.0, seconds - 0.5, 0.25):
        notes = [48 + 12 * rng.integers(0, 2) + scale[rng.integers(0, 7)] for _ in range(3)]
        seg = (t >= start) & (t < start + 0.6)
        for n in notes:
            f0 = 440.0 * 2 ** ((n - 69 + detune_cents / 100.0) / 12)
            for h, amp in enumerate((1.0, 0.5, 0.3, 0.2), 1):
                y[seg] += amp * np.sin(2 * np.pi * h * f0 * t[seg])
    return (0.1 * y / np.abs(y).max()).astype(np.float32)


@pytest.mark.parametrize("cents", [30.0, -20.0, 0.0, 12.0])
def test_the_tuning_of_a_backing_is_where_its_notes_sit_off_a440(cents):
    got, conf = melody.tuning_of(tones(cents), 16000)
    assert got == pytest.approx(cents, abs=2.0)
    assert conf > 0.5


def test_the_spectral_peaks_of_a_tone_are_at_its_frequency():
    t = np.arange(16000) / 16000
    freqs, mags = melody.spectral_peaks((0.3 * np.sin(2 * np.pi * 523.0 * t)).astype(np.float32), 16000)
    assert len(freqs) > 20
    assert np.median(freqs[mags >= np.median(mags)]) == pytest.approx(523.0, abs=0.5)      # a third of a cent per Hz and a bit
    assert melody.spectral_peaks(np.zeros(8000, dtype=np.float32), 16000)[0].size == 0


def test_a_backing_of_noise_has_no_tuning_to_speak_of():
    noise = np.random.default_rng(1).normal(0, 0.1, 16000 * 20).astype(np.float32)
    assert melody.tuning_of(noise, 16000)[1] < melody.BACKING_SURE


def test_the_tuning_is_the_backings_where_it_is_sure_then_the_voices_then_nothing():
    assert melody.choose_tuning((25.0, 0.8), (-5.0, 0.9)) == (25.0, 0.8, "backing")
    assert melody.choose_tuning((25.0, 0.1), (-5.0, 0.9)) == (-5.0, 0.9, "lead")
    assert melody.choose_tuning((25.0, 0.1), (-5.0, 0.1)) == (0.0, 0.0, None)
    assert melody.choose_tuning(None, None) == (0.0, 0.0, None)


def test_the_voices_own_tuning_is_the_circular_mean_of_its_fractions():
    rng = np.random.default_rng(5)
    pitch = 60 + rng.integers(-5, 6, 2000) + 0.35 + rng.normal(0, 0.05, 2000)        # 35 cents sharp
    got, conf = melody.lead_tuning(pitch, np.ones(2000))
    assert got == pytest.approx(35, abs=2) and conf > 0.8
    got, conf = melody.lead_tuning(rng.uniform(55, 70, 2000), np.ones(2000))             # no grid at all
    assert conf < 0.1
    assert melody.lead_tuning(pitch, np.full(2000, 0.1)) == (0.0, 0.0)                    # not sure of a frame
    assert melody.lead_tuning(pitch[:50], np.ones(50)) == (0.0, 0.0)                      # too little to say


def line(*parts, tuning=0.0):
    """A pitch line from (MIDI, seconds) parts (None for none), a frame each 10 ms; a loud level and sure frames."""
    midi = np.concatenate([np.full(int(round(s / melody.FRAME)), np.nan if m is None else m + tuning / 100.0, dtype=np.float32)
                           for m, s in parts])
    return midi, np.full(len(midi), -20.0, dtype=np.float32), np.ones(len(midi), dtype=np.float32)


def test_notes_are_rounded_against_the_songs_own_tuning():
    midi, level, conf = line((60.3, 0.5), (None, 0.2), (62.3, 0.5))
    on_a440 = melody.notes_from(midi, level, conf)
    assert [n[2:4] for n in on_a440] == [[60, 30], [62, 30]]
    on_song = melody.notes_from(midi, level, conf, tuning=30.0)
    assert [n[2:4] for n in on_song] == [[60, 0], [62, 0]]
    midi, level, conf = line((60.45, 0.5), tuning=0)
    assert melody.notes_from(midi, level, conf, tuning=20.0)[0][2:4] == [60, 25]          # 60.45 less 20 cents is 60.25


def test_a_notes_conf_is_the_mean_of_its_frames():
    midi, level, conf = line((60, 0.5))
    conf[:25] = 0.0                                              # a half of its frames outside the sung words
    assert melody.notes_from(midi, level, conf)[0][5] == pytest.approx(0.5, abs=0.02)
    assert melody.notes_from(midi, level)[0][5] == 1.0           # no word on it: sure


def note(start, length, pitch, level=0.5, conf=0.9):
    n = int(round(pitch))
    return [start, length, n, int(round((pitch - n) * 100)), level, conf]


def test_a_fragment_is_folded_into_the_longer_neighbour_it_sits_against():
    notes = [note(0.0, 0.4, 60), note(0.4, 0.09, 60.5), note(0.5, 0.4, 64)]
    got = melody.merge_fragments(notes)
    assert len(got) == 2
    assert got[0][0] == 0.0 and got[0][1] == pytest.approx(0.49, abs=0.001)
    assert melody._pitch(got[0]) == pytest.approx((0.4 * 60 + 0.09 * 60.5) / 0.49, abs=0.01)    # weighed by their lengths
    assert got[0][4] == 0.5 and got[1] == notes[2]


def test_a_fragment_takes_the_nearer_neighbour_in_pitch():
    notes = [note(0.0, 0.4, 60.0), note(0.4, 0.09, 60.7), note(0.5, 0.4, 61.0)]
    got = melody.merge_fragments(notes)
    assert len(got) == 2 and got[0] == notes[0]
    assert got[1][0] == 0.4 and got[1][1] == pytest.approx(0.5, abs=0.001)
    assert melody._pitch(got[1]) == pytest.approx((0.09 * 60.7 + 0.4 * 61.0) / 0.49, abs=0.01)


@pytest.mark.parametrize("notes", [
    [note(0.0, 0.4, 60), note(0.4, 0.09, 62.5), note(0.5, 0.4, 64)],        # a tone and a half from either
    [note(0.0, 0.4, 60), note(0.5, 0.09, 60), note(0.7, 0.4, 60)],          # 0.1 s from the one and 0.11 from the other
    [note(0.0, 0.07, 60), note(0.07, 0.07, 60.3)],                          # neither neighbour is longer
    [note(0.0, 0.4, 60), note(0.4, 0.1, 60)],                               # 0.1 s is not a fragment
])
def test_a_fragment_that_fits_no_neighbour_stays(notes):
    assert melody.merge_fragments(notes) == notes


def test_fragments_fold_into_one_another_down_a_run():
    notes = [note(0.0, 0.5, 60), note(0.5, 0.06, 60.2), note(0.56, 0.06, 60.3), note(0.62, 0.06, 60.1)]
    got = melody.merge_fragments(notes)
    assert len(got) == 1 and got[0][0] == 0.0 and got[0][1] == pytest.approx(0.68, abs=0.001)


def test_a_doubtful_note_takes_the_side_of_the_semitone_that_is_in_the_key():
    c_major = melody.key_scale("C major")
    got = melody.snap_to_key([note(0, 1, 61.4)], c_major)[0]           # C# 40 cents up, 10 from the edge to D: D is in the key, C# is not
    assert got[2] == 62 and got[3] == -60
    got = melody.snap_to_key([note(0, 1, 66.45)], c_major)[0]          # F# and G
    assert got[2] == 67 and got[3] == -55
    got = melody.snap_to_key([note(0, 1, 70.45)], c_major)[0]          # A# and B
    assert got[2] == 71 and got[3] == -55
    got = melody.snap_to_key([note(0, 1, 60.4)], c_major)[0]           # C and C#: the nearer, C, is the key's anyway
    assert got[2] == 60 and got[3] == 40
    got = melody.snap_to_key([note(0, 1, 61.6)], c_major)[0]           # C# and D, nearer D
    assert got[2] == 62 and got[3] == -40
    assert melody.snap_to_key([note(0, 1, 60.55)], melody.key_scale("F# major"))[0][2] == 61      # C is not in F# major, C# is
    assert melody.snap_to_key([note(0, 1, 60.4)], melody.key_scale("A minor"))[0][2] == 60
    snapped = melody.snap_to_key([note(2.0, 0.5, 61.4, level=0.7, conf=0.6)], c_major)[0]
    assert snapped[:2] == [2.0, 0.5] and snapped[4:] == [0.7, 0.6]     # only the note and its cents change


def test_a_note_with_both_sides_or_neither_in_the_key_stays_on_the_nearer():
    c_major = melody.key_scale("C major")
    assert melody.snap_to_key([note(0, 1, 64.55)], c_major)[0][2:4] == [65, -45]      # E and F: both are in C major
    assert melody.snap_to_key([note(0, 1, 59.55)], c_major)[0][2:4] == [60, -45]      # B and C: both
    pentatonic = frozenset({0, 2, 4, 7, 9})                                          # F and F# are both out of it
    assert melody.snap_to_key([note(0, 1, 65.45)], pentatonic)[0][2:4] == [65, 45]
    assert melody.snap_to_key([note(0, 1, 65.55)], pentatonic)[0][2:4] == [66, -45]


def test_a_note_in_no_doubt_is_left_alone_and_no_key_snaps_nothing():
    c_major = melody.key_scale("C major")
    n = note(0, 1, 61.25)                                          # a C#, 25 cents up: not within 20 of an edge
    assert melody.snap_to_key([n], c_major) == [n]
    n = note(0, 1, 61.45)
    assert melody.snap_to_key([n], None) == [n] and melody.snap_to_key([n], frozenset()) == [n]


def test_keys_as_chords_json_writes_them():
    assert melody.key_scale("A major") == frozenset({9, 11, 1, 2, 4, 6, 8})
    assert melody.key_scale("C minor") == frozenset({0, 2, 3, 5, 7, 8, 10})
    assert melody.key_scale("Eb major") == melody.key_scale("D# major")
    assert melody.key_scale("F#") == melody.key_scale("F# major")
    assert melody.key_scale(None) is None and melody.key_scale("") is None and melody.key_scale("unknown") is None


def test_the_sung_windows_are_the_words_widened_and_joined():
    words = [(10.0, 10.5), (10.6, 11.0), (20.0, 20.2), (0.1, 0.3)]
    np.testing.assert_allclose(melody.sung_windows(words), [(0.0, 0.55), (9.75, 11.25), (19.75, 20.45)])
    mask = melody.sung_mask(1500, 0.02, melody.sung_windows(words))
    assert mask[int(10.0 / 0.02)] and mask[int(9.8 / 0.02)] and not mask[int(9.7 / 0.02)]
    assert mask[int(11.2 / 0.02)] and not mask[int(11.3 / 0.02)] and not mask[int(15 / 0.02)]
    assert melody.sung_mask(10, 0.02, None).all()
    assert melody.words_of({"lines": [[{"start": 1, "end": 2}, {"start": 3}, {"start": 5, "end": 4}], [{"start": 6, "end": 7}]]}) == [(1.0, 2.0), (6.0, 7.0)]
    assert melody.words_of({}) == [] and melody.words_of({"lines": None}) == []


def test_the_contours_conf_is_the_larger_of_a_steps_frames_and_nothing_where_unsung_or_unvoiced():
    midi = np.array([60, 60, 61, np.nan, 62, 62, 63], dtype=np.float32)
    conf = np.array([0.2, 0.9, 0.5, 0.0, 0.4, 0.3, 0.7], dtype=np.float32)
    got = melody.contour(midi, conf)
    assert got["step"] == 0.02
    assert got["pitch"] == [600, 610, 620, 630] and got["conf"] == [90, 50, 40, 70]
    sung = np.array([True, False, True, True])
    assert melody.contour(midi, conf, sung)["conf"] == [90, 0, 40, 70]
    assert melody.contour(midi, np.array([0.2, 0.9, 0.5, 0.9, 0.4, 0.3, 0.7], dtype=np.float32))["conf"][1] == 90   # a pitch frame
    midi[2] = np.nan                                                                       # no pitch: none sure
    assert melody.contour(midi, conf)["pitch"][1] is None and melody.contour(midi, conf)["conf"][1] == 0
    assert melody.contour(midi)["conf"] == [0, 0, 0, 0]


def made_up_song(tuning=30.0, stray=True):
    """A pitch line with a few phrases (C major) and the words they are sung to, and a blip well outside them."""
    parts = [(None, 1.0), (60, 0.5), (62, 0.5), (64, 0.5), (None, 1.0), (65, 0.4), (67, 0.6), (None, 1.0)]
    if stray:
        parts += [(61, 0.5), (None, 0.5)]
    midi, level, conf = line(*parts, tuning=tuning)
    words = [(1.0, 1.5), (1.5, 2.0), (2.0, 2.5), (3.5, 3.9), (3.9, 4.5)]
    return midi, level, conf, words


def test_the_whole_file_in_version_2():
    midi, level, conf, words = made_up_song()
    data = melody.build(midi, conf * 0.8, level, origin="voice", backing=(30.0, 0.9), key="C major", words=words)
    assert list(data) == ["version", "model", "source", "tuning", "tuning_conf", "tuning_from", "key", "range", "notes",
                          "contour", "reliability"]
    assert data["version"] == 2 and data["model"] == "rmvpe" and data["source"] == "voice"
    assert data["tuning"] == 30.0 and data["tuning_conf"] == 0.9 and data["tuning_from"] == "backing" and data["key"] == "C major"
    # the notes on the song's grid: the pitches were 30 cents up, so they are C D E F G, in tune
    assert [n[2] for n in data["notes"]][:5] == [60, 62, 64, 65, 67]
    assert all(len(n) == 6 and abs(n[3]) <= 5 for n in data["notes"][:5])
    assert data["notes"][0][5] == pytest.approx(0.8, abs=0.02)                  # sure, and sung
    assert data["notes"][-1][2] == 61 and data["notes"][-1][5] == 0.0           # the blip: outside the words, conf 0
    assert data["range"] == [60, 67]
    line_ = data["contour"]
    assert line_["step"] == 0.02 and len(line_["conf"]) == len(line_["pitch"])
    assert all(c == 0 for c, p in zip(line_["conf"], line_["pitch"]) if p is None)
    assert max(line_["conf"]) == 80 and line_["conf"][int(1.2 / 0.02)] == 80
    assert line_["pitch"][int(5.7 / 0.02)] is not None and line_["conf"][int(5.7 / 0.02)] == 0       # the blip has a pitch, and no conf
    assert line_["pitch"][int(1.2 / 0.02)] == 603                               # the contour keeps the raw pitch, on A440's scale
    r = data["reliability"]
    assert set(r) == {"score", "covered", "stray", "in_key", "conf", "words_no_note"}
    assert r["in_key"] == pytest.approx(0.9, abs=0.1) and r["stray"] == pytest.approx(0.2, abs=0.1)
    assert 0 <= r["score"] <= 1 and r["words_no_note"] == 0.0 and r["covered"] == 1.0
    assert json.loads(json.dumps(data)) == data                                  # plain JSON all through


def test_a_song_with_no_chords_no_timings_and_no_tuning_is_still_a_file():
    midi, level, conf = line(*[(55 + (7 * k) % 12 + (0.37 * k) % 1.0, 0.4) for k in range(40)])       # fractions all over: no grid to speak of
    data = melody.build(midi, conf, level, origin="mix")
    assert data["key"] is None and data["tuning"] == 0.0 and data["tuning_from"] is None and data["source"] == "mix"
    r = data["reliability"]
    assert r["in_key"] is None and r["covered"] is None and r["stray"] is None and r["words_no_note"] is None
    assert r["conf"] == 1.0 and r["score"] is not None
    assert all(c in (0, 100) for c in data["contour"]["conf"]) and len(data["notes"]) == 40
    empty = melody.build(np.full(300, np.nan, dtype=np.float32), np.zeros(300, dtype=np.float32), np.full(300, -20.0, dtype=np.float32), origin="mix")
    assert empty["notes"] == [] and empty["range"] is None and empty["reliability"]["score"] is None


def test_the_voices_own_tuning_stands_in_where_the_backing_says_too_little():
    parts = [(None, 0.5)] + [(m, 0.4) for m in (60, 62, 64, 65, 67, 69, 71, 72) * 6]
    midi, level, conf = line(*parts, tuning=-22.0)
    data = melody.build(midi, conf, level, origin="voice", backing=(10.0, 0.05))
    assert data["tuning_from"] == "lead" and data["tuning"] == pytest.approx(-22.0, abs=1.0)
    assert all(n[3] in (0, 1, -1) for n in data["notes"])


def reliability_of(words, pitch_parts, notes, key="C major"):
    midi, _, conf = line(*pitch_parts)
    return melody.reliability(notes, midi[::2].astype(float), conf[::2], words, melody.key_scale(key))


def test_reliability_gets_worse_with_stray_pitch_gaps_in_the_words_and_notes_out_of_the_key():
    words = [(1.0, 2.0), (3.0, 4.0)]
    notes = [note(1.0, 1.0, 60), note(3.0, 1.0, 62)]
    good = reliability_of(words, [(None, 1.0), (60, 1.0), (None, 1.0), (62, 1.0)], notes)
    assert good["covered"] == 1.0 and good["stray"] == 0.0 and good["in_key"] == 1.0 and good["words_no_note"] == 0.0 and good["score"] > 0.95
    stray = reliability_of(words, [(None, 1.0), (60, 1.0), (None, 1.0), (62, 1.0), (None, 1.0), (64, 2.0)], notes)
    assert stray["stray"] > 0.3 and stray["score"] < good["score"]
    holes = reliability_of(words, [(None, 1.0), (60, 0.4), (None, 0.6), (None, 1.0), (62, 1.0)], notes)
    assert holes["covered"] == pytest.approx(0.7, abs=0.05) and holes["score"] < good["score"]
    off = reliability_of(words, [(None, 1.0), (60, 1.0), (None, 1.0), (62, 1.0)], [note(1.0, 1.0, 61), note(3.0, 1.0, 66)])
    assert off["in_key"] == 0.0 and off["score"] < good["score"]
    missing = reliability_of(words, [(None, 1.0), (60, 1.0), (None, 1.0), (62, 1.0)], [note(1.0, 1.0, 60)])
    assert missing["words_no_note"] == 0.5 and missing["score"] < good["score"]
    no_key = reliability_of(words, [(None, 1.0), (60, 1.0), (None, 1.0), (62, 1.0)], notes, key=None)
    assert no_key["in_key"] is None and no_key["score"] == pytest.approx(good["score"], abs=0.02)


def test_the_score_follows_each_part_and_can_be_made_again_from_them():
    best = {"covered": 1.0, "stray": 0.0, "in_key": 1.0, "conf": 1.0, "words_no_note": 0.0}
    assert melody.score_of(best) == 1.0
    assert melody.score_of({k: None for k in best}) is None
    for part, worse in (("covered", 0.4), ("stray", 0.5), ("in_key", 0.5), ("conf", 0.2), ("words_no_note", 0.5)):
        assert melody.score_of({**best, part: worse}) < melody.score_of({**best, part: best[part]})
    # a part that is not there is left out, the others weighing the more
    assert melody.score_of({**best, "in_key": None}) == 1.0
    data = {"reliability": {**best, "score": 0.12}}
    assert melody.rescore(data)["reliability"]["score"] == 1.0 and data["reliability"]["score"] == 0.12


def test_tab_is_made_from_a_version_1_or_a_version_2_melody():
    v1 = {"version": 1, "notes": [[0.0, 0.5, 64, 3, 0.5], [0.5, 0.5, 66, -2, 0.6]], "contour": {"step": 0.02, "pitch": [640]}}
    v2 = {"version": 2, "notes": [[0.0, 0.5, 64, 3, 0.5, 0.9], [0.5, 0.5, 66, -2, 0.6, 0.8]], "tuning": 12.0}
    one, two = tab.make(None, v1), tab.make({"chords": [[0.0, 2.0, "Am"]]}, v2)
    assert [n[:2] for n in one["notes"]] == [[0.0, 0.5], [0.5, 0.5]] and one["notes"] == two["notes"]
    assert two["chords"][0][2] == "Am"


def test_a_file_is_made_again_when_what_it_is_made_from_is_newer(tmp_path: Path):
    target, source = tmp_path / "melody.json", tmp_path / "source.mkv"
    assert newer(target, source)                                      # not there yet
    target.write_text("{}")
    source.write_text("x")
    old = time.time() - 100
    os.utime(source, (old, old))
    assert not newer(target, source, tmp_path / "missing.wav")        # newer than what it is made from, and a missing source is no reason
    os.utime(target, (old - 100, old - 100))
    assert newer(target, source)
