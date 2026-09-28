import numpy as np
import pytest

from karaokifex.music import chords, melody, tab


def line(*parts):
    """A pitch line from (MIDI or None, seconds) parts, a frame each 10 ms, and a flat loud level."""
    midi = np.concatenate([np.full(int(round(s / melody.FRAME)), np.nan if m is None else m, dtype=np.float32)
                           for m, s in parts])
    return midi, np.full(len(midi), -20.0, dtype=np.float32)


def test_notes_split_on_a_change_of_pitch_and_bridge_short_gaps():
    midi, level = line((60, 0.5), (None, 0.03), (60, 0.3), (64, 0.4), (None, 0.3), (67, 0.2))
    notes = melody.notes_from(midi, level)
    assert [n[2] for n in notes] == [60, 64, 67]
    assert notes[0][1] == pytest.approx(0.83, abs=0.03)       # the 30 ms gap bridged into one note
    assert notes[1][0] == pytest.approx(0.83, abs=0.03)


def test_a_dip_in_the_voice_on_one_pitch_is_two_syllables():
    midi, level = line((62, 1.0))
    level[45:52] = -40.0                                        # a 20 dB dip, 70 ms
    assert len(melody.notes_from(midi, level)) == 2


def test_quiet_frames_and_short_blips_are_not_notes():
    midi, level = line((60, 0.5), (72, 0.04), (60, 0.5))
    level[:] = -20.0
    assert [n[2] for n in melody.notes_from(midi, level)] == [60]
    midi, level = line((60, 0.5))
    level[:] = -90.0
    level[0] = -20.0
    assert melody.notes_from(midi, level) == []


def test_viterbi_holds_a_chord_through_a_flicker():
    odds = np.full((60, 170), 1e-4)
    odds[:, 9 * 14 + 1] = 0.9                                   # A for 60 frames ...
    odds[30:32, 9 * 14 + 1], odds[30:32, 4 * 14 + 1] = 0.2, 0.7  # ... E for two frames in the middle
    assert set(chords.viterbi(odds).tolist()) == {9 * 14 + 1}


def test_key_and_spelling():
    f, bb, c, dm = 5 * 14 + 1, 10 * 14 + 1, 0 * 14 + 1, 2 * 14
    key = chords.key_of([(f, 8), (bb, 4), (c, 4), (dm, 2)])
    assert key == (5, "major")
    assert chords.name(bb, key) == "Bb" and chords.name(dm, key) == "Dm"
    assert chords.name(10 * 14 + 6, (11, "major")) == "A#m7"
    assert chords.name(169, key) == "N"


@pytest.mark.parametrize("kind", chords.KINDS)
@pytest.mark.parametrize("root", range(12))
def test_every_chord_has_a_shape_with_only_its_notes(root, kind):
    name = chords.ROOTS[root] + chords.WRITTEN[kind]
    frets = tab.shape(name)
    if frets is None:
        pytest.skip(f"no shape for {name}")
    sounded = {(open_ + f) % 12 for open_, f in zip(tab.TUNING, frets) if f != tab.X}
    tones = {(root + t) % 12 for t in chords.TONES[kind]}
    assert sounded <= tones, name
    assert root in sounded, name
    assert max(frets) <= 16


def test_open_chords_are_preferred_and_flats_parse():
    assert tab.written(tab.shape("Am")) == "x02210"
    assert tab.written(tab.shape("Bb")) == "x13331"
    assert tab.shape("N") is None


def test_the_melody_moves_as_little_as_it_can():
    notes = [64, 66, 67, 69, 71, 72]                            # E F# G A B C, up the high strings
    places = tab.finger(notes)
    assert all(p is not None for p in places)
    frets = [p[1] for p in places]
    # from the open high e up that string (0 2 3 5 7 8), the least sliding there is for it
    assert sum(abs(a - b) for a, b in zip(frets, frets[1:])) <= 8
    assert {p[0] for p in places} <= {1, 2}
    assert tab.octave_shift([76, 79, 84, 88, 91]) in (0, -1)    # a high voice comes down onto the neck if needed


@pytest.mark.parametrize("name,expected", [("C", "x32010"), ("G", "210003"), ("D", "xx0132"), ("Am", "x02310"),
                                           ("E", "023100"), ("F", "134211")])
def test_the_common_chords_take_their_usual_fingers(name, expected):
    assert tab.written(tab.fingers(tab.shape(name))) == expected


def test_the_hand_moves_only_when_a_note_is_out_of_reach():
    # frets 5 6 7 8 in one place, then 10 moves the hand up, then an open string
    places = [(1, 5), (1, 6), (1, 7), (1, 8), (1, 10), (2, 0)]
    assert tab.hand(places) == [1, 2, 3, 4, 4, 0]
