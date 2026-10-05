"""Known notes (notes.py, music/align.py): reading UltraStar charts and MIDI files, finding the sources, aligning a file's
notes to the melody we detected, accepting or not, using them, and the report. Made-up tunes and files only."""

import json
import struct
from pathlib import Path

import numpy as np
import pytest
from click.testing import CliRunner

from karaokifex import notes
from karaokifex.music import align


# ---------------------------------------------------------------- a made-up tune, and the voice that sings it
def tune(seed: int = 7, phrases: int = 14) -> list[tuple[float, float, int]]:
    """(start, length, MIDI note) of a made-up melody in C major: phrases of a walk up and down the scale, a rest between."""
    rng = np.random.default_rng(seed)
    scale = [0, 2, 4, 5, 7, 9, 11]
    t, degree, out = 3.0, 4, []
    for _ in range(phrases):
        for _ in range(int(rng.integers(6, 11))):
            degree = int(np.clip(degree + int(rng.choice([-3, -2, -1, -1, 0, 1, 1, 2, 3])), 0, 13))
            length = float(rng.choice([0.18, 0.25, 0.35, 0.5, 0.75, 1.0]))
            out.append((round(t, 3), length, 55 + 12 * (degree // 7) + scale[degree % 7]))
            t += length + 0.03
        t += float(rng.uniform(0.8, 2.2))
    return out


def sing(song: list[tuple[float, float, int]], seed: int = 3, tuning: float = 0.0) -> dict:
    """A melody.json (version 2) for a voice singing `song`: a pitch line every 20 ms with vibrato, a scoop up into each
    note and some error, a gap between the notes, and the notes a detector would find."""
    rng = np.random.default_rng(seed)
    n = int((song[-1][0] + song[-1][1] + 2.0) / 0.02)
    pitch = np.full(n, np.nan)
    found = []
    for start, length, midi in song:
        a, b = int(round(start / 0.02)), int(round((start + length - 0.03) / 0.02))
        t = (np.arange(a, b) - a) * 0.02
        pitch[a:b] = midi + 0.15 * np.sin(2 * np.pi * 5.5 * t) * (t > 0.15) - 0.6 * np.exp(-t / 0.04) + tuning / 100 + rng.normal(0, 0.05, b - a)
        found.append([round(start + float(rng.normal(0, 0.01)), 3), round(length - 0.03, 3), midi, 0, 0.6, 0.9])
    return {"version": 2, "model": "rmvpe", "source": "voice", "tuning": tuning, "tuning_conf": 0.8, "tuning_from": "backing",
            "key": "C major", "range": [min(m for _, _, m in song), max(m for _, _, m in song)], "notes": found,
            "contour": {"step": 0.02, "pitch": [None if np.isnan(v) else int(round(v * 10)) for v in pitch],
                        "conf": [0 if np.isnan(v) else 90 for v in pitch]},
            "reliability": {"score": 0.9}}


def folder_of(tmp_path: Path, song, name: str = "Made Up - Tune", **kwargs) -> Path:
    folder = tmp_path / name
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "melody.json").write_text(json.dumps(sing(song, **kwargs)))
    return folder


def perturbed(song, *, shift=0.0, tempo=1.0, transpose=0, delete=0.0, move=0.0, seed=1) -> list[tuple[float, float, int, None]]:
    """The notes a file would have of `song`: its clock `shift` s behind the recording's and running at `tempo` times its
    speed, `transpose` semitones lower, some notes missing, some a semitone off."""
    rng = np.random.default_rng(seed)
    out = []
    for start, length, midi in song:
        if rng.random() < delete:
            continue
        if rng.random() < move:
            midi += int(rng.choice([-1, 1]))
        if (start - shift) * tempo >= 0:
            out.append(((start - shift) * tempo, length * tempo, midi - transpose, None))
    return out


# ---------------------------------------------------------------- files to read
TICKS = 960         # ticks a second at 480 a quarter and 120 beats a minute


def vlq(n: int) -> bytes:
    out = [n & 0x7F]
    n >>= 7
    while n:
        out.append((n & 0x7F) | 0x80)
        n >>= 7
    return bytes(out[::-1])


def smf(tracks: list[list[tuple[int, bytes]]], division: int = 480, fmt: int = 1) -> bytes:
    """A standard MIDI file from tracks of (tick, raw event bytes)."""
    body = b""
    for events in tracks:
        data, last = b"", 0
        for tick, raw in sorted(events, key=lambda e: e[0]):
            data += vlq(tick - last) + raw
            last = tick
        data += vlq(0) + b"\xff\x2f\x00"
        body += b"MTrk" + struct.pack(">I", len(data)) + data
    return b"MThd" + struct.pack(">IHHH", 6, fmt, len(tracks), division) + body


def line_track(song, channel: int = 0, name: str | None = None, program: int | None = None, lyrics: bool = False) -> list[tuple[int, bytes]]:
    events = []
    if name:
        events.append((0, b"\xff\x03" + bytes([len(name)]) + name.encode()))
    if program is not None:
        events.append((0, bytes([0xC0 | channel, program])))
    for start, length, midi, *_ in song:
        a, b = round(start * TICKS), round((start + length) * TICKS)
        if lyrics:
            events.append((a, b"\xff\x05\x02la"))
        events.append((a, bytes([0x90 | channel, midi, 90])))
        events.append((b, bytes([0x80 | channel, midi, 0])))
    return events


def accompaniment(song) -> list[list[tuple[int, bytes]]]:
    """Chords (threes at once), a bass and drums for `song`'s length: nothing a voice sings."""
    end = song[-1][0] + song[-1][1]
    chords, bass, drums = [], [], []
    for k, t in enumerate(np.arange(0.0, end, 2.0)):
        root = [48, 53, 55, 45][k % 4]
        for off in (0, 4, 7):
            chords += [(round(t * TICKS), bytes([0x91, root + off, 70])), (round((t + 1.9) * TICKS), bytes([0x81, root + off, 0]))]
        for b in range(4):
            bass += [(round((t + 0.5 * b) * TICKS), bytes([0x92, root - 12, 80])), (round((t + 0.5 * b + 0.4) * TICKS), bytes([0x82, root - 12, 0]))]
            drums += [(round((t + 0.5 * b) * TICKS), bytes([0x99, 36, 100])), (round((t + 0.5 * b + 0.1) * TICKS), bytes([0x89, 36, 0]))]
    return [chords, bass, drums]


def ultrastar(song, bpm: float = 240.0, gap_ms: int = 2000, relative: bool = False, comma: bool = False) -> str:
    """An UltraStar chart of `song` (MIDI note 60 is pitch 0): a beat is 15/bpm s; one line a phrase."""
    beat = 15.0 / bpm
    rows = [f"#TITLE:Made up", f"#ARTIST:Nobody", f"#BPM:{str(bpm).replace('.', ',') if comma else bpm}", f"#GAP:{gap_ms}"]
    if relative:
        rows.append("#RELATIVE:YES")
    phrase, origin = [], 0
    notes_ = [(round((s - gap_ms / 1000.0) / beat), max(1, round(l / beat)), m - 60) for s, l, m, *_ in song]
    for i, (b, l, p) in enumerate(notes_):
        phrase.append((b, l, p))
        last = i == len(notes_) - 1
        if last or notes_[i + 1][0] - (b + l) > 12:
            for pb, pl, pp in phrase:
                rows.append(f": {pb - origin if relative else pb} {pl} {pp} la")
            if not last:
                nxt = notes_[i + 1][0]
                rows.append(f"- {b + l - origin} {nxt - origin}" if relative else f"- {b + l}")
                origin = nxt if relative else origin
            phrase = []
    rows.append("E")
    return "\n".join(rows) + "\n"


# ---------------------------------------------------------------- UltraStar
def test_an_ultrastar_chart_is_timed_by_its_bpm_and_gap_and_only_its_pitched_notes_are_kept():
    chart = "\n".join(["#TITLE:Made up", "#BPM:120,5", "#GAP:2500", ": 0 4 0 Hel", "* 4 4 2 lo", "F 8 2 0 la", "R 10 3 0 hey",
                       "G 13 2 0 go", "- 16", ": 16 4 -3 you", "E"])
    (line,) = notes.parse_ultrastar(chart)
    beat = 15.0 / 120.5
    assert [n[2] for n in line.notes] == [60, 62, 57]
    assert [n[3] for n in line.notes] == ["Hel", "lo", "you"]
    assert line.notes[0][0] == pytest.approx(2.5) and line.notes[0][1] == pytest.approx(4 * beat)
    assert line.notes[1][0] == pytest.approx(2.5 + 4 * beat)
    assert line.notes[2][0] == pytest.approx(2.5 + 16 * beat)


def test_a_relative_chart_counts_each_lines_beats_from_where_the_line_before_set_it():
    chart = "#BPM:240\n#GAP:0\n#RELATIVE:YES\n: 0 4 0 a\n: 4 4 2 b\n- 8 10\n: 0 4 4 c\n: 4 4 5 d\n- 8 10\nE\n"
    (line,) = notes.parse_ultrastar(chart)
    beat = 15.0 / 240
    assert [round(n[0] / beat) for n in line.notes] == [0, 4, 10, 14]
    assert [n[2] for n in line.notes] == [60, 62, 64, 65]


def test_a_chart_may_change_its_bpm_and_may_be_a_duet_in_any_encoding():
    chart = "#BPM:120\n#GAP:1000\n: 0 4 0 a\nB 8 240\n: 8 4 2 b\n: 12 4 4 c\nE\n"
    (line,) = notes.parse_ultrastar(chart)
    assert line.notes[1][0] == pytest.approx(1.0 + 8 * 15 / 120) and line.notes[1][1] == pytest.approx(4 * 15 / 240)
    assert line.notes[2][0] == pytest.approx(1.0 + 8 * 15 / 120 + 4 * 15 / 240)
    duet = "#BPM:120\n#GAP:0\nP1\n: 0 4 0 a\nP2\n: 2 4 5 b\nE\n"
    lines = notes.parse_ultrastar(duet)
    assert [l.label for l in lines] == ["P1", "P2"] and lines[1].notes[0][2] == 65
    assert notes.parse_ultrastar("#BPM:100\n#GAP:0\n: 0 4 0 caf\xe9\nE\n".encode("cp1252"))[0].notes[0][3] == "caf\xe9"
    assert notes.parse_ultrastar(b"\xef\xbb\xbf#BPM:100\n#GAP:0\n: 0 4 0 na\xc3\xafve\nE\n")[0].notes[0][3] == "na\xefve"
    assert notes.parse_ultrastar("no bpm here\n: 0 4 0 a\n") == [] and notes.parse_ultrastar("#BPM:0\n: 0 4 0 a\n") == []


def test_a_chart_may_separate_its_fields_with_tabs_and_leaves_a_syllable_out():
    (line,) = notes.parse_ultrastar("#BPM:120\r\n#GAP:0\r\n:\t0\t4\t0\tHel\r\n: 4 4 2\r\n*  8  4  4  lo wor\r\nE\r\n")
    assert [(n[2], n[3]) for n in line.notes] == [(60, "Hel"), (62, None), (64, "lo wor")]


def test_an_ultrastar_chart_made_of_a_song_comes_back_as_its_notes():
    song = tune()
    (line,) = notes.parse_ultrastar(ultrastar([(s, l, m) for s, l, m in song], comma=True))
    assert len(line.notes) == len(song)
    assert np.median([abs(a[0] - b[0]) for a, b in zip(line.notes, song)]) < 0.04       # a beat is 62 ms
    assert [n[2] for n in line.notes] == [m for _, _, m in song]


# ---------------------------------------------------------------- MIDI
def test_a_midi_file_is_read_with_its_tempo_map_and_running_status():
    melody = [(0, b"\xff\x03\x05Lead!"), (0, b"\x90\x3c\x50"), (480, b"\x80\x3c\x00"), (480, b"\x3e\x50"), (960, b"\x3e\x00"),
              (960, b"\x90\x40\x50"), (1440, b"\x90\x40\x00")]            # the second note on in running status, off by velocity 0
    tempo = [(0, b"\xff\x51\x03\x07\xa1\x20"), (960, b"\xff\x51\x03\x03\xd0\x90")]    # 500000 us a quarter, then 250000
    division, tracks = notes.parse_smf(smf([tempo, melody]))
    assert division == 480 and len(tracks) == 2
    assert [e[:2] for e in tracks[0]] == [(0, "tempo"), (960, "tempo")]
    kinds = [e[1] for e in tracks[1]]
    assert kinds.count("note") == 6 and tracks[1][0][1] == "text"
    seconds = notes._seconds(np.array([0, 480, 960, 1440]), 480, [(0, 500000), (960, 250000)])
    assert seconds.tolist() == pytest.approx([0.0, 0.5, 1.0, 1.25])
    assert notes._seconds(np.array([480]), -960, []).tolist() == [0.5]       # a SMPTE file: ticks a second


def test_a_damaged_or_wrapped_midi_file_is_read_as_far_as_it_goes():
    song = [(0.5 * k, 0.4, 60 + k % 5) for k in range(30)]
    data = smf([line_track(song)])
    assert len(notes.parse_smf(b"RIFF\x00\x00\x00\x00RMIDdata\x00\x00\x00\x00" + data)[1][0]) > 50
    cut = notes.parse_smf(data[:-60])[1][0]
    assert 20 < len(cut) < 60
    with pytest.raises(ValueError):
        notes.parse_smf(b"not a midi file")


def test_the_melody_of_a_midi_is_the_line_that_is_one_note_at_a_time_and_not_the_drums_or_the_chords():
    song = tune()
    tracks = [[(0, b"\xff\x51\x03\x07\xa1\x20")]] + accompaniment(song) + [line_track(song, channel=3, name="Voice", lyrics=True)]
    lines = notes.midi_lines(smf(tracks))
    assert lines[0].label.startswith("track 5") and "Voice" in lines[0].label           # the chords (polyphonic) and the drums are no candidates
    assert all("track 4" not in l.label and "track 2" not in l.label for l in lines)     # drums: channel 10; chords: three at once
    assert len(lines[0].notes) == len(song) and lines[0].notes[0][0] == pytest.approx(song[0][0], abs=0.002)
    assert lines[0].notes[0][2] == song[0][2]


# ---------------------------------------------------------------- aligning a file with the voice
@pytest.mark.parametrize("name,kw,transpose", [
    ("as it is", {}, 0),
    ("an intro of 8 s", {"shift": 8.0}, 0),
    ("another tempo", {"tempo": 0.93}, 0),
    ("two semitones low", {"transpose": 2}, 2),
    ("an octave high", {"transpose": -12}, -12),
    ("a fifth of it missing", {"delete": 0.2}, 0),
    ("a tenth a semitone off", {"move": 0.1}, 0),
    ("all of it", {"shift": 8.0, "tempo": 0.93, "transpose": 2, "delete": 0.2, "move": 0.1}, 2),
])
def test_a_perturbed_midi_of_the_song_is_accepted_with_its_transposition_and_a_small_onset_error(tmp_path, name, kw, transpose):
    song = tune()
    ours = notes.ours_of(folder_of(tmp_path, song))
    answer = notes.Answer("midi-lib", "midi", "x.mid", smf([line_track(perturbed(song, **kw))]))
    doc = notes.check_answer(answer, ours)
    assert doc["accepted"], doc.get("reason")
    assert doc["transpose"] == transpose
    assert doc["onset_median_ms"] < 80 and doc["agree_known"] > 0.7 and doc["same_song"]
    assert doc["warp"]["scale"] == pytest.approx(1 / kw.get("tempo", 1.0), abs=0.03)
    assert doc["warp"]["offset"] == pytest.approx(kw.get("shift", 0.0) * kw.get("tempo", 1.0) / kw.get("tempo", 1.0), abs=0.4)
    assert doc["pitch_within_1"] > 0.9 and doc["matched"] > 0.7 * doc["notes_known"]


def test_a_perturbed_ultrastar_chart_is_accepted_too(tmp_path):
    song = tune()
    ours = notes.ours_of(folder_of(tmp_path, song))
    chart = ultrastar(perturbed(song, transpose=12, delete=0.1, move=0.1), gap_ms=1500)
    doc = notes.check_answer(notes.Answer("charts", "ultrastar", "x.txt", chart.encode()), ours)
    assert doc["accepted"], doc.get("reason")
    assert doc["transpose"] == 12 and doc["onset_median_ms"] < 80
    assert doc["warp"]["offset"] == pytest.approx(0.0, abs=0.4)


def test_how_much_of_the_file_lies_in_the_sung_words_is_said_where_the_song_has_timings(tmp_path):
    song = tune()
    folder = folder_of(tmp_path, song)
    ours = notes.ours_of(folder)
    assert ours.windows is None
    answer = notes.Answer("midi-lib", "midi", "x.mid", smf([line_track(perturbed(song, shift=1.0))]))
    assert notes.check_answer(answer, ours)["in_words"] is None
    words = [{"start": s, "end": s + l, "text": "la"} for s, l, _ in song if s < 40]       # words for the first notes only
    (folder / "timings.json").write_text(json.dumps({"lines": [words]}))
    ours = notes.ours_of(folder)
    assert ours.windows and ours.windows[0][0] == pytest.approx(song[0][0] - 0.25)
    doc = notes.check_answer(answer, ours)
    assert 0.3 < doc["in_words"] < 0.7 and doc["accepted"]                                   # the notes after 40 s are outside every word


def test_the_melody_of_another_song_is_not_accepted(tmp_path):
    ours = notes.ours_of(folder_of(tmp_path, tune(seed=7)))
    answer = notes.Answer("midi-lib", "midi", "x.mid", smf([line_track(tune(seed=21, phrases=12))]))
    doc = notes.check_answer(answer, ours)
    assert not doc["accepted"] and doc["agree_known"] < 0.45 and not doc["same_song"]
    assert "semitone" in doc["reason"] and doc["notes_known"] > 50


def test_a_midi_with_no_melody_in_it_is_not_accepted_and_the_melody_is_found_among_the_tracks(tmp_path):
    song = tune()
    ours = notes.ours_of(folder_of(tmp_path, song))
    chords, bass, drums = accompaniment(song)
    doc = notes.check_answer(notes.Answer("midi-lib", "midi", "x.mid", smf([chords, drums])), ours)       # threes at once, and drums
    assert not doc["accepted"] and "no melody-like line" in doc["reason"]
    doc = notes.check_answer(notes.Answer("midi-lib", "midi", "x.mid", smf(accompaniment(song))), ours)    # a bass is a line, and not this song's
    assert not doc["accepted"] and doc["agree_known"] < 0.3 and not doc["same_song"]
    # the melody on track 3 among a tempo track, chords, bass and drums, with no name to go by
    tracks = [[(0, b"\xff\x51\x03\x07\xa1\x20")], accompaniment(song)[0], line_track(perturbed(song, shift=2.0), channel=2), accompaniment(song)[1], accompaniment(song)[2]]
    doc = notes.check_answer(notes.Answer("midi-lib", "midi", "x.mid", smf(tracks)), ours)
    assert doc["accepted"] and doc["candidate"].startswith("track 3")


def test_too_little_to_compare_is_said_so(tmp_path):
    ours = notes.ours_of(folder_of(tmp_path, tune()))
    short = notes.Answer("x", "midi", "x.mid", smf([line_track([(0.5 * k, 0.4, 60 + k % 4) for k in range(25)])]))
    ok = notes.check_answer(short, ours)
    assert not ok["accepted"] and ok["reason"]
    empty = notes.Answer("x", "ultrastar", "x.txt", b"#BPM:100\n#GAP:0\nE\n")
    assert "no melody-like line" in notes.check_answer(empty, ours)["reason"]
    garbled = notes.Answer("x", "midi", "x.mid", b"MThd\x00\x00")
    assert "not readable" in notes.check_answer(garbled, ours)["reason"]
    no_division = notes.Answer("x", "midi", "x.mid", smf([line_track(tune())], division=0))
    assert "not readable" in notes.check_answer(no_division, ours)["reason"]


# ---------------------------------------------------------------- the sources, and where the files are
def test_the_sources_file_has_urls_and_paths_and_refuses_what_is_neither(tmp_path):
    f = tmp_path / "note-sources.toml"
    f.write_text('[[source]]\nname = "u"\nurl = "https://e.test/{artist_slug}/{song_q}"\nkind = "midi"\nheaders = { "X-A" = "b" }\n'
                 '[[source]]\nname = "p"\npath = "/tmp/notes/{artist} - {song}.*"\n')
    u, p = notes.load_sources(f)
    assert u.kind == "midi" and u.headers == {"X-A": "b"} and p.kind == "auto"
    assert u.address("The Made Ups", "A Song!") == "https://e.test/the-made-ups/A%20Song%21"
    assert p.pattern("Made [Up]", "Tune") == "/tmp/notes/Made [[]Up] - Tune.*"
    assert p.root == Path("/tmp/notes") and u.root is None
    for bad in ('name = "a"\nurl = "https://e.test/"\nkind = "scroll"', 'name = "a"', 'name = "a"\nurl = "https://e.test/"\npath = "/x"',
                'name = "a"\nurl = "https://e.test/{album}"', 'name = "a"\npath = "/x/{artist}/{}.mid"'):
        f.write_text(f"[[source]]\n{bad}\n")
        with pytest.raises(ValueError):
            notes.load_sources(f)


def test_the_sources_file_is_looked_for_where_the_tabs_one_is(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(Path, "home", lambda: tmp_path / "home")
    monkeypatch.delenv("KARAOKIFEX_NOTE_SOURCES", raising=False)
    assert notes.sources_file() is None
    (tmp_path / "note-sources.toml").write_text("")
    assert notes.sources_file() == Path("note-sources.toml")
    other = tmp_path / "other.toml"
    other.write_text("")
    monkeypatch.setenv("KARAOKIFEX_NOTE_SOURCES", str(other))
    assert notes.sources_file() == other
    assert notes.sources_file(tmp_path / "note-sources.toml") == tmp_path / "note-sources.toml"


def test_the_example_sources_file_is_valid_and_says_what_it_is():
    example = Path(__file__).parent.parent / "note-sources.example.toml"
    sources = notes.load_sources(example)
    assert [s.name for s in sources] == ["mine", "midi-collection", "example-midi"]
    assert sources[0].root == Path("/data/notes") and sources[1].kind == "midi" and sources[2].url.startswith("https://midi.example.com/")
    assert "note-sources.toml" in (Path(__file__).parent.parent / ".gitignore").read_text()


def touch(path: Path, data: bytes = b"MThd") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def test_a_path_names_its_files_exactly_or_by_the_folded_names_down_its_own_folders(tmp_path):
    flat = notes.Source("flat", path=str(tmp_path / "flat" / "{artist} - {song}.*"))
    touch(tmp_path / "flat" / "Beatles - Yesterday.mid")
    touch(tmp_path / "flat" / "Beatles - Yesterday.jpg")
    assert [p.name for p in notes.find_files(flat, "Beatles", "Yesterday")] == ["Beatles - Yesterday.mid"]       # exact; a picture is no notes
    touch(tmp_path / "flat" / "Sigur Rós - Hoppípolla.txt")
    touch(tmp_path / "flat" / "Beyonce feat. Jay-Z - Crazy in Love.kar")
    touch(tmp_path / "flat" / "Simon and Garfunkel - The Boxer.mid")
    assert [p.name for p in notes.find_files(flat, "Sigur Ros", "Hoppipolla")] == ["Sigur Rós - Hoppípolla.txt"]       # accents
    assert [p.name for p in notes.find_files(flat, "BEYONCÉ", "Crazy In Love (Remastered)")] == ["Beyonce feat. Jay-Z - Crazy in Love.kar"]
    assert [p.name for p in notes.find_files(flat, "Simon & Garfunkel", "Boxer")] == ["Simon and Garfunkel - The Boxer.mid"]       # "The " and "&"
    assert [p.name for p in notes.find_files(flat, "Simon & Garfunkel", "Sound of Silence")] == []
    assert notes.find_files(flat, "Nobody", "Nothing") == []
    nested = notes.Source("midi-lib", path=str(tmp_path / "collection" / "{artist}" / "{song}*.mid"))
    touch(tmp_path / "collection" / "Beatles, The" / "Yesterday.1.mid")
    assert notes.find_files(nested, "The Beatles", "Hey Jude") == []
    assert notes.find_files(notes.Source("gone", path=str(tmp_path / "missing" / "{artist}" / "{song}.mid")), "A", "B") == []


def test_all_the_versions_of_a_song_in_a_nested_library_are_found(tmp_path):
    nested = notes.Source("midi-lib", path=str(tmp_path / "collection" / "{artist}" / "{song}*.mid"))
    for name in ("Yesterday.1.mid", "Yesterday.2.mid", "Yesterday.10.mid", "Yesterday Once More.mid", "Help.1.mid"):
        touch(tmp_path / "collection" / "Beatles, The" / name)
    found = notes.find_files(nested, "Beatles, The", "Yesterday")           # named as the folder is: the template's own glob
    assert sorted(p.name for p in found) == ["Yesterday Once More.mid", "Yesterday.1.mid", "Yesterday.10.mid", "Yesterday.2.mid"]
    found = notes.find_files(nested, "The Beatles", "Yesterday")            # not as the folder is: the folded names, the files by the template's own glob
    assert sorted(p.name for p in found) == ["Yesterday Once More.mid", "Yesterday.1.mid", "Yesterday.10.mid", "Yesterday.2.mid"]
    found = notes.find_files(nested, "The Beatles", "Yesterday Once More")
    assert [p.name for p in found] == ["Yesterday Once More.mid"]


def test_a_folder_with_the_files_dropped_into_it_is_looked_at_first_and_a_source_with_no_folder_is_skipped_quietly(tmp_path):
    song = tune()
    folder = folder_of(tmp_path, song)
    (folder / "notes").mkdir()
    (folder / "notes" / "chart.txt").write_text(ultrastar(perturbed(song, transpose=2)))
    (folder / "notes" / "readme.md").write_text("not notes")
    (folder / "notes" / "b.kar").write_bytes(smf([line_track(perturbed(song))]))
    assert [p.name for p in notes.dropped_in(folder)] == ["b.kar", "chart.txt"]
    sources = [notes.Source("gone", path=str(tmp_path / "nowhere" / "{artist} - {song}.*")), notes.Source("web", url="https://e.test/{song_slug}.mid")]
    asked = notes.asked(folder, sources, only_local=False)
    assert asked == [("dropped-in", str(folder / "notes" / "b.kar")), ("dropped-in", str(folder / "notes" / "chart.txt")), ("web", "https://e.test/tune.mid")]
    assert [a for a in notes.asked(folder, sources, only_local=True) if a[0] == "web"] == []
    calls = []
    got = list(notes.answers(folder, sources, only_local=False, fetch=lambda url, headers: calls.append(url)))
    assert [name for name, _ in got] == ["dropped-in"] and calls == ["https://e.test/tune.mid"]
    assert [a.kind for a in got[0][1]] == ["midi", "ultrastar"]
    list(notes.answers(folder, sources, only_local=True, fetch=lambda url, headers: calls.append("again")))
    assert calls == ["https://e.test/tune.mid"]


def test_an_address_that_answers_a_page_is_not_an_answer(tmp_path):
    folder = folder_of(tmp_path, tune())
    web = [notes.Source("web", url="https://e.test/{song_slug}")]
    assert list(notes.answers(folder, web, fetch=lambda u, h: b"<html>not found, sorry</html>")) == []
    assert list(notes.answers(folder, web, fetch=lambda u, h: None)) == []
    (name, (answer,)), = notes.answers(folder, web, fetch=lambda u, h: b"MThd....")
    assert answer.kind == "midi" and answer.file == "https://e.test/tune"


# ---------------------------------------------------------------- asking, writing, using
def test_asking_writes_the_check_and_the_accepted_notes_and_uses_them(tmp_path):
    song = tune()
    folder = folder_of(tmp_path, song)
    midi = smf([line_track(perturbed(song, shift=4.0, transpose=12, delete=0.1))])
    web = [notes.Source("wrong", url="https://e.test/wrong"), notes.Source("good", url="https://e.test/good", kind="midi")]
    bodies = {"https://e.test/wrong": smf([line_track(tune(seed=21))]), "https://e.test/good": midi}
    doc = notes.make(folder, web, fetch=lambda url, headers: bodies[url])
    assert doc["accepted"] and doc["source"] == "good" and doc["transpose"] == 12
    (other,) = doc["others"]
    assert other["source"] == "wrong" and other["file"] == "https://e.test/wrong" and not other["accepted"] and other["reason"]
    assert other["same_song"] is False and "pitch_exact" in other and "others" not in other
    check = json.loads((folder / "note-check.json").read_text())
    for key in ("version", "source", "kind", "file", "accepted", "transpose", "offset", "warp", "notes_known", "notes_ours", "matched",
                "pitch_exact", "pitch_within_1", "onset_median_ms", "onset_p90_ms", "covered_known", "covered_ours", "wrong", "missing", "extra"):
        assert key in check, key
    source = json.loads((folder / "notes-source.json").read_text())
    assert source["version"] == 1 and source["source"] == "good" and source["kind"] == "midi" and source["url"] == "https://e.test/good"
    assert source["transpose"] == 12 and all(len(n) == 4 for n in source["notes"]) and 0.7 * len(song) < len(source["notes"]) <= len(song)
    nearest = [min(song, key=lambda s: abs(s[0] - n[0])) for n in source["notes"]]
    assert np.mean([abs(n[0] - s[0]) < 0.2 and n[2] == s[2] for n, s in zip(source["notes"], nearest)]) > 0.9       # on the recording's clock and pitches
    melody = json.loads((folder / "melody.json").read_text())
    assert melody["known"]["source"] == "good" and len(melody["notes_detected"]) == len(song)
    # asked again only when forced
    assert notes.make(folder, web, fetch=lambda url, headers: pytest.fail("asked again"))["source"] == "good"
    assert notes.make(folder, [], force=True, fetch=lambda url, headers: pytest.fail("no sources"))is None


def test_a_rejected_answer_is_kept_in_the_check_and_gives_no_notes(tmp_path):
    folder = folder_of(tmp_path, tune())
    web = [notes.Source("wrong", url="https://e.test/wrong")]
    doc = notes.make(folder, web, fetch=lambda url, headers: smf([line_track(tune(seed=21))]))
    assert not doc["accepted"] and doc["reason"]
    assert (folder / "note-check.json").exists() and not (folder / "notes-source.json").exists()
    assert "known" not in json.loads((folder / "melody.json").read_text())


def test_a_hand_dropped_file_that_is_accepted_ends_the_asking(tmp_path):
    song = tune()
    folder = folder_of(tmp_path, song)
    (folder / "notes").mkdir()
    (folder / "notes" / "mine.txt").write_text(ultrastar(perturbed(song, transpose=2)))
    doc = notes.make(folder, [notes.Source("web", url="https://e.test/x")], fetch=lambda url, headers: pytest.fail("asked"))
    assert doc["source"] == "dropped-in" and doc["accepted"] and doc["transpose"] == 2
    assert json.loads((folder / "notes-source.json").read_text())["path"].endswith("mine.txt")


def test_a_song_with_no_melody_or_no_proper_name_is_left_alone(tmp_path):
    bare = tmp_path / "Made Up - Tune"
    bare.mkdir()
    assert notes.make(bare, [notes.Source("w", url="https://e.test/x")], fetch=lambda u, h: pytest.fail("asked")) is None
    odd = folder_of(tmp_path, tune(), name="No Dash In This Name")
    assert notes.make(odd, [notes.Source("w", url="https://e.test/x")], fetch=lambda u, h: pytest.fail("asked")) is None


def detected_song(tmp_path):
    song = tune()
    folder = folder_of(tmp_path, song)
    return song, folder


def test_apply_puts_the_known_notes_over_the_detected_ones_and_can_be_run_again(tmp_path):
    song, folder = detected_song(tmp_path)
    melody = json.loads((folder / "melody.json").read_text())
    detected = [list(n) for n in melody["notes"]]                    # one for each note of the song, in order
    # the file has every fifth note a semitone above what is sung, lacks two notes the voice sang, and has one the voice does not sing
    known = [[s, l, m + (1 if k % 5 == 0 else 0), None] for k, (s, l, m) in enumerate(song) if k not in (10, 20)]
    quiet = [song[-1][0] + song[-1][1] + 1.0, 0.5, 64, None]
    (folder / "notes-source.json").write_text(json.dumps({"version": 1, "source": "mine", "kind": "midi", "notes": known + [quiet]}))
    melody["notes"] = [n for k, n in enumerate(detected) if k != 31]                # the detector lost one that the file has
    (folder / "melody.json").write_text(json.dumps(melody))
    assert notes.apply(folder)
    once = (folder / "melody.json").read_text()
    out = json.loads(once)
    assert out["known"] == {"source": "mine", "kind": "midi", "matched": len(song) - 3, "added": 1}
    assert len(out["notes_detected"]) == len(song) - 1 and all(len(n) == 6 for n in out["notes"])
    assert [n[0] for n in out["notes"]] == sorted(n[0] for n in out["notes"])
    at = {round(n[0], 2): n for n in out["notes"]}
    for k, (s, l, m) in enumerate(song):
        n = at.get(round(detected[k][0], 2))
        if k == 31:
            continue
        if k in (10, 20):                                            # nothing known for it: kept as detected, less sure
            assert n[2] == m and n[5] <= 0.5
        else:
            assert n[2] == m + (1 if k % 5 == 0 else 0) and n[5] >= 0.9           # the known note, and sure of it
            assert n[3] == (-100 if k % 5 == 0 else 0) or abs(n[3]) <= 30           # its cents against what was sung
    (added,) = [n for n in out["notes"] if n[5] == 0.6]                            # the lost one, put back where the voice is
    assert added[2] == song[31][2] and abs(added[0] - song[31][0]) < 0.05 and abs(added[3]) <= 30
    assert not any(n[0] > song[-1][0] + song[-1][1] for n in out["notes"])          # nothing was sung under the quiet one
    assert out["range"] == [min(n[2] for n in out["notes"]), max(n[2] for n in out["notes"])]
    assert notes.apply(folder) and (folder / "melody.json").read_text() == once          # again: the same, made from notes_detected
    (folder / "notes-source.json").unlink()                                              # the notes as detected are put back
    assert notes.apply(folder)
    back = json.loads((folder / "melody.json").read_text())
    assert back["notes"] == out["notes_detected"] and "known" not in back and "notes_detected" not in back
    assert not notes.apply(folder)                                                      # nothing to do


def test_apply_takes_its_cents_from_the_detected_pitch_and_adds_a_missing_note_where_the_line_is_voiced(tmp_path):
    song, folder = detected_song(tmp_path)
    melody = json.loads((folder / "melody.json").read_text())
    gone = melody["notes"].pop(12)                                   # the detector missed one that the voice sang
    melody["notes"][5][3] = 35                                         # a note detected 35 cents up
    (folder / "melody.json").write_text(json.dumps(melody))
    known = [[s, l, m, None] for s, l, m in song]
    known[5][2] = melody["notes"][5][2] + 1                          # the known one is a semitone higher: the voice is 65 cents under it
    (folder / "notes-source.json").write_text(json.dumps({"source": "mine", "kind": "ultrastar", "notes": known}))
    notes.apply(folder)
    out = json.loads((folder / "melody.json").read_text())
    five = next(n for n in out["notes"] if abs(n[0] - melody["notes"][5][0]) < 1e-6)
    assert five[2] == known[5][2] and five[3] == -65
    added = next(n for n in out["notes"] if abs(n[0] - gone[0]) < 0.1)
    assert added[5] == 0.6 and added[2] == gone[2] and abs(added[3]) < 30
    assert notes.known_stale(folder) is False
    (folder / "notes-source.json").write_text(json.dumps({"source": "other", "kind": "ultrastar", "notes": known}))
    assert notes.known_stale(folder)
    (folder / "notes-source.json").unlink()
    assert notes.known_stale(folder)                                  # the file still carries known notes
    notes.apply(folder)
    assert notes.known_stale(folder) is False


# ---------------------------------------------------------------- the figure
def test_the_report_counts_the_songs_and_says_how_right_our_notes_are(tmp_path):
    song = tune()
    rows = [("a", True, 0.9, 0.97, 30, 0.8), ("b", True, 0.7, 0.9, 50, 0.7), ("c", False, 0.5, 0.8, 90, 0.5), ("d", False, 0.1, 0.4, None, 0.1)]
    folders = []
    for name, accepted, exact, within, onset, agree in rows:
        folder = tmp_path / f"Artist - {name}"
        folder.mkdir()
        (folder / "note-check.json").write_text(json.dumps({"accepted": accepted, "same_song": agree >= notes.SAME_SONG, "pitch_exact": exact,
                                                             "pitch_within_1": within, "onset_median_ms": onset, "agree_known": agree}))
        if accepted:
            (folder / "notes-source.json").write_text("{}")
        folders.append(folder)
    (tmp_path / "Artist - none").mkdir()
    folders.append(tmp_path / "Artist - none")
    checks, known = notes.collect(folders)
    assert len(checks) == 4 and known == 2
    s = notes.summary(checks)
    assert (s["songs"], s["accepted"], s["rejected"], s["answers"], s["same_song"], s["obviously_another"]) == (4, 2, 2, 4, 3, 1)
    assert s["of_accepted"]["pitch_exact"] == {"n": 2, "median": pytest.approx(0.8), "p10": pytest.approx(0.72), "p90": pytest.approx(0.88)}
    assert s["of_all_same_song"]["pitch_exact"]["n"] == 3 and s["of_all_same_song"]["pitch_exact"]["median"] == pytest.approx(0.7)
    assert s["of_all_same_song"]["onset_median_ms"]["n"] == 3
    text = notes.format_report(checks, known)
    assert "4 songs have an answer" in text and "2 songs accepted" in text and "pitch_exact" in text
    assert notes.summary([])["of_accepted"]["pitch_exact"]["median"] is None


def test_the_report_counts_every_answer_a_song_has_not_only_the_one_kept(tmp_path):
    folder = tmp_path / "Artist - Song"
    folder.mkdir()
    other = {"accepted": False, "same_song": True, "pitch_exact": 0.3, "pitch_within_1": 0.5, "onset_median_ms": 90, "agree_known": 0.45}
    elsewhere = {"accepted": False, "same_song": False, "pitch_exact": 0.1, "pitch_within_1": 0.2, "onset_median_ms": None, "agree_known": 0.1}
    (folder / "note-check.json").write_text(json.dumps({"accepted": True, "same_song": True, "pitch_exact": 0.9, "pitch_within_1": 0.95, "onset_median_ms": 20,
                                                         "agree_known": 0.8, "others": [other, elsewhere]}))
    checks, _ = notes.collect([folder])
    s = notes.summary(checks)
    assert (s["songs"], s["accepted"], s["answers"], s["same_song"], s["obviously_another"]) == (1, 1, 3, 2, 1)
    assert s["of_accepted"]["pitch_exact"]["n"] == 1 and s["of_accepted"]["pitch_exact"]["median"] == pytest.approx(0.9)
    assert s["of_all_same_song"]["pitch_exact"]["n"] == 2 and s["of_all_same_song"]["pitch_exact"]["median"] == pytest.approx(0.6)


def test_the_command_lists_what_it_would_ask_and_reports(tmp_path):
    song = tune()
    folder = folder_of(tmp_path, song)
    sources = tmp_path / "note-sources.toml"
    sources.write_text('[[source]]\nname = "web"\nurl = "https://e.test/{artist_slug}/{song_slug}.mid"\n'
                       f'[[source]]\nname = "here"\npath = "{tmp_path.as_posix()}/lib/{{artist}} - {{song}}.*"\n')
    touch(tmp_path / "lib" / "Made Up - Tune.mid")
    runner = CliRunner()
    out = runner.invoke(notes.main, [str(folder), "--sources", str(sources), "--dry-run"]).output.splitlines()
    assert "Made Up - Tune: web: https://e.test/made-up/tune.mid" in out
    (here,) = [line for line in out if line.startswith("Made Up - Tune: here: ")]
    assert Path(here.removeprefix("Made Up - Tune: here: ")) == tmp_path / "lib" / "Made Up - Tune.mid"
    out = runner.invoke(notes.main, [str(folder), "--sources", str(sources), "--dry-run", "--only-local"]).output
    assert "web" not in out and "here" in out
    (folder / "note-check.json").write_text(json.dumps({"accepted": True, "same_song": True, "pitch_exact": 0.8, "pitch_within_1": 0.9,
                                                         "onset_median_ms": 40, "agree_known": 0.7}))
    out = runner.invoke(notes.main, ["--report", "--library", str(tmp_path)]).output
    assert "1 songs have an answer" in out and "80%" in out
    assert runner.invoke(notes.main, []).exit_code != 0
