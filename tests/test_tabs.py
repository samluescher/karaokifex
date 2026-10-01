"""Downloaded chords and tabs (tabs.py): reading what a source says, and the check against ours. Made-up songs only."""

import json
from pathlib import Path

from karaokifex import tabs

OVER = """[Verse 1]
C            G
Made up words to a made up tune
Am           F
Nobody wrote this one

[Chorus]
F   C
La la la
"""

PRO = """{title: Made up}
[C]Made up words to a [G]made up tune
[Am]Nobody wrote [F]this one
"""

TAB = """Intro
e|--0--1--0--|
B|--1--1--1--|
G|--0--2--0--|
D|--2--3--2--|
A|--3--3--3--|
E|-----------|
"""


def test_chords_over_lyrics_take_each_chord_with_its_column_and_the_words_under_it():
    got = tabs.parse(OVER, "chords-over-lyrics")
    assert got["chords"] == ["C", "G", "Am", "F"]
    assert [s["name"] for s in got["sections"]] == ["Verse 1", "Chorus"]
    first = got["sections"][0]["lines"][0]
    assert first["text"] == "Made up words to a made up tune"
    assert first["chords"] == [{"name": "C", "at": 0}, {"name": "G", "at": 13}]


def test_chordpro_has_its_chords_in_the_words():
    got = tabs.parse(PRO, "chordpro")
    assert got["chords"] == ["C", "G", "Am", "F"]
    line = got["sections"][0]["lines"][0]
    assert line["text"] == "Made up words to a made up tune"
    assert line["chords"][1] == {"name": "G", "at": 19}


def test_an_ascii_tab_is_kept_as_written():
    got = tabs.parse(TAB, "ascii-tab")
    block = got["sections"][0]["lines"][0]["tab"]
    assert len(block) == 6 and block[0].startswith("e|")


def test_what_is_and_is_not_a_chord():
    for name in ("C", "F#m7", "Bb", "G/B", "Dsus4", "Cmaj7", "Am"):
        assert tabs.is_chord(name), name
    for word in ("Made", "tune", "Hello", "I"):
        assert not tabs.is_chord(word), word


def test_the_url_is_made_from_the_names():
    s = tabs.Source("x", "https://e.test/{artist_slug}/{song_slug}?a={artist_q}", "chordpro")
    assert s.address("The Made Ups", "A Song!") == "https://e.test/the-made-ups/a-song?a=The%20Made%20Ups"


def test_downloaded_is_right_and_ours_is_checked_against_it(tmp_path: Path):
    folder = tmp_path / "Made Up - Tune"
    folder.mkdir()
    (folder / "chords.json").write_text(json.dumps({"chords": [[0, 1, "C"], [1, 1, "N"], [2, 1, "Em"], [3, 1, "G"]]}))
    doc = tabs.download(folder, [tabs.Source("none", "https://nothing.invalid/{song_slug}"), tabs.Source("mine", "https://e.test/{song_slug}")],
                        get=lambda url, headers: OVER if "e.test" in url else None)
    assert doc and doc["source"] == "mine" and doc["chords"] == ["C", "G", "Am", "F"]
    assert json.loads((folder / "tab-source.json").read_text())["url"] == "https://e.test/tune"
    report = json.loads((folder / "tab-check.json").read_text())
    assert report == {"source": "mine", "agree": ["C", "G"], "wrong": ["Em"], "missing": ["Am", "F"]}


def test_a_source_with_no_chords_in_its_answer_is_passed_over(tmp_path: Path):
    folder = tmp_path / "Made Up - Tune"
    folder.mkdir()
    assert tabs.download(folder, [tabs.Source("bad", "https://e.test/x")], get=lambda u, h: "<html>not found, sorry</html>") is None
    assert not (folder / "tab-source.json").exists()


def test_already_downloaded_is_left_unless_forced(tmp_path: Path):
    folder = tmp_path / "Made Up - Tune"
    folder.mkdir()
    (folder / "tab-source.json").write_text('{"chords": ["A"]}')
    assert tabs.download(folder, [tabs.Source("s", "https://e.test/x")], get=lambda u, h: OVER) == {"chords": ["A"]}
    assert tabs.download(folder, [tabs.Source("s", "https://e.test/x")], get=lambda u, h: OVER, force=True)["chords"][0] == "C"


def test_the_sources_file_is_read_and_a_wrong_kind_refused(tmp_path: Path):
    f = tmp_path / "tab-sources.toml"
    f.write_text('[[source]]\nname = "a"\nurl = "https://e.test/{song}"\nkind = "chordpro"\n')
    assert tabs.load_sources(f)[0].kind == "chordpro"
    f.write_text('[[source]]\nname = "a"\nurl = "https://e.test/"\nkind = "scroll"\n')
    try:
        tabs.load_sources(f)
    except ValueError as e:
        assert "kind" in str(e)
    else:
        raise AssertionError("a wrong kind was accepted")
