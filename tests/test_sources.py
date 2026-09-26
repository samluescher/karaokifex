from pathlib import Path

from karaokifex.sources import bandcamp, lyrics_web
from karaokifex.steps import download, lyrics


def test_known_site_block_is_taken_and_cleaned():
    page = ('<html><body><nav>Home</nav><div id="lyrics">[Chorus]<br>Oh la la, the night is young<br>'
            + "<br>".join(f"line number {i} of the song" for i in range(10))
            + '<br>(Geigensolo)<br>Written by: Somebody</div><footer>© site</footer></body></html>')
    found = lyrics_web.lyrics_of(page, "https://www.songtexte.com/songtext/x/y.html")
    assert found[0] == "Oh la la, the night is young"
    assert "[Chorus]" not in found and "(Geigensolo)" not in found
    assert not any("Written by" in line for line in found)


def test_an_unknown_site_gives_its_longest_run_of_short_lines():
    prose = "<p>This is a long paragraph of prose about the band that goes on and on for many many words indeed.</p>"
    verse = "<p>" + "<br>".join(f"a short sung line {i}" for i in range(9)) + "</p>"
    assert len(lyrics_web.lyrics_of(prose + verse + prose, "https://example.org/x")) == 9


def test_the_version_the_others_agree_with_wins():
    same = ["we go down to the river", "and the river takes us home"] * 5
    other = ["something else entirely here", "nothing like the rest of them"] * 5
    picked = lyrics_web.pick([("a", other), ("b", same), ("c", same)])
    assert picked[0] == "b"


def test_given_lyrics_read_as_plain_or_lrc(tmp_path: Path):
    plain = tmp_path / "plain.txt"
    plain.write_text("first line\n\nsecond line\n", encoding="utf-8")
    got = lyrics.from_file(plain, "A", "S")
    assert not got.synced and [line.text for line in got.lines] == ["first line", "second line"]
    assert lyrics.describe(got).startswith("given")
    lrc = tmp_path / "song.lrc"
    lrc.write_text("[00:01.00]first line\n[00:03.50]second line\n", encoding="utf-8")
    assert lyrics.from_file(lrc, "A", "S").synced


def test_a_local_file_is_a_source_a_link_is_not(tmp_path: Path):
    video = tmp_path / "Artist - Song.mp4"
    video.write_bytes(b"x")
    assert download.local_file(str(video)) == video
    assert download.local_file(video.as_uri()) == video
    assert download.local_file("https://www.youtube.com/watch?v=abc") is None
    assert download.local_file(str(tmp_path / "missing.mp4")) is None
    Path(f"{video}.info.json").write_text('{"id": "bandcamp:x.bandcamp.com/track/y", "title": "T"}', encoding="utf-8")
    assert download.sidecar(video)["id"] == "bandcamp:x.bandcamp.com/track/y"


def test_bandcamp_keys_and_covers():
    assert bandcamp.source_key("https://DisasterFantasy.bandcamp.com/track/Anywhere/?from=x") == \
        "bandcamp:disasterfantasy.bandcamp.com/track/anywhere"
    info = {"thumbnails": [{"url": "https://f4.bcbits.com/img/a123_5.jpg"}, {"url": "https://f4.bcbits.com/img/a123_10.jpg"}]}
    assert bandcamp.cover_url(info) == "https://f4.bcbits.com/img/a123_0.jpg"
