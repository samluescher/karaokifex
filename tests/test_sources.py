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
    Path(f"{video}.info.json").write_text('{"id": "bandcamp:x.bandcamp.com/track/y", "title": "T", "made": "still"}',
                                          encoding="utf-8")
    assert download.sidecar(video)["id"] == "bandcamp:x.bandcamp.com/track/y"
    from karaokifex.models import VideoInfo
    # made rides along into the song's info.json, and an info.json with fields it doesn't know still reads
    info = VideoInfo(id="bandcamp:x", title="T", made="still")
    assert VideoInfo.from_dict({**info.to_dict(), "later": 1}).made == "still"


def test_bandcamp_keys_and_covers():
    assert bandcamp.source_key("https://DisasterFantasy.bandcamp.com/track/Anywhere/?from=x") == \
        "bandcamp:disasterfantasy.bandcamp.com/track/anywhere"
    info = {"thumbnails": [{"url": "https://f4.bcbits.com/img/a123_5.jpg"}, {"url": "https://f4.bcbits.com/img/a123_10.jpg"}]}
    assert bandcamp.cover_url(info) == "https://f4.bcbits.com/img/a123_0.jpg"


def test_any_encoding_is_read_as_unicode():
    word = "Bümpliz – Gränni"
    assert lyrics.decode_text(word.encode("utf-8")) == (word, "utf-8")
    assert lyrics.decode_text(word.encode("utf-8-sig"))[0] == word
    assert lyrics.decode_text(word.encode("utf-16"))[0] == word
    assert lyrics.decode_text(("ä Kafi am Pischterand " * 4).encode("cp1252"), "windows-1252")[0].startswith("ä Kafi")
    # a page naming Latin-1 but written in UTF-8: UTF-8 wins
    assert lyrics.decode_text(word.encode("utf-8"), "iso-8859-1")[0] == word
    # no charset named: Latin-script text in Windows-1252, Cyrillic guessed
    assert lyrics.decode_text("Äs isch Zyt für üs zwöi".encode("cp1252")) == ("Äs isch Zyt für üs zwöi", "cp1252")
    assert lyrics.decode_text(("Где же ты, моя любимая, где же ты " * 3).encode("cp1251"))[0].startswith("Где же ты")
    # "u" and a combining diaeresis come out as the one character
    assert lyrics.decode_text("Bümpliz".encode("utf-8"))[0] == "Bümpliz"


def test_given_lyrics_skip_notes_and_any_encoding(tmp_path: Path):
    given = tmp_path / "given.txt"
    given.write_bytes("# taken from: https://example.org\n# agrees 80%: x\nÄs isch Zyt\n\nfür üs zwöi\n".encode("cp1252"))
    got = lyrics.from_file(given, "A", "S")
    assert [line.text for line in got.lines] == ["Äs isch Zyt", "für üs zwöi"]


def test_web_lyrics_say_where_they_came_from():
    kept = ["we go down to the river", "and the river takes us home"] * 5
    pages = [{"url": "https://genius.com/x", "lines": kept, "how": 'its lyrics block (the element with data-lyrics-container="true")',
              "removed": ["[Chorus]"]},
             {"url": "https://songtexte.de/x", "lines": kept[:8], "how": "the longest run of short lines on it", "removed": []},
             {"url": "https://www.songtexte.com/x", "why": "HTTP 202, no page (a bot check)"}]
    head = lyrics_web.web_header("A - S", "nothing for it (asked first)", "https://genius.com/x", pages)
    assert all(line.startswith("#") for line in head)
    text = "\n".join(head)
    assert "taken from: https://genius.com/x" in text and 'data-lyrics-container="true"' in text
    assert "cleaned out: [Chorus]" in text and "agrees 100%: https://songtexte.de/x" in text
    assert "looked at, nothing: https://www.songtexte.com/x (HTTP 202" in text


def test_lrclib_is_asked_before_the_web(tmp_path: Path, monkeypatch):
    found = lyrics.Lyrics(lrclib_id=7, artist="A", track="S", album=None, duration=None, synced=True,
                          lines=(lyrics.LyricLine(1.5, "first"), lyrics.LyricLine(63.25, "second")))
    monkeypatch.setattr(lyrics, "fetch_lyrics", lambda *a, **k: [found])
    monkeypatch.setattr(lyrics_web, "find", lambda *a, **k: (_ for _ in ()).throw(AssertionError("the web searched")))
    out = tmp_path / "out.txt"
    assert lyrics_web.main(["A - S", "-o", str(out)]) == 0
    written = out.read_text(encoding="utf-8").splitlines()
    assert written[0].startswith("# A - S: lyrics from lrclib #7") and written[2:] == ["[00:01.50]first", "[01:03.25]second"]
    assert [line.text for line in lyrics.from_file(out, "A", "S").lines] == ["first", "second"]


def test_the_pipeline_asks_lrclib_before_a_given_file(tmp_path: Path, monkeypatch):
    from types import SimpleNamespace
    from karaokifex import pipeline
    given = tmp_path / "given.txt"
    given.write_text("given one\ngiven two\n", encoding="utf-8")
    job = SimpleNamespace(artist="A", song="S", info=SimpleNamespace(duration=None),
                          config=SimpleNamespace(lyrics_file=given), workspace=SimpleNamespace(lyrics_json=tmp_path / "lyrics.json"))
    ctx = SimpleNamespace(note=lambda *_: None)
    hit = lyrics.Lyrics(lrclib_id=7, artist="A", track="S", album=None, duration=None, synced=False,
                        lines=(lyrics.LyricLine(None, "from lrclib"),))
    monkeypatch.setattr(lyrics, "fetch_lyrics", lambda *a, **k: [hit])
    pipeline._lyrics(job, ctx)
    assert lyrics.load_lyrics(job.workspace.lyrics_json)[0].lrclib_id == 7
    monkeypatch.setattr(lyrics, "fetch_lyrics", lambda *a, **k: [])
    assert "lrclib miss" in pipeline._lyrics(job, ctx)
    assert [line.text for line in lyrics.load_lyrics(job.workspace.lyrics_json)[0].lines] == ["given one", "given two"]


def test_only_the_songs_own_page_counts():
    song = "Ä Kafi am Pischterand"
    assert lyrics_web.about_song("https://www.musixmatch.com/lyrics/X/%C3%84-Kafi-am-Pischterand", "", song)
    assert lyrics_web.about_song("https://example.org/p/123", "<title>Kafi am Pischterand – Songtext</title>", song)
    assert not lyrics_web.about_song("https://www.songtexte.com/artist/patent-ochsner-23d6cce7.html",
                                     "<title>Patent Ochsner Songtexte</title>", song)
    assert lyrics_web.about_song("https://genius.com/Patent-ochsner-w-nuss-vo-bumpliz-lyrics", "", "W. Nuss vo Bümpliz")


def test_a_slideshow_covers_the_track_and_the_cover_comes_round():
    assert bandcamp.slides(240, 3) == 14             # 20 s slides overlapping by 2 s: 18 s each after the first
    assert bandcamp.slides(10, 3) == 1
    assert bandcamp.order(3, 7) == [0, 1, 2, 3, 1, 0, 2]
    assert bandcamp.order(0, 3) == [0, 0, 0]
    graph = bandcamp.slideshow_filter(1080, 3)
    assert graph.count("zoompan") == 3 and "offset=18.000" in graph and "offset=36.000" in graph and graph.endswith("[v]")


def test_press_photos_name_the_artist_and_are_no_covers():
    from karaokifex.sources import photos
    assert photos.names("Disaster Fantasy", "Dark Disco Duo Disaster Fantasy Debut Single")
    assert photos.names("Disaster Fantasy", "https://post-punk.com/dark-disco-duo-disaster-fantasy-debut/")
    assert not photos.names("Disaster Fantasy", "DESASTER Announces New Album")
    for name in ("Anywhere-Cover-scaled.jpg", "DF_TheHourglass-Artwork--scaled.jpg", "DF_VinylImage_860x.jpg", "tour-poster.png"):
        assert photos.NOT_A_PHOTO.search(name)
    assert not photos.NOT_A_PHOTO.search("Disaster-Fantasy.jpg")
    assert photos.ALBUM_ART.search("/img/a4085427402_10.jpg") and not photos.ALBUM_ART.search("/img/0030665963_0.jpg")
    assert photos.near(0b1011, 0b1001) and not photos.near(0, (1 << 20) - 1)
