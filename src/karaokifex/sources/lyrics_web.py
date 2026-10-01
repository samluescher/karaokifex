"""Lyrics from the web, for a song lrclib hasn't got (karaokifex-lyrics-web).

    karaokifex-lyrics-web "Artist - Song" [-o lyrics.txt] [--show]

lrclib first, always, as karaokifex itself does: a song it has is written from there (LRC when
synced), and the web is not searched. Only for a song it hasn't got, this searches the web for its lyrics ("<artist> <song> lyrics", and "songtext" too, where songs in
German and Swiss German are written down), fetches the pages of the lyric sites among the results,
and takes the lyrics out of each page that is the song's own (its address or title names the song;
an artist's page, listing song titles, reads like verses): the page's own lyrics block where the site is known, else the
longest run of short lines on it. Each is cleaned -- section labels ([Chorus], Refrain:), notes,
credits and the site's own lines out, a blank line between verses -- and the one most of the others
agree with is kept (a site's version that nobody else has is likelier wrong). Written as a flat text
file, a line a line, to hand karaokifex --lyrics-file, which aligns it to the voice like lrclib's.

Every file says where it came from, in # lines at the top (karaokifex skips them): lrclib's answer,
the page the lyrics were taken from and how -- the site's lyrics block and its marker, or the longest
run of short lines -- the other pages and how far they agreed, those that gave nothing and why, and
what the cleaning took out. Text is Unicode throughout: every page's encoding is detected (the
charset it names only a hint), lines are NFC ("ü" one character, never "u" and a combining mark), and
the file is written as UTF-8.

A miss exits 1 and writes nothing; karaokifex then transcribes the voice itself.
"""
from __future__ import annotations

import argparse
import html
import logging
import re
import sys
import time
import unicodedata
from html.parser import HTMLParser
from pathlib import Path
from typing import Callable
from urllib.parse import parse_qs, quote_plus, unquote, urlparse

import requests

from karaokifex.steps import lyrics
from karaokifex.timing import normalize

log = logging.getLogger(__name__)

USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/140.0 Safari/537.36"
SEARCH = "https://html.duckduckgo.com/html/?q={q}"
# the lyric sites worth a look, most trusted first; any other page among the results is still tried
SITES = ("songtexte.com", "genius.com", "lyrics.com", "songtexte-mania", "lyricstranslate.com", "musixmatch.com",
         "letras.", "azlyrics.com", "lyrics.az", "songlyrics.com", "liedtext", "songtext")
MAX_PAGES = 8
# a lyrics line: a few words, not a sentence of prose
LINE_WORDS = (1, 14)
MIN_LINES = 8
# where a known site keeps its lyrics: the attribute and value its lyrics' element carries
KNOWN = {
    "genius.com": ("data-lyrics-container", "true"),
    "songtexte.com": ("id", "lyrics"),
    "lyrics.com": ("id", "lyric-body-text"),
    "lyricstranslate.com": ("id", "song-body"),
}
LABEL = re.compile(r"^\s*[\[(]?\s*(chorus|refrain|verse|strophe|bridge|intro|outro|pre-?chorus|hook|interlude|"
                   r"\d+\.?\s*(strophe|verse)|x\d+|\d+x)\b[^\n]*[\])]?\s*:?\s*$", re.I)
JUNK = re.compile(r"(songtext|lyrics|writer|written by|komponist|text:|musik:|copyright|©|embed|you might also like|"
                  r"see .* live|get tickets|translation|übersetzung|contributors?|advertisement|cookie|javascript)", re.I)

Get = Callable[..., requests.Response]


class _Lines(HTMLParser):
    """A page as lines of text, breaking where the page breaks (br, p, div, li, headings); and the
    text inside the element marked `mark` (attribute, value) on its own, where there is one."""
    BREAK = {"br", "p", "div", "li", "h1", "h2", "h3", "h4", "tr", "section", "article"}
    SKIP = {"script", "style", "noscript", "svg", "head", "button", "nav", "footer", "form"}

    def __init__(self, mark: tuple[str, str] | None = None):
        super().__init__(convert_charrefs=True)
        self.mark, self.lines, self.marked = mark, [""], []
        self.depth_skip = 0
        self.in_mark = 0          # depth inside a marked element (0: outside)
        self.depth = 0

    def handle_starttag(self, tag, attrs):
        self.depth += 1
        if tag in self.SKIP:
            self.depth_skip += 1
        if self.in_mark:
            self.in_mark += 1
        elif self.mark and dict(attrs).get(self.mark[0]) == self.mark[1]:
            self.in_mark = 1
            self.marked.append("")
        if tag in self.BREAK:
            self._break()

    def handle_endtag(self, tag):
        self.depth -= 1
        if tag in self.SKIP and self.depth_skip:
            self.depth_skip -= 1
        if tag in self.BREAK:
            self._break()
        if self.in_mark:
            self.in_mark -= 1
            if not self.in_mark:
                self.marked.append("")

    def handle_startendtag(self, tag, attrs):
        if tag == "br":
            self._break()

    def handle_data(self, data):
        if self.depth_skip:
            return
        text = re.sub(r"\s+", " ", data)
        self.lines[-1] += text
        if self.in_mark:
            self.marked[-1] += text

    def _break(self):
        self.lines.append("")
        if self.in_mark:
            self.marked.append("")


def page_lines(page: str, mark: tuple[str, str] | None = None) -> tuple[list[str], list[str]]:
    """The page's lines, and those of its marked element (empty without one)."""
    parser = _Lines(mark)
    parser.feed(page)
    tidy = lambda ls: [line.strip() for line in ls]
    return tidy(parser.lines), tidy(parser.marked)


def _lyric_line(line: str) -> bool:
    words = len(line.split())
    return LINE_WORDS[0] <= words <= LINE_WORDS[1] and not line.endswith((".com", ".de", "»")) and not JUNK.search(line)


def longest_run(lines: list[str]) -> list[str]:
    """The longest run of short lines (blank lines between verses allowed, one at a time)."""
    best: list[str] = []
    run: list[str] = []
    for line in lines + ["\x00"]:
        if line and _lyric_line(line):
            run.append(line)
        elif not line and run and run[-1]:
            run.append("")
        else:
            words = sum(bool(x) for x in run)
            if words > sum(bool(x) for x in best):
                best = run
            run = []
    return best


def clean(lines: list[str], removed: list[str] | None = None) -> list[str]:
    """Labels, notes and junk out (each put in `removed`, if given); one blank line between verses, none at the ends."""
    out: list[str] = []
    for line in lines:
        line = unicodedata.normalize("NFC", html.unescape(line)).strip().strip("\u200b")
        # a label, a note in brackets on its own ([Instrumental], (Geigensolo)), or the site's own words
        if LABEL.match(line) or re.fullmatch(r"[\[(][^\])]*[\])]", line) or (line and JUNK.search(line) and len(line.split()) > 2):
            if removed is not None and line and line not in removed:
                removed.append(line)
            line = ""
        line = re.sub(r"\s*[\[(](x\d+|\d+x|chorus|refrain)[\])]\s*$", "", line, flags=re.I)
        if line or (out and out[-1]):
            out.append(line)
    while out and not out[-1]:
        out.pop()
    return out


def read_page(page: str, url: str) -> tuple[list[str], str, list[str]]:
    """The lyrics on a page (none: fewer than MIN_LINES), how they were found, and what the cleaning took out."""
    host = urlparse(url).netloc.lower()
    mark = next((m for site, m in KNOWN.items() if site in host), None)
    lines, marked = page_lines(page, mark)
    removed: list[str] = []
    if sum(bool(x) for x in marked) >= MIN_LINES:
        found, how = clean(marked, removed), f'its lyrics block (the element with {mark[0]}="{mark[1]}")'
    else:
        found, how = clean(longest_run(lines), removed), "the longest run of short lines on it"
    return (found if sum(bool(x) for x in found) >= MIN_LINES else []), how, removed


def lyrics_of(page: str, url: str) -> list[str]:
    return read_page(page, url)[0]


def page_text(r: requests.Response) -> tuple[str, str]:
    """The page as Unicode and the encoding it was read as: detected (lyrics.decode_text), the charset
    it names -- its header's, else its <meta>'s -- only a hint. (requests reads a page whose header names
    none as Latin-1, and "ü" comes out "Ã¼".)"""
    named = re.search(r"charset=([\w-]+)", r.headers.get("content-type", ""), re.I)
    meta = re.search(rb'<meta[^>]+charset=["\']?([\w-]+)', r.content[:4096], re.I)
    declared = named.group(1) if named else meta.group(1).decode("ascii") if meta else None
    return lyrics.decode_text(r.content, declared)


def about_song(url: str, page: str, song: str) -> bool:
    """Whether a page is the song's own -- its address or its title names the song (most of its words) --
    and not an artist's or an album's page, whose list of song titles reads like verses."""
    title = re.search(r"<title[^>]*>(.*?)</title>", page, re.I | re.S)
    said = {normalize(w) for w in re.split(r"[\W_]+", unquote(urlparse(url).path) + " " + (html.unescape(title.group(1)) if title else ""))}
    words = [w for w in (normalize(w) for w in song.split()) if len(w) > 2] or [normalize(song)]
    return sum(w in said for w in words) * 2 >= len(words)


def _words(lines: list[str]) -> set[str]:
    return {w for line in lines for w in re.findall(r"\w+", line.lower()) if len(w) > 2}


def agreement(a: list[str], b: list[str]) -> float:
    """How far two versions agree: the words they share, of all the words either has (0..1)."""
    x, y = _words(a), _words(b)
    return len(x & y) / max(1, len(x | y))


def pick(candidates: list[tuple[str, list[str]]]) -> tuple[str, list[str]] | None:
    """The version most of the others agree with (their words in common); alone, the first."""
    if not candidates:
        return None
    total = lambda i: sum(agreement(candidates[i][1], c[1]) for j, c in enumerate(candidates) if j != i)
    return candidates[max(range(len(candidates)), key=lambda i: (total(i), -i))]


def search(artist: str, song: str, *, get: Get = requests.get) -> list[str]:
    """Result links for the song's lyrics, the known lyric sites first, each page once."""
    links: list[str] = []
    for words in ("lyrics", "songtext"):
        q = quote_plus(f"{artist} {song} {words}")
        try:
            r = get(SEARCH.format(q=q), headers={"User-Agent": USER_AGENT}, timeout=20)
            r.raise_for_status()
        except requests.RequestException as error:
            log.warning("search failed (%s): %s", words, error)
            continue
        for href in re.findall(r'class="result__a"[^>]*href="([^"]+)"', r.text):
            href = html.unescape(href)
            if "uddg=" in href:          # DuckDuckGo's redirect: the real link is in it
                href = unquote(parse_qs(urlparse(href).query).get("uddg", [href])[0])
            if href.startswith("http") and href not in links:
                links.append(href)
    known = [link for site in SITES for link in links if site in urlparse(link).netloc.lower()]
    return list(dict.fromkeys(known + links))[:MAX_PAGES]


def find(artist: str, song: str, *, get: Get = requests.get) -> tuple[tuple[str, list[str]] | None, list[dict]]:
    """The version kept, (url, lines), or None; and every page looked at, with its lines or why none."""
    candidates, pages = [], []
    for url in search(artist, song, get=get):
        try:
            r = get(url, headers={"User-Agent": USER_AGENT}, timeout=20)
            r.raise_for_status()
        except requests.RequestException as error:
            status = getattr(getattr(error, "response", None), "status_code", None)
            pages.append({"url": url, "why": f"HTTP {status}" if status else type(error).__name__})
            continue
        if r.status_code != 200:     # 202 and an empty page: a bot check, not got past
            pages.append({"url": url, "why": f"HTTP {r.status_code}, no page (a bot check)"})
            continue
        text, encoding = page_text(r)
        if not about_song(url, text, song):
            pages.append({"url": url, "why": "not the song's page (its address and title don't name it)"})
            continue
        lines, how, removed = read_page(text, url)
        how += f", read as {encoding}"
        log.info("%s: %d lines", url, sum(bool(x) for x in lines))
        if lines:
            candidates.append((url, lines))
            pages.append({"url": url, "lines": lines, "how": how, "removed": removed})
        else:
            pages.append({"url": url, "why": f"fewer than {MIN_LINES} lines, by {how}"})
    return pick(candidates), pages


def count(lines: list[str]) -> int:
    return sum(bool(x) for x in lines)


def web_header(name: str, lrclib: str, kept: str, pages: list[dict]) -> list[str]:
    """Where lyrics found on the web came from and how, as # lines (karaokifex --lyrics-file skips them)."""
    got = next(p for p in pages if p["url"] == kept)
    out = [f"# {name}: lyrics found on the web by karaokifex-lyrics-web, {time.strftime('%Y-%m-%d')}",
           "# (lines starting with # are notes; karaokifex --lyrics-file skips them)",
           f"# lrclib: {lrclib}",
           f"# taken from: {kept}",
           f"#   found by: {got['how']}; {count(got['lines'])} lines"]
    if got["removed"]:
        out.append(f"#   cleaned out: {' | '.join(got['removed'][:12])}" + (" | ..." if len(got["removed"]) > 12 else ""))
    others = [p for p in pages if p["url"] != kept]
    for p in others:
        if "lines" in p:
            out.append(f"# agrees {agreement(got['lines'], p['lines']):.0%}: {p['url']} ({count(p['lines'])} lines, by {p['how']})")
    for p in others:
        if "why" in p:
            out.append(f"# looked at, nothing: {p['url']} ({p['why']})")
    return out


def lrc_line(line: lyrics.LyricLine) -> str:
    minutes, seconds = divmod(line.start or 0.0, 60)
    return f"[{int(minutes):02d}:{seconds:05.2f}]{line.text}"


def safe(name: str) -> str:
    return re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", name).strip(" .") or "lyrics"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="karaokifex-lyrics-web", description=__doc__.split("\n\n")[0])
    parser.add_argument("name", help='"Artist - Song"')
    parser.add_argument("-o", "--output", type=Path, help="the text file to write (default: <Artist - Song>.txt)")
    parser.add_argument("--show", action="store_true", help="print the lyrics found too")
    parser.add_argument("-v", "--verbose", action="store_true")
    a = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO if a.verbose else logging.WARNING, format="%(message)s")
    for stream in (sys.stdout, sys.stderr):      # a Windows console or pipe is cp1252
        stream.reconfigure(encoding="utf-8", errors="replace")
    artist, _, song = a.name.partition(" - ")
    if not song:
        parser.error('the name is "Artist - Song"')
    artist, song = unicodedata.normalize("NFC", artist.strip()), unicodedata.normalize("NFC", song.strip())
    target = a.output or Path(f"{safe(a.name)}.txt")
    # the lyrics database first, always, as karaokifex does: the web only for a song lrclib hasn't got
    known = lyrics.fetch_lyrics(artist, song, None)
    if known:
        best = known[0]
        kind = "synced" if best.synced else "plain"
        head = [f"# {a.name}: lyrics from lrclib #{best.lrclib_id} ({kind}: {best.artist} - {best.track}), "
                f"{time.strftime('%Y-%m-%d')}; the web not searched",
                "# (lines starting with # are notes; karaokifex --lyrics-file skips them; karaokifex asks lrclib itself anyway)"]
        lines = [lrc_line(line) if best.synced else line.text for line in best.lines]
        target.write_text("\n".join(head + lines) + "\n", encoding="utf-8")
        print(f"{target}: {len(lines)} lines, from lrclib #{best.lrclib_id} ({kind}); the web not searched")
    else:
        found, pages = find(artist, song)
        if not found:
            print(f"no lyrics for {a.name}: none on lrclib, and none readable on the web", file=sys.stderr)
            for p in pages:
                print(f"  {p['url']}: {p.get('why') or 'read'}", file=sys.stderr)
            return 1
        url, lines = found
        head = web_header(a.name, "nothing for it (asked first)", url, pages)
        target.write_text("\n".join(head + lines) + "\n", encoding="utf-8")
        print(f"{target}: {count(lines)} lines, from {url}")
    if a.show:
        print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
