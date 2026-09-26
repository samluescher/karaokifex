"""Lyrics from the web, for a song lrclib hasn't got (karaokifex-lyrics-web).

    karaokifex-lyrics-web "Artist - Song" [-o lyrics.txt] [--show]

Searches the web for the song's lyrics ("<artist> <song> lyrics", and "songtext" too, where songs in
German and Swiss German are written down), fetches the pages of the lyric sites among the results,
and takes the lyrics out of each: the page's own lyrics block where the site is known, else the
longest run of short lines on it. Each is cleaned -- section labels ([Chorus], Refrain:), notes,
credits and the site's own lines out, a blank line between verses -- and the one most of the others
agree with is kept (a site's version that nobody else has is likelier wrong). Written as a flat text
file, a line a line, to hand karaokifex --lyrics-file, which aligns it to the voice like lrclib's.

A miss exits 1 and writes nothing; karaokifex then transcribes the voice itself.
"""
from __future__ import annotations

import argparse
import html
import logging
import re
import sys
from html.parser import HTMLParser
from pathlib import Path
from typing import Callable
from urllib.parse import parse_qs, quote_plus, unquote, urlparse

import requests

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


def clean(lines: list[str]) -> list[str]:
    """Labels, notes and junk out; one blank line between verses, none at the ends."""
    out: list[str] = []
    for line in lines:
        line = html.unescape(line).strip().strip("​")
        # a label, a note in brackets on its own ([Instrumental], (Geigensolo)), or the site's own words
        if LABEL.match(line) or re.fullmatch(r"[\[(][^\])]*[\])]", line) or (line and JUNK.search(line) and len(line.split()) > 2):
            line = ""
        line = re.sub(r"\s*[\[(](x\d+|\d+x|chorus|refrain)[\])]\s*$", "", line, flags=re.I)
        if line or (out and out[-1]):
            out.append(line)
    while out and not out[-1]:
        out.pop()
    return out


def lyrics_of(page: str, url: str) -> list[str]:
    host = urlparse(url).netloc.lower()
    mark = next((m for site, m in KNOWN.items() if site in host), None)
    lines, marked = page_lines(page, mark)
    found = clean(marked) if sum(bool(x) for x in marked) >= MIN_LINES else clean(longest_run(lines))
    return found if sum(bool(x) for x in found) >= MIN_LINES else []


def _words(lines: list[str]) -> set[str]:
    return {w for line in lines for w in re.findall(r"\w+", line.lower()) if len(w) > 2}


def pick(candidates: list[tuple[str, list[str]]]) -> tuple[str, list[str]] | None:
    """The version most of the others agree with (their words in common); alone, the first."""
    if not candidates:
        return None
    sets = [_words(lines) for _, lines in candidates]
    def agreement(i: int) -> float:
        return sum(len(sets[i] & sets[j]) / max(1, len(sets[i] | sets[j])) for j in range(len(sets)) if j != i)
    return candidates[max(range(len(candidates)), key=lambda i: (agreement(i), -i))]


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


def find(artist: str, song: str, *, get: Get = requests.get) -> tuple[str, list[str]] | None:
    candidates = []
    for url in search(artist, song, get=get):
        try:
            r = get(url, headers={"User-Agent": USER_AGENT}, timeout=20)
            r.raise_for_status()
        except requests.RequestException as error:
            log.info("%s: %s", url, error)
            continue
        lines = lyrics_of(r.text, url)
        log.info("%s: %d lines", url, sum(bool(x) for x in lines))
        if lines:
            candidates.append((url, lines))
    return pick(candidates)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="karaokifex-lyrics-web", description=__doc__.split("\n\n")[0])
    parser.add_argument("name", help='"Artist - Song"')
    parser.add_argument("-o", "--output", type=Path, help="the text file to write (default: <Artist - Song>.txt)")
    parser.add_argument("--show", action="store_true", help="print the lyrics found too")
    parser.add_argument("-v", "--verbose", action="store_true")
    a = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO if a.verbose else logging.WARNING, format="%(message)s")
    artist, _, song = a.name.partition(" - ")
    if not song:
        parser.error('the name is "Artist - Song"')
    found = find(artist.strip(), song.strip())
    if not found:
        print(f"no lyrics found on the web for {a.name}", file=sys.stderr)
        return 1
    url, lines = found
    target = a.output or Path(f"{a.name}.txt")
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"{target}: {sum(bool(x) for x in lines)} lines, from {url}")
    if a.show:
        print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
