"""Press photos of an artist, found on the web, for a video made from pictures (karaokifex-bandcamp).

    press_photos(artist, into, bandcamp=host, limit=6, cache=dir) -> [Photo(path, image, page, how), ...]

First the artist's own: the band photo on their Bandcamp page, and their picture on Deezer (its open
API: the artist's own image, 1000x1000). Then pages about them: the official links MusicBrainz knows
(its open API), and a web search (DuckDuckGo's HTML search, which allows it, a query every two
seconds) for press, their label, their own sites -- keeping only pages whose title or address names
them -- and from each page its share picture (og:image) and
the photos in it that name the artist in their alt text or file name, or that a camera named (DSC_8332.jpg:
in an article about the artist, most likely a photo of them). A site whose robots.txt says
no is left alone; nothing that asks for proof of being a person is answered.

Kept: photos large enough for a 1080p frame (the short side at least MIN_SHORT pixels), not an album
cover (Bandcamp's a<number> images: the video has its cover already) nor a picture whose name or alt
text calls it a cover, artwork, a vinyl or product shot, a poster or a logo, and not a near copy of one
already kept (an 8x8 average hash, the same picture at another size or crop). Each one says where it
was found (page) and how (the band photo, a page's share picture, a photo on the page).

With a cache folder, the photos found for an artist are kept there with where each came from, and the
next track by the same artist takes them from it rather than searching again (for CACHE_DAYS).
"""
from __future__ import annotations

import hashlib
import html
import json
import logging
import re
import shutil
import subprocess
import time
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from typing import Callable
from urllib.parse import parse_qs, quote_plus, unquote, urljoin, urlparse
from urllib.robotparser import RobotFileParser

import requests

log = logging.getLogger(__name__)

USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/140.0 Safari/537.36"
SEARCH = "https://html.duckduckgo.com/html/?q={q}"
QUERIES = ('"{artist}" band', '"{artist}" band press photo', '"{artist}" interview', '"{artist}" live', '"{artist}" new single')
# sites whose pages need a login or build themselves in the browser: nothing to read there
SKIP = ("bandcamp.com", "youtube.", "youtu.be", "spotify.", "music.apple.", "instagram.", "facebook.", "tiktok.",
        "twitter.", "x.com", "soundcloud.", "deezer.", "tidal.", "amazon.", "shazam.")
MAX_PAGES = 14
PAUSE = 2.0              # seconds between two web searches
CACHE_DAYS = 14
MB_AGENT = "karaokifex/0.1 (https://github.com/samluescher/karaokifex)"
MAX_CANDIDATES = 24
MIN_SHORT = 800          # pixels on a photo's short side: a 1080p frame shows it without blowing it up blurry
MAX_BYTES = 15 * 2**20
NEAR = 10                # bits apart (of 64) under which two average hashes are one picture
ALBUM_ART = re.compile(r"(^|/|_)a\d{8,}_\d+")
# a photo as a camera names it (DSC_8332, IMG_1234, _MG_0042, DSCF1234, P1010001, 0Z5A9141): on a page about the
# artist, most likely a photo of them, though its name doesn't say so
CAMERA = re.compile(r"^(dsc[_f]?\d|img[_-]?\d|_mg_\d|dscf\d|p\d{7}|[0-9a-z]{4}\d{4}\b|\d{8}[_-]\d{6})", re.I)
# what a picture of a record, a product or a gig is called: a cover, artwork, a vinyl shot, merch, a poster...
NOT_A_PHOTO = re.compile(r"(cover|artwork|art[-_ ]work|vinyl|\blp\b|\bcd\b|merch|product|packshot|tracklist|poster|flyer|logo|banner)", re.I)
Get = Callable[..., requests.Response]


@dataclass
class Photo:
    path: Path
    image: str          # where the picture itself was
    page: str           # the page it was found on
    how: str            # the band photo, a page's share picture, a photo on the page


def plain(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", html.unescape(text or "").lower()).strip()


def names(artist: str, *texts: str) -> bool:
    """Whether any of the texts names the artist: the whole name, as words in a title or an address."""
    a = plain(artist)
    return bool(a) and any(f" {a} " in f" {plain(unquote(t))} " for t in texts if t)


class _Images(HTMLParser):
    """A page's title, its share picture (og:image, twitter:image) and its <img>s with their alt text."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.title, self.share, self.images, self._in_title = "", [], [], False

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag == "title":
            self._in_title = True
        elif tag == "meta" and (a.get("property") or a.get("name") or "").lower() in ("og:image", "twitter:image") and a.get("content"):
            self.share.append(a["content"])
        elif tag == "img":
            # the largest of a srcset, else the src (a lazy page's data-src)
            src = a.get("src") or a.get("data-src") or ""
            if a.get("srcset"):
                sizes = [(int(m.group(2)), m.group(1)) for m in re.finditer(r"(\S+)\s+(\d+)w", a["srcset"])]
                if sizes:
                    src = max(sizes)[1]
            if src:
                self.images.append((src, a.get("alt") or ""))

    def handle_endtag(self, tag):
        if tag == "title":
            self._in_title = False

    def handle_data(self, data):
        if self._in_title:
            self.title += data


def search(artist: str, *, get: Get = requests.get) -> list[tuple[str, str]]:
    """(page, title) for the web's pages about the artist, those that name them, each once."""
    found: dict[str, str] = {}
    for n, query in enumerate(QUERIES):
        if n:
            time.sleep(PAUSE)
        try:
            r = get(SEARCH.format(q=quote_plus(query.format(artist=artist))), headers={"User-Agent": USER_AGENT}, timeout=20)
            r.raise_for_status()
        except requests.RequestException as error:
            log.info("search failed: %s", error)
            continue
        # asked to prove it's a person (an "anomaly" page): not answered; no more searching for now
        if r.status_code == 202 or "anomaly" in r.text[:20000].lower() and "result__a" not in r.text:
            log.warning("the web search asks for proof of being a person: no more searching now (the photos found so far stay)")
            break
        for href, title in re.findall(r'class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>', r.text, re.S):
            href = html.unescape(href)
            if "uddg=" in href:
                href = unquote(parse_qs(urlparse(href).query).get("uddg", [href])[0])
            title = re.sub(r"<[^>]+>", "", html.unescape(title))
            host = urlparse(href).netloc.lower()
            if href.startswith("http") and not any(s in host for s in SKIP) and names(artist, title, href):
                found.setdefault(href, title)
    return list(found.items())[:MAX_PAGES]


_robots: dict[str, RobotFileParser | None] = {}


def allowed(url: str, *, get: Get = requests.get) -> bool:
    """What the site's robots.txt says of fetching this page (no robots.txt: yes)."""
    u = urlparse(url)
    root = f"{u.scheme}://{u.netloc}"
    if root not in _robots:
        try:
            r = get(f"{root}/robots.txt", headers={"User-Agent": USER_AGENT}, timeout=10)
            rp = RobotFileParser()
            rp.parse(r.text.splitlines() if r.status_code == 200 else [])
            _robots[root] = rp
        except requests.RequestException:
            _robots[root] = None
    rp = _robots[root]
    return rp is None or rp.can_fetch(USER_AGENT, url)


def band_photo(host: str, *, get: Get = requests.get) -> str | None:
    """The band photo on the artist's Bandcamp page, at its full size."""
    try:
        r = get(f"https://{host}/", headers={"User-Agent": USER_AGENT}, timeout=20)
        r.raise_for_status()
    except requests.RequestException:
        return None
    m = re.search(r'class="popupImage"[^>]*href="([^"]+)"', r.text) or re.search(r'<img[^>]*src="([^"]+)"[^>]*class="band-photo"', r.text)
    return re.sub(r"_\d+\.(jpg|png)$", r"_0.\1", m.group(1)) if m else None


def deezer_photo(artist: str, *, get: Get = requests.get) -> str | None:
    """The artist's picture on Deezer (its open API), at 1000x1000; none for a name it doesn't have exactly, or
    for its placeholder (an image with no id)."""
    try:
        r = get("https://api.deezer.com/search/artist", params={"q": artist}, timeout=20)
        r.raise_for_status()
        hit = next((a for a in r.json().get("data", []) if plain(a.get("name", "")) == plain(artist)), None)
    except (requests.RequestException, ValueError):
        return None
    url = (hit or {}).get("picture_xl") or ""
    return url if "/artist/" in url and "/artist//" not in url else None


def official_pages(artist: str, *, get: Get = requests.get) -> list[tuple[str, str]]:
    """The artist's official links on MusicBrainz (its open API: an exact name, its best match), as pages to read."""
    try:
        r = get("https://musicbrainz.org/ws/2/artist/", params={"query": f'artist:"{artist}"', "fmt": "json", "limit": 3},
                headers={"User-Agent": MB_AGENT}, timeout=20)
        r.raise_for_status()
        hit = next((a for a in r.json().get("artists", []) if plain(a.get("name", "")) == plain(artist) and a.get("score", 0) >= 95), None)
        if not hit:
            return []
        time.sleep(1.1)                       # MusicBrainz asks for a request a second at most
        r = get(f"https://musicbrainz.org/ws/2/artist/{hit['id']}", params={"inc": "url-rels", "fmt": "json"},
                headers={"User-Agent": MB_AGENT}, timeout=20)
        r.raise_for_status()
    except (requests.RequestException, ValueError):
        return []
    urls = [rel.get("url", {}).get("resource", "") for rel in r.json().get("relations", [])
            if rel.get("type") in ("official homepage", "social network", "image", "fanpage", "biography", "interview")]
    return [(u, "an official link on MusicBrainz") for u in urls if u.startswith("http") and not any(s in urlparse(u).netloc for s in SKIP)]


def candidates(artist: str, pages: list[tuple[str, str]], *, get: Get = requests.get) -> list[tuple[str, str, str]]:
    """(image, page, how) from the pages: each one's share picture, and its photos naming the artist."""
    out: list[tuple[str, str, str]] = []
    for page, _ in pages:
        if not allowed(page, get=get):
            log.info("%s: its robots.txt says no", page)
            continue
        try:
            r = get(page, headers={"User-Agent": USER_AGENT}, timeout=20)
            r.raise_for_status()
        except requests.RequestException as error:
            log.info("%s: %s", page, error)
            continue
        p = _Images()
        try:
            p.feed(r.text)
        except Exception:  # noqa: BLE001 -- a page that doesn't parse gives nothing
            continue
        for src in p.share:
            out.append((urljoin(page, src), page, "the page's share picture"))
        for src, alt in p.images:
            if NOT_A_PHOTO.search(alt):
                continue
            name = unquote(Path(urlparse(src).path).name)
            if names(artist, alt, name.replace("-", " ").replace("_", " ")):
                out.append((urljoin(page, src), page, "a photo on the page"))
            elif CAMERA.match(name):
                out.append((urljoin(page, src), page, "a camera's photo on a page about them"))
    seen, unique = set(), []
    for c in out:
        name = unquote(Path(urlparse(c[0]).path).name)
        if c[0] not in seen and not ALBUM_ART.search(urlparse(c[0]).path) and not NOT_A_PHOTO.search(name):
            seen.add(c[0])
            unique.append(c)
    return unique[:MAX_CANDIDATES]


def size_of(path: Path, ffprobe: str = "ffprobe") -> tuple[int, int]:
    out = subprocess.run([ffprobe, "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height",
                          "-of", "csv=p=0", str(path)], capture_output=True, text=True).stdout.strip()
    try:
        w, h = (int(x) for x in out.split(",")[:2])
        return w, h
    except ValueError:
        return 0, 0


def ahash(path: Path, ffmpeg: str = "ffmpeg") -> int | None:
    """An 8x8 average hash: the picture's light, bit by bit, above or below its mean."""
    raw = subprocess.run([ffmpeg, "-v", "error", "-i", str(path), "-vf", "scale=8:8:flags=area,format=gray", "-frames:v", "1",
                          "-f", "rawvideo", "-"], capture_output=True).stdout
    if len(raw) < 64:
        return None
    mean = sum(raw[:64]) / 64
    return sum(1 << i for i, v in enumerate(raw[:64]) if v > mean)


def near(a: int, b: int) -> bool:
    return bin(a ^ b).count("1") <= NEAR


def press_photos(artist: str, into: Path, *, bandcamp: str | None = None, cover: Path | None = None, limit: int = 6,
                 cache: Path | None = None, get: Get = requests.get, ffmpeg: str = "ffmpeg", ffprobe: str = "ffprobe") -> list[Photo]:
    """The artist's press photos, downloaded into `into` (see the top)."""
    into.mkdir(parents=True, exist_ok=True)
    if cache and (kept := from_cache(cache, into)) is not None:
        log.info("%d photos of %s from the cache (%s)", len(kept), artist, cache)
        return kept[:limit]
    found: list[tuple[str, str, str]] = []
    if bandcamp and (bp := band_photo(bandcamp, get=get)):
        found.append((bp, f"https://{bandcamp}/", "the band photo on their Bandcamp page"))
    if dz := deezer_photo(artist, get=get):
        found.append((dz, "https://www.deezer.com/", "their picture on Deezer"))
    found += candidates(artist, official_pages(artist, get=get) + search(artist, get=get), get=get)
    kept: list[Photo] = []
    # the cover is in the video already: a copy of it found on a page is no new photo
    hashes = [h for h in [ahash(cover, ffmpeg) if cover else None] if h is not None]
    for image, page, how in found:
        if len(kept) >= limit:
            break
        try:
            r = get(image, headers={"User-Agent": USER_AGENT, "Referer": page}, timeout=30)
            r.raise_for_status()
        except requests.RequestException:
            continue
        if not r.headers.get("content-type", "").startswith("image/") or len(r.content) > MAX_BYTES:
            continue
        ext = {"image/png": ".png", "image/webp": ".webp"}.get(r.headers["content-type"].split(";")[0], ".jpg")
        path = into / f"photo-{hashlib.sha1(image.encode()).hexdigest()[:10]}{ext}"
        path.write_bytes(r.content)
        w, h = size_of(path, ffprobe)
        hsh = ahash(path, ffmpeg)
        if min(w, h) < MIN_SHORT or hsh is None or any(near(hsh, x) for x in hashes):
            path.unlink(missing_ok=True)
            continue
        hashes.append(hsh)
        kept.append(Photo(path, image, page, how))
        log.info("photo %s (%dx%d) from %s", image, w, h, page)
    if cache:
        to_cache(cache, kept)
    return kept


def to_cache(cache: Path, photos: list[Photo]) -> None:
    """The photos found for an artist, kept with where each came from (photos.json)."""
    cache.mkdir(parents=True, exist_ok=True)
    for p in photos:
        shutil.copy2(p.path, cache / p.path.name)
    (cache / "photos.json").write_text(json.dumps({"found": time.time(), "photos": [
        {"file": p.path.name, "image": p.image, "page": p.page, "how": p.how} for p in photos]}, indent=1), encoding="utf-8")


def from_cache(cache: Path, into: Path) -> list[Photo] | None:
    """The photos kept for an artist, copied into `into`; None when there are none, or they are older than CACHE_DAYS."""
    try:
        doc = json.loads((cache / "photos.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if time.time() - doc.get("found", 0) > CACHE_DAYS * 86400:
        return None
    out = []
    for p in doc.get("photos", []):
        if (cache / p["file"]).exists():
            shutil.copy2(cache / p["file"], into / p["file"])
            out.append(Photo(into / p["file"], p["image"], p["page"], p["how"]))
    return out
