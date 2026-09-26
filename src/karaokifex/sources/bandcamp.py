"""A Bandcamp track as a video, for a song that has no video of its own (karaokifex-bandcamp).

    karaokifex-bandcamp <track url> [-o dir] [--height 1080] [--photos 6]

Bandcamp gives a track's sound and its cover art, and the web has press photos of the artist
(photos.py: their Bandcamp band photo first, then pages about them). This makes a video of them for
karaokifex, which then treats it like any other: a slow slideshow for as long as the track lasts --
the cover first, then the photos, each SLIDE seconds, whole over a blurred copy of itself filling the
frame, slowly zooming in or out, crossfading into the next, the cover coming round again -- with the
track's sound. With no photos found, the cover alone, zooming slowly. Beside it,
<file>.info.json says what the source said -- its key (bandcamp:<host>/track/<name>), title,
artist, track, album, duration -- that the video was made from stills (made: still), which
karaokifex reads for a local file and keeps in the song's info.json, so a player knows the picture
is slow and calm and may liven it up -- and where each picture came from (photos: the image, the
page it was on, how it was found). Each picture is composed once and zoomed from there, so a track
takes a fraction of its length to make.

    karaokifex "<the video>" -a Artist -s Song ...      (or Compute's make_song.py with it)

The lyrics: karaokifex looks on lrclib as always, and karaokifex-lyrics-web finds some on the web
for --lyrics-file; with none anywhere, karaokifex transcribes the voice it separates.
Prints the video's path.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from urllib.parse import urlparse

import requests
import yt_dlp

from karaokifex.sources import photos as press


def ffprobe_of(ffmpeg: str) -> str:
    """The ffprobe beside an ffmpeg (its own folder), else the one on the path."""
    p = Path(ffmpeg)
    probe = p.with_name(p.name.replace("ffmpeg", "ffprobe"))
    return str(probe) if probe != p and probe.exists() else shutil.which("ffprobe") or "ffprobe"

HEIGHT = 1080
FPS = 25
ZOOM = 0.12          # how far in the cover zooms over the track (1 + ZOOM at its end)
SLIDE = 20           # seconds a picture is on in a slideshow, its crossfades included
FADE = 2             # seconds of crossfade between two pictures
SLIDE_ZOOM = 0.08    # how far a picture zooms in (or out) while it is on
COVER_EVERY = 4      # the cover comes round again after this many photos


def source_key(url: str) -> str:
    """The library's key for a Bandcamp track: the lowercased host and the track's path, no query."""
    u = urlparse(url)
    return f"bandcamp:{u.netloc.lower()}{u.path.rstrip('/').lower()}"


def safe(name: str) -> str:
    return re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", name).strip(" .") or "track"


def cover_url(info: dict) -> str | None:
    """The cover at its full size: Bandcamp's image address with the size code _0 (the original)."""
    thumbs = [t.get("url") for t in info.get("thumbnails") or [] if t.get("url")] or [info.get("thumbnail")]
    url = next((t for t in reversed(thumbs) if t), None)
    return re.sub(r"_\d+\.(jpg|png)$", r"_0.\1", url) if url else None


def still_filter(height: int) -> str:
    """The picture, once: the cover whole over a blurred copy of itself filling a 16:9 frame, at twice
    the video's size, so the zoom that follows has pixels to spare."""
    w, h = height * 32 // 9, height * 2
    return (f"[0:v]scale={w}:{h}:force_original_aspect_ratio=increase,crop={w}:{h},boxblur=80:2,eq=brightness=-0.12[bg];"
            f"[0:v]scale={h}:{h}:force_original_aspect_ratio=decrease[fg];[bg][fg]overlay=(W-w)/2:(H-h)/2[v]")


def video_filter(height: int, seconds: float) -> str:
    """That picture, zooming in slowly and drifting for the track's length: every frame from the one
    picture decoded once (zoompan's d), rather than the cover decoded, scaled and blurred afresh 25
    times a second, which took a track ten times its length."""
    width = height * 16 // 9
    frames = max(1, int(seconds * FPS) + FPS)
    zoom = f"1+{ZOOM}*on/{frames}"
    return (f"[0:v]zoompan=z='{zoom}':x='iw/2-(iw/zoom/2)+sin(on/{FPS * 9})*iw*0.02':y='ih/2-(ih/zoom/2)+cos(on/{FPS * 11})*ih*0.02'"
            f":d={frames}:s={width}x{height}:fps={FPS},format=yuv420p[v]")


def slides(duration: float, count: int, slide: float = SLIDE, fade: float = FADE) -> int:
    """How many slides of `slide` seconds, each overlapping the last by `fade`, cover `duration`."""
    return max(1, math.ceil((duration - fade) / (slide - fade)))


def order(photos: int, n: int) -> list[int]:
    """Which picture each of n slides shows: 0 the cover, 1.. the photos, the cover first and again after every
    COVER_EVERY photos, the photos in turn."""
    if not photos:
        return [0] * n
    out, p = [], 0
    for k in range(n):
        if k % (COVER_EVERY + 1) == 0:
            out.append(0)
        else:
            out.append(1 + p % photos)
            p += 1
    return out


def slideshow_filter(height: int, n: int, slide: float = SLIDE, fade: float = FADE) -> str:
    """n composed pictures (inputs 0..n-1, each one frame) as a slideshow: each zoompan'd from its one decoded
    frame for `slide` seconds -- in on the even ones, out on the odd, drifting a little -- crossfading into the next
    at k * (slide - fade) seconds."""
    width = height * 16 // 9
    frames = int(slide * FPS)
    parts = []
    for k in range(n):
        z = f"1+{SLIDE_ZOOM}*on/{frames}" if k % 2 == 0 else f"1+{SLIDE_ZOOM}-{SLIDE_ZOOM}*on/{frames}"
        drift = f"(on/{frames}-0.5)*iw*{0.03 if k % 4 < 2 else -0.03}"
        parts.append(f"[{k}:v]zoompan=z='{z}':x='iw/2-(iw/zoom/2)+{drift}':y='ih/2-(ih/zoom/2)':d={frames}"
                     f":s={width}x{height}:fps={FPS},setsar=1,format=yuv420p[s{k}]")
    if n == 1:
        parts.append("[s0]null[v]")
        return ";".join(parts)
    last = "s0"
    for k in range(1, n):
        label = "v" if k == n - 1 else f"x{k}"
        parts.append(f"[{last}][s{k}]xfade=transition=fade:duration={fade}:offset={k * (slide - fade):.3f}[{label}]")
        last = label
    return ";".join(parts)


def make(url: str, out: Path, *, height: int = HEIGHT, ffmpeg: str = "ffmpeg", photos: int = 6) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="karaokifex-bandcamp-") as tmp:
        tmp_dir = Path(tmp)
        options = {"quiet": True, "noprogress": True, "noplaylist": True, "format": "bestaudio",
                   "outtmpl": str(tmp_dir / "audio.%(ext)s")}
        with yt_dlp.YoutubeDL(options) as ydl:
            info = ydl.sanitize_info(ydl.extract_info(url, download=True))
        audio = next(tmp_dir.glob("audio.*"))
        artist = info.get("artist") or info.get("uploader") or info.get("creator") or ""
        track = info.get("track") or info.get("title") or "track"
        duration = float(info.get("duration") or 0)
        art = cover_url(info)
        if not art:
            raise SystemExit(f"{url}: no cover art to make a picture of")
        cover = tmp_dir / f"cover{Path(urlparse(art).path).suffix or '.jpg'}"
        response = requests.get(art, timeout=30)
        response.raise_for_status()
        cover.write_bytes(response.content)
        target = out / f"{safe(f'{artist} - {track}')}.mp4"
        # the artist's press photos, and each picture composed once: the cover first
        # (kept per artist beside the videos, so the next track of theirs doesn't search again)
        found = press.press_photos(artist, tmp_dir / "photos", bandcamp=urlparse(url).netloc.lower(), cover=cover,
                                   limit=photos, cache=out / ".press-photos" / safe(artist), ffmpeg=ffmpeg,
                                   ffprobe=ffprobe_of(ffmpeg)) if photos else []
        stills = []
        for i, picture in enumerate([cover, *(p.path for p in found)]):
            still = tmp_dir / f"still-{i}.png"
            subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-i", str(picture), "-filter_complex", still_filter(height),
                            "-map", "[v]", "-frames:v", "1", str(still)], check=True)
            stills.append(still)
        seconds = duration or 600
        if found:
            n = slides(seconds, len(found))
            shows = [stills[i] for i in order(len(found), n)]
            inputs = [arg for s in shows for arg in ("-i", str(s))]
            graph = slideshow_filter(height, n)
        else:
            shows, inputs, graph = [stills[0]], ["-i", str(stills[0])], video_filter(height, seconds)
        command = [ffmpeg, "-y", "-loglevel", "error", *inputs, "-i", str(audio), "-filter_complex", graph,
                   "-map", "[v]", "-map", f"{len(shows)}:a", "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
                   "-tune", "stillimage", "-c:a", "aac", "-b:a", "256k", "-shortest", "-movflags", "+faststart",
                   str(target)]
        subprocess.run(command, check=True)
    sidecar = {"id": source_key(url), "title": f"{artist} - {track}", "artist": artist, "track": track,
               "album": info.get("album"), "uploader": info.get("uploader") or artist, "duration": duration or None,
               "webpage_url": url, "made": "still", "cover": art,
               "photos": [{"image": p.image, "page": p.page, "how": p.how} for p in found]}
    Path(f"{target}.info.json").write_text(json.dumps(sidecar, indent=1, ensure_ascii=False), encoding="utf-8")
    return target


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="karaokifex-bandcamp", description=__doc__.split("\n\n")[0])
    parser.add_argument("url", help="a Bandcamp track's page")
    parser.add_argument("-o", "--output-dir", type=Path, default=Path("."))
    parser.add_argument("--height", type=int, default=HEIGHT, help=f"the video's height (default {HEIGHT})")
    parser.add_argument("--ffmpeg", default=shutil.which("ffmpeg") or "ffmpeg")
    parser.add_argument("--photos", type=int, default=6, help="press photos of the artist to look for (default 6; 0: the cover alone)")
    parser.add_argument("-v", "--verbose", action="store_true")
    a = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO if a.verbose else logging.WARNING, format="%(message)s")
    print(make(a.url, a.output_dir, height=a.height, ffmpeg=a.ffmpeg, photos=a.photos))
    return 0


if __name__ == "__main__":
    sys.exit(main())
