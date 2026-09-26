"""A Bandcamp track as a video, for a song that has no video of its own (karaokifex-bandcamp).

    karaokifex-bandcamp <track url> [-o dir] [--height 1080]

Bandcamp gives a track's sound and its cover art. This makes a video of them for karaokifex, which
then treats it like any other: the cover, whole, over a blurred copy of itself filling the frame,
slowly zooming in and drifting, for as long as the track lasts, with the track's sound. Beside it,
<file>.info.json says what the source said -- its key (bandcamp:<host>/track/<name>), title,
artist, track, album, duration -- which karaokifex reads for a local file. The picture is composed
once and zoomed from there, so a track takes a fraction of its length to make.

    karaokifex "<the video>" -a Artist -s Song ...      (or Compute's make_song.py with it)

The lyrics: karaokifex looks on lrclib as always, and karaokifex-lyrics-web finds some on the web
for --lyrics-file; with none anywhere, karaokifex transcribes the voice it separates.
Prints the video's path.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from urllib.parse import urlparse

import requests
import yt_dlp

HEIGHT = 1080
FPS = 25
ZOOM = 0.12          # how far in the cover zooms over the track (1 + ZOOM at its end)


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


def make(url: str, out: Path, *, height: int = HEIGHT, ffmpeg: str = "ffmpeg") -> Path:
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
        still = tmp_dir / "still.png"
        subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-i", str(cover), "-filter_complex", still_filter(height),
                        "-map", "[v]", "-frames:v", "1", str(still)], check=True)
        command = [ffmpeg, "-y", "-loglevel", "error", "-i", str(still),
                   "-i", str(audio), "-filter_complex", video_filter(height, duration or 600),
                   "-map", "[v]", "-map", "1:a", "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
                   "-tune", "stillimage", "-c:a", "aac", "-b:a", "256k", "-shortest", "-movflags", "+faststart",
                   str(target)]
        subprocess.run(command, check=True)
    sidecar = {"id": source_key(url), "title": f"{artist} - {track}", "artist": artist, "track": track,
               "album": info.get("album"), "uploader": info.get("uploader") or artist, "duration": duration or None,
               "webpage_url": url}
    Path(f"{target}.info.json").write_text(json.dumps(sidecar, indent=1, ensure_ascii=False), encoding="utf-8")
    return target


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="karaokifex-bandcamp", description=__doc__.split("\n\n")[0])
    parser.add_argument("url", help="a Bandcamp track's page")
    parser.add_argument("-o", "--output-dir", type=Path, default=Path("."))
    parser.add_argument("--height", type=int, default=HEIGHT, help=f"the video's height (default {HEIGHT})")
    parser.add_argument("--ffmpeg", default=shutil.which("ffmpeg") or "ffmpeg")
    a = parser.parse_args(argv)
    print(make(a.url, a.output_dir, height=a.height, ffmpeg=a.ffmpeg))
    return 0


if __name__ == "__main__":
    sys.exit(main())
