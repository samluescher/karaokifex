"""Metadata lookup and download with yt-dlp -- or a video already on this machine.

A source can be a file here instead of a link: its path, or a file:// URL (karaokifex-bandcamp makes
one from a track's audio and cover). Its metadata is what ffprobe reads of it, and what its source
said, from <file>.info.json beside it where there is one: id (a source key such as
bandcamp:<host>/track/<name>), title, artist, track, uploader, duration, and made ("still": a
still picture made into a video, as karaokifex-bandcamp makes them).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlparse
from urllib.request import url2pathname

import ffmpeg
import yt_dlp

from karaokifex.models import VideoInfo

log = logging.getLogger(__name__)

# Best video + best audio (merged), or the best single file if that's all there is.
FORMAT = "bv*+ba/b"
# For --browser-friendly: H.264 wins when it comes at the best resolution and frame rate on offer,
# so the render can often copy the video instead of encoding it.
PREFER_H264 = ["res", "fps", "vcodec:h264"]

ProgressCallback = Callable[[float | None, str], None]


class _YtdlpLogger:
    """Routes yt-dlp's console output into our logging (and thus the task-tagged log)."""

    def debug(self, message: str) -> None:
        log.debug(message.removeprefix("[debug] "))

    def info(self, message: str) -> None:
        log.debug(message)

    def warning(self, message: str) -> None:
        log.warning(message)

    def error(self, message: str) -> None:
        log.error(message)


def _options(**extra: Any) -> dict[str, Any]:
    return {
        "quiet": True,
        "noprogress": True,
        "noplaylist": True,
        "logger": _YtdlpLogger(),
        "format": FORMAT,
        # YouTube needs a JavaScript runtime; yt-dlp only enables deno by default.
        "js_runtimes": {"deno": {}, "node": {}},
        **extra,
    }


def local_file(url: str) -> Path | None:
    """The file a source names, when it is one on this machine (a path or a file:// URL); None for a link."""
    if url.startswith("file:"):
        path = Path(url2pathname(urlparse(url).path))
    elif "://" in url:
        return None
    else:
        path = Path(url)
    try:
        return path if path.is_file() else None
    except OSError:
        return None


def sidecar(path: Path) -> dict[str, Any]:
    """What a local file's source said of it: <file>.info.json, else <stem>.info.json, else nothing."""
    for side in (path.with_name(path.name + ".info.json"), path.with_suffix(".info.json")):
        try:
            return json.loads(side.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
    return {}


def _probe_local(path: Path, ffprobe: str) -> VideoInfo:
    streams = ffmpeg.probe(str(path), cmd=ffprobe)
    video = next((s for s in streams.get("streams", []) if s.get("codec_type") == "video"), {})
    side = sidecar(path)
    return VideoInfo(
        id=side.get("id") or f"file:{path.stem}",
        title=side.get("title") or path.stem,
        duration=side.get("duration") or float((streams.get("format") or {}).get("duration") or 0) or None,
        width=video.get("width"),
        height=video.get("height"),
        artist=side.get("artist"),
        track=side.get("track"),
        uploader=side.get("uploader"),
        made=side.get("made"),
    )


def probe(url: str, *, ffprobe: str = "ffprobe") -> VideoInfo:
    """Fetch metadata (title, duration, resolution, …) without downloading anything."""
    if (path := local_file(url)) is not None:
        return _probe_local(path, ffprobe)
    with yt_dlp.YoutubeDL(_options()) as ydl:
        info = ydl.extract_info(url, download=False)
        return VideoInfo.from_ytdlp(ydl.sanitize_info(info))


def download(url: str, target: Path, on_progress: ProgressCallback, *, prefer_h264: bool = False,
             ffmpeg_path: str = "ffmpeg") -> Path:
    """Download best video + best audio into `target` (always an .mkv); a local file is copied in as it is."""
    if (path := local_file(url)) is not None:
        on_progress(None, f"copying {path.name}")
        ffmpeg.input(str(path)).output(str(target), c="copy", map=0).run(cmd=ffmpeg_path, overwrite_output=True, quiet=True)
        on_progress(1.0, "copied")
        return target

    def progress_hook(status: dict[str, Any]) -> None:
        if status["status"] == "downloading":
            total = status.get("total_bytes") or status.get("total_bytes_estimate")
            fraction = status.get("downloaded_bytes", 0) / total if total else None
            stream = "video" if (status.get("info_dict") or {}).get("vcodec", "none") != "none" else "audio"
            speed = status.get("speed")
            speed_text = f" · {speed / 1_048_576:.1f} MiB/s" if speed else ""
            on_progress(fraction, f"{stream} stream{speed_text}")
        elif status["status"] == "finished":
            on_progress(1.0, "stream downloaded")

    def postprocessor_hook(status: dict[str, Any]) -> None:
        if status["status"] == "started":
            on_progress(None, f"post-processing ({status.get('postprocessor', 'ffmpeg')})…")

    options = _options(
        outtmpl=str(target.with_suffix("")) + ".%(ext)s",
        merge_output_format="mkv",
        postprocessors=[{"key": "FFmpegVideoRemuxer", "preferedformat": "mkv"}],
        progress_hooks=[progress_hook],
        postprocessor_hooks=[postprocessor_hook],
        **({"format_sort": PREFER_H264} if prefer_h264 else {}),
    )
    with yt_dlp.YoutubeDL(options) as ydl:
        ydl.download([url])
    if not target.exists():
        raise RuntimeError(f"yt-dlp finished but {target.name} is missing")
    return target
