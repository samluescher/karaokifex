"""The karaokifex pipeline: which steps exist, what they need, and what can run in parallel.

    probe ─┬─ lyrics ─────────────────────┬──────────────┬──────────────┐
           ├─ load_whisper ───────────────┤              │              │
           └─ download ─┬─ extract_audio ─ separate_karaoke ─┬─ transcribe ─ force_align ─ subtitles ─ render
                        │                                    └─ vocal_activity ┘ (also → subtitles)  │
                        └─ extract_video ────────────────────────────────────────────────────────────┘

One separation pass yields both stems the rest needs: the backing track (for
render) and the lead vocals (for transcription, vocal activity and forced
alignment). `transcribe` listens twice, with and without the lyrics as prompt,
and fills what the first pass missed from the second. With --mix-vote, `transcribe_mix` also transcribes the full mix.
With --no-burn-lyrics, `render` doesn't wait for `subtitles`, which still writes
the lyrics files. With --palette, `palette` samples frames of the extracted video
for its dominant colours and how much its picture changes (motion). With --describe, `describe` asks MusicBrainz about the
song (album, year, genres, writers, language) once the lyrics are in. With
--quality, `quality` reads the download's streams and the renders' once they
are made, before the download is deleted. With --keep-source, `original` puts the download's own
sound back under the video, in the output format.
`probe` runs first on its own (its metadata names the song folder); the task
runner handles the rest.
"""

from __future__ import annotations

import json
import logging
import time
from collections import Counter
from dataclasses import asdict, dataclass, replace
from functools import partial
from pathlib import Path
from typing import Any

import ffmpeg
import requests
from rich.live import Live

from karaokifex.activity import Activity
from karaokifex.ass import build_ass
from karaokifex.config import DEFAULT_BACKING_MODEL, Config
from karaokifex.console import TaskBoard, console, register_tasks
from karaokifex.gpu import free_gpu_memory
from karaokifex.metadata import guess_artist_song, name_guesses, title_segments
from karaokifex.models import Lyrics, TimedWord, VideoInfo
from karaokifex import dialects, level, quality
from karaokifex.palette import dominant_colors, motion, without_bars
from karaokifex.runner import RunReport, Task, TaskContext, TaskRunner, current_task
from karaokifex.steps import download, lyrics, media, musicbrainz, separation, transcription
from karaokifex.timing import (
    BACKING,
    Alignment,
    AlignmentPlan,
    align_lyrics,
    backing_heard,
    fill_unlyricked,
    filter_heard,
    forced_requests,
    kept_lines,
    lines_from_words,
    merge_heard,
    plan_alignment,
    video_order,
)
from karaokifex.workspace import Workspace, partial_path, sanitize_name

log = logging.getLogger(__name__)

LOW_MATCH_WARNING = 0.3  # below this quality the lyrics are probably a different version of the song


@dataclass(frozen=True)
class Job:
    """Everything the steps need to know about the song being processed."""

    config: Config
    workspace: Workspace
    info: VideoInfo
    artist: str
    song: str
    device: str
    ffmpeg: media.FfmpegBinary = media.FfmpegBinary()
    kept: bool = False  # made again from its kept download and details: YouTube not asked (kept_download)

    @property
    def title(self) -> str:
        return f"{self.artist} – {self.song}"

    @property
    def output_video(self) -> Path:
        if not self.config.burn_lyrics:
            video = self.workspace.plain_video
        else:
            video = self.workspace.debug_video if self.config.debug_ass else self.workspace.final_video
        return self._in_output_format(video)

    @property
    def original_video(self) -> Path:
        return self._in_output_format(self.workspace.original_video)

    def _in_output_format(self, video: Path) -> Path:
        return video.with_suffix(".mp4") if self.config.browser_friendly else video


@dataclass(frozen=True)
class PipelineResult:
    job: Job
    report: RunReport

    @property
    def ok(self) -> bool:
        return self.report.ok


def prepare(config: Config) -> Job:
    """Probe the video and set up its working folder."""
    token = current_task.set("probe")
    try:
        info = kept_download(config)
        if info:
            log.info("made again from its kept download and details: %s not asked", config.url)
        else:
            log.info("looking up %s", config.url)
            info = download.probe(config.url, ffprobe=media.FfmpegBinary(config.ffmpeg or "ffmpeg").ffprobe)
        artist, song = guess_artist_song(info, config.artist, config.song)
        if config.musicbrainz and not (config.artist and config.song):
            artist, song = canonical_names(info, config.artist, config.song, (artist, song))
        workspace = Workspace.create(config.output_dir, f"{artist} - {song}")
        workspace.info_json.write_text(json.dumps(info.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
        device = config.resolve_device()
        ffmpeg = media.find_ffmpeg(config.ffmpeg)
        log.info("“%s” (%s) → %s – %s", info.title, _format_length(info.duration), artist, song)
        log.info("working folder: %s · device: %s", workspace.root, device)
        log.info("ffmpeg: %s · %s", ffmpeg.path, ffmpeg.describe())
        return Job(config, workspace, info, artist, song, device, ffmpeg, kept=bool(kept_download(config)))
    finally:
        current_task.reset(token)


def kept_download(config: Config) -> VideoInfo | None:
    """A song made again (-a and -s naming its folder) whose folder keeps the download of this very video
    (--keep-download) and its details (info.json): those details, so neither the probe nor the download asks
    the video's site again. None for anything else."""
    if not (config.artist and config.song):
        return None
    root = config.output_dir / sanitize_name(f"{config.artist} - {config.song}")
    try:
        data = json.loads((root / "info.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not (root / "source.mkv").exists() or not data.get("id") or str(data["id"]) not in config.url:
        return None
    return VideoInfo.from_dict(data)


def canonical_names(info: VideoInfo, artist: str | None, song: str | None, guess: tuple[str, str],
                    *, get: musicbrainz.HttpGet = requests.get) -> tuple[str, str]:
    """Artist and song as MusicBrainz spells them, if it confirms them; otherwise `guess`. Given names win."""
    segments = title_segments(info.title)
    # A title that doesn't split into parts may still hold both names ('Smashing Pumpkins Mayonaise').
    text = segments[0] if len(segments) == 1 and not (artist or song or info.artist) else None
    try:
        match = musicbrainz.lookup(name_guesses(info, artist, song), text, get=get)
    except (requests.RequestException, ValueError) as error:
        log.warning("MusicBrainz lookup failed (%s) — keeping “%s – %s”", error, *guess)
        return guess
    if match is None:
        log.info("MusicBrainz isn't sure about this one — keeping “%s – %s”", *guess)
        return guess
    names = (artist or match.artist, song or match.song)
    log.info("MusicBrainz: “%s – %s” (%d matching recordings%s)", *names, match.recordings,
             "" if names == guess else f"; the video's own guess was “{guess[0]} – {guess[1]}”")
    return names


def build_tasks(job: Job) -> list[Task]:
    ws, cfg = job.workspace, job.config
    transcripts = (ws.transcript_json, ws.transcript_mix_json) if cfg.mix_vote else (ws.transcript_json,)
    tasks = [
        # (the lyrics stay through --force: a song made again is made from the lyrics it has, and lrclib asked nothing,
        # unless --fresh-lyrics asks for its versions anew)
        Task("lyrics", partial(_lyrics, job), outputs=(ws.lyrics_json,), description="lrclib lookup",
             keep=not job.config.fresh_lyrics),
        # (made again from the download it kept, the same video: the download is reused, and said to be, not done again)
        Task("download", partial(_download, job), outputs=(ws.source,), keep=bool(job.kept),
             description="yt-dlp: best video" + (" (H.264 if as good)" if cfg.browser_friendly else "") + " + audio"),
        Task("load_whisper", partial(_load_whisper, job), outputs=transcripts,
             description=f"whisperx {cfg.whisper_model}"),
        Task("extract_audio", partial(_extract_audio, job), deps=("download",), outputs=(ws.audio,),
             description="ffmpeg → audio.wav"),
        Task("extract_video", partial(_extract_video, job), deps=("download",), outputs=(ws.video,),
             description="ffmpeg → video.mkv (no audio)"),
        Task("separate_karaoke", partial(_separate_karaoke, job), deps=("extract_audio",),
             outputs=(ws.karaoke_backing, ws.karaoke_lead), gpu=True, model=True, description=" + ".join(cfg.karaoke_models)),
        Task("vocal_activity", partial(_vocal_activity, job), deps=("separate_karaoke",),
             outputs=(ws.lead_activity,), description="when the lead vocals are audible"),
        Task("transcribe", partial(_transcribe, job), deps=("separate_karaoke", "load_whisper", "lyrics"),
             outputs=(ws.transcript_json,), gpu=True, model=True, description="whisperx on the lead vocals, with and without the lyrics as prompt"),
    ]
    if cfg.backing_vocals:
        tasks += [
            Task("separate_backing", partial(_separate_backing, job), deps=("separate_karaoke",),
                 outputs=(ws.backing_vocals, ws.backing_activity), gpu=True, model=True,
                 description=f"{DEFAULT_BACKING_MODEL}: the backing vocals out of the karaoke backing"),
            # after transcribe, to take over its model
            Task("transcribe_backing", partial(_transcribe_backing, job),
                 deps=("separate_backing", "load_whisper", "lyrics", "transcribe"),
                 outputs=(ws.transcript_backing_json,), gpu=True, model=True,
                 description="whisperx on the backing vocals, with and without the lyrics as prompt"),
        ]
    if cfg.mix_vote:
        # Depends on transcribe so it can take over the loaded model instead of loading it twice.
        tasks.append(Task("transcribe_mix", partial(_transcribe_mix, job),
                          deps=("extract_audio", "load_whisper", "lyrics", "transcribe"),
                          outputs=(ws.transcript_mix_json,), gpu=True, model=True, description="whisperx on the full mix"))
    if cfg.keep_source:
        tasks.append(Task("original", partial(_original, job),
                          deps=("download", "extract_video") + (("extract_audio",) if cfg.lift_quiet else ()),
                          outputs=(job.original_video,), gpu=job.ffmpeg.gpu,
                          description="the original video with its own sound"))
    if cfg.palette:
        tasks.append(Task("palette", partial(_palette, job), deps=("extract_video",), outputs=(ws.metadata_json,),
                          description="dominant colours of the video"))
    if cfg.describe:
        tasks.append(Task("describe", partial(_describe, job), deps=("lyrics",), outputs=(ws.song_json,),
                          description="MusicBrainz: album, year, genres, writers, language"))
    if cfg.quality:
        tasks.append(Task("quality", partial(_quality, job),
                          deps=("download", "render") + (("original",) if cfg.keep_source else ()),
                          outputs=(ws.quality_json,), description="resolution, frame rate, codecs, bitrates"))
    backing = ("transcribe_backing",) if cfg.backing_vocals else ()
    subtitle_deps = ("lyrics", "transcribe", "vocal_activity", "force_align") + backing
    # Without burned-in lyrics the render doesn't wait for them; the subtitles step still writes the lyrics files.
    render_deps = ("extract_video", "separate_karaoke") + (("subtitles",) if cfg.burn_lyrics else ())
    if not cfg.burn_lyrics:
        render_description = "karaoke audio, picture as it is (no lyrics burned in)"
    else:
        render_description = "darken, karaoke audio, burn in subtitles" + (" (debug colours)" if cfg.debug_ass else "")
    tasks += [
        Task("force_align", partial(_force_align, job), deps=("lyrics", "transcribe", "vocal_activity") + backing,
             outputs=(ws.forced_json,), gpu=True, model=True, description="wav2vec2 alignment of the known lyrics"),
        Task("subtitles", partial(_subtitles, job),
             deps=subtitle_deps + (("transcribe_mix",) if cfg.mix_vote else ()),
             outputs=(ws.subtitles, ws.debug_subtitles, ws.timings_json),
             description="lyrics + word timings → karaoke ASS"),
        Task("render", partial(_render, job), deps=render_deps, outputs=(job.output_video,), gpu=job.ffmpeg.gpu,
             description=render_description),
    ]
    return tasks


def run_pipeline(config: Config) -> PipelineResult:
    job = prepare(config)
    tasks = build_tasks(job)
    register_tasks(task.name for task in tasks)
    board = TaskBoard(job.title, [(task.name, task.description) for task in tasks])
    runner = TaskRunner(tasks, gpu_slots=config.gpu_jobs, gpu_lock=config.gpu_lock, force=config.force, observer=board)
    with Live(board, console=console, refresh_per_second=8):
        report = runner.run()
    return PipelineResult(job, report)


# --- task implementations ------------------------------------------------------------


def _lyrics(job: Job, ctx: TaskContext) -> str:
    """lrclib first, always; lyrics given with the song (--lyrics-file) only when it has none; then whisperx."""
    ctx.note(f"searching “{job.artist} – {job.song}”…")
    found = _only_swiss_german(job, lyrics.fetch_lyrics(job.artist, job.song, job.info.duration))
    if found and job.config.lyrics_file:
        log.info("lrclib has the song: the lyrics given (%s) are not used", job.config.lyrics_file.name)
    if not found and job.config.lyrics_file:
        given = lyrics.from_file(job.config.lyrics_file, job.artist, job.song)
        lyrics.save_lyrics([given] if given.lines else [], job.workspace.lyrics_json)
        log.info("none on lrclib: lyrics given, %s, %d lines", job.config.lyrics_file.name, len(given.lines))
        if given.lines:
            return lyrics.describe(given) + " (lrclib miss)"
    lyrics.save_lyrics(found, job.workspace.lyrics_json)
    if not found:
        log.warning("no lyrics on lrclib — the whisperx transcription will be used instead")
        return "whisperx transcription (lrclib miss)"
    for candidate in found:
        log.info("found %s (%d lines, %s)", lyrics.describe(candidate), len(candidate.lines),
                 _format_length(candidate.duration))
    others = f" (+{len(found) - 1} alternatives)" if len(found) > 1 else ""
    return lyrics.describe(found[0]) + others


def _only_swiss_german(job: Job, found: list[Lyrics]) -> list[Lyrics]:
    """For a Swiss German song (--language gsw), only lyrics that read as Swiss German: Standard German words
    for it are no lyrics of it."""
    if not dialects.swiss(job.config.language):
        return found
    keep = [c for c in found if dialects.reads_swiss_german(" ".join(line.text for line in c.lines))]
    for c in found:
        if c not in keep:
            log.info("%s doesn't read as Swiss German: not used for a Swiss German song", lyrics.describe(c))
    return keep


def _swiss_german_lyrics(job: Job, words: list[TimedWord]) -> None:
    """A Swiss German song with no lyrics: whisper's lines, when they read as Standard German written back in Swiss
    German by the chat model (--llm-url), kept as lyrics.gsw.txt with where they came from and aligned like any
    lyrics. Lines that read as Swiss German already are the transcription's own, as for any song without lyrics."""
    lines = [" ".join(w.text.strip() for w in line.words) for line in lines_from_words(words)]
    text = " ".join(lines)
    if not lines or dialects.reads_swiss_german(text):
        log.info("whisper's lines read as Swiss German: used as they are")
        return
    ctx_model = job.config.llm_model or "the chat model"
    rewritten = dialects.rewrite(lines, job.config.llm_url, job.config.llm_model) if job.config.llm_url else None
    how = f"{ctx_model} wrote each line back in Swiss German, and its common words left in Standard German were swapped"
    by = ctx_model
    if not rewritten:
        # no model, or none that gave Swiss German back: the common words swapped, which reads as Swiss German at least
        rewritten = [dialects.swiss_words(line) for line in lines]
        how = "its common words were swapped for their Swiss German (ich -> i, ist -> isch, nicht -> nöd...)"
        by = "swapping its common words"
        if not dialects.reads_swiss_german(" ".join(rewritten)):
            log.warning("whisper wrote this Swiss German song in Standard German, and swapping its words didn't make it "
                        "read as Swiss German: its lines are used as heard; give its lyrics with --lyrics-file")
            return
    path = job.workspace.swiss_lyrics
    header = [f"# {job.artist} - {job.song}: Swiss German lyrics made from the singing, {time.strftime('%Y-%m-%d')}",
              f"# whisper heard it in German and wrote mostly Standard German; {how}",
              "# (the words as sung, most likely: not a published text; karaokifex --lyrics-file skips # lines)"]
    path.write_text("\n".join(header + rewritten) + "\n", encoding="utf-8")
    made = lyrics.from_file(path, job.artist, job.song)
    lyrics.save_lyrics([made], job.workspace.lyrics_json)
    log.info("%d lines written back in Swiss German by %s (%s)", len(rewritten), by, path.name)


def _download(job: Job, ctx: TaskContext) -> None:
    def on_progress(fraction: float | None, note: str) -> None:
        ctx.progress(fraction)
        ctx.note(note)

    if job.kept and job.workspace.source.exists():
        ctx.note("the kept download")  # made again from it (kept_download): the site not asked
        return
    download.download(job.config.url, job.workspace.source, on_progress, prefer_h264=job.config.browser_friendly,
                      ffmpeg_path=job.ffmpeg.path)
    log.info("downloaded %s (%.0f MiB)", job.workspace.source.name, job.workspace.source.stat().st_size / 1_048_576)


def _extract_audio(job: Job, ctx: TaskContext) -> None:
    media.extract_audio(job.workspace.source, job.workspace.audio, binary=job.ffmpeg.path,
                        duration=job.info.duration, on_progress=ctx.progress)


def _extract_video(job: Job, ctx: TaskContext) -> None:
    media.extract_video(job.workspace.source, job.workspace.video, binary=job.ffmpeg.path,
                        duration=job.info.duration, on_progress=ctx.progress)


def _palette(job: Job, ctx: TaskContext) -> None:
    ws = job.workspace
    ctx.note("sampling frames…")
    frames = media.sample_frames(ws.video, binary=job.ffmpeg.path, ffprobe=job.ffmpeg.ffprobe,
                                 duration=job.info.duration)
    if not len(frames):
        raise RuntimeError(f"no frame of {ws.video.name} could be decoded")
    cropped = without_bars(frames)
    swatches, change = dominant_colors(cropped), motion(cropped)
    _write_json(ws.metadata_json, {"palette": {"colors": [swatch.to_dict() for swatch in swatches],
                                               "frames": len(frames), "motion": round(change, 3)}})
    log.info("dominant colours: %s; motion %.1f", ", ".join(f"{swatch.hex} {swatch.weight:.0%}" for swatch in swatches), change)


def _describe(job: Job, ctx: TaskContext) -> str:
    ctx.note("asking MusicBrainz…")
    candidates = lyrics.load_lyrics(job.workspace.lyrics_json)
    try:
        released = json.loads(job.workspace.info_json.read_text(encoding="utf-8")).get("released")
    except (OSError, ValueError):
        released = None
    details = describe_song(job.workspace.song_json, job.artist, job.song,
                            job.config.language or (lyrics.guess_language(candidates[0].lines) if candidates else None),
                            released=released)
    return ", ".join(str(x) for x in (details.get("album"), details.get("year"), *details.get("genres", [])[:2]) if x) \
        or "MusicBrainz knows nothing more"


def describe_song(target: Path, artist: str, song: str, language: str | None = None, *,
                  released: str | None = None, get: musicbrainz.HttpGet = requests.get) -> dict[str, Any]:
    """Write song.json: the song's names and what MusicBrainz knows of it. The language is the one
    the song's work is in, else `language` (the lyrics' or the singing's). MusicBrainz not answering
    leaves the rest out, and says so. A song MusicBrainz knows no recording of (a band only on Bandcamp)
    takes its year from when its source says it came out (`released`, YYYY-MM-DD), with year_from saying so."""
    about: dict[str, Any] = {"artist": artist, "song": song}
    try:
        details = musicbrainz.describe(artist, song, get=get)
    except (requests.RequestException, ValueError) as error:
        log.warning("MusicBrainz didn't answer (%s): no album, year, genres or writers this time", error)
        about["musicbrainz"] = None
    else:
        if details is None:
            log.info("MusicBrainz knows no recording of “%s – %s”%s", artist, song,
                     f": its year from its source, out {released}" if released else "")
            if released and released[:4].isdigit():
                about.update(year=int(released[:4]), released=released, year_from="source")
        else:
            about.update(details.to_dict())
            log.info("MusicBrainz: first out %s, on %s · %s · by %s · sung in %s", details.year or "?", details.album or "no album",
                     ", ".join(details.genres) or "no genres", ", ".join(w.name for w in details.writers) or "?",
                     details.language or "?")
    about["language"] = about.get("language") or language
    _write_json(target, about)
    return about


def _quality(job: Job, ctx: TaskContext) -> str:
    ws = job.workspace
    ffprobe = job.ffmpeg.ffprobe
    # the download as it came: never re-encoded, and kept as source.mkv with --keep-download
    source = {**quality.summary(ws.source, ffprobe=ffprobe), "from": "the download", "reencoded": False,
              **({"file": ws.source.name} if job.config.keep_download else {})}
    renders = {"karaoke": job.output_video, **({"original": job.original_video} if job.config.keep_source else {})}
    q = quality.write_quality(ws.quality_json, source=source, renders=renders, ffprobe=ffprobe)
    v, a = source.get("video") or {}, source.get("audio") or {}
    video_rate = f" {round(v['bitrate'] / 1000)} kb/s" if v.get("bitrate") else ""
    audio_rate = f"{round(a['bitrate'] / 1000)} kb/s" if a.get("bitrate") else "? kb/s"
    text = (f"source {v.get('width')}x{v.get('height')} {v.get('codec')} {v.get('fps') or '?'} fps{video_rate}, "
            f"audio {a.get('codec')} {audio_rate}" + (", upscaled" if q["upscaled"] else ""))
    log.info("quality: %s", text)
    return text


def _separate_karaoke(job: Job, ctx: TaskContext) -> None:
    """Each model's lead and backing (kept per model, so a re-run resumes after the last one done),
    then the lead averaged at the song's own level and the backing the song minus it (separation.combine)."""
    ws, cfg = job.workspace, job.config
    pairs = []
    for number, model in enumerate(cfg.karaoke_models):
        lead, backing = ws.stems_dir / f"lead.{number}.wav", ws.stems_dir / f"backing.{number}.wav"
        if not (lead.exists() and backing.exists()):
            prefix = f"{number + 1}/{len(cfg.karaoke_models)} " if len(cfg.karaoke_models) > 1 else ""
            separation.separate(ws.audio, model=model, model_dir=cfg.model_dir,
                                stems={"instrumental": backing, "vocals": lead}, overlap=cfg.separation_overlap,
                                fp16=cfg.fp16, verbose=cfg.verbose, on_stage=lambda note: ctx.note(prefix + note))
        pairs.append((lead, backing))
    ctx.note("the song minus the lead…")
    gains = separation.combine(ws.audio, pairs, backing=ws.karaoke_backing, lead=ws.karaoke_lead)
    log.info("lead vocals at the song's level (gains %s), the karaoke the song minus their average",
             ", ".join(f"{g:.2f}" for g in gains))


def _vocal_activity(job: Job, ctx: TaskContext) -> None:
    import soundfile  # only needed here

    samples, rate = soundfile.read(str(job.workspace.karaoke_lead), dtype="float32")
    activity = Activity.compute(samples, rate)
    partial_file = partial_path(job.workspace.lead_activity)
    activity.save(partial_file)
    partial_file.replace(job.workspace.lead_activity)
    log.info("lead vocals audible %.0f%% of the time", 100 * float(activity.voiced.mean()) if activity.duration else 0)


def _load_whisper(job: Job, ctx: TaskContext) -> object:
    ctx.note("loading model (downloaded on first use)…")
    return transcription.load_model(job.config.whisper_model, job.device, dialects.whisper_language(job.config.language))


FALLBACK_LANGUAGE = "en"  # what whisper hears a song in again when the language it guessed has no alignment model


def _song_language(job: Job, candidates: list[Lyrics]) -> str | None:
    """--language wins; otherwise the lyrics' language, which is more reliable than listening to singing."""
    if job.config.language or not candidates:
        return job.config.language
    if language := lyrics.guess_language(candidates[0].lines):
        log.info("language from the lyrics: %s", language)
    return language


def _whisper_model(job: Job, ctx: TaskContext, *upstream: str) -> Any:
    # take(): the runner must not keep the model alive — it would hog GPU memory during render.
    for task in upstream:
        if (model := ctx.take(task)) is not None:
            return model
    ctx.note("loading model…")  # upstream was cached (an old transcript existed) but we must transcribe again
    return transcription.load_model(job.config.whisper_model, job.device, dialects.whisper_language(job.config.language))


def _transcribe_into(job: Job, ctx: TaskContext, model: Any, audio: Path, target: Path) -> None:
    candidates = lyrics.load_lyrics(job.workspace.lyrics_json)
    swiss = dialects.swiss(job.config.language)
    # a Swiss German song with no lyrics: whisper prompted in Swiss German, which keeps it nearer the dialect
    prompt = lyrics.prompt_text(candidates[0].lines) if candidates else dialects.PROMPT if swiss else None
    language = dialects.whisper_language(_song_language(job, candidates))
    # the song's language unknown: the backing sings in the lead's (whisper's guess on backing singers alone strays further),
    # and a guess with no alignment model is heard again in English rather than failing the song
    if language is None and target == job.workspace.transcript_backing_json and job.workspace.transcript_json.exists():
        language = transcription.load_transcript(job.workspace.transcript_json)[1] or None
    words, language = transcription.transcribe(model, audio, job.device, language=language, prompt=prompt,
                                               fallback=FALLBACK_LANGUAGE, on_stage=ctx.note)
    log.info("heard %d words in %s (language: %s)", len(words), audio.name, language)
    unprompted = {job.workspace.transcript_json: job.workspace.transcript_noprompt_json,
                  job.workspace.transcript_backing_json: job.workspace.transcript_backing_noprompt_json}
    if prompt and target in unprompted:
        # Prompted with the lyrics whisper hears their words better but skips repeats; a pass without the
        # prompt fills the places the first heard nothing.
        plain, _ = transcription.transcribe(model, audio, job.device, language=language, prompt=None,
                                            on_stage=lambda stage: ctx.note(f"{stage} (no prompt)"))
        transcription.save_transcript(plain, language, unprompted[target])
        merged = merge_heard(words, plain)
        log.info("%d more words heard without the lyrics as prompt", len(merged) - len(words))
        words = merged
    transcription.save_transcript(words, language, target)
    if swiss and not candidates and target == job.workspace.transcript_json:
        _swiss_german_lyrics(job, words)


def _transcribe(job: Job, ctx: TaskContext) -> Any:
    model = _whisper_model(job, ctx, "load_whisper")
    _transcribe_into(job, ctx, model, job.workspace.karaoke_lead, job.workspace.transcript_json)
    if job.config.mix_vote or job.config.backing_vocals:
        return model  # handed over to transcribe_mix or transcribe_backing, which frees it
    del model
    free_gpu_memory()
    return None


def _transcribe_mix(job: Job, ctx: TaskContext) -> None:
    model = _whisper_model(job, ctx, "transcribe", "load_whisper")
    _transcribe_into(job, ctx, model, job.workspace.audio, job.workspace.transcript_mix_json)
    del model
    free_gpu_memory()


def _separate_backing(job: Job, ctx: TaskContext) -> None:
    """The backing vocals: the vocals of the karaoke backing, which the karaoke model leaves the backing singers in."""
    import soundfile

    ws, cfg = job.workspace, job.config
    rest = ws.stems_dir / "backing_without_vocals.wav"  # the model's other half: not wanted, not left lying about
    separation.separate(ws.karaoke_backing, model=DEFAULT_BACKING_MODEL, model_dir=cfg.model_dir,
                        stems={"vocals": ws.backing_vocals, "instrumental": rest}, overlap=cfg.separation_overlap,
                        fp16=cfg.fp16, verbose=cfg.verbose, on_stage=ctx.note)
    rest.unlink(missing_ok=True)
    samples, rate = soundfile.read(str(ws.backing_vocals), dtype="float32")
    activity = Activity.compute(samples, rate)
    partial_file = partial_path(ws.backing_activity)
    activity.save(partial_file)
    partial_file.replace(ws.backing_activity)
    log.info("backing vocals audible %.0f%% of the time", 100 * float(activity.voiced.mean()) if activity.duration else 0)


def _transcribe_backing(job: Job, ctx: TaskContext) -> None:
    model = _whisper_model(job, ctx, "transcribe", "load_whisper")
    _transcribe_into(job, ctx, model, job.workspace.backing_vocals, job.workspace.transcript_backing_json)
    del model
    free_gpu_memory()


def _heard(ws: Workspace) -> tuple[list[TimedWord], str, Activity | None, Activity | None]:
    """What was heard: the lead's words, the backing vocals' (backing_heard) with them when they were transcribed,
    the language, and the lead's and the backing's activity."""
    words, language = transcription.load_transcript(ws.transcript_json)
    activity = Activity.load(ws.lead_activity) if ws.lead_activity.exists() else None
    backing_activity = None
    if ws.transcript_backing_json.exists() and ws.backing_activity.exists():
        backing_activity = Activity.load(ws.backing_activity)
        backing = backing_heard(transcription.load_transcript(ws.transcript_backing_json)[0], backing_activity, words)
        words = sorted([*words, *backing], key=lambda w: w.start)
    return words, language, activity, backing_activity


def _force_align(job: Job, ctx: TaskContext) -> None:
    ws = job.workspace
    candidates = lyrics.load_lyrics(ws.lyrics_json)
    words, language, activity, backing_activity = _heard(ws)
    candidates = in_video_order(candidates, words, language, activity)
    plans = [plan_alignment(c.lines, words, activity, language=language, backing_activity=backing_activity)
             for c in candidates]
    # the backing lines on the backing vocals, when there are, the others on the lead
    split = backing_activity is not None and ws.backing_vocals.exists()
    results = {}
    for audio, which in ((ws.karaoke_lead, False if split else None), (ws.backing_vocals, True)):
        if which is True and not split:
            continue
        requests = [((number, index, positions), line_words, window)
                    for number, (candidate, plan) in enumerate(zip(candidates, plans))
                    for index, positions, line_words, window in forced_requests(candidate.lines, plan, backing=which)]
        results |= transcription.force_align(requests, audio, language, job.device, on_stage=ctx.note)

    entries = []
    for number, (candidate, plan) in enumerate(zip(candidates, plans)):
        kept = kept_lines(candidate.lines)
        lines: list[list[dict[str, Any] | None] | None] = [None] * len(kept)
        # a line's lead words and its backing part are aligned apart, each on its own voice: put back by position
        for (owner, index, positions), placed in results.items():
            if owner != number or placed is None:
                continue
            if lines[index] is None:
                lines[index] = [None] * len(kept[index][1])
            for position, word in zip(positions, placed):
                lines[index][position] = asdict(word)
        aligned = sum(line is not None for line in lines)
        log.info("%s: %s · %d lines force-aligned, %d cut", lyrics.describe(candidate), plan.time_map.describe(),
                 aligned, len(plan.cut))
        entries.append({"lrclib_id": candidate.lrclib_id, "plan": plan.to_dict(), "lines": lines})
    _write_json(ws.forced_json, {"candidates": entries})


def in_video_order(candidates: list[Lyrics], words: list[TimedWord], language: str | None,
                   activity: Activity | None) -> list[Lyrics]:
    """Each candidate's lines in the order the video sings them (timing.video_order), where it moves blocks; said in the
    log. Run alike before forced alignment and the subtitles, from the same transcript, so their line indices agree."""
    out = []
    for candidate in candidates:
        lines, moved = video_order(candidate.lines, words, activity, language=language)
        if moved:
            blocks = ", ".join(f"lines {a + 1}-{b + 1} at {int(t // 60)}:{t % 60:04.1f}" for a, b, t in moved)
            log.info("%s: %d block%s sung elsewhere than the lyrics have them: %s", lyrics.describe(candidate), len(moved),
                     "" if len(moved) == 1 else "s", blocks)
            candidate = replace(candidate, lines=tuple(lines))
        out.append(candidate)
    return out


def load_forced(path: Path) -> list[dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))["candidates"] if path.exists() else []


def forced_lines(entry: dict[str, Any]) -> list[list[TimedWord | None] | None]:
    """Each line's forced words; a word left out of its line's alignment (its backing part, unaligned) is None."""
    return [None if line is None else [None if word is None else TimedWord(**word) for word in line]
            for line in entry["lines"]]


def align_candidates(candidates: list[Lyrics], words: list[TimedWord], language: str | None,
                     activity: Activity | None, forced: list[dict[str, Any]], *,
                     mix: list[TimedWord] | None = None, use_word_tags: bool = True,
                     backing_activity: Activity | None = None) -> list[tuple[Lyrics, Alignment]]:
    """Align every lyrics candidate; best (by how well it fits the audio) first."""
    results = []
    for number, candidate in enumerate(candidates):
        entry = forced[number] if number < len(forced) and forced[number]["lrclib_id"] == candidate.lrclib_id else None
        plan = AlignmentPlan.from_dict(entry["plan"]) if entry else None
        alignment = align_lyrics(candidate.lines, words, plan=plan, forced=forced_lines(entry) if entry else None,
                                 activity=activity, language=language, mix_heard=mix, use_word_tags=use_word_tags,
                                 backing_activity=backing_activity)
        results.append((candidate, alignment))
    return sorted(results, key=lambda item: -item[1].quality)  # stable: lrclib's ranking breaks ties


def _subtitles(job: Job, ctx: TaskContext) -> str:
    ws = job.workspace
    candidates = lyrics.load_lyrics(ws.lyrics_json)
    words, language, activity, backing_activity = _heard(ws)
    candidates = in_video_order(candidates, words, language, activity)
    mix = transcription.load_transcript(ws.transcript_mix_json)[0] if job.config.mix_vote else None
    if candidates:
        ranked = align_candidates(candidates, words, language, activity, load_forced(ws.forced_json), mix=mix,
                                  backing_activity=backing_activity)
        for candidate, alignment in ranked:
            log.info("%s: quality %.2f (%.0f%% heard, forced score %s, %d lines cut)", lyrics.describe(candidate),
                     alignment.quality, alignment.match_ratio * 100,
                     "–" if alignment.forced_score is None else f"{alignment.forced_score:.2f}", alignment.cut)
        chosen, alignment = ranked[0]
        # what's sung where the lyrics have nothing (a rap they left out), from what was heard there
        lead_words = [w for w in words if w.source != BACKING]  # what the lead sings that the lyrics lack
        lines = fill_unlyricked(alignment.lines, lead_words, activity)
        if len(lines) > len(alignment.lines):
            log.info("%d lines added from the transcription where the lyrics have none", len(lines) - len(alignment.lines))
        sources = Counter(word.source for line in lines for word in line.words)
        log.info("word timing sources: %s", ", ".join(f"{name} {count}" for name, count in sources.most_common()))
        if alignment.quality < LOW_MATCH_WARNING:
            log.warning("the lyrics fit the audio poorly — they may belong to a different version of the song")
        description = lyrics.describe(chosen)
    else:
        lines = lines_from_words(filter_heard([w for w in words if w.source != BACKING], activity))
        log.info("built %d lines from the transcription", len(lines))
        description = "whisperx transcription (lrclib miss)"
    if not lines:
        raise RuntimeError("nothing to display: no lyrics found and no words transcribed")

    width, height = job.info.width or 1920, job.info.height or 1080
    for path, debug in ((ws.subtitles, False), (ws.debug_subtitles, True)):
        _write_text(path, build_ass(lines, width=width, height=height, title=job.title, debug=debug))
    _write_json(ws.timings_json, {"lyrics": description, "language": "gsw" if dialects.swiss(job.config.language) else language,
                                  "lines": [[asdict(word) for word in line.words] for line in lines]})
    log.info("wrote %d karaoke lines to %s", len(lines), ws.subtitles.name)
    return description


def _source_info(job: Job) -> media.SourceInfo:
    try:
        return media.probe_source(job.workspace.video, job.workspace.source, ffprobe=job.ffmpeg.ffprobe)
    except (ffmpeg.Error, OSError) as error:
        log.warning("couldn't inspect the source video (%s) — encoding at constant quality", error)
        return media.SourceInfo()


_LEVELS: dict[tuple[str, float], float | None] = {}


def _song_level(job: Job) -> float | None:
    """The song's loudness (EBU R128 LUFS), measured once for the original and the karaoke both."""
    audio = job.workspace.audio
    key = (str(audio), audio.stat().st_mtime if audio.exists() else 0.0)
    if key not in _LEVELS:
        _LEVELS[key] = media.loudness(audio, binary=job.ffmpeg.path)
    return _LEVELS[key]


def _original(job: Job, ctx: TaskContext) -> str:
    """The download in the output format: the same picture treatment as the karaoke video, the source's own sound
    (copied, unless the song is too quiet and lifted: --lift-quiet)."""
    ws = job.workspace
    gain = level.lift(_song_level(job), job.config.quiet_floor) if job.config.lift_quiet else 0.0
    if gain:
        log.info("loudness: the song %.1f LUFS, quieter than %.0f: lifted %+.1f dB, the original and the karaoke alike",
                 _song_level(job), job.config.quiet_floor, gain)
    encoding = media.render(ws.video, ws.source, None, job.original_video, tool=job.ffmpeg, source=_source_info(job),
                            target_height=job.config.target_height, browser=job.config.browser_friendly,
                            copy_audio=not gain, gain_db=gain or None, duration=job.info.duration,
                            on_progress=ctx.progress)
    size = job.original_video.stat().st_size / 1_048_576
    log.info("wrote %s (%.0f MiB) with %s", job.original_video.name, size, encoding)
    return encoding


def _render(job: Job, ctx: TaskContext) -> str:
    ws = job.workspace
    source = _source_info(job)
    gain = None
    if job.config.match_loudness or job.config.lift_quiet:
        song, karaoke = _song_level(job), media.loudness(ws.karaoke_backing, binary=job.ffmpeg.path)
        _, gain = level.gains(song, karaoke, match=job.config.match_loudness, lift_quiet=job.config.lift_quiet,
                              floor=job.config.quiet_floor)
        if job.config.match_loudness and song is not None and karaoke is not None:
            log.info("loudness: the song %.1f LUFS, the karaoke %.1f: %+.1f dB%s", song, karaoke, gain,
                     " to match and lift" if level.lift(song, job.config.quiet_floor) and job.config.lift_quiet else " to match")
        gain = gain or None
    if not job.config.burn_lyrics:
        subtitles = None
    else:
        subtitles = ws.debug_subtitles if job.config.debug_ass else ws.subtitles
    encoding = media.render(ws.video, ws.karaoke_backing, subtitles, job.output_video, tool=job.ffmpeg,
                            source=source, lead=ws.karaoke_lead, lead_volume=job.config.lead_volume,
                            darken=job.config.darken, target_height=job.config.target_height,
                            browser=job.config.browser_friendly, duration=job.info.duration, gain_db=gain,
                            on_progress=ctx.progress)
    size = job.output_video.stat().st_size / 1_048_576
    log.info("rendered %s (%.0f MiB) with %s", job.output_video.name, size, encoding)
    return encoding


def _write_text(path: Path, text: str) -> None:
    partial_file = partial_path(path)
    partial_file.write_text(text, encoding="utf-8")
    partial_file.replace(path)


def _write_json(path: Path, data: Any) -> None:
    _write_text(path, json.dumps(data, indent=1, ensure_ascii=False))


def _format_length(seconds: float | None) -> str:
    if not seconds:
        return "unknown length"
    minutes, secs = divmod(int(round(seconds)), 60)
    return f"{minutes}:{secs:02d}"
