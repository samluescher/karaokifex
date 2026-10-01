"""The lead melody as notes: melody.json, from the lead voice's pitch.

The voice is the song less its karaoke backing: karaokifex made the backing as the song (ffmpeg's decode at
44.1 kHz, float) minus the lead it separated, so the same decode less the backing gives that lead back exactly,
with nothing separated again. It is kept as stems/vocals.flac (mono, 44.1 kHz), as the drums and kick are. A song
without a backing is read whole: RMVPE follows the voice in a full mix too, a little less surely.

RMVPE gives the voice's pitch every 10 ms. The notes are its runs: gaps under BRIDGE s bridged, a new note where
the pitch moves more than JUMP semitones from the note's and stays there HOLD s, or where the voice dips DIP dB
and comes back (a new syllable on the same pitch), and shorter than SHORTEST s dropped. Frames where the voice
is QUIET dB under its loudest are left out: what bleeds through there isn't sung.

melody.json:
  version, model ("rmvpe"), source ("voice" or "mix")
  notes     [[start s, length s, MIDI note, cents off it, level 0-1], ...] in time order
  range     [lowest, highest] MIDI note
  contour   {step: 0.02, pitch: [MIDI x 10, or null where unvoiced, ...]}: the pitch line, for drawing
"""

from __future__ import annotations

import json
import logging
import subprocess
from pathlib import Path

import numpy as np

log = logging.getLogger("karaokifex")

VERSION = 1
FRAME = 0.01        # s: RMVPE's frame
BRIDGE = 0.05       # s: gaps up to this are bridged within a note
JUMP = 0.7          # semitones from the note's pitch that start a new note ...
HOLD = 0.06         # s: ... once the pitch has stayed there this long
DIP = 6.0           # dB the voice drops, and comes back, between two syllables on one pitch
SHORTEST = 0.08     # s: shorter notes are dropped
QUIET = 45.0        # dB under the voice's loudest: frames quieter are not sung
RATE = 44100


def decode(source: Path, *, binary: str = "ffmpeg") -> np.ndarray:
    """The song's sound as karaokifex's extract_audio has it: 44.1 kHz stereo float, (samples, 2)."""
    raw = subprocess.run([binary, "-v", "error", "-nostdin", "-i", str(source), "-vn", "-acodec", "pcm_f32le",
                          "-ar", str(RATE), "-ac", "2", "-f", "f32le", "-"], check=True, capture_output=True).stdout
    return np.frombuffer(raw, dtype=np.float32).reshape(-1, 2)


def source_of(folder: Path) -> Path | None:
    found = sorted(p for p in folder.glob("source.*") if ".partial" not in p.name)
    return found[0] if found else None


def voice(folder: Path, *, binary: str = "ffmpeg") -> tuple[np.ndarray, str]:
    """The lead voice, mono at 44.1 kHz, and where it came from: the song less its backing ("voice"), or the
    whole song where there is no backing ("mix")."""
    import soundfile as sf
    song = decode(source_of(folder), binary=binary)
    backing = folder / "stems" / "karaoke_backing.wav"
    if not backing.exists():
        return song.mean(axis=1), "mix"
    rest, rate = sf.read(str(backing), dtype="float32", always_2d=True)
    if rate != RATE:
        raise RuntimeError(f"{backing}: {rate} Hz, expected {RATE}")
    n = min(len(song), len(rest))
    return (song[:n] - rest[:n]).mean(axis=1), "voice"


def keep_voice(folder: Path, samples: np.ndarray) -> Path:
    """stems/vocals.flac: the lead voice, mono 16-bit at 44.1 kHz (some 12 MB a song)."""
    import soundfile as sf
    from karaokifex.workspace import partial_path
    path = folder / "stems" / "vocals.flac"
    path.parent.mkdir(exist_ok=True)
    partial = partial_path(path)
    sf.write(str(partial), np.clip(samples, -1, 1), RATE, format="FLAC", subtype="PCM_16")
    partial.replace(path)
    return path


def level_db(samples_16k: np.ndarray, frames: int) -> np.ndarray:
    """The voice's level in dB each 10 ms, `frames` long (RMVPE's frames: centred, 160 samples apart)."""
    import librosa
    rms = librosa.feature.rms(y=samples_16k, frame_length=640, hop_length=160, center=True)[0]
    db = 20 * np.log10(np.maximum(rms, 1e-7))
    out = np.full(frames, db.min() if len(db) else -140.0, dtype=np.float32)
    out[: min(frames, len(db))] = db[:frames]
    return out


def to_midi(f0: np.ndarray) -> np.ndarray:
    midi = np.full(len(f0), np.nan, dtype=np.float32)
    on = f0 > 0
    midi[on] = 69 + 12 * np.log2(f0[on] / 440.0)
    return midi


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    edges = np.flatnonzero(np.diff(np.concatenate(([0], mask.astype(np.int8), [0]))))
    return list(zip(edges[::2], edges[1::2]))


def notes_from(midi: np.ndarray, level: np.ndarray) -> list[list]:
    """The notes of a pitch line (MIDI, NaN unvoiced, a frame each 10 ms) with the voice's level in dB."""
    from scipy.ndimage import median_filter
    midi = midi.copy()
    loud = level > np.nanmax(level) - QUIET if len(level) else np.zeros(0, bool)
    midi[~loud[: len(midi)]] = np.nan
    voiced = ~np.isnan(midi)
    # short gaps bridged: the pitch drawn straight across them
    bridge = int(round(BRIDGE / FRAME))
    for a, b in _runs(~voiced):
        if a > 0 and b < len(midi) and b - a <= bridge:
            midi[a:b] = np.linspace(midi[a - 1], midi[b], b - a + 2)[1:-1]
    voiced = ~np.isnan(midi)
    hold, shortest = int(round(HOLD / FRAME)), int(round(SHORTEST / FRAME))
    peak = float(np.nanmax(level)) if len(level) else 0.0
    notes: list[list] = []

    def close(a: int, b: int) -> None:
        if b - a < shortest:
            return
        seg = midi[a:b]
        pitch = float(np.median(seg))
        n = int(round(pitch))
        vol = float(np.clip((np.mean(level[a:b]) - (peak - QUIET)) / QUIET, 0, 1))
        notes.append([round(a * FRAME, 3), round((b - a) * FRAME, 3), n, int(round((pitch - n) * 100)), round(vol, 2)])

    for a, b in _runs(voiced):
        line = median_filter(midi[a:b], size=5, mode="nearest") if b - a >= 5 else midi[a:b]
        start, away, top = 0, 0, level[a]
        for i in range(1, b - a):
            ref = float(np.median(line[max(start, i - 15):i]))
            away = away + 1 if abs(line[i] - ref) > JUMP else 0
            top = max(top, level[a + i])
            if away >= hold:                                  # the pitch has moved on: a new note where it left
                close(a + start, a + i - away + 1)
                start, away, top = i - away + 1, 0, level[a + i]
                continue
            # a dip and a rise on one pitch: a new syllable, split at the dip's bottom
            if i - start > shortest and level[a + i] > level[a + i - 1] and top - level[a + i - 1] >= DIP:
                low = start + int(np.argmin(level[a + start:a + i]))
                if low - start >= shortest and level[a + i] - level[a + low] >= DIP * 0.66:
                    close(a + start, a + low)
                    start, top = low, level[a + i]
        close(a + start, b)
    return notes


def contour(midi: np.ndarray, step: int = 2) -> dict:
    line = midi[::step]
    return {"step": round(step * FRAME, 3),
            "pitch": [None if np.isnan(v) else int(round(v * 10)) for v in line]}


def analyse(folder: Path, rmvpe, *, binary: str = "ffmpeg") -> dict:
    """melody.json's content for the song in `folder`, with `rmvpe` (_rmvpe.Rmvpe) loaded."""
    import librosa
    sound, origin = voice(folder, binary=binary)
    if origin == "voice":
        keep_voice(folder, sound)
    at16 = librosa.resample(sound, orig_sr=RATE, target_sr=16000, res_type="soxr_hq")
    f0, _confidence = rmvpe.pitch(at16)
    midi = to_midi(f0)
    level = level_db(at16, len(midi))
    notes = notes_from(midi, level)
    pitches = [n[2] for n in notes]
    return {"version": VERSION, "model": "rmvpe", "source": origin,
            "range": [min(pitches), max(pitches)] if pitches else None,
            "notes": notes, "contour": contour(np.where(~np.isnan(midi) & (level > level.max() - QUIET), midi, np.nan))}


def write(folder: Path, name: str, data: dict) -> Path:
    from karaokifex.workspace import partial_path
    path = folder / name
    partial = partial_path(path)
    partial.write_text(json.dumps(data, separators=(",", ":")), encoding="utf-8")
    partial.replace(path)
    return path
