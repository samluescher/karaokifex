"""The lead melody as notes: melody.json, from the lead voice's pitch.

The voice is the song less its karaoke backing: karaokifex made the backing as the song (ffmpeg's decode at
44.1 kHz, float) minus the lead it separated, so the same decode less the backing gives that lead back exactly,
with nothing separated again. It is kept as stems/vocals.flac (mono, 44.1 kHz), as the drums and kick are. A song
without a backing is read whole: RMVPE follows the voice in a full mix too, a little less surely.

RMVPE gives the voice's pitch every 10 ms, and how sure it is of each frame (its salience, 0-1). The notes are its
runs: gaps under BRIDGE s bridged, a new note where the pitch moves more than JUMP semitones from the note's and
stays there HOLD s, or where the voice dips DIP dB and comes back (a new syllable on the same pitch), and shorter
than SHORTEST s dropped. Frames where the voice is QUIET dB under its loudest are left out: what bleeds through
there isn't sung.

Which semitone a note is on is the song's own: records are not all at A440, so the notes are rounded against the
song's tuning, the pitch less tuning/100, not against A440. The tuning is where the pitches of the karaoke backing
pile up between A440's semitones: librosa.estimate_tuning's method (the offsets from A440's semitones of the spectral
peaks of the louder half, their histogram of 1 cent, its peak, taken at the middle of its bin), done in numpy
(spectral_peaks; librosa's own peak finder crashes Python on Windows with numba 0.67), on reference.flac when the song
has one, else stems/karaoke_backing.wav, the middle TUNING_SECONDS of it. tuning_conf is how far above an even spread
that peak stands: the share of the offsets within 5 cents of it, from BACKING_FLOOR (0.11, an even spread) to
BACKING_FULL taken to 0-1. Where the backing says too little (tuning_conf under BACKING_SURE: a song of drums and
noise) the voice's own pitches say it, the circular mean of their fractions over the frames that are sure (LEAD_SURE)
and sung, with their concentration R (0 even, 1 all one offset) from LEAD_FLOOR to LEAD_FULL as the confidence, taken
only from LEAD_TRUST up; where neither is trustworthy the tuning is 0 and tuning_from null.

Then the notes are cleaned. A fragment under FRAGMENT s is folded into the neighbour before or after it (the gap at
most NEIGHBOUR_GAP s) when that neighbour is longer and within NEIGHBOUR_SPAN semitone of it, the two lengths
weighing their pitch, level and conf. Then, with the key from chords.json (the chords come first: "A major",
"C minor"), a note whose pitch is within DOUBT cents of the edge between two semitones takes whichever of the two
that is in the key's scale (natural major or minor) when exactly one is, and keeps the nearer otherwise; with no
chords.json there is no snapping. A note's conf is the mean of RMVPE's confidence over its frames, the frames
outside the sung words counting 0: the sung words are every word's window in timings.json widened by WIDEN s each
side (no timings.json: all of it is sung).

reliability says how far the file as a whole can be trusted, each part 0-1 (null where the song lacks what it needs):
  covered        the median, over the sung words, of the share of a word's 20 ms frames that have a pitch
  stray          the share of the pitch's frames that lie outside the sung windows (lower is better)
  in_key         the share of the notes' length that is in the key, after the snapping (the key is the chords')
  conf           the mean of the contour's conf over the pitch's frames inside the sung windows
  words_no_note  the share of the words that no note overlaps at all (lower is better)
  score          the five, each taken to 0-1 between the value that scores 0 and the one that scores 1, weighed, over
                 the parts the song has (the weights then made to add up to 1):
                   covered 0.60 -> 0.97 (weight 0.30)      stray 0.35 -> 0.02 (0.20)      in_key 0.65 -> 0.95 (0.20)
                   conf 0.45 -> 0.82 (0.20)                words_no_note 0.30 -> 0.00 (0.10)
                 Set on the library's parts so that its weakest twentieth falls under 0.5 and a tenth under 0.7:
                 what scores low is rap, ad-libs and dance tracks, where the words are not sung to the pitch line;
                 the parts are kept so that rescore() can redo the sum when SCORE changes.

melody.json (version 2; version 1 had no tuning, tuning_conf, tuning_from, key, reliability or conf, and its notes
were rounded against A440):
  version, model ("rmvpe"), source ("voice" or "mix")
  tuning        the song's tuning against A440 in cents (+ is sharp), 0 where nothing says
  tuning_conf   how sure of it, 0-1; tuning_from "backing", "lead" or null (see above)
  key           the key the notes were snapped to and in_key counted against, or null
  notes         [[start s, length s, MIDI note, cents off it, level 0-1, conf 0-1], ...] in time order: the MIDI
                note a whole number on the song's own grid, the cents the note's pitch less that note less
                tuning/100 (so the note's pitch over A440's is note + cents/100 + tuning/100 semitones; the
                cents mostly under 50, a note snapped to its key can reach 70)
  range         [lowest, highest] MIDI note
  contour       {step: 0.02, pitch: [MIDI x 10, or null where unvoiced, ...] (absolute, on A440's scale, as RMVPE
                gave it), conf: [0-100 a frame, 0 where the pitch is null or the frame is outside the sung words]}
  reliability   {score, covered, stray, in_key, conf, words_no_note}: see above

karaokifex.music.notes (karaokifex-notes) adds notes_detected and known to this file where notes of a known source
agree with the voice; both are left out here.
"""

from __future__ import annotations

import json
import logging
import re
import subprocess
from pathlib import Path

import numpy as np

log = logging.getLogger("karaokifex")

VERSION = 2
FRAME = 0.01        # s: RMVPE's frame
STEP = 2            # RMVPE frames to the contour's frame (20 ms)
BRIDGE = 0.05       # s: gaps up to this are bridged within a note
JUMP = 0.7          # semitones from the note's pitch that start a new note ...
HOLD = 0.06         # s: ... once the pitch has stayed there this long
DIP = 6.0           # dB the voice drops, and comes back, between two syllables on one pitch
SHORTEST = 0.08     # s: shorter notes are dropped
QUIET = 45.0        # dB under the voice's loudest: frames quieter are not sung
RATE = 44100
FRAGMENT = 0.10     # s: a shorter note is a fragment, folded into a neighbour where one fits
NEIGHBOUR_GAP = 0.06    # s: the most a fragment may lie from the neighbour it is folded into
NEIGHBOUR_SPAN = 1.0    # semitones: the most the two may differ
DOUBT = 20          # cents from the edge between two semitones within which a note may take the key's side
WIDEN = 0.25        # s: a word's window is widened by this each side to count as sung
TUNING_RATE = 16000     # Hz: the sound the tuning is read from (reference.flac's own rate)
TUNING_SECONDS = 360    # s: of the middle of the backing, at most
BACKING_FLOOR, BACKING_FULL = 0.11, 0.40    # share of the offsets within 5 cents of the peak: an even spread, and all but certain
BACKING_SURE = 0.20     # tuning_conf from which the backing is believed
LEAD_SURE = 0.5         # RMVPE's confidence a frame needs to count for the voice's own tuning
LEAD_FLOOR, LEAD_FULL = 0.08, 0.40      # the voice's concentration R, taken to a confidence 0-1 between them
LEAD_TRUST = 0.15       # R from which the voice's own tuning is believed (a median error of about 3 cents)
# reliability.score: part -> (the value that scores 0, the value that scores 1, weight)
SCORE = {"covered": (0.60, 0.97, 0.30), "stray": (0.35, 0.02, 0.20), "in_key": (0.65, 0.95, 0.20),
         "conf": (0.45, 0.82, 0.20), "words_no_note": (0.30, 0.0, 0.10)}

MAJOR, MINOR = (0, 2, 4, 5, 7, 9, 11), (0, 2, 3, 5, 7, 8, 10)
TONICS = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
KEY = re.compile(r"^\s*([A-Ga-g])([#b]?)\s*(major|minor|maj|min|m)?\s*$")


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


# ---------------------------------------------------------------- the song's own: key, sung words, tuning
def key_scale(key: str | None) -> frozenset[int] | None:
    """The pitch classes (0 C .. 11 B) of the scale of a key as chords.json writes it ("A major", "Eb major",
    "C# minor"; a bare tonic is major), natural major or natural minor; None where there is no key."""
    m = KEY.match(key or "")
    if not m:
        return None
    tonic = (TONICS[m.group(1).upper()] + {"#": 1, "b": -1, "": 0}[m.group(2)]) % 12
    mode = (m.group(3) or "major").lower()
    return frozenset((tonic + d) % 12 for d in (MINOR if mode in ("minor", "min", "m") else MAJOR))


def words_of(timings: dict) -> list[tuple[float, float]]:
    """Every word's (start, end) in seconds in a timings.json (lines of words with start and end), a word with
    no end or none after its start left out."""
    out = []
    for line in (timings or {}).get("lines") or []:
        for w in line if isinstance(line, list) else (line or {}).get("words") or []:
            if isinstance(w, dict) and w.get("start") is not None and w.get("end") is not None and w["end"] > w["start"]:
                out.append((float(w["start"]), float(w["end"])))
    return out


def sung_windows(words: list[tuple[float, float]], widen: float = WIDEN) -> list[tuple[float, float]]:
    """The words' windows widened by `widen` each side, those that touch joined, in time order."""
    out: list[list[float]] = []
    for a, b in sorted(words):
        a, b = max(0.0, a - widen), b + widen
        if out and a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return [(a, b) for a, b in out]


def sung_mask(n: int, step: float, windows: list[tuple[float, float]] | None) -> np.ndarray:
    """A bool a frame (`step` s apart, `n` of them) for whether the frame lies in one of the windows; None for
    no windows is every frame."""
    if windows is None:
        return np.ones(n, dtype=bool)
    mask = np.zeros(n, dtype=bool)
    for a, b in windows:
        mask[min(n, int(np.ceil(a / step - 1e-9))):min(n, int(np.floor(b / step + 1e-9)) + 1)] = True
    return mask


def peak_offset(freqs: np.ndarray) -> tuple[float, float]:
    """Where, in cents against A440's semitones, the frequencies pile up (the peak of a histogram of 1 cent, at
    its bin's middle) and how far above an even spread they pile up (0-1)."""
    freqs = freqs[freqs > 0]
    if len(freqs) < 500:
        return 0.0, 0.0
    semis = 12 * np.log2(freqs / 440.0)
    counts, _ = np.histogram(semis - np.round(semis), bins=np.linspace(-0.5, 0.5, 101))
    peak = int(np.argmax(counts))
    near = np.take(counts, np.arange(peak - 5, peak + 6), mode="wrap").sum() / counts.sum()
    return peak + 0.5 - 50.0, float(np.clip((near - BACKING_FLOOR) / (BACKING_FULL - BACKING_FLOOR), 0, 1))


def spectral_peaks(y: np.ndarray, sr: int, n_fft: int = 2048, fmin: float = 150.0, fmax: float = 4000.0,
                   threshold: float = 0.1) -> tuple[np.ndarray, np.ndarray]:
    """(frequencies in Hz, magnitudes) of the spectral peaks of `y`, as librosa.piptrack finds them: in each frame
    (Hann window of n_fft, hop n_fft/4) the local maxima of the magnitude between fmin and fmax that stand over
    `threshold` of the frame's largest, their frequency and magnitude from a parabola through the peak and its
    neighbours. (piptrack's own local-maximum step is a numba ufunc that crashes the interpreter on Windows with
    numba 0.67, so this is the same sum in numpy.)"""
    from scipy.signal import get_window
    hop = n_fft // 4
    window = get_window("hann", n_fft).astype(np.float32)
    y = np.pad(np.asarray(y, dtype=np.float32), n_fft // 2)
    frames = np.lib.stride_tricks.sliding_window_view(y, n_fft)[::hop]
    bins = np.fft.rfftfreq(n_fft, 1.0 / sr)
    wanted = (bins >= fmin) & (bins < fmax)
    wanted[[0, -1]] = False
    freqs, mags = [], []
    for a in range(0, len(frames), 512):
        spec = np.abs(np.fft.rfft(frames[a:a + 512] * window, axis=1)).T.astype(np.float32)        # (bins, frames)
        mid = spec[1:-1]
        slope = 0.5 * (spec[2:] - spec[:-2])
        bend = 2 * mid - spec[2:] - spec[:-2]
        shift = slope / (bend + (np.abs(bend) < np.finfo(np.float32).tiny))
        high = spec * (spec > threshold * spec.max(axis=0, keepdims=True))
        peak = np.zeros(spec.shape, dtype=bool)
        peak[1:-1] = (high[1:-1] > high[:-2]) & (high[1:-1] >= high[2:])
        peak &= wanted[:, None]
        rows, cols = np.nonzero(peak[1:-1])
        freqs.append((rows + 1 + shift[rows, cols]) * sr / n_fft)
        mags.append(mid[rows, cols] + 0.5 * slope[rows, cols] * shift[rows, cols])
    return (np.concatenate(freqs), np.concatenate(mags)) if freqs else (np.zeros(0), np.zeros(0))


def tuning_of(y: np.ndarray, sr: int) -> tuple[float, float]:
    """(cents against A440 (+ sharp), 0-1) for the mono sound `y`: librosa's estimate_tuning at a resolution of
    1 cent (the spectral peaks of the louder half, their offsets from A440's semitones), and how sure."""
    if sr != TUNING_RATE:
        import librosa
        y = librosa.resample(y, orig_sr=sr, target_sr=TUNING_RATE, res_type="soxr_hq")
    freqs, mags = spectral_peaks(y, TUNING_RATE)
    if not len(freqs):
        return 0.0, 0.0
    return peak_offset(freqs[mags >= np.median(mags)])


def backing_tuning(folder: Path) -> tuple[float, float] | None:
    """The song's tuning from its backing, or None where it has none to read."""
    import soundfile as sf
    path = next((p for p in (folder / "reference.flac", folder / "stems" / "karaoke_backing.wav") if p.exists()), None)
    if path is None:
        return None
    with sf.SoundFile(str(path)) as f:
        want = int(TUNING_SECONDS * f.samplerate)
        start = max(0, (f.frames - want) // 2)
        f.seek(start)
        y = f.read(min(want, f.frames - start), dtype="float32", always_2d=True).mean(axis=1)
        rate = f.samplerate
    return tuning_of(y, rate)


def lead_tuning(pitch: np.ndarray, conf: np.ndarray) -> tuple[float, float]:
    """The voice's own tuning: the circular mean of the pitches' fractions (cents against A440's semitones) over
    the frames that are voiced and have a conf of LEAD_SURE or more, and its concentration R taken to 0-1."""
    use = ~np.isnan(pitch) & (conf >= LEAD_SURE)
    if use.sum() < 200:
        return 0.0, 0.0
    z = np.exp(2j * np.pi * pitch[use].astype(np.float64)).mean()
    r = float(abs(z))
    return float(np.angle(z) / (2 * np.pi) * 100), float(np.clip((r - LEAD_FLOOR) / (LEAD_FULL - LEAD_FLOOR), 0, 1))


def choose_tuning(backing: tuple[float, float] | None, lead: tuple[float, float] | None) -> tuple[float, float, str | None]:
    """(cents, confidence, where from): the backing's where it is sure, else the voice's where that is, else 0."""
    if backing and backing[1] >= BACKING_SURE:
        return round(backing[0], 1), round(backing[1], 2), "backing"
    if lead and lead[1] >= (LEAD_TRUST - LEAD_FLOOR) / (LEAD_FULL - LEAD_FLOOR):
        return round(lead[0], 1), round(lead[1], 2), "lead"
    return 0.0, 0.0, None


# ---------------------------------------------------------------- the notes
def notes_from(midi: np.ndarray, level: np.ndarray, conf: np.ndarray | None = None, tuning: float = 0.0) -> list[list]:
    """The notes of a pitch line (MIDI on A440's scale, NaN unvoiced, a frame each 10 ms) with the voice's level in
    dB and RMVPE's confidence a frame (none: all sure), rounded against the song's tuning (cents against A440):
    [[start, length, note, cents, level, conf], ...], before they are cleaned."""
    from scipy.ndimage import median_filter
    midi = midi.copy() - np.float32(tuning / 100.0)                 # on the song's own grid
    conf = np.ones(len(midi), dtype=np.float32) if conf is None else conf
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
        notes.append([round(a * FRAME, 3), round((b - a) * FRAME, 3), n, int(round((pitch - n) * 100)), round(vol, 2),
                      round(float(np.mean(conf[a:b])), 2)])

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


def _pitch(note: list) -> float:
    """A note's pitch on the song's grid, in semitones."""
    return note[2] + note[3] / 100.0


def _fuse(a: list, b: list) -> list:
    """Two notes (a before b) as one over both of them and what lies between, their pitch, level and conf weighed
    by their lengths."""
    la, lb = a[1], b[1]
    pitch = (la * _pitch(a) + lb * _pitch(b)) / (la + lb)
    note = int(round(pitch))
    return [round(a[0], 3), round(b[0] + b[1] - a[0], 3), note, int(round((pitch - note) * 100)),
            round((la * a[4] + lb * b[4]) / (la + lb), 2), round((la * a[5] + lb * b[5]) / (la + lb), 2)]


def merge_fragments(notes: list[list]) -> list[list]:
    """A note under FRAGMENT s folded into the neighbour before or after it (at most NEIGHBOUR_GAP s away) that is
    longer and within NEIGHBOUR_SPAN semitone of it: the nearer in pitch where both are, the one before on a tie."""
    out = [list(n) for n in notes]
    i = 0
    while i < len(out):
        n = out[i]
        best = None
        if n[1] < FRAGMENT:
            for j in (i - 1, i + 1):
                if not 0 <= j < len(out):
                    continue
                m = out[j]
                gap = n[0] - (m[0] + m[1]) if j < i else m[0] - (n[0] + n[1])
                dist = abs(_pitch(m) - _pitch(n))
                if gap <= NEIGHBOUR_GAP + 1e-9 and m[1] > n[1] and dist <= NEIGHBOUR_SPAN + 1e-9 and (best is None or dist < best[0]):
                    best = (dist, j)
        if best is None:
            i += 1
            continue
        j = best[1]
        lo = min(i, j)
        out[lo:lo + 2] = [_fuse(out[lo], out[lo + 1])]
        i = max(0, lo - 1)                      # the new note and the one before it, looked at again
    return out


def snap_to_key(notes: list[list], scale: frozenset[int] | None) -> list[list]:
    """A note within DOUBT cents of the edge between two semitones (so it could be either) takes the one that is in
    the key where only one is; the others stay on the nearer. No scale: as they are."""
    if not scale:
        return [list(n) for n in notes]
    out = []
    for n in notes:
        n = list(n)
        if abs(n[3]) >= 50 - DOUBT:
            other = n[2] + (1 if n[3] > 0 else -1)
            if other % 12 in scale and n[2] % 12 not in scale:
                n[3] -= (other - n[2]) * 100
                n[2] = other
        out.append(n)
    return out


def clean(notes: list[list], scale: frozenset[int] | None) -> list[list]:
    """The notes as melody.json has them: fragments folded into their neighbours, the doubtful ones snapped to the key."""
    return snap_to_key(merge_fragments(notes), scale)


def contour(midi: np.ndarray, conf: np.ndarray | None = None, sung: np.ndarray | None = None, step: int = STEP) -> dict:
    """The pitch line each `step` frames (20 ms): the pitch as ints of 0.1 semitone or None, and conf, 0-100 (the
    larger of the frames the step spans), 0 where there is no pitch or the frame is outside the sung words."""
    line = midi[::step]
    out_conf = np.zeros(len(line), dtype=np.int64)
    if conf is not None:
        pad = np.concatenate([conf, np.zeros((-len(conf)) % step, dtype=conf.dtype)])
        out_conf = np.round(100 * pad.reshape(-1, step).max(axis=1)[: len(line)]).astype(np.int64)
    if sung is not None:
        out_conf[~sung[: len(line)]] = 0
    out_conf[np.isnan(line)] = 0
    return {"step": round(step * FRAME, 3), "pitch": [None if np.isnan(v) else int(round(v * 10)) for v in line],
            "conf": [int(c) for c in np.clip(out_conf, 0, 100)]}


# ---------------------------------------------------------------- how far it can be trusted
def reliability(notes: list[list], pitch: np.ndarray, conf: np.ndarray, words: list[tuple[float, float]] | None,
                scale: frozenset[int] | None, step: float = STEP * FRAME) -> dict:
    """reliability's parts and score for notes and a contour (`pitch` MIDI with NaN for none, `conf` 0-1, a frame each
    `step` s) with the song's words (start, end), None where it has none, and the scale of its key, None where none."""
    voiced = ~np.isnan(pitch)
    parts: dict = {"covered": None, "stray": None, "in_key": None, "conf": None, "words_no_note": None}
    if len(notes) and scale:
        length = np.array([n[1] for n in notes])
        parts["in_key"] = float((length * np.array([n[2] % 12 in scale for n in notes])).sum() / length.sum())
    if words:
        sung = sung_mask(len(pitch), step, sung_windows(words))
        parts["stray"] = float((voiced & ~sung).sum() / max(1, voiced.sum()))
        inside = voiced & sung
        parts["conf"] = float(conf[inside].mean()) if inside.any() else 0.0
        shares = [seg.mean() for seg in (voiced[int(a / step):int(b / step)] for a, b in words) if len(seg)]
        parts["covered"] = float(np.median(shares)) if shares else None
        start = np.array([n[0] for n in notes])
        end = start + np.array([n[1] for n in notes])
        parts["words_no_note"] = float(sum(not ((start < b) & (end > a)).any() for a, b in words) / len(words))
    elif voiced.any():
        parts["conf"] = float(conf[voiced].mean())
    parts["score"] = score_of(parts)
    return {k: None if v is None else round(v, 2) for k, v in parts.items()}


def score_of(parts: dict) -> float | None:
    """reliability's score from its parts: each taken to 0-1 between its floor and its full mark (SCORE) and
    weighed, over the parts that are there; None where none are."""
    got = [(min(1.0, max(0.0, (parts[k] - lo) / (hi - lo))), w) for k, (lo, hi, w) in SCORE.items() if parts.get(k) is not None]
    return None if not got else sum(s * w for s, w in got) / sum(w for _, w in got)


def rescore(data: dict) -> dict:
    """A melody.json's content with reliability's score made again from its parts (a change of SCORE needs no new
    analysis)."""
    r = dict(data.get("reliability") or {})
    s = score_of(r)
    r["score"] = None if s is None else round(s, 2)
    return {**data, "reliability": r}


# ---------------------------------------------------------------- the file
def read_key(folder: Path) -> str | None:
    """The key in the folder's chords.json, None where there is none."""
    try:
        return json.loads((folder / "chords.json").read_text(encoding="utf-8")).get("key") or None
    except (OSError, ValueError, AttributeError):
        return None


def read_words(folder: Path) -> list[tuple[float, float]] | None:
    """Every word's (start, end) in the folder's timings.json, None where it has none."""
    try:
        return words_of(json.loads((folder / "timings.json").read_text(encoding="utf-8"))) or None
    except (OSError, ValueError, AttributeError):
        return None


def build(midi: np.ndarray, conf: np.ndarray, level: np.ndarray, *, origin: str, backing: tuple[float, float] | None = None,
          key: str | None = None, words: list[tuple[float, float]] | None = None) -> dict:
    """melody.json's content from RMVPE's pitch (MIDI, NaN unvoiced), its confidence and the voice's level, a
    frame each 10 ms: `backing` the backing's (tuning, conf) or None, `key` chords.json's, `words` timings.json's."""
    windows = sung_windows(words) if words else None
    conf10 = np.where(sung_mask(len(midi), FRAME, windows), conf, 0).astype(np.float32)
    shown = np.where(~np.isnan(midi) & (level > level.max() - QUIET), midi, np.nan)
    line = contour(shown, conf10, None)
    pitch20 = shown[::STEP].astype(np.float64)               # (not the contour's, which is rounded to 0.1 semitone)
    conf20 = np.array(line["conf"]) / 100.0
    tuning, tuning_conf, tuning_from = choose_tuning(backing, lead_tuning(pitch20, conf20))
    scale = key_scale(key)
    notes = clean(notes_from(midi, level, conf10, tuning), scale)
    pitches = [n[2] for n in notes]
    return {"version": VERSION, "model": "rmvpe", "source": origin, "tuning": tuning, "tuning_conf": tuning_conf,
            "tuning_from": tuning_from, "key": key if scale else None,
            "range": [min(pitches), max(pitches)] if pitches else None, "notes": notes, "contour": line,
            "reliability": reliability(notes, pitch20, conf20, words, scale)}


def analyse(folder: Path, rmvpe, *, binary: str = "ffmpeg") -> dict:
    """melody.json's content for the song in `folder`, with `rmvpe` (_rmvpe.Rmvpe) loaded."""
    import librosa
    sound, origin = voice(folder, binary=binary)
    if origin == "voice":
        keep_voice(folder, sound)
    at16 = librosa.resample(sound, orig_sr=RATE, target_sr=16000, res_type="soxr_hq")
    f0, confidence = rmvpe.pitch(at16)
    midi = to_midi(f0)
    level = level_db(at16, len(midi))
    return build(midi, confidence, level, origin=origin, backing=backing_tuning(folder), key=read_key(folder),
                 words=read_words(folder))


def write(folder: Path, name: str, data: dict) -> Path:
    from karaokifex.workspace import partial_path
    path = folder / name
    partial = partial_path(path)
    partial.write_text(json.dumps(data, separators=(",", ":")), encoding="utf-8")
    partial.replace(path)
    return path
