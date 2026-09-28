"""The chords: chords.json, from BTC's large vocabulary on the karaoke backing (the song without its lead voice).

BTC reads a constant-Q spectrum of the sound at 22.05 kHz, 144 bins at 24 a octave, a frame each 2048 samples
(FRAME, about 93 ms), in 10-second stretches of 108 frames, and names one of 170 chords a frame: 12 roots times
14 kinds (major, minor, sevenths, sixths, sus, dim, aug ...), X (something, not a chord it knows) and N (none).
Frame by frame it flickers, so the path through its odds is taken as a whole (Viterbi), a change of chord costing
as much as SWITCH makes it: a chord holds unless the music says otherwise for a while. Then the key, from how long
each chord's notes sound (Krumhansl's profiles), and the roots spelled for it: Bb in F, A# in B.

chords.json:
  version, model ("btc-large-voca"), source ("backing" or "mix")
  key       "A minor", "Eb major" ...
  chords    [[start s, length s, name ("Am7", "Eb", "N"), label (BTC's, "A:min7"), confidence 0-1], ...]
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

log = logging.getLogger("karaokifex")

VERSION = 1
SR = 22050
HOP = 2048
WINDOW = 108
FRAME = 10 / WINDOW      # s: BTC's frame as its own code times it, each 10 s stretch giving 108 of them
SWITCH = 0.04            # a frame's odds of a change of chord, for the whole path (Viterbi)
SHORTEST = 0.35          # s: a chord shorter is folded into the one before

CONFIG = {"feature_size": 144, "timestep": WINDOW, "num_chords": 170, "input_dropout": 0.2, "layer_dropout": 0.2,
          "attention_dropout": 0.2, "relu_dropout": 0.2, "num_layers": 8, "num_heads": 4, "hidden_size": 128,
          "total_key_depth": 128, "total_value_depth": 128, "filter_size": 128, "loss": "ce", "probs_out": False}

ROOTS = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
FLATS = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]
KINDS = ["min", "maj", "dim", "aug", "min6", "maj6", "min7", "minmaj7", "maj7", "7", "dim7", "hdim7", "sus2", "sus4"]
# how each kind is written after its root, and the notes it has (semitones over the root)
WRITTEN = {"maj": "", "min": "m", "dim": "dim", "aug": "aug", "min6": "m6", "maj6": "6", "min7": "m7",
           "minmaj7": "m(maj7)", "maj7": "maj7", "7": "7", "dim7": "dim7", "hdim7": "m7b5", "sus2": "sus2", "sus4": "sus4"}
TONES = {"maj": (0, 4, 7), "min": (0, 3, 7), "dim": (0, 3, 6), "aug": (0, 4, 8), "min6": (0, 3, 7, 9), "maj6": (0, 4, 7, 9),
         "min7": (0, 3, 7, 10), "minmaj7": (0, 3, 7, 11), "maj7": (0, 4, 7, 11), "7": (0, 4, 7, 10), "dim7": (0, 3, 6, 9),
         "hdim7": (0, 3, 6, 10), "sus2": (0, 2, 7), "sus4": (0, 5, 7)}
# Krumhansl-Kessler key profiles
MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
# keys whose spelling takes flats: F and down the circle of fifths, and their relative minors
FLAT_MAJORS = {5, 10, 3, 8, 1, 6}          # F Bb Eb Ab Db Gb
FLAT_MINORS = {2, 7, 0, 5, 10, 3}          # D G C F Bb Eb minor


def label(i: int) -> tuple[int | None, str]:
    """BTC's chord `i` as (root 0-11, kind), or (None, "N"/"X")."""
    if i == 169:
        return None, "N"
    if i == 168:
        return None, "X"
    return i // 14, KINDS[i % 14]


def btc_label(i: int) -> str:
    root, kind = label(i)
    if root is None:
        return kind
    return ROOTS[root] if kind == "maj" else f"{ROOTS[root]}:{kind}"


def features(y: np.ndarray) -> np.ndarray:
    """BTC's input, as its own code makes it: the CQT of each 10 s on its own, joined, log magnitude; (frames, 144)."""
    import librosa
    step = SR * 10
    parts = [librosa.cqt(y[a:a + step], sr=SR, n_bins=144, bins_per_octave=24, hop_length=HOP)
             for a in range(0, max(len(y), 1), step) if len(y[a:a + step]) > HOP]
    cqt = np.concatenate(parts, axis=1) if parts else np.zeros((144, 1))
    return np.log(np.abs(cqt) + 1e-6).T.astype(np.float32)


class Btc:
    def __init__(self, weights: Path, device: str = "cuda") -> None:
        import torch
        from karaokifex.music._btc import BTC_model
        self.torch, self.device = torch, torch.device(device)
        ckpt = torch.load(str(weights), map_location="cpu", weights_only=False)
        self.mean, self.std = float(ckpt["mean"]), float(ckpt["std"])
        model = BTC_model(config=CONFIG)
        model.load_state_dict(ckpt["model"])
        self.model = model.eval().to(self.device)

    def odds(self, feats: np.ndarray) -> np.ndarray:
        """(frames, 170): each chord's probability a frame."""
        torch = self.torch
        n = len(feats)
        x = (feats - self.mean) / self.std
        x = np.pad(x, ((0, (-n) % WINDOW), (0, 0)))
        with torch.no_grad():
            t = torch.from_numpy(x.reshape(-1, WINDOW, 144)).to(self.device)
            hidden, _ = self.model.self_attn_layers(t)
            logits = self.model.output_layer.output_projection(hidden)
            probs = torch.softmax(logits, -1).reshape(-1, 170)[:n]
        return probs.float().cpu().numpy()


def viterbi(odds: np.ndarray, switch: float = SWITCH) -> np.ndarray:
    """The likeliest chord a frame over the whole song, a change costing log(switch / 169) against log(1 - switch)."""
    n, k = odds.shape
    logp = np.log(np.maximum(odds, 1e-9))
    stay, move = np.log(1 - switch), np.log(switch / (k - 1))
    score = logp[0].copy()
    back = np.zeros((n, k), dtype=np.int16)
    for t in range(1, n):
        best = int(score.argmax())
        kept = score + stay
        moved = score[best] + move
        back[t] = np.where(kept >= moved, np.arange(k), best)
        score = np.maximum(kept, moved) + logp[t]
    path = np.zeros(n, dtype=np.int16)
    path[-1] = int(score.argmax())
    for t in range(n - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def key_of(segments: list[tuple[int, float]]) -> tuple[int, str]:
    """The key (tonic 0-11, "major"/"minor") from (chord index, seconds) pairs: each chord's notes weighted by how
    long they sound, against Krumhansl's profiles in every key."""
    weight = np.zeros(12)
    for i, secs in segments:
        root, kind = label(i)
        if root is not None:
            for t in TONES[kind]:
                weight[(root + t) % 12] += secs * (1.5 if t == 0 else 1.0)
    if not weight.any():
        return 0, "major"
    best = max(((np.corrcoef(np.roll(prof, tonic), weight)[0, 1], tonic, mode)
                for prof, mode in ((MAJOR, "major"), (MINOR, "minor")) for tonic in range(12)))
    return best[1], best[2]


def spell(root: int, key: tuple[int, str]) -> str:
    flat = key[0] in (FLAT_MAJORS if key[1] == "major" else FLAT_MINORS)
    return (FLATS if flat else ROOTS)[root]


def name(i: int, key: tuple[int, str]) -> str:
    root, kind = label(i)
    if root is None:
        return kind
    return spell(root, key) + WRITTEN[kind]


def chords_from(odds: np.ndarray) -> tuple[str, list[list]]:
    path = viterbi(odds)
    runs, start = [], 0
    for t in range(1, len(path) + 1):
        if t == len(path) or path[t] != path[start]:
            runs.append([int(path[start]), start, t])
            start = t
    # a chord shorter than SHORTEST folded into the one before it (the first into the one after)
    shortest = max(1, int(round(SHORTEST / FRAME)))
    merged: list[list] = []
    for r in runs:
        if merged and (r[2] - r[1] < shortest or r[0] == merged[-1][0]):
            merged[-1][2] = r[2]
        else:
            merged.append(r)
    if len(merged) > 1 and merged[0][2] - merged[0][1] < shortest:
        merged[1][1] = merged[0][1]
        merged.pop(0)
    key = key_of([(i, (b - a) * FRAME) for i, a, b in merged])
    out = [[round(a * FRAME, 3), round((b - a) * FRAME, 3), name(i, key), btc_label(i),
            round(float(odds[a:b, i].mean()), 2)] for i, a, b in merged]
    return f"{spell(key[0], key)} {key[1]}", out


def analyse(folder: Path, btc: Btc, *, binary: str = "ffmpeg") -> dict:
    """chords.json's content for the song in `folder`, with `btc` loaded."""
    import librosa
    import soundfile as sf
    from karaokifex.music.melody import decode, source_of
    backing = folder / "stems" / "karaoke_backing.wav"
    if backing.exists():
        sound, rate = sf.read(str(backing), dtype="float32", always_2d=True)
        origin = "backing"
    else:
        sound, rate, origin = decode(source_of(folder), binary=binary), 44100, "mix"
    y = librosa.resample(sound.mean(axis=1), orig_sr=rate, target_sr=SR, res_type="soxr_hq")
    key, chords = chords_from(btc.odds(features(y)))
    return {"version": VERSION, "model": "btc-large-voca", "source": origin, "key": key, "chords": chords}
