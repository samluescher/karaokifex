"""python -m karaokifex.music <song folder>... | --library <root> [--only "Artist - Song"]...
                             [--steps melody,chords,tab] [--again] [--gpu-lock FILE] [--free-gb 2.5] [--cpu]

One song at a time. The models are loaded for a song and let go after it, inside the GPU lock karaokifex's runs
share (--gpu-lock, else KARAOKIFEX_GPU_LOCK), so a song being made and this take turns on the card. Before a
song, it waits while the card has less than --free-gb free (the line model and Chatterbox may be on it), up to
WAIT_MOST s, then does that song on the CPU. A file is made again when what it's made from is newer, or with
--again; a song that fails is logged and the next one taken.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

from karaokifex.music import chords as chords_mod
from karaokifex.music import melody as melody_mod
from karaokifex.music import tab as tab_mod

log = logging.getLogger("karaokifex")
WAIT_MOST = 20 * 60


class NoLock:
    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False


def newer(target: Path, *sources: Path) -> bool:
    """`target` missing, or older than any of `sources` that exist."""
    if not target.exists():
        return True
    t = target.stat().st_mtime
    return any(s.exists() and s.stat().st_mtime > t for s in sources)


def free_gb() -> float | None:
    try:
        import torch
        if not torch.cuda.is_available():
            return None
        free, _ = torch.cuda.mem_get_info()
        return free / 1e9
    except Exception:  # noqa: BLE001
        return None


def device_for(want_free: float, cpu: bool) -> str:
    if cpu:
        return "cpu"
    waited = 0.0
    while True:
        free = free_gb()
        if free is None:
            return "cpu"
        if free >= want_free:
            return "cuda"
        if waited >= WAIT_MOST:
            log.warning("the card has %.1f GB free after %d min: this song on the CPU", free, waited // 60)
            return "cpu"
        if waited == 0:
            log.info("the card has %.1f GB free, wanting %.1f: waiting", free, want_free)
        time.sleep(30)
        waited += 30


def songs(args) -> list[Path]:
    if args.library:
        root = Path(args.library)
        found = sorted(p for p in root.iterdir() if p.is_dir() and melody_mod.source_of(p))
        if args.only:
            found = [p for p in found if p.name in set(args.only)]
        return found
    return [Path(f) for f in args.folders]


def run(folder: Path, steps: set[str], again: bool, lock, want_free: float, cpu: bool) -> dict:
    source = melody_mod.source_of(folder)
    backing = folder / "stems" / "karaoke_backing.wav"
    melody_json, chords_json, tab_json = folder / "melody.json", folder / "chords.json", folder / "tab.json"
    need_melody = "melody" in steps and (again or newer(melody_json, source, backing))
    need_chords = "chords" in steps and (again or newer(chords_json, source, backing))
    done: dict = {}
    if need_melody or need_chords:
        from karaokifex.gpu import free_gpu_memory
        from karaokifex.music.weights import weights
        with lock:
            device = device_for(want_free, cpu)
            if need_melody:
                from karaokifex.music._rmvpe import Rmvpe
                t0 = time.perf_counter()
                model = Rmvpe(weights("rmvpe.pt"), device)
                data = melody_mod.analyse(folder, model)
                del model
                free_gpu_memory()
                melody_mod.write(folder, "melody.json", data)
                done["melody"] = f"{len(data['notes'])} notes from the {data['source']}, {time.perf_counter() - t0:.1f} s"
            if need_chords:
                t0 = time.perf_counter()
                model = chords_mod.Btc(weights("btc_model_large_voca.pt"), device)
                data = chords_mod.analyse(folder, model)
                del model
                free_gpu_memory()
                melody_mod.write(folder, "chords.json", data)
                done["chords"] = f"{len(data['chords'])} chords in {data['key']}, {time.perf_counter() - t0:.1f} s"
    if "tab" in steps and (again or newer(tab_json, melody_json, chords_json)) and (melody_json.exists() or chords_json.exists()):
        read = lambda p: json.loads(p.read_text(encoding="utf-8")) if p.exists() else None
        data = tab_mod.make(read(chords_json), read(melody_json))
        melody_mod.write(folder, "tab.json", data)
        done["tab"] = f"{len(data['chords'])} chords, {len(data['notes'])} notes"
    return done


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m karaokifex.music", description=__doc__.split("\n\n")[1])
    ap.add_argument("folders", nargs="*")
    ap.add_argument("--library")
    ap.add_argument("--only", action="append")
    ap.add_argument("--steps", default="melody,chords,tab")
    ap.add_argument("--again", action="store_true")
    ap.add_argument("--gpu-lock", default=os.environ.get("KARAOKIFEX_GPU_LOCK"))
    ap.add_argument("--free-gb", type=float, default=2.5)
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S", stream=sys.stdout)
    from karaokifex.gpu import SharedGpu
    lock = SharedGpu(Path(args.gpu_lock)) if args.gpu_lock else NoLock()
    steps = {s.strip() for s in args.steps.split(",") if s.strip()}
    todo = songs(args)
    log.info("%d songs: %s", len(todo), ", ".join(sorted(steps)))
    failed = 0
    for n, folder in enumerate(todo, 1):
        try:
            done = run(folder, steps, args.again, lock, args.free_gb, args.cpu)
            log.info("[%d/%d] %s: %s", n, len(todo), folder.name, "; ".join(f"{k} {v}" for k, v in done.items()) or "up to date")
        except Exception as e:  # noqa: BLE001
            failed += 1
            log.exception("[%d/%d] %s failed: %s", n, len(todo), folder.name, e)
    log.info("done: %d songs, %d failed", len(todo), failed)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
