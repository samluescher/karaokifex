"""The music models' trained weights, fetched once on first use into MODELS (KARAOKIFEX_MUSIC_MODELS, else
~/.cache/karaokifex-music) and checked by their size:

  rmvpe.pt                  181 MB   RMVPE's, from RVC's model repository on Hugging Face (MIT)
  btc_model_large_voca.pt    12 MB   BTC's large-vocabulary model, from its GitHub repository (MIT)
"""

from __future__ import annotations

import logging
import os
import urllib.request
from pathlib import Path

log = logging.getLogger("karaokifex")

MODELS = Path(os.environ.get("KARAOKIFEX_MUSIC_MODELS", Path.home() / ".cache" / "karaokifex-music"))
WEIGHTS = {
    "rmvpe.pt": ("https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/rmvpe.pt", 181_184_272),
    "btc_model_large_voca.pt": ("https://github.com/jayg996/BTC-ISMIR19/raw/master/test/btc_model_large_voca.pt",
                                12_229_576),
}


def weights(name: str) -> Path:
    """The file of `name`'s weights, fetched first if it isn't here whole."""
    url, size = WEIGHTS[name]
    path = MODELS / name
    if path.exists() and path.stat().st_size == size:
        return path
    MODELS.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    log.info("fetching %s (%.0f MB) from %s", name, size / 1e6, url)
    with urllib.request.urlopen(url, timeout=60) as r, open(partial, "wb") as f:
        while chunk := r.read(1 << 20):
            f.write(chunk)
    got = partial.stat().st_size
    if got != size:
        partial.unlink()
        raise RuntimeError(f"{name}: fetched {got} bytes, expected {size}")
    partial.replace(path)
    return path
