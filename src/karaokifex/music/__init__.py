"""The music in a finished song: its lead melody as notes, its chords, and a guitar tab from both, for the
Timeline's note and tab tracks. Run over song folders after they are made (they read the song's own audio and
its karaoke backing, and write beside them):

    python -m karaokifex.music <song folder>... [--steps melody,chords,known,tab] [--again]
    python -m karaokifex.music --library <root> [--only "Artist - Song"]... [--gpu-lock FILE]

  melody.py   melody.json: the lead voice's notes, from RMVPE's pitch of the voice (the song less its backing), on
              the song's own tuning and snapped to its key, with the pitch line, how sure of it, and how far to trust it
  chords.py   chords.json: the chords and the key, from BTC's large vocabulary (170 chords) on the backing
  align.py    the known notes of a source laid over our melody: transposition, time warp, agreement (karaokifex-notes)
  tab.py      tab.json: a guitar tab of our own, chord shapes and the melody as single notes
  weights.py  the two models' trained weights, fetched once on first use
  _rmvpe.py, _btc.py   the models' networks, vendored (MIT) so their weights load as they are

Each file is made again when what it is made from is newer, or with --again. The chords are made before the melody,
which reads their key. The known step puts the notes of a source of yours (notes-source.json, from karaokifex-notes) over the
melody's: no model, no GPU.
"""
