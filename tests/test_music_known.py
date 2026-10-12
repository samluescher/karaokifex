"""The known step of python -m karaokifex.music (music/__main__.py): the notes of a source of ours put into melody.json
(notes.apply) between the melody and the tab, and taken out again. Made-up tune only."""

import json

from karaokifex.music.__main__ import NoLock, run
from test_notes import detected_song


def test_the_known_step_of_the_music_command_puts_the_notes_in_and_the_tab_follows(tmp_path):
    song, folder = detected_song(tmp_path)
    (folder / "chords.json").write_text(json.dumps({"key": "C major", "chords": [[0.0, 90.0, "C", "C", 0.9]]}))
    assert run(folder, {"known", "tab"}, False, NoLock(), 2.5, True).keys() == {"tab"}            # nothing known: nothing to do
    known = [[s, l, m + 1, None] for s, l, m in song]
    (folder / "notes-source.json").write_text(json.dumps({"version": 1, "source": "mine", "kind": "midi", "notes": known}))
    done = run(folder, {"known", "tab"}, False, NoLock(), 2.5, True)
    assert done["known"].startswith(f"{len(song)} detected notes put right by mine") and "tab" in done
    assert [n[2] for n in json.loads((folder / "melody.json").read_text())["notes"]][:3] == [m + 1 for _, _, m in song][:3]
    assert "known" not in run(folder, {"known"}, False, NoLock(), 2.5, True)                      # up to date
    (folder / "notes-source.json").unlink()
    assert run(folder, {"known"}, False, NoLock(), 2.5, True)["known"] == "back to the notes as detected"
    assert json.loads((folder / "melody.json").read_text())["notes"][0][2] == song[0][2]
