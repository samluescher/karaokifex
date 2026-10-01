"""Swiss German (--language gsw): a language whisper doesn't know as one of its own.

Whisper hears Swiss German as German and, more often than not, writes it down in Standard German --
"Ein Kaffee am Pistenrand" for "Ä Kafi am Pischterand" -- and lrclib may have a song's words in
Standard German too. For a Swiss German song neither is its lyrics. So:

  - lyrics from lrclib (or the web) count only when they read as Swiss German (reads_swiss_german);
  - whisper runs as German (whisper_language), prompted with Swiss German (PROMPT), which keeps its
    spelling nearer the dialect;
  - a transcription that still reads as Standard German is written back in Swiss German spelling by a
    chat model (rewrite: an OpenAI-style server, such as a local llama.cpp one), line by line, and
    those lines are then aligned to the voice like any lyrics.

How Swiss German reads: by its common words (isch, nöd, chli, gsi, üs...) against their Standard
German counterparts (ist, nicht, klein, gewesen, uns...) -- the share of the dialect's among those
found. A few lines have enough of them to tell.

    swiss(code) -> bool                 gsw, de-CH, "swiss german", "schwiizerdütsch"...
    whisper_language(code) -> str|None  "de" for Swiss German, else the code
    swiss_share(text) -> float|None     the dialect's share of the marker words (None: too few)
    reads_swiss_german(text), reads_standard_german(text) -> bool
    rewrite(lines, url, model) -> [lines] or None
    swiss_words(text) -> text           the common words swapped for their Swiss German (ich -> i, ist -> isch...)

A chat model given only an instruction may hand the lines back unchanged (an 8B one did); worked examples
(EXAMPLES) help it, and swiss_words then swaps whatever common word it left in Standard German, so what
comes back always reads as Swiss German -- and without a model at all, swiss_words alone does.
"""
from __future__ import annotations

import logging
import re
from typing import Any, Callable, Sequence

import requests

log = logging.getLogger(__name__)

CODES = {"gsw", "de-ch", "de_ch", "swiss german", "swissgerman", "schweizerdeutsch", "schwiizerdütsch", "schwiizertüütsch",
         "schwyzerdütsch", "alemannic", "als"}
# words only the dialect writes so (spellings vary from valley to valley: the common ones)
DIALECT = set("""
isch ischt nöd nid nüd nöt chli chlii chlei gsi gsii gseh gsee gseit gmacht ghört gfunde gloffe gange gsunge
hesch hets hät händ hend bisch simmer semmer sind's üs üsi üses öis ois öise mis mini dis dini sis sini
öppis öpper eifach hüt hütt chunnt chunt chum chumm gah gaht gang wotsch wosch wosch chasch chaschs chönd
nüt nüüt nümm nümme scho gäll lueg luege säge sägsch gits gits git's wäg wänn wenn's
chalt chopf chind chuchi chile chnü chunsch chömed dörf dörfsch
ha hei hei's gha hesch's lönd loh la lah
ou
""".split())
# their Standard German counterparts, and words the dialect doesn't use
STANDARD = set("""
ist nicht nichts klein kleine kleinen gewesen gesehen gesagt gemacht gehört gefunden gelaufen gegangen gesungen
hast hat haben habe bist sind uns unser unsere unseren mein meine meinen dein deine sein seine etwas jemand einfach
heute kommt komm gehen geht gehe willst kannst können nichts nimmer schon gibt weg wenn kalt kopf kind kinder küche
kirche knie kommst kommen darf darfst lassen lass lasst ein eine einen einem einer der ich wir auch
""".split())
MIN_MARKERS = 4
PROMPT = ("Das isch es Lied uf Schwiizerdütsch, gschribe wie mer's singt: i ha, mir sind, du bisch, es isch, "
          "nöd, chli, öppis, gsi, hüt, üs, eifach, Chuchi, Chopf.")

Post = Callable[..., Any]
# Standard German words and their Swiss German, as most dialects write them (Zurich's where they differ)
WORDS = {
    "ich": "i", "ist": "isch", "nicht": "nöd", "nichts": "nüt", "ein": "e", "eine": "e", "einen": "en", "einem": "emne",
    "einer": "ere", "wir": "mir", "uns": "üs", "unser": "üse", "unsere": "üsi", "haben": "händ", "habe": "ha", "hast": "hesch",
    "hat": "hät", "bin": "bi", "bist": "bisch", "gehen": "gönd", "geht": "gaht", "gehe": "gang", "kommen": "chömed",
    "kommt": "chunnt", "komm": "chum", "können": "chönd", "kann": "cha", "kannst": "chasch", "darf": "dörf", "heute": "hüt",
    "etwas": "öppis", "jemand": "öpper", "auch": "au", "noch": "no", "schon": "scho", "gewesen": "gsi", "gesehen": "gseh",
    "gesagt": "gseit", "gemacht": "gmacht", "gehabt": "gha", "gehört": "ghört", "gefunden": "gfunde", "gegangen": "gange", "gesungen": "gsunge", "klein": "chli",
    "kleine": "chlii", "kleinen": "chliine", "kleines": "chliises", "einfach": "eifach", "kaffee": "kafi", "küche": "chuchi",
    "kopf": "chopf", "kind": "chind", "kinder": "chind", "kalt": "chalt", "wenn": "wänn", "weg": "wäg", "lass": "lah",
    "lassen": "lah", "der": "de", "mein": "min", "meine": "mini", "dein": "din", "deine": "dini", "sein": "sin",
    "seine": "sini", "gibt": "git", "sehen": "gseh", "sagen": "säge", "schauen": "luege", "schau": "lueg",
}
EXAMPLES = [
    ("Ich habe heute nichts gemacht", "I ha hüt nüt gmacht"),
    ("Wir gehen noch einen Kaffee trinken", "Mir gönd no en Kafi go trinke"),
    ("Das ist nicht so einfach gewesen", "Das isch nöd so eifach gsi"),
    ("Komm, wir haben ein kleines Haus am Berg", "Chum, mir händ es chliises Huus am Bärg"),
    ("Kannst du mir etwas sagen?", "Chasch mer öppis säge?"),
]

# every Swiss German word the swap writes reads as the dialect too (i, en, chliine, Kafi...)
DIALECT |= {v for v in WORDS.values() if v not in STANDARD}


def swiss_words(text: str) -> str:
    """The common Standard German words in a line swapped for their Swiss German, capitals kept."""
    def swap(m: re.Match) -> str:
        w = m.group(0)
        s = WORDS.get(w.lower())
        if s is None:
            return w
        return s[:1].upper() + s[1:] if w[:1].isupper() else s
    return re.sub(r"[A-Za-zÄÖÜäöüß]+", swap, text)


def swiss(code: str | None) -> bool:
    return bool(code) and code.strip().lower() in CODES


def whisper_language(code: str | None) -> str | None:
    """The language whisper and the aligner are run in: German for Swiss German, else as given."""
    return "de" if swiss(code) else code


def _words(text: str) -> list[str]:
    return re.findall(r"[a-zäöüéèàâ']+", text.lower())


def swiss_share(text: str) -> float | None:
    """The dialect's share of the marker words in a text; None with fewer than MIN_MARKERS of them."""
    words = _words(text)
    d = sum(w in DIALECT for w in words)
    s = sum(w in STANDARD for w in words)
    return d / (d + s) if d + s >= MIN_MARKERS else None


def reads_swiss_german(text: str) -> bool:
    share = swiss_share(text)
    return share is not None and share >= 0.6


def reads_standard_german(text: str) -> bool:
    share = swiss_share(text)
    return share is not None and share < 0.4


SYSTEM = ("You write Swiss German (Schwiizerdütsch) the way it is sung and written in Swiss dialect songs. "
          "You are given lines a speech recogniser wrote in Standard German while listening to a song sung in Swiss "
          "German. Write each line back in Swiss German spelling, as the singer most likely sang it: the same meaning, "
          "the same order, one line out for each line in, nothing added, nothing left out, no translation notes. "
          "Use the common Swiss spellings (isch, nöd, chli, gsi, i, mir, üs, öppis, hüt, Chuchi, Chopf). "
          "Reply with the numbered lines only, as they came in.")


def rewrite(lines: Sequence[str], url: str, model: str | None = None, *, post: Post = requests.post,
            timeout: float = 120) -> list[str] | None:
    """The lines in Swiss German spelling, by a chat model at `url` (OpenAI-style, /chat/completions); None when it
    answers with another number of lines, or with lines that don't read as Swiss German."""
    numbered = "\n".join(f"{i + 1}. {line}" for i, line in enumerate(lines))
    shots = []
    for a, b in EXAMPLES:
        shots += [{"role": "user", "content": f"1. {a}"}, {"role": "assistant", "content": f"1. {b}"}]
    body = {"model": model or "default", "temperature": 0.2,
            "messages": [{"role": "system", "content": SYSTEM}, *shots, {"role": "user", "content": numbered}]}
    try:
        r = post(f"{url.rstrip('/')}/chat/completions", json=body, timeout=timeout)
        r.raise_for_status()
        text = r.json()["choices"][0]["message"]["content"]
    except (requests.RequestException, KeyError, IndexError, ValueError) as error:
        log.warning("no Swiss German rewrite: %s", error)
        return None
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    got: dict[int, str] = {}
    for line in text.splitlines():
        m = re.match(r"\s*(\d+)[.)]\s*(.+?)\s*$", line)
        if m and 1 <= int(m.group(1)) <= len(lines):
            got.setdefault(int(m.group(1)), m.group(2))
    out = [got.get(i + 1, "") for i in range(len(lines))]
    if len(got) != len(lines) or not all(out):
        log.warning("the Swiss German rewrite came back with %d of %d lines: not used", len(got), len(lines))
        return None
    # what the model left in Standard German: its common words swapped for their Swiss German
    out = [swiss_words(line) for line in out]
    if not reads_swiss_german(" ".join(out)):
        log.warning("the Swiss German rewrite doesn't read as Swiss German: not used")
        return None
    return out
