"""The word matcher (timing._match_pairs) row by row in numpy pairs the words exactly as the cell-by-cell programme did,
on made-up songs: near-miss spellings, one word heard as two and two as one, words that sound alike, backing vocals,
line windows and cut lines; and a long song's matching takes seconds."""

from __future__ import annotations

import math
import random
import time

from karaokifex.models import TimedWord
from karaokifex.timing import (BACKING, BACKING_FOR_LEAD, MERGE_MIN, _Match, _similarity, _Token, _weight, _match_pairs,
                               normalize)

WORDS = ["all", "right", "alright", "there", "their", "they're", "night", "nite", "colour", "color", "gonna", "going",
         "to", "la", "love", "loved", "lover", "be", "bee", "sea", "see", "wanna", "want", "dance", "dancing", "danced",
         "yeah", "yea", "oh", "ohh", "hold", "holding", "on", "only", "lonely", "baby", "babe", "way", "away", "a"]


# the matcher as it was, cell by cell: the reference the numpy one has to agree with
def reference_match_pairs(tokens: Sequence[_Token], heard: Sequence[TimedWord], windows: Sequence[Window | None],
                 slack: float, language: str | None = None, *, skip: frozenset[int] = frozenset()
                 ) -> list[_Match]:
    """Order-preserving pairing of lyric and heard words maximising total (score-weighted) similarity.

    An LCS-style dynamic programme that also allows one lyric word sung as two heard
    words ("alright" / "all right") and vice versa. A pair is only allowed when the
    heard word lies within its lyric line's window (± slack); the lines in `skip` (cut) pair with nothing; a word
    heard in the backing vocals counts in full for a lyric word in parentheses, at BACKING_FOR_LEAD for another
    (lyric sites leave many a backing part unmarked, and the backing echoes the lead).
    """
    phonetic = language == "en"
    heard_norm = [normalize(word.text) for word in heard]
    weights = [_weight(word) for word in heard]
    backing = [word.source == BACKING for word in heard]  # heard in the backing vocals: for backing words only
    rows, columns = len(tokens) + 1, len(heard) + 1
    score = [[0.0] * columns for _ in range(rows)]
    move = [bytearray(columns) for _ in range(rows)]  # 0 skip lyric, 1 skip heard, 2 = 1:1, 3 = 1:2, 4 = 2:1
    bounds = []
    for token in tokens:
        window = windows[token.line]
        if token.line in skip:
            bounds.append((math.inf, -math.inf))
        else:
            bounds.append((window[0] - slack, window[1] + slack) if window else (-math.inf, math.inf))

    for i in range(1, rows):
        token = tokens[i - 1]
        low, high = bounds[i - 1]
        previous, row, moves = score[i - 1], score[i], move[i]
        before = score[i - 2] if i >= 2 else None
        pair_ok = before is not None and tokens[i - 2].line == token.line
        for j in range(1, columns):
            if previous[j] >= row[j - 1]:
                best, how = previous[j], 0
            else:
                best, how = row[j - 1], 1
            start = heard[j - 1].start
            if low <= start <= high:
                # a word heard in the backing vocals counts for a lyric word outside parentheses at a part only
                w1 = weights[j - 1] * (BACKING_FOR_LEAD if backing[j - 1] and not token.backing else 1.0)
                similarity = _similarity(token.norm, heard_norm[j - 1], phonetic)
                if similarity and previous[j - 1] + similarity * w1 > best:
                    best, how = previous[j - 1] + similarity * w1, 2
                if j >= 2 and token.norm[:1] == heard_norm[j - 2][:1] and low <= heard[j - 2].start:
                    w2 = weights[j - 2] * (BACKING_FOR_LEAD if backing[j - 2] and not token.backing else 1.0)
                    similarity = _similarity(token.norm, heard_norm[j - 2] + heard_norm[j - 1], phonetic)
                    gain = similarity * (w2 + w1) / 2
                    if similarity >= MERGE_MIN and previous[j - 2] + gain > best:
                        best, how = previous[j - 2] + gain, 3
                if pair_ok and tokens[i - 2].norm[:1] == heard_norm[j - 1][:1]:
                    w0 = weights[j - 1] * (BACKING_FOR_LEAD if backing[j - 1] and not tokens[i - 2].backing else 1.0)
                    similarity = _similarity(tokens[i - 2].norm + token.norm, heard_norm[j - 1], phonetic)
                    gain = similarity * min(w0, w1) * 1.5  # two lyric words explained by one heard word
                    if similarity >= MERGE_MIN and before[j - 1] + gain > best:  # type: ignore[index]
                        best, how = before[j - 1] + gain, 4  # type: ignore[index]
            row[j], moves[j] = best, how

    matches: list[_Match] = []
    i, j = rows - 1, columns - 1
    while i > 0 and j > 0:
        how = move[i][j]
        if how == 0:
            i -= 1
        elif how == 1:
            j -= 1
        elif how == 2:
            matches.append(_Match(i - 1, 1, j - 1, 1))
            i, j = i - 1, j - 1
        elif how == 3:
            matches.append(_Match(i - 1, 1, j - 2, 2))
            i, j = i - 1, j - 2
        else:
            matches.append(_Match(i - 2, 2, j - 1, 1))
            i, j = i - 2, j - 1
    return matches[::-1]




def song(rng: random.Random, lines: int, synced: bool):
    tokens, windows, heard, t = [], [], [], 0.0
    for line in range(lines):
        words = [rng.choice(WORDS) for _ in range(rng.randint(1, 7))]
        backing = rng.random() < 0.15
        start = t
        for position, word in enumerate(words):
            tokens.append(_Token(line, position, word, normalize(word), backing))
            sung = rng.random()
            if sung < 0.7:      # heard as it is, or near it
                said = word if rng.random() < 0.7 else rng.choice(WORDS)
                heard.append(TimedWord(said, t, t + 0.3, rng.choice([None, rng.random()]),
                                       BACKING if rng.random() < 0.1 else None))
            elif sung < 0.8:    # heard as two
                heard.append(TimedWord(word[:len(word) // 2] or word, t, t + 0.15, rng.random()))
                heard.append(TimedWord(word[len(word) // 2:] or word, t + 0.15, t + 0.3, rng.random()))
            if rng.random() < 0.1:   # something nobody wrote down
                heard.append(TimedWord(rng.choice(WORDS), t + 0.3, t + 0.4, rng.random()))
            t += 0.5
        windows.append((start + rng.uniform(-2, 2), t + rng.uniform(-2, 2)) if synced else None)
        t += rng.uniform(0, 3)
    return tokens, heard, windows


def test_the_numpy_matcher_pairs_exactly_as_the_cell_by_cell_one():
    rng = random.Random(7)
    for case in range(120):
        tokens, heard, windows = song(rng, rng.randint(1, 14), synced=case % 3 != 0)
        skip = frozenset(rng.sample(range(len(windows)), k=min(len(windows), rng.randint(0, 2)))) if case % 4 == 0 else frozenset()
        slack = rng.choice([0.0, 1.0, 5.0, 15.0])
        language = rng.choice(["en", "de", None])
        assert _match_pairs(tokens, heard, windows, slack, language, skip=skip) == \
            reference_match_pairs(tokens, heard, windows, slack, language, skip=skip), case


def test_nothing_to_pair_is_no_pairs():
    tokens, heard, windows = song(random.Random(1), 3, synced=True)
    assert _match_pairs([], heard, windows, 5.0) == []
    assert _match_pairs(tokens, [], windows, 5.0) == []


def test_a_long_unsynced_song_is_matched_in_seconds():
    rng = random.Random(3)
    tokens, heard, windows = song(rng, 600, synced=False)       # some 2 400 words each side, nowhere to look first
    began = time.perf_counter()
    matches = _match_pairs(tokens, heard, windows, 5.0, "en")
    assert time.perf_counter() - began < 20
    assert len(matches) > len(tokens) / 3
