import numpy as np
import pytest

from karaokifex.activity import HOP, Activity
from karaokifex.models import LyricLine, TimedLine, TimedWord
from karaokifex.timing import (
    PHONETIC,
    AlignmentPlan,
    _similarity,
    align_lyrics,
    backing_heard,
    fill_unlyricked,
    filter_heard,
    forced_requests,
    heard_loops,
    lines_from_words,
    merge_heard,
    normalize,
    plan_alignment,
    split_words,
    video_order,
)


def heard(*items: tuple[str, float, float]) -> list[TimedWord]:
    return [TimedWord(text, start, end) for text, start, end in items]


def all_words(alignment):
    return [word for line in alignment.lines for word in line.words]


def test_normalize():
    assert normalize("Don’t!") == "don't"
    assert normalize("Café") == "cafe"
    assert normalize("'Cause") == "cause"
    assert normalize("...") == ""


def test_split_words_glues_punctuation_to_neighbours():
    assert split_words("Hello - world ...") == ["Hello -", "world ..."]
    assert split_words("... and then") == ["... and", "then"]
    assert split_words("♪ ♪") == []


def test_exact_match_uses_whisper_times():
    lines = [LyricLine(1.0, "Hello world"), LyricLine(3.0, "Goodbye moon")]
    words = heard(("hello", 1.1, 1.5), ("world", 1.6, 2.0), ("goodbye", 3.1, 3.6), ("moon", 3.7, 4.2))
    result = align_lyrics(lines, words)
    assert result.matched == result.total == 4
    assert [line.text for line in result.lines] == ["Hello world", "Goodbye moon"]
    assert result.lines[0].words[0] == TimedWord("Hello", 1.1, 1.5, source="whisper")
    assert result.lines[1].words[1].start == 3.7


def test_missing_word_is_interpolated_between_neighbours():
    lines = [LyricLine(1.0, "Hello world"), LyricLine(3.0, "Goodbye moon")]
    words = heard(("hello", 1.1, 1.5), ("goodbye", 3.1, 3.6), ("moon", 3.7, 4.2))
    result = align_lyrics(lines, words)
    world = result.lines[0].words[1]
    assert result.matched == 3
    assert 1.5 <= world.start < world.end <= 3.1


def test_extra_and_misspelled_words():
    lines = [LyricLine(None, "The colour of love")]
    words = heard(("uh", 0.2, 0.4), ("the", 1.0, 1.2), ("color", 1.3, 1.7), ("of", 1.8, 1.9),
                  ("love", 2.0, 2.6), ("yeah", 3.0, 3.5))
    result = align_lyrics(lines, words)
    assert result.matched == 4
    assert [w.start for w in all_words(result)] == [1.0, 1.3, 1.8, 2.0]


def test_repeated_chorus_matches_the_right_occurrence():
    lines = [LyricLine(10.0, "Na na hey"), LyricLine(20.0, "Something else"), LyricLine(50.0, "Na na hey")]
    words = heard(("something", 20.1, 20.5), ("else", 20.6, 21.0),
                  ("na", 50.1, 50.3), ("na", 50.4, 50.6), ("hey", 50.7, 51.0))
    result = align_lyrics(lines, words)
    assert result.matched == 5
    assert result.lines[2].words[0].start == 50.1
    # The unheard first chorus falls back to lrclib's timestamp.
    assert 10.0 <= result.lines[0].words[0].start < 20.0


def test_constant_offset_between_lyrics_and_video_is_compensated():
    # The video has a 12 s longer intro than the lrclib version.
    lines = [LyricLine(float(t), text) for t, text in [(1, "one two"), (3, "three four"), (5, "five six"), (7, "seven eight")]]
    words = heard(*[(word, 12 + t + i * 0.4, 12 + t + i * 0.4 + 0.3)
                    for t, text in [(1, "one two"), (3, "three four"), (5, "five six"), (7, "seven eight")]
                    for i, word in enumerate(text.split())])
    result = align_lyrics(lines, words)
    assert result.matched == 8


def test_plain_lyrics_without_timestamps():
    lines = [LyricLine(None, "one two"), LyricLine(None, "three four")]
    words = heard(("one", 1, 1.3), ("two", 1.4, 1.8), ("three", 2, 2.4), ("four", 2.5, 3))
    assert align_lyrics(lines, words).matched == 4


def test_unheard_plain_line_lands_in_the_gap():
    lines = [LyricLine(None, "one two"), LyricLine(None, "three four"), LyricLine(None, "five six")]
    words = heard(("one", 1, 1.4), ("two", 1.5, 2), ("five", 10, 10.4), ("six", 10.5, 11))
    middle = align_lyrics(lines, words).lines[1]
    assert 2.0 <= middle.start and middle.end <= 10.0


def test_times_are_monotonic_and_positive():
    lines = [LyricLine(None, "a b c d e f g")]
    words = heard(("a", 1, 1.2), ("c", 0.5, 0.7), ("d", 2, 2.2), ("x", 2.3, 2.4), ("g", 3, 3.2))
    result = all_words(align_lyrics(lines, words))
    assert len(result) == 7
    assert all(w.end > w.start for w in result)
    assert all(a.start <= b.start for a, b in zip(result, result[1:]))


def test_edge_cases():
    assert align_lyrics([], []).total == 0
    lonely = align_lyrics([LyricLine(None, "hi")], [])
    assert lonely.matched == 0 and lonely.lines[0].words[0].start == 0.0


def voice(duration: float, *voiced: tuple[float, float]) -> Activity:
    """A fake lead-vocal activity: singing in the given intervals, silence elsewhere."""
    frames = round(duration / HOP)
    mask = np.zeros(frames, dtype=bool)
    for start, end in voiced:
        mask[round(start / HOP): round(end / HOP)] = True
    return Activity(np.where(mask, -20.0, -80.0), mask, HOP)


def test_good_forced_alignment_beats_interpolation():
    lines = [LyricLine(1.0, "one two"), LyricLine(3.0, "three four")]
    words = heard(("one", 1.1, 1.4), ("two", 1.5, 1.9))
    forced = [None, [TimedWord("three", 3.2, 3.5, 0.8), TimedWord("four", 3.6, 4.0, 0.7)]]
    result = align_lyrics(lines, words, forced=forced)
    three = result.lines[1].words[0]
    assert (three.start, three.source) == (3.2, "forced")
    assert result.forced == 2 and result.matched == 2
    assert result.forced_score == pytest.approx(0.75)


def test_poor_forced_line_keeps_whisper_times():
    lines = [LyricLine(1.0, "one two")]
    words = heard(("one", 1.1, 1.4), ("two", 1.5, 1.9))
    forced = [[TimedWord("one", 2.0, 2.2, 0.05), TimedWord("two", 2.3, 2.6, 0.05)]]
    result = align_lyrics(lines, words, forced=forced)
    assert [w.source for w in all_words(result)] == ["whisper", "whisper"]
    assert all_words(result)[0].start == 1.1


def test_heard_words_in_silence_are_dropped():
    words = heard(("one", 1.0, 1.4), ("ghost", 5.0, 5.3))
    assert [w.text for w in filter_heard(words, voice(8, (0.9, 2.0)))] == ["one"]
    assert len(filter_heard(words)) == 2


def test_lines_on_silence_were_cut_from_the_video():
    texts = ["one two", "three four", "five six", "seven eight", "nine ten", "eleven twelve", "red blue",
             "green gold"]
    lines = [LyricLine(10.0 * (i + 1), text) for i, text in enumerate(texts)]
    sung = [i for i in range(len(texts)) if i not in (2, 3)]  # the video has no "five six" / "seven eight"
    words = heard(*[(word, 10.0 * (i + 1) + 0.1 + k * 0.5, 10.0 * (i + 1) + 0.5 + k * 0.5)
                    for i in sung for k, word in enumerate(texts[i].split())])
    activity = voice(100, *[(10.0 * (i + 1), 10.0 * (i + 1) + 1.2) for i in sung])
    result = align_lyrics(lines, words, activity=activity)
    assert [line.text for line in result.lines] == [texts[i] for i in sung]
    assert result.cut == 2


SIX = ["one two", "three four", "five six", "seven eight", "nine ten", "eleven twelve"]


def six_lines():
    return [LyricLine(10.0 * (i + 1), text) for i, text in enumerate(SIX)]


def six_voiced():
    return voice(80, *[(10.0 * (i + 1), 10.0 * (i + 1) + 1.0) for i in range(len(SIX))])


def test_misleading_onset_correlation_loses_to_what_whisper_heard(monkeypatch):
    monkeypatch.setattr("karaokifex.timing.xcorr_offset", lambda *args, **kwargs: 59.6)
    words = heard(*[(word, 10.0 * (i + 1) + 0.1 + k * 0.4, 10.0 * (i + 1) + 0.4 + k * 0.4)
                    for i, text in enumerate(SIX) for k, word in enumerate(text.split())])
    plan = plan_alignment(six_lines(), words, six_voiced())
    assert plan.time_map.offset == pytest.approx(0.0, abs=0.3)
    assert not plan.cut
    assert align_lyrics(six_lines(), words, activity=six_voiced()).matched == 12


def test_nothing_is_cut_when_no_heard_words_back_the_time_map(monkeypatch):
    monkeypatch.setattr("karaokifex.timing.xcorr_offset", lambda *args, **kwargs: 59.6)
    plan = plan_alignment(six_lines(), [], six_voiced())  # the prior maps every line onto silence
    assert not plan.cut


BLOCK = ["red kite flies", "blue boat sails", "green tree grows"]


def sung_lines(*lines: tuple[str, float]) -> tuple[list[TimedWord], Activity]:
    """Each line's words heard 0.4 s apart from its start, and the lead voiced while they are sung."""
    words = [TimedWord(word, start + 0.4 * k, start + 0.4 * k + 0.3, 0.9)
             for text, start in lines for k, word in enumerate(text.split())]
    return words, voice(60, *[(start, start + 0.4 * len(text.split())) for text, start in lines])


def test_a_video_that_leaves_out_one_of_two_identical_blocks_is_followed_through_the_cut():
    # lrclib (the album): the block twice, a chorus, the block again, an outro; the video sings the block
    # once, the chorus, the block, and stops
    texts = [*BLOCK, *BLOCK, "shout it out loud", *BLOCK, "fade away now", "fade away now"]
    lines = [LyricLine(4.0 * i, text) for i, text in enumerate(texts)]
    video = [(BLOCK[0], 5.0), (BLOCK[1], 9.0), (BLOCK[2], 13.0), ("shout it out loud", 17.0),
             (BLOCK[0], 21.0), (BLOCK[1], 25.0), (BLOCK[2], 29.0)]
    words, activity = sung_lines(*video)
    result = align_lyrics(lines, words, activity=activity)
    assert [line.text for line in result.lines] == [text for text, _ in video]
    assert [line.start for line in result.lines] == pytest.approx([start for _, start in video], abs=0.05)
    assert result.cut == 5


def test_blocks_the_video_sings_in_another_order_come_back_in_its_order():
    # lrclib (the album): three blocks A B C, 4 s a line; the video sings A, then C, then B, each line 4 s apart too
    A = ["red kite flies high", "blue boat sails far", "green tree grows tall"]
    B = ["old clock ticks slow", "warm bread smells good", "cold rain falls down"]
    C = ["small dog barks loud", "bright star shines on", "long road runs west"]
    lines = [LyricLine(10.0 + 4.0 * i, text) for i, text in enumerate([*A, *B, *C])]
    video = [(text, 5.0 + 4.0 * i) for i, text in enumerate([*A, *C, *B])]
    words, activity = sung_lines(*video)
    ordered, moved = video_order(lines, words, activity, language="en")
    assert [line.text for line in ordered] == [text for text, _ in video]
    assert [line.start for line in ordered] == pytest.approx([start for _, start in video], abs=0.3)
    assert len(moved) == 1 and (moved[0][0], moved[0][1]) == (6, 8)
    result = align_lyrics(ordered, words, activity=activity, language="en")
    assert [line.text for line in result.lines] == [text for text, _ in video]
    assert [line.start for line in result.lines] == pytest.approx([start for _, start in video], abs=0.05)
    assert result.cut == 0


VERSE = ["red kite flies high", "blue boat sails far", "green tree grows tall"]
CHORUS = ["old clock ticks slow", "warm bread smells good", "cold rain falls down", "small dog barks loud"]
VERSE_TWO = ["bright star shines on", "long road runs west", "deep well holds cool"]


def looped(text: str, start: float, copies: int, every: float = 3.0) -> list[TimedWord]:
    """Whisper looping: `text` written `copies` times from `start`, one every `every` seconds."""
    return [TimedWord(word, start + every * n + 0.4 * k, start + every * n + 0.4 * k + 0.3, 0.6)
            for n in range(copies) for k, word in enumerate(text.split())]


def test_heard_loops_flags_one_phrase_over_and_over_where_the_lyrics_differ():
    lines = [LyricLine(4.0 * i, text) for i, text in enumerate([*VERSE, *CHORUS, *VERSE_TWO])]
    verse, _ = sung_lines(*[(text, 4.0 * i) for i, text in enumerate(VERSE)])
    loop = looped(CHORUS[0], 12.0, 4)
    words = [*verse, *loop, *sung_lines(*[(text, 28.0 + 4.0 * i) for i, text in enumerate(VERSE_TWO)])[0]]
    flagged = heard_loops(words, lines)
    assert flagged == frozenset(range(len(verse), len(verse) + len(loop)))
    # a phrase the lyrics sing as often in a row is sung so: "na na na na" four times over is no loop
    chant = [LyricLine(0.0, "na na na na na na na na"), LyricLine(4.0, "na na na na na na na na"), LyricLine(8.0, "then we go home")]
    assert heard_loops(looped("na na na na", 0.0, 4, every=1.6), chant) == frozenset()
    # two copies are no loop either
    assert heard_loops(looped(CHORUS[0], 12.0, 2), lines) == frozenset()


def test_a_chorus_whisper_loops_on_keeps_the_clock_and_cuts_nothing():
    # the lead sings verse, chorus (four different lines) and a second verse, the video 2 s behind lrclib; whisper
    # hears the verses and writes the first chorus line over and over through the chorus
    lines = [LyricLine(4.0 * i, text) for i, text in enumerate([*VERSE, *CHORUS, *VERSE_TWO])]
    video = [(text, 2.0 + 4.0 * i) for i, text in enumerate([*VERSE, *CHORUS, *VERSE_TWO])]
    heard_verses = [w for text, start in video if text not in CHORUS for w in sung_lines((text, start))[0]]
    words = sorted([*heard_verses, *looped(CHORUS[0], 14.0, 5)], key=lambda w: w.start)
    activity = voice(60, (2.0, 12.0), (14.0, 30.0), (30.0, 40.0))  # the chorus sung through, breathy
    plan = plan_alignment(lines, words, activity, language="en")
    assert plan.cut == frozenset()
    assert plan.time_map.corrections == ()
    assert plan.ignored is not None and len(plan.ignored) == 5 * len(CHORUS[0].split())
    result = align_lyrics(lines, words, plan=plan, activity=activity, language="en")
    assert [line.text for line in result.lines] == [text for text, _ in video]
    assert [line.start for line in result.lines] == pytest.approx([start for _, start in video], abs=0.5)
    # the looped words make no lines of their own either
    assert fill_unlyricked(result.lines, words, activity, lines) == result.lines


def test_the_prompt_echoed_over_a_chorus_places_nothing_and_cuts_nothing():
    # whisper hears the first verse, then writes it again (its prompt) where the chorus is sung
    lines = [LyricLine(4.0 * i, text) for i, text in enumerate([*VERSE, *CHORUS])]
    video = [(text, 2.0 + 4.0 * i) for i, text in enumerate([*VERSE, *CHORUS])]
    heard_verse, _ = sung_lines(*[(text, 2.0 + 4.0 * i) for i, text in enumerate(VERSE)])
    echo, _ = sung_lines(*[(text, 15.0 + 4.0 * i) for i, text in enumerate(VERSE)])
    words = [*heard_verse, *echo]
    activity = voice(60, (2.0, 12.0), (14.0, 30.0))  # the chorus sung through
    plan = plan_alignment(lines, words, activity, language="en")
    assert plan.cut == frozenset()
    assert plan.ignored is not None and len(plan.ignored) == len(echo)
    result = align_lyrics(lines, words, plan=plan, activity=activity, language="en")
    assert [line.text for line in result.lines] == [text for text, _ in video]
    assert [line.start for line in result.lines] == pytest.approx([start for _, start in video], abs=0.5)


def test_quality_prefers_the_lyrics_version_whose_lines_cover_the_singing():
    # two versions of the lyrics: one with the bridge the video sings, one without it; whisper hears neither bridge
    full = [LyricLine(4.0 * i, text) for i, text in enumerate([*VERSE, *CHORUS, *VERSE_TWO])]
    without = [LyricLine(4.0 * i, text) for i, text in enumerate(VERSE)] + \
              [LyricLine(4.0 * (i + len(VERSE) + len(CHORUS)), text) for i, text in enumerate(VERSE_TWO)]
    video = [(text, 2.0 + 4.0 * i) for i, text in enumerate([*VERSE, *CHORUS, *VERSE_TWO])]
    words = [w for text, start in video if text not in CHORUS for w in sung_lines((text, start))[0]]
    activity = voice(60, *[(start, start + 2.5) for _, start in video])
    plans = [plan_alignment(version, words, activity, language="en") for version in (full, without)]
    assert plans[0].misfit < plans[1].misfit
    qualities = [align_lyrics(version, words, plan=plan, activity=activity, language="en").quality
                 for version, plan in zip((full, without), plans)]
    assert qualities[0] > qualities[1]


def test_a_lead_line_heard_only_in_the_backing_vocals_is_not_placed_there():
    # the album has a vamp between two sung lines; the video skips it, but the backing singers hum two of its words under
    # the next line's start: the vamp is cut, not laid over that line
    lines = [LyricLine(10.0, "red kite flies high"), LyricLine(14.0, "warm bread smells good"), LyricLine(18.0, "blue boat sails far"),
             LyricLine(22.0, "small dog barks loud")]
    words, activity = sung_lines(("red kite flies high", 5.0), ("blue boat sails far", 12.0), ("small dog barks loud", 16.0))
    backing = [TimedWord("bread", 11.6, 11.8, 0.9, source="backing"), TimedWord("good", 11.9, 12.1, 0.9, source="backing")]
    heard_all = sorted([*words, *backing], key=lambda w: w.start)
    result = align_lyrics(lines, heard_all, activity=activity, language="en")
    assert [line.text for line in result.lines] == ["red kite flies high", "blue boat sails far", "small dog barks loud"]
    assert result.cut == 1


def test_a_line_the_backing_sings_alone_is_still_placed_on_it():
    # the lyrics leave the backing's own line unmarked; the lead is silent while the backing sings it
    lines = [LyricLine(10.0, "red kite flies high"), LyricLine(14.0, "warm bread smells good"), LyricLine(18.0, "blue boat sails far"),
             LyricLine(22.0, "small dog barks loud")]
    words, activity = sung_lines(("red kite flies high", 5.0), ("blue boat sails far", 13.0), ("small dog barks loud", 17.0))
    backing = [TimedWord(w, 9.0 + 0.4 * k, 9.3 + 0.4 * k, 0.9, source="backing") for k, w in enumerate("warm bread smells good".split())]
    result = align_lyrics(lines, sorted([*words, *backing], key=lambda w: w.start), activity=activity, language="en")
    assert [line.text for line in result.lines] == ["red kite flies high", "warm bread smells good", "blue boat sails far", "small dog barks loud"]
    assert result.lines[1].start == pytest.approx(9.0, abs=0.3)


def test_a_video_in_the_lyrics_order_is_left_as_it_is():
    texts = [*BLOCK, "shout it out loud", *BLOCK]
    lines = [LyricLine(4.0 * i, text) for i, text in enumerate(texts)]
    words, activity = sung_lines(*[(text, 5.0 + 4.0 * i) for i, text in enumerate(texts)])
    ordered, moved = video_order(lines, words, activity)
    assert moved == [] and [line.start for line in ordered] == [line.start for line in lines]


def test_both_transcriptions_are_kept_but_a_word_heard_by_both_only_once():
    prompted = heard(("echo", 1.0, 1.5), ("echo", 3.0, 3.5), ("river", 9.0, 9.5))
    unprompted = heard(("echo", 1.05, 1.5), ("echoes", 3.1, 3.4), ("echo", 5.0, 5.5), ("ocean", 9.0, 9.4))
    assert [(w.text, w.start) for w in merge_heard(prompted, unprompted)] == [
        ("echo", 1.0), ("echo", 3.0), ("echo", 5.0), ("river", 9.0), ("ocean", 9.0)]


def test_prompted_words_over_a_phrase_heard_otherwise_are_dropped():
    # the prompted pass writes a lyric line where something else is sung; the unprompted pass hears that
    prompted = heard(("under", 1.0, 1.2), ("my", 1.3, 1.4), ("bright", 1.5, 1.8), ("sky", 1.9, 2.2),
                     ("red", 5.0, 5.2), ("kite", 5.3, 5.5), ("flies", 5.6, 5.9))
    unprompted = heard(("round", 1.0, 1.2), ("the", 1.3, 1.4), ("old", 1.5, 1.8), ("well", 1.9, 2.2),
                       ("red", 5.0, 5.2), ("kit", 5.3, 5.5), ("flies", 5.6, 5.9))
    assert [w.text for w in merge_heard(prompted, unprompted)] == ["round", "the", "old", "well", "red", "kite", "flies"]


def test_backing_lines_are_placed_on_the_backing_vocals_and_lead_lines_keep_the_lead():
    lines = [LyricLine(10.0, "red kite flies"), LyricLine(14.0, "(over the hill)"), LyricLine(18.0, "blue boat sails")]
    lead = heard(("red", 10.0, 10.3), ("kite", 10.4, 10.7), ("flies", 10.8, 11.1),
                 ("blue", 18.0, 18.3), ("boat", 18.4, 18.7), ("sails", 18.8, 19.1))
    lead_voice = voice(30, (10.0, 11.2), (18.0, 19.2))
    # the backing sings its line late, at 15.5, and echoes the lead's "blue boat sails" at 20
    backing_voice = voice(30, (15.5, 16.8), (20.0, 21.2))
    backing = backing_heard(heard(("over", 15.5, 15.8), ("the", 15.9, 16.1), ("hill", 16.2, 16.6),
                                  ("blue", 20.0, 20.3), ("boat", 20.4, 20.7), ("sails", 20.8, 21.1),
                                  ("noise", 25.0, 25.3)), backing_voice)  # "noise" sung nowhere: filtered out
    assert [w.text for w in backing][-1] == "sails"
    result = align_lyrics(lines, sorted([*lead, *backing], key=lambda w: w.start), activity=lead_voice,
                          backing_activity=backing_voice)
    starts = [line.start for line in result.lines]
    assert starts[1] == pytest.approx(15.5, abs=0.1)
    assert starts[2] == pytest.approx(18.0, abs=0.1)


def test_unsure_words_a_long_pause_before_the_rest_of_their_line_are_timed_with_it():
    lines = [LyricLine(10.0, "red kite flies"), LyricLine(14.0, "blue boat sails home")]
    words = [TimedWord("red", 10.0, 10.3, 0.9), TimedWord("kite", 10.4, 10.7, 0.9), TimedWord("flies", 10.8, 11.1, 0.9),
             TimedWord("blue", 11.3, 11.35, 0.0),  # squeezed in where whisper skipped
             TimedWord("boat", 15.6, 15.9, 0.9), TimedWord("sails", 16.0, 16.3, 0.9), TimedWord("home", 16.4, 16.7, 0.9)]
    result = align_lyrics(lines, words)
    assert result.lines[1].start > 14.5


def test_a_breath_inside_a_line_keeps_its_sure_words_where_they_are():
    lines = [LyricLine(10.0, "red kite flies over the green hill")]
    words = [TimedWord(w, s, s + 0.3, 0.9) for w, s in (("red", 10.0), ("kite", 10.4), ("flies", 10.8),
                                                        ("over", 15.0), ("the", 15.4), ("green", 15.8), ("hill", 16.2))]
    assert align_lyrics(lines, words).lines[0].start == pytest.approx(10.0)


def test_held_last_note_extends_the_line():
    lines = [LyricLine(None, "one two"), LyricLine(None, "three")]
    words = heard(("one", 1.0, 1.3), ("two", 1.4, 1.7), ("three", 10.0, 10.5))
    result = align_lyrics(lines, words, activity=voice(12, (1.0, 3.0), (10.0, 10.6)))
    assert result.lines[0].end == pytest.approx(3.0, abs=HOP)


def test_interpolated_line_lands_where_someone_sings():
    lines = [LyricLine(None, "one two"), LyricLine(None, "three four"), LyricLine(None, "five six")]
    words = heard(("one", 1, 1.4), ("two", 1.5, 2), ("five", 10, 10.4), ("six", 10.5, 11))
    result = align_lyrics(lines, words, activity=voice(12, (1, 2), (6, 7), (10, 11)))
    middle = result.lines[1]
    assert 6.0 - HOP <= middle.start and middle.end <= 7.0 + HOP
    assert {w.source for w in middle.words} == {"interpolated"}


def test_one_lyric_word_sung_as_two_heard_words_and_back():
    result = align_lyrics([LyricLine(None, "alright now")], heard(("all", 1.0, 1.2), ("right", 1.3, 1.6),
                                                                  ("now", 1.7, 2.0)))
    assert result.matched == 2
    assert (all_words(result)[0].start, all_words(result)[0].end) == (1.0, 1.6)
    result = align_lyrics([LyricLine(None, "want to go")], heard(("wanna", 1.0, 1.4), ("go", 1.5, 1.8)))
    assert result.matched == 3
    assert all_words(result)[1].end == pytest.approx(1.4)


def test_phonetic_similarity_only_for_english():
    assert _similarity("cue", "queue") == 0.0
    assert _similarity("cue", "queue", True) == PHONETIC


def test_low_score_heard_words_count_less():
    lines = [LyricLine(None, "la")]
    words = [TimedWord("la", 1.0, 1.2, 0.01), TimedWord("la", 2.0, 2.2, 0.9)]
    assert all_words(align_lyrics(lines, words))[0].start == 2.0


def test_enhanced_lrc_tags_time_words():
    lines = [LyricLine(1.0, "one two", (1.0, 1.5), 2.0)]
    words = all_words(align_lyrics(lines, []))
    assert [(w.start, w.source) for w in words] == [(1.0, "lrc-tag"), (1.5, "lrc-tag")]
    assert all_words(align_lyrics(lines, [], use_word_tags=False))[0].source == "lrc-line"


def test_full_mix_transcription_votes():
    lines = [LyricLine(None, "one two three")]
    lead = heard(("one", 1.0, 1.3), ("three", 2.0, 2.3))
    mix = [TimedWord("one", 1.1, 1.4, 0.5), TimedWord("two", 1.5, 1.8, 0.5)]
    result = align_lyrics(lines, lead, mix_heard=mix)
    assert result.matched == 3
    assert all_words(result)[1].start == 1.5


def test_plan_roundtrip_and_forced_requests():
    lines = [LyricLine(10.0, "one two"), LyricLine(20.0, "three four", (20.0, 20.5)), LyricLine(30.0, "five six")]
    words = heard(("one", 10.1, 10.5), ("two", 10.6, 11.0), ("five", 30.1, 30.5), ("six", 30.6, 31.0))
    plan = plan_alignment(lines, words)
    assert AlignmentPlan.from_dict(plan.to_dict()) == plan
    requests = forced_requests(lines, plan)
    assert [index for index, _, _, _ in requests] == [0, 2]  # line 1 has word tags
    assert requests[0][1:3] == ((0, 1), ["one", "two"])
    low, high = requests[0][3]
    assert low <= 10.1 and high >= 11.0


def test_lines_from_words_splits_on_pauses_sentences_and_length():
    words = heard(("Hello", 0, 0.4), ("there", 0.5, 0.9), ("friend.", 1.0, 1.4), ("How", 1.5, 1.7),
                  ("are", 1.8, 2.0), ("you", 5.0, 5.4))
    assert [line.text for line in lines_from_words(words)] == ["Hello there friend.", "How are", "you"]
    many = heard(*[(f"w{i}", i * 0.3, i * 0.3 + 0.2) for i in range(10)])
    assert [len(line.words) for line in lines_from_words(many, max_words=8)] == [8, 2]


def lines_at(*spans: tuple[str, float, float]) -> list[TimedLine]:
    return [TimedLine((TimedWord(text, start, end, 1.0, "forced"),)) for text, start, end in spans]


def rap(start: float, count: int, step: float = 0.5) -> list[TimedWord]:
    return [TimedWord(f"rap{k}", start + k * step, start + k * step + 0.4, 0.9) for k in range(count)]


def test_a_sung_stretch_without_lyrics_is_filled_from_what_was_heard():
    lines = lines_at(("verse", 1.0, 3.0), ("chorus", 12.0, 14.0))
    filled = fill_unlyricked(lines, rap(4.0, 12), voice(20.0, (1.0, 3.0), (4.0, 10.0), (12.0, 14.0)))
    heard_words = [w for line in filled for w in line.words if w.source == "heard"]
    assert len(heard_words) == 12 and heard_words[0].start == 4.0
    assert [line.start for line in filled] == sorted(line.start for line in filled)
    assert filled[0].words[0].text == "verse" and filled[-1].words[0].text == "chorus"


def test_no_fill_in_short_gaps_silence_or_for_a_few_words():
    lines = lines_at(("verse", 1.0, 3.0), ("chorus", 5.0, 7.0))
    assert fill_unlyricked(lines, rap(3.4, 6, 0.2), voice(10.0, (1.0, 7.0))) == lines  # the gap is too short
    lines = lines_at(("verse", 1.0, 3.0), ("chorus", 20.0, 22.0))
    assert fill_unlyricked(lines, rap(5.0, 4), voice(25.0, (1.0, 22.0))) == lines  # too few words
    assert fill_unlyricked(lines, rap(5.0, 12), voice(25.0, (1.0, 3.0), (20.0, 22.0))) == lines  # heard in silence


def test_words_heard_inside_a_lyric_line_are_not_added_again():
    lines = lines_at(("long line", 2.0, 12.0))
    assert fill_unlyricked(lines, rap(3.0, 12), voice(15.0, (2.0, 12.0))) == lines


def test_forced_requests_split_a_line_s_backing_part():
    lines = [LyricLine(10.0, "one two (three four)"), LyricLine(20.0, "(five six)"), LyricLine(30.0, "seven eight")]
    words = heard(("one", 10.1, 10.5), ("two", 10.6, 11.0), ("seven", 30.1, 30.5), ("eight", 30.6, 31.0))
    plan = plan_alignment(lines, words)
    lead = {index: (positions, line_words) for index, positions, line_words, _ in forced_requests(lines, plan, backing=False)}
    backing = {index: (positions, line_words) for index, positions, line_words, _ in forced_requests(lines, plan, backing=True)}
    alone = {index: positions for index, positions, _, _ in forced_requests(lines, plan)}
    assert lead == {0: ((0, 1), ["one", "two"]), 2: ((0, 1), ["seven", "eight"])}
    assert backing == {0: ((2, 3), ["three", "four"]), 1: ((0, 1), ["five", "six"])}
    # without a backing stem the backing part is left out of the lead's alignment; backing lines go on the lead whole
    assert alone == {0: (0, 1), 1: (0, 1), 2: (0, 1)}


def test_a_backing_part_left_out_of_forced_alignment_keeps_its_own_timing():
    lines = [LyricLine(None, "one two (three four)")]
    lead = heard(("one", 1.0, 1.3), ("two", 1.4, 1.8))
    backing = [TimedWord("three", 2.5, 2.8, 0.9, "backing"), TimedWord("four", 2.9, 3.3, 0.9, "backing")]
    forced = [[TimedWord("one", 1.05, 1.3, 0.9), TimedWord("two", 1.45, 1.8, 0.9), None, None]]
    result = all_words(align_lyrics(lines, sorted([*lead, *backing], key=lambda w: w.start), forced=forced))
    assert [round(w.start, 2) for w in result] == [1.05, 1.45, 2.5, 2.9]
    assert [w.source for w in result] == ["forced", "forced", "whisper", "whisper"]


def test_a_backing_part_sung_over_the_lead_keeps_its_times():
    lines = [LyricLine(None, "one two three (four five six)")]
    lead = heard(("one", 1.0, 1.4), ("two", 1.5, 1.9), ("three", 2.0, 4.0))
    forced = [[TimedWord("one", 1.0, 1.4, 0.9), TimedWord("two", 1.5, 1.9, 0.9), TimedWord("three", 2.0, 4.0, 0.9),
               TimedWord("four", 2.5, 2.8, 0.9), TimedWord("five", 2.9, 3.2, 0.9), TimedWord("six", 3.3, 3.8, 0.9)]]
    words = all_words(align_lyrics(lines, lead, forced=forced))
    assert [round(w.start, 2) for w in words] == [1.0, 1.5, 2.0, 2.5, 2.9, 3.3]
    assert words[2].end == pytest.approx(4.0)
