from karaokifex import dialects


def test_swiss_german_is_its_own_language_whisper_runs_as_german():
    assert dialects.swiss("gsw") and dialects.swiss("de-CH") and dialects.swiss("Swiss German")
    assert not dialects.swiss("de") and not dialects.swiss(None)
    assert dialects.whisper_language("gsw") == "de" and dialects.whisper_language("en") == "en"


def test_swiss_german_reads_as_such_and_standard_german_doesnt():
    swiss = "I ha hüt es chli Kafi gha, es isch nöd so eifach gsi, mir sind no chli müed"
    standard = "Ich habe heute einen kleinen Kaffee gehabt, es ist nicht so einfach gewesen, wir sind noch müde"
    assert dialects.reads_swiss_german(swiss) and not dialects.reads_standard_german(swiss)
    assert dialects.reads_standard_german(standard) and not dialects.reads_swiss_german(standard)
    assert dialects.swiss_share("la la la") is None


class _Answer:
    def __init__(self, content):
        self.content = content

    def raise_for_status(self):
        pass

    def json(self):
        return {"choices": [{"message": {"content": self.content}}]}


def test_a_rewrite_comes_back_line_for_line_in_swiss_german_or_not_at_all():
    lines = ["Ich habe heute nichts gemacht", "es ist nicht so einfach"]
    good = lambda *a, **k: _Answer("<think></think>1. I ha hüt nüt gmacht\n2. es isch nöd so eifach")
    assert dialects.rewrite(lines, "http://x/v1", "m", post=good) == ["I ha hüt nüt gmacht", "es isch nöd so eifach"]
    short = lambda *a, **k: _Answer("1. I ha hüt nüt gmacht")
    assert dialects.rewrite(lines, "http://x/v1", post=short) is None


def test_the_common_words_swapped_read_as_swiss_german():
    line = "Ich habe heute nicht einen kleinen Kaffee gehabt, wir sind noch hier"
    swapped = dialects.swiss_words(line)
    assert swapped.startswith("I ha hüt nöd en chliine Kafi")
    assert dialects.reads_swiss_german(swapped)


def test_a_rewrite_left_in_standard_german_is_swapped_after():
    lines = ["Ich habe heute nichts gemacht", "es ist nicht so einfach"]
    unchanged = lambda *a, **k: _Answer("1. Ich habe heute nichts gemacht\n2. es ist nicht so einfach")
    assert dialects.rewrite(lines, "http://x/v1", post=unchanged) == ["I ha hüt nüt gmacht", "es isch nöd so eifach"]
