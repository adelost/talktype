"""Tests for sentence-boundary chunking logic."""
from dataclasses import dataclass
from bdd_pytest import unit, scenario, expect
from talktype.chunking import find_last_sentence_boundary


@dataclass
class Word:
    """Mimics Whisper's TranscriptionWord object."""
    word: str
    start: float
    end: float


def words_and_text(word_data, text):
    return [Word(*w) for w in word_data], text


@unit
def test_finds_boundary_at_period():
    scenario("splits at the last period in a two-sentence transcription",
        given=("whisper output with two sentences", lambda: words_and_text(
            [("Vi", 0.0, 0.3), ("testar", 0.3, 0.8), ("igen", 0.8, 1.2),
             ("Ja", 1.5, 1.7), ("det", 1.7, 2.0), ("funkar", 2.0, 2.5)],
            "Vi testar igen. Ja det funkar",
        )),
        when=("finding boundary", lambda ctx: find_last_sentence_boundary(*ctx)),
        then=("cuts at the period with timestamp from word 3", lambda result, _: (
            expect(result).not_.to_be_none(),
            expect(result[0]).to_be("Vi testar igen."),
            expect(result[1]).to_be(1.2),
        )),
    )


@unit
def test_finds_boundary_at_question_mark():
    scenario("splits at a question mark",
        given=("a question followed by more words", lambda: words_and_text(
            [("Hur", 0.0, 0.3), ("mår", 0.3, 0.6), ("du", 0.6, 0.9),
             ("Jag", 1.2, 1.4), ("mår", 1.4, 1.7)],
            "Hur mår du? Jag mår",
        )),
        when=("finding boundary", lambda ctx: find_last_sentence_boundary(*ctx)),
        then=("cuts at the question mark", lambda result, _: (
            expect(result).not_.to_be_none(),
            expect(result[0]).to_be("Hur mår du?"),
            expect(result[1]).to_be(0.9),
        )),
    )


@unit
def test_no_boundary_without_punctuation():
    scenario("returns None when no sentence-ending punctuation exists",
        given=("a partial sentence", lambda: words_and_text(
            [("vi", 0.0, 0.3), ("testar", 0.3, 0.8), ("nu", 0.8, 1.0)],
            "vi testar nu",
        )),
        when=("finding boundary", lambda ctx: find_last_sentence_boundary(*ctx)),
        then=("None — wait for more audio", lambda result, _:
            expect(result).to_be_none(),
        ),
    )


@unit
def test_empty_words_returns_none():
    scenario("returns None with empty word list",
        given=("no words", lambda: ([], "some text.")),
        when=("finding boundary", lambda ctx: find_last_sentence_boundary(*ctx)),
        then=("None", lambda result, _: expect(result).to_be_none()),
    )


@unit
def test_multiple_sentences_picks_last():
    scenario("returns all text up to the last sentence boundary",
        given=("three sentences then trailing words", lambda: words_and_text(
            [("Ett", 0.0, 0.3), ("Två", 0.5, 0.8), ("Tre", 1.0, 1.3),
             ("men", 1.5, 1.7), ("mer", 1.7, 2.0)],
            "Ett. Två. Tre. men mer",
        )),
        when=("finding boundary", lambda ctx: find_last_sentence_boundary(*ctx)),
        then=("includes all three sentences, timestamp at word 3", lambda result, _: (
            expect(result[0]).to_be("Ett. Två. Tre."),
            expect(result[1]).to_be(1.3),
        )),
    )


@unit
def test_works_with_dict_words():
    scenario("handles dict-style words for backwards compat",
        given=("words as dicts", lambda: (
            [{"word": "Hej", "start": 0.0, "end": 0.5},
             {"word": "där", "start": 0.5, "end": 1.0}],
            "Hej där.",
        )),
        when=("finding boundary", lambda ctx: find_last_sentence_boundary(*ctx)),
        then=("finds boundary using dict access", lambda result, _: (
            expect(result).not_.to_be_none(),
            expect(result[0]).to_be("Hej där."),
            expect(result[1]).to_be(1.0),
        )),
    )


@unit
def test_exclamation_mark():
    scenario("treats exclamation mark as sentence boundary",
        given=("an exclamation then trailing words", lambda: words_and_text(
            [("Kör", 0.0, 0.3), ("bara", 0.3, 0.6), ("kör", 0.6, 0.9),
             ("nu", 1.0, 1.2)],
            "Kör bara kör! nu",
        )),
        when=("finding boundary", lambda ctx: find_last_sentence_boundary(*ctx)),
        then=("cuts at the exclamation", lambda result, _: (
            expect(result[0]).to_be("Kör bara kör!"),
            expect(result[1]).to_be(0.9),
        )),
    )
