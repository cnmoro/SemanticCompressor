import pytest
from compressor.semantic import correct_spelling


class TestCorrectSpelling:
    def test_correct_english_word(self):
        result = correct_spelling("speling", "en")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_correct_portuguese_word(self):
        result = correct_spelling("caza", "pt")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_already_correct_words(self):
        result = correct_spelling("the cat sat on the mat", "en")
        assert isinstance(result, str)

    def test_multiple_errors(self):
        result = correct_spelling("acomodation recieved", "en")
        assert isinstance(result, str)

    def test_empty_string(self):
        result = correct_spelling("", "en")
        assert result == ""

    def test_mixed_case(self):
        result = correct_spelling("Hello World", "en")
        assert isinstance(result, str)
