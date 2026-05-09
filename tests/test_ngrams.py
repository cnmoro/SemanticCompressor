import pytest
from compressor.semantic import compute_and_remove_repeated_ngrams


class TestComputeAndRemoveRepeatedNgrams:
    def test_no_repetition(self):
        text = "the quick brown fox jumps over the lazy dog"
        result = compute_and_remove_repeated_ngrams(text)
        assert result == text

    def test_repeated_trigram_removed(self):
        text = "this is a test this is a test this is a test hello world"
        result = compute_and_remove_repeated_ngrams(text, ngram_size=3, threshold=1)
        assert "this is a" in result
        assert text.count("this is a") > result.count("this is a")

    def test_different_ngram_size(self):
        text = "a b c d a b c d a b c d hello"
        result = compute_and_remove_repeated_ngrams(text, ngram_size=2, threshold=1)
        assert "a b" in result
        assert result.count("a b") == 1

    def test_high_threshold_no_removal(self):
        text = "a a a b b b c c c"
        result = compute_and_remove_repeated_ngrams(text, ngram_size=1, threshold=10)
        assert result == text

    def test_empty_string(self):
        assert compute_and_remove_repeated_ngrams("") == ""

    def test_single_word(self):
        assert compute_and_remove_repeated_ngrams("hello") == "hello"

    def test_repeated_five_times_removed(self):
        text = "repeat me please " * 6 + "unique ending"
        result = compute_and_remove_repeated_ngrams(text, ngram_size=3, threshold=2)
        assert "unique ending" in result
        assert "repeat me please" in result

    def test_ngram_size_larger_than_text(self):
        text = "short text"
        result = compute_and_remove_repeated_ngrams(text, ngram_size=10)
        assert result == text

    def test_multiple_unique_repeated_ngrams(self):
        text = "abc def ghi " * 4 + "xyz " + "123 456 789 " * 4
        result = compute_and_remove_repeated_ngrams(text, ngram_size=3, threshold=2)
        assert "xyz" in result
        assert result.count("abc def ghi") == 1

    def test_whitespace_handling(self):
        text = "hello    world    hello    world    hello    world"
        result = compute_and_remove_repeated_ngrams(text, ngram_size=3, threshold=1)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_only_unique_ngrams(self):
        text = "a b c d e f g h i j k l m n o p"
        result = compute_and_remove_repeated_ngrams(text, ngram_size=3, threshold=2)
        assert result == text
