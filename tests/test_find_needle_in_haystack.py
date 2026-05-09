import pytest
from compressor.semantic import find_needle_in_haystack


@pytest.mark.need_model
class TestFindNeedleInHaystack:
    def test_find_exact_needle(self, sample_text_dense):
        needle = "Deep learning models"
        result = find_needle_in_haystack(
            haystack=sample_text_dense,
            needle=needle,
            block_size=50,
            embedding_mode='textual'
        )
        assert isinstance(result, str)
        assert len(result) > 0
        assert needle.lower() in result.lower()

    def test_find_with_semantic_mode(self, sample_text_dense):
        needle = "neural networks"
        result = find_needle_in_haystack(
            haystack=sample_text_dense,
            needle=needle,
            block_size=50,
            embedding_mode='semantic'
        )
        assert isinstance(result, str)
        assert len(result) > 0

    def test_find_with_both_mode(self, sample_text_dense):
        needle = "Computer vision"
        result = find_needle_in_haystack(
            haystack=sample_text_dense,
            needle=needle,
            block_size=50,
            embedding_mode='both'
        )
        assert isinstance(result, str)
        assert len(result) > 0

    def test_needle_not_present_still_returns_something(self, sample_text_dense):
        needle = "quantum physics theories"
        result = find_needle_in_haystack(
            haystack=sample_text_dense,
            needle=needle,
            block_size=50,
            embedding_mode='textual'
        )
        assert isinstance(result, str)
        assert len(result) > 0

    def test_exact_match_high_confidence(self):
        haystack = "Python is a programming language. It is widely used. Many developers love it."
        needle = "programming language"
        result = find_needle_in_haystack(
            haystack=haystack,
            needle=needle,
            block_size=30,
            embedding_mode='textual'
        )
        assert isinstance(result, str)
        assert "programming" in result.lower()

    def test_with_stemming(self, sample_text_dense):
        needle = "learning models"
        result = find_needle_in_haystack(
            haystack=sample_text_dense,
            needle=needle,
            block_size=50,
            embedding_mode='textual',
            use_stemming=True
        )
        assert isinstance(result, str)
        assert len(result) > 0

    def test_with_spelling_correction(self, sample_text_dense):
        needle = "algorithms"
        result = find_needle_in_haystack(
            haystack=sample_text_dense,
            needle=needle,
            block_size=50,
            embedding_mode='textual',
            correct_spelling_needle=True
        )
        assert isinstance(result, str)
        assert len(result) > 0

    def test_empty_haystack(self):
        needle = "test"
        result = find_needle_in_haystack(
            haystack="",
            needle=needle,
            block_size=50,
            embedding_mode='textual'
        )
        assert isinstance(result, str)

    def test_empty_needle(self, sample_text_dense):
        result = find_needle_in_haystack(
            haystack=sample_text_dense,
            needle="",
            block_size=50,
            embedding_mode='textual'
        )
        assert isinstance(result, str)

    def test_custom_weights(self, sample_text_dense):
        result = find_needle_in_haystack(
            haystack=sample_text_dense,
            needle="machine learning",
            block_size=50,
            embedding_mode='both',
            semantic_embeddings_weight=0.7,
            textual_embeddings_weight=0.3
        )
        assert isinstance(result, str)
        assert len(result) > 0

    def test_invalid_embedding_mode(self, sample_text_dense):
        result = find_needle_in_haystack(
            haystack=sample_text_dense,
            needle="test",
            block_size=50,
            embedding_mode='invalid'
        )
        assert isinstance(result, str)
        assert result == sample_text_dense

    def test_large_block_size(self, sample_text_dense):
        result = find_needle_in_haystack(
            haystack=sample_text_dense,
            needle="test",
            block_size=500,
            embedding_mode='textual'
        )
        assert isinstance(result, str)

    def test_portuguese_needle(self, sample_text_pt):
        needle = "aprendizado de máquina"
        result = find_needle_in_haystack(
            haystack=sample_text_pt,
            needle=needle,
            block_size=50,
            embedding_mode='textual'
        )
        assert isinstance(result, str)
        assert len(result) > 0
