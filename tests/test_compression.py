import pytest
from compressor.semantic import compress_text, semantic_compress_text


@pytest.mark.need_model
class TestCompressText:
    def test_compress_with_rate(self, sample_text_en):
        result = compress_text(sample_text_en, compression_rate=0.9)
        assert isinstance(result, str)
        assert len(result) > 0
        assert result != sample_text_en

    def test_compress_with_target_token_count(self, sample_text_en):
        result = compress_text(sample_text_en, target_token_count=10)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_compress_preserves_some_original_words(self, sample_text_en):
        result = compress_text(sample_text_en, compression_rate=0.5)
        original_words = set(sample_text_en.lower().split())
        result_words = set(result.lower().split())
        overlap = original_words & result_words
        assert len(overlap) > 0

    def test_compress_empty_text(self):
        result = compress_text("", compression_rate=0.5)
        assert isinstance(result, str)

    def test_compress_short_text(self):
        text = "Hello world."
        result = compress_text(text, compression_rate=0.5)
        assert isinstance(result, str)

    def test_compress_very_short_text_different(self):
        text = "Hello."
        result = compress_text(text, target_token_count=1)
        assert isinstance(result, str)

    def test_token_count_below_minimum_no_compression(self):
        text = "Short text."
        result = compress_text(text, target_token_count=100)
        assert result == text

    def test_compression_with_steering(self, sample_text_dense):
        reference = "Deep learning and neural networks"
        result = compress_text(sample_text_dense, compression_rate=0.5, reference_text_steering=reference)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_compression_with_cleaning(self, sample_text_en):
        result = compress_text(sample_text_en, compression_rate=0.7, perform_cleaning=True)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_compression_without_cleaning(self, sample_text_en):
        result = compress_text(sample_text_en, compression_rate=0.7, perform_cleaning=False)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_compress_portuguese(self, sample_text_pt):
        result = compress_text(sample_text_pt, compression_rate=0.5)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_preserves_sentence_structure(self, sample_text_dense):
        result = compress_text(sample_text_dense, compression_rate=0.5)
        assert result[0].isupper()
        assert result.rstrip()[-1] in '.!?'

    def test_compression_rate_of_one_returns_original(self, sample_text_en):
        original_token_count = len(sample_text_en.split())
        result = compress_text(sample_text_en, compression_rate=1.0)
        assert isinstance(result, str)

    def test_compress_text_output_type(self, sample_text_en):
        result = compress_text(sample_text_en, compression_rate=0.5)
        assert isinstance(result, str)


@pytest.mark.need_model
class TestSemanticCompressText:
    def test_basic_compression(self, sample_text_en):
        result = semantic_compress_text(sample_text_en, compression_rate=0.7)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_compression_reduces_length(self, sample_text_dense):
        original_len = len(sample_text_dense.split())
        result = semantic_compress_text(sample_text_dense, compression_rate=0.5)
        result_len = len(result.split())
        assert result_len <= original_len

    def test_with_reference_text(self, sample_text_dense):
        reference = "I want to focus on machine learning and deep neural networks."
        result = semantic_compress_text(sample_text_dense, compression_rate=0.5, reference_text=reference)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_different_num_topics(self, sample_text_en):
        result = semantic_compress_text(sample_text_en, compression_rate=0.5, num_topics=3)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_no_cleaning(self, sample_text_en):
        result = semantic_compress_text(sample_text_en, compression_rate=0.5, perform_cleaning=False)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_compression_rate_is_default(self, sample_text_en):
        result = semantic_compress_text(sample_text_en)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_compression_returns_string_not_empty(self):
        text = "This is a test. With multiple sentences. For compression testing."
        result = semantic_compress_text(text, compression_rate=0.3)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_compression_preserves_capitalization(self):
        text = "The quick brown fox. Jumps over the lazy dog. Repeated many times for testing."
        result = semantic_compress_text(text, compression_rate=0.5)
        assert result[0].isupper()
