import pytest
import numpy as np
from compressor.semantic import extract_textual_embeddings, calculate_similarity


class TestExtractTextualEmbeddings:
    def test_returns_list(self, sample_text_en):
        result = extract_textual_embeddings(sample_text_en)
        assert isinstance(result, list)
        assert len(result) > 0

    def test_fixed_size_output(self):
        v1 = extract_textual_embeddings("short text")
        v2 = extract_textual_embeddings("A much longer text" * 100)
        assert len(v1) == len(v2)

    def test_similar_texts_have_similar_embeddings(self):
        v1 = extract_textual_embeddings("machine learning")
        v2 = extract_textual_embeddings("machine learning")
        assert v1 == v2

    def test_different_texts_different_embeddings(self):
        v1 = extract_textual_embeddings("cat")
        v2 = extract_textual_embeddings("astrophysics")
        result = calculate_similarity(v1, v2)
        assert result < 0.99

    def test_empty_string(self):
        result = extract_textual_embeddings("")
        assert isinstance(result, list)
        assert len(result) > 0

    def test_numeric_values(self):
        result = extract_textual_embeddings("test")
        for val in result:
            assert isinstance(val, (int, float))


class TestCalculateSimilarity:
    def test_identical_vectors(self):
        v = [1.0, 0.0, 0.0]
        result = calculate_similarity(v, v)
        assert result == pytest.approx(1.0)

    def test_orthogonal_vectors(self):
        v1 = [1.0, 0.0, 0.0]
        v2 = [0.0, 1.0, 0.0]
        result = calculate_similarity(v1, v2)
        assert result == pytest.approx(0.0, abs=1e-10)

    def test_opposite_vectors(self):
        v1 = [1.0, 0.0]
        v2 = [-1.0, 0.0]
        result = calculate_similarity(v1, v2)
        assert result == pytest.approx(-1.0)

    def test_similarity_between_texts(self):
        v1 = extract_textual_embeddings("hello world")
        v2 = extract_textual_embeddings("hello world")
        similarity = calculate_similarity(v1, v2)
        assert similarity == pytest.approx(1.0)
        assert isinstance(similarity, float)
        assert 0.0 <= abs(similarity) <= 1.0

    def test_different_dimensions_different_texts(self):
        v1 = extract_textual_embeddings("python programming")
        v2 = extract_textual_embeddings("cooking recipes")
        result = calculate_similarity(v1, v2)
        assert isinstance(result, float)

    def test_symmetric(self):
        v1 = [0.5, 0.2, 0.8]
        v2 = [0.1, 0.9, 0.3]
        assert calculate_similarity(v1, v2) == pytest.approx(calculate_similarity(v2, v1))
