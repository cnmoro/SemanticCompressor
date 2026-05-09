import pytest
import numpy as np
from compressor.semantic import extract_semantic_embeddings, calculate_similarity


@pytest.mark.need_model
class TestExtractSemanticEmbeddings:
    def test_returns_numpy_array(self, sample_text_en):
        result = extract_semantic_embeddings(sample_text_en)
        assert isinstance(result, np.ndarray)
        assert len(result) > 0

    def test_fixed_size_output(self):
        v1 = extract_semantic_embeddings("short text")
        v2 = extract_semantic_embeddings("A much longer text " * 100)
        assert len(v1) == len(v2)

    def test_identical_texts_same_embedding(self):
        v1 = extract_semantic_embeddings("This is a test.")
        v2 = extract_semantic_embeddings("This is a test.")
        assert np.allclose(v1, v2)

    def test_different_texts_different_embeddings(self):
        v1 = extract_semantic_embeddings("I love programming")
        v2 = extract_semantic_embeddings("I hate programming")
        assert not np.allclose(v1, v2)

    def test_semantic_embedding_works_with_similarity(self, sample_text_en):
        emb = extract_semantic_embeddings(sample_text_en)
        similarity = calculate_similarity(emb.tolist(), emb.tolist())
        assert similarity == pytest.approx(1.0)

    def test_semantic_similarity_reasonable(self):
        emb1 = extract_semantic_embeddings("Machine learning is amazing")
        emb2 = extract_semantic_embeddings("Artificial intelligence is great")
        emb3 = extract_semantic_embeddings("I like to eat pizza")
        sim_similar = calculate_similarity(emb1.tolist(), emb2.tolist())
        sim_different = calculate_similarity(emb1.tolist(), emb3.tolist())
        assert sim_similar > sim_different
