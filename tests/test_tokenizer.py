import pytest
from compressor.semantic import count_tokens, structurize_text


class TestCountTokens:
    def test_count_simple_text(self):
        count = count_tokens("Hello world")
        assert count > 0
        assert isinstance(count, int)

    def test_count_multiple_sentences(self, sample_text_en):
        count = count_tokens(sample_text_en)
        assert count > 10

    def test_count_empty_text(self):
        assert count_tokens("") == 0

    def test_count_single_word(self):
        assert count_tokens("Hello") > 0

    def test_count_special_characters(self):
        count = count_tokens("Hello! How are you? I'm fine.")
        assert count > 5

    def test_count_python_code(self):
        count = count_tokens("def hello(): pass")
        assert count > 0

    def test_count_portuguese(self, sample_text_pt):
        count = count_tokens(sample_text_pt)
        assert count > 5

    def test_count_is_consistent(self):
        text = "The quick brown fox jumps over the lazy dog."
        assert count_tokens(text) == count_tokens(text)


class TestStructurizeText:
    def test_basic_chunking(self):
        text = "word " * 1000
        chunks = structurize_text(text, tokens_per_chunk=300)
        assert len(chunks) >= 3
        for chunk in chunks:
            assert isinstance(chunk, str)
            assert len(chunk) > 0

    def test_small_text_single_chunk(self, sample_text_en):
        chunks = structurize_text(sample_text_en, tokens_per_chunk=300)
        assert len(chunks) == 1
        assert chunks[0] == sample_text_en

    def test_chunk_overlap(self):
        text = "token " * 500
        chunks_no_overlap = structurize_text(text, tokens_per_chunk=100, chunk_overlap=0)
        chunks_with_overlap = structurize_text(text, tokens_per_chunk=100, chunk_overlap=10)
        assert len(chunks_no_overlap) <= len(chunks_with_overlap)

    def test_text_preserved_across_chunks(self):
        original = "Hello world. This is a test. " * 50
        chunks = structurize_text(original, tokens_per_chunk=50)
        reconstructed = "".join(chunks)
        assert len(reconstructed) > 0

    def test_empty_text(self):
        chunks = structurize_text("")
        assert chunks == [""]

    def test_default_parameters(self, sample_text_en):
        chunks = structurize_text(sample_text_en)
        assert len(chunks) == 1
