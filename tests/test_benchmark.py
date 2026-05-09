import time
import pytest


def format_time(seconds):
    if seconds < 1:
        return f"{seconds * 1000:.1f}ms"
    return f"{seconds:.2f}s"


class TestImportBenchmark:
    def test_import_is_instant(self):
        import subprocess, sys
        result = subprocess.run(
            [sys.executable, "-c", "import time; t=time.time(); from compressor.semantic import clean_text, count_tokens, stem_text, detect_language, extract_textual_embeddings, calculate_similarity, compute_and_remove_repeated_ngrams, correct_spelling; print(f'{time.time()-t:.4f}s')"],
            capture_output=True, text=True, timeout=30
        )
        import_time = float(result.stdout.strip().rstrip('s'))
        print(f"Module import: {format_time(import_time)}")
        assert import_time < 0.1, f"Import too slow: {format_time(import_time)}"

    def test_full_module_import_is_instant(self):
        import subprocess, sys
        result = subprocess.run(
            [sys.executable, "-c", "import time; t=time.time(); import compressor.semantic; print(f'{time.time()-t:.4f}s')"],
            capture_output=True, text=True, timeout=30
        )
        import_time = float(result.stdout.strip().rstrip('s'))
        print(f"Full module import: {format_time(import_time)}")
        assert import_time < 0.1, f"Full module import too slow: {format_time(import_time)}"


class TestCleanTextBenchmark:
    def test_small_text(self, sample_text_en):
        from compressor.semantic import clean_text
        times = []
        for _ in range(100):
            t = time.time()
            clean_text(sample_text_en)
            times.append(time.time() - t)
        avg = sum(times) / len(times)
        print(f"clean_text (100x, small): avg={format_time(avg)}, total={format_time(sum(times))}")
        assert avg < 0.001

    def test_large_text(self):
        from compressor.semantic import clean_text
        text = "Hello world. " * 10000 + "Test! " * 1000
        t = time.time()
        clean_text(text)
        elapsed = time.time() - t
        print(f"clean_text (10k words): {format_time(elapsed)}")
        assert elapsed < 0.5


class TestDetectLanguageBenchmark:
    def test_language_detection(self, sample_text_en, sample_text_pt):
        from compressor.semantic import detect_language
        times_en = []
        for _ in range(10):
            t = time.time()
            detect_language(sample_text_en)
            times_en.append(time.time() - t)
        avg_en = sum(times_en) / len(times_en)
        times_pt = []
        for _ in range(10):
            t = time.time()
            detect_language(sample_text_pt)
            times_pt.append(time.time() - t)
        avg_pt = sum(times_pt) / len(times_pt)
        print(f"detect_language (EN): avg={format_time(avg_en)}")
        print(f"detect_language (PT): avg={format_time(avg_pt)}")


class TestStemTextBenchmark:
    def test_stemming(self):
        from compressor.semantic import stem_text
        text = "running runner ran jumps jumped jumping cats dogs horses quickly nicely " * 100
        stem_text("warm up", 'en')
        t = time.time()
        stem_text(text, 'en')
        elapsed = time.time() - t
        print(f"stem_text (1000 words): {format_time(elapsed)}")
        assert elapsed < 1.0


class TestCountTokensBenchmark:
    def test_token_counting(self, sample_text_en):
        from compressor.semantic import count_tokens
        times = []
        for _ in range(100):
            t = time.time()
            count_tokens(sample_text_en)
            times.append(time.time() - t)
        avg = sum(times) / len(times)
        print(f"count_tokens (100x): avg={format_time(avg)}, total={format_time(sum(times))}")


class TestExtractTextualEmbeddingsBenchmark:
    def test_textual_embeddings(self):
        from compressor.semantic import extract_textual_embeddings
        text = "This is a test sentence for extracting textual embeddings."
        t = time.time()
        extract_textual_embeddings(text)
        first = time.time() - t
        times = []
        for _ in range(50):
            t = time.time()
            extract_textual_embeddings(text)
            times.append(time.time() - t)
        avg = sum(times) / len(times)
        print(f"extract_textual_embeddings: first={format_time(first)}, avg_next={format_time(avg)}")
        assert avg < 0.01


class TestComputeNgramsBenchmark:
    def test_ngram_removal(self):
        from compressor.semantic import compute_and_remove_repeated_ngrams
        text = "this is a test this is a test this is a test " * 100 + "unique ending here"
        t = time.time()
        compute_and_remove_repeated_ngrams(text)
        elapsed = time.time() - t
        print(f"compute_and_remove_repeated_ngrams (300 n-grams): {format_time(elapsed)}")
        assert elapsed < 0.5


@pytest.mark.need_model
class TestCompressionBenchmark:
    def test_compress_small(self, sample_text_dense):
        from compressor.semantic import compress_text
        t = time.time()
        result = compress_text(sample_text_dense, compression_rate=0.5)
        elapsed = time.time() - t
        print(f"compress_text (small, 1st call): {format_time(elapsed)}")
        assert isinstance(result, str)

    def test_compress_medium(self):
        from compressor.semantic import compress_text
        text = "The quick brown fox jumps over the lazy dog. " * 50
        # Warm up
        compress_text(text, compression_rate=0.5)
        t = time.time()
        compress_text(text, compression_rate=0.5)
        elapsed = time.time() - t
        print(f"compress_text (medium, 2nd call): {format_time(elapsed)}")

    def test_compress_large(self):
        from compressor.semantic import compress_text
        text = ("Artificial intelligence is transforming the world. " * 200
                + "Machine learning enables computers to learn from data. " * 200)
        # Warm up
        compress_text(text[:1000], compression_rate=0.5)
        t = time.time()
        compress_text(text, compression_rate=0.5)
        elapsed = time.time() - t
        print(f"compress_text (large, 400 sentences): {format_time(elapsed)}")


@pytest.mark.need_model
class TestFindNeedleBenchmark:
    def test_find_needle(self, sample_text_dense):
        from compressor.semantic import find_needle_in_haystack
        t = time.time()
        find_needle_in_haystack(
            haystack=sample_text_dense,
            needle="neural networks",
            block_size=50,
            embedding_mode='textual'
        )
        elapsed = time.time() - t
        print(f"find_needle_in_haystack (textual): {format_time(elapsed)}")

    def test_find_needle_semantic(self, sample_text_dense):
        from compressor.semantic import find_needle_in_haystack
        t = time.time()
        find_needle_in_haystack(
            haystack=sample_text_dense,
            needle="neural networks",
            block_size=50,
            embedding_mode='semantic'
        )
        elapsed = time.time() - t
        print(f"find_needle_in_haystack (semantic): {format_time(elapsed)}")


def pytest_benchmark_report():
    """Run all benchmarks and print summary."""
    pass
