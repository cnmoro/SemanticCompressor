import os
import sys
import pytest
import importlib.resources

_NLTK_DATA_PATH = None
try:
    _NLTK_DATA_PATH = str(importlib.resources.files('compressor').joinpath('resources/nltk_data'))
except Exception:
    _NLTK_DATA_PATH = os.path.join(
        os.path.dirname(__file__), '..', 'compressor', 'resources', 'nltk_data'
    )

os.environ['NLTK_DATA'] = _NLTK_DATA_PATH

from compressor.semantic import (
    clean_text,
    detect_language,
    stem_text,
    count_tokens,
    structurize_text,
    extract_textual_embeddings,
    calculate_similarity,
    compute_and_remove_repeated_ngrams,
    correct_spelling,
)


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "need_model: mark test as requiring the model2vec model (skipped if not available)",
    )


def _check_model_available():
    try:
        from compressor.semantic import _get_embedding_model
        _get_embedding_model()
        return True
    except Exception:
        return False


def pytest_collection_modifyitems(config, items):
    model_available = _check_model_available()
    if not model_available:
        skip_need_model = pytest.mark.skip(reason="model2vec model not available")
        for item in items:
            if "need_model" in item.keywords:
                item.add_marker(skip_need_model)


@pytest.fixture
def sample_text_en():
    return (
        "The quick brown fox jumps over the lazy dog. "
        "This is a test sentence for the semantic compressor. "
        "Natural language processing is a fascinating field. "
        "Machine learning algorithms can analyze text data efficiently."
    )


@pytest.fixture
def sample_text_pt():
    return (
        "O rato roeu a roupa do rei de Roma. "
        "Esta é uma frase de teste para o compressor semântico. "
        "Processamento de linguagem natural é uma área fascinante. "
        "Algoritmos de aprendizado de máquina podem analisar texto eficientemente."
    )


@pytest.fixture
def sample_text_noisy():
    return "Hello,   World!!!   This is... a   very  noisy??? text---with | weird • characters [and] (parens)."


@pytest.fixture
def sample_text_hyphenated():
    return "This is a hyphen- ated word that should be re- paired.\n\nSecond paragraph here."


@pytest.fixture
def sample_text_dense():
    return (
        "Artificial intelligence has transformed the modern world. "
        "Deep learning models can recognize patterns in complex data. "
        "Neural networks are inspired by the human brain. "
        "Natural language understanding enables machines to read text. "
        "Computer vision allows machines to interpret images and video."
    )
