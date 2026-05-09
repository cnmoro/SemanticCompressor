import os, sys, importlib.resources, functools, re

_NLTK_DATA_PATH = None
try:
    _NLTK_DATA_PATH = str(importlib.resources.files('compressor').joinpath('resources/nltk_data'))
except Exception:
    _NLTK_DATA_PATH = os.path.join(os.path.dirname(__file__), 'resources', 'nltk_data')

os.environ['NLTK_DATA'] = _NLTK_DATA_PATH

from collections import Counter

_PUNCT_REATTACH = re.compile(r'\s+([.!,\?;:])')
_PUNCT_BOUNDARY = re.compile(r'([.!,\?;:])(?=\S)')
_HYPHENATION = re.compile(r'(\w)-\s*\n\s*(\w)')
_NOISE_CHARS = re.compile(r'[\|\•\[\]\(\)\"“”]')
_LEADING_HYPHEN = re.compile(r'(?m)^\s*-\s*')
_STRAY_HYPHEN = re.compile(r'(?<!\w)-(?!\w)')
_REPEATED_PUNCT = re.compile(r'([!?.,;:]){2,}')
_MULTI_SPACE = re.compile(r'[ \t]+')
_MULTI_NEWLINE = re.compile(r'\n{2,}')
_AGGRESSIVE_CLEAN = re.compile(r'[^A-Za-zÀ-ÿ\s\.\,\;\:\?\!]')
_MULTI_SPACE2 = re.compile(r'\s{2,}')

_EN_STOPWORDS_PATH = str(importlib.resources.files('compressor').joinpath('resources/en_stopwords.pkl'))
_PT_STOPWORDS_PATH = str(importlib.resources.files('compressor').joinpath('resources/pt_stopwords.pkl'))


@functools.lru_cache(maxsize=1)
def _ensure_nltk_ready():
    import nltk.data
    nltk.data.path.insert(0, _NLTK_DATA_PATH)


@functools.lru_cache(maxsize=1)
def _get_tokenizer():
    from compressor.minbpe.regex import RegexTokenizer
    return RegexTokenizer()


@functools.lru_cache(maxsize=1)
def _get_english_stemmer():
    _ensure_nltk_ready()
    from nltk.stem import PorterStemmer
    return PorterStemmer()


@functools.lru_cache(maxsize=1)
def _get_portuguese_stemmer():
    _ensure_nltk_ready()
    from nltk.stem import RSLPStemmer
    return RSLPStemmer()


@functools.lru_cache(maxsize=1)
def _get_language_detector():
    from lingua import Language, LanguageDetectorBuilder
    return LanguageDetectorBuilder.from_languages(
        Language.ENGLISH, Language.PORTUGUESE
    ).build()


@functools.lru_cache(maxsize=1)
def _get_language_enums():
    from lingua import Language
    return Language


@functools.lru_cache(maxsize=1)
def _get_english_stopwords():
    import pickle
    return pickle.load(open(_EN_STOPWORDS_PATH, "rb"))


@functools.lru_cache(maxsize=1)
def _get_portuguese_stopwords():
    import pickle
    return pickle.load(open(_PT_STOPWORDS_PATH, "rb"))


@functools.lru_cache(maxsize=1)
def _get_embedding_model():
    from model2vec import StaticModel
    return StaticModel.from_pretrained("cnmoro/static-nomic-eng-ptbr-tiny")


@functools.lru_cache(maxsize=1)
def _get_hashing_vectorizer():
    from sklearn.feature_extraction.text import HashingVectorizer
    return HashingVectorizer(ngram_range=(1, 6), analyzer='char', n_features=512)


@functools.lru_cache(maxsize=1)
def _get_sent_tokenize():
    _ensure_nltk_ready()
    from nltk.tokenize import sent_tokenize
    return sent_tokenize


def clean_text(text: str) -> str:
    text = _HYPHENATION.sub(r'\1\2', text)
    text = _NOISE_CHARS.sub(' ', text)
    text = _LEADING_HYPHEN.sub('', text)
    text = _STRAY_HYPHEN.sub(' ', text)
    text = _REPEATED_PUNCT.sub(r'\1', text)
    text = _MULTI_SPACE.sub(' ', text)
    text = _MULTI_NEWLINE.sub('\n', text).strip()

    alpha_ratio = sum(c.isalpha() for c in text) / max(len(text), 1)
    if alpha_ratio < 0.8:
        text = _AGGRESSIVE_CLEAN.sub(' ', text)
        text = _MULTI_SPACE2.sub(' ', text).strip()

    text = _PUNCT_REATTACH.sub(r'\1', text)
    text = _PUNCT_BOUNDARY.sub(r'\1 ', text)
    return text


def extract_textual_embeddings(text):
    v = _get_hashing_vectorizer()
    import numpy as np
    return np.asarray(v.transform([text]).sum(axis=0)).ravel().tolist()


def extract_semantic_embeddings(text):
    return _get_embedding_model().encode([text])[0]


def structurize_text(full_text, tokens_per_chunk=300, chunk_overlap=0):
    tok = _get_tokenizer()
    chunks = []
    current_chunk = []
    current_chunk_length = 0
    tokens = tok.encode(full_text)
    for i, token in enumerate(tokens):
        if current_chunk_length + 1 > tokens_per_chunk:
            chunks.append(current_chunk)
            current_chunk = tokens[i - chunk_overlap:i] if i > chunk_overlap else []
            current_chunk_length = len(current_chunk)
        current_chunk.append(token)
        current_chunk_length += 1
    chunks.append(current_chunk)
    return [tok.decode(chunk) for chunk in chunks]


def count_tokens(text):
    return len(_get_tokenizer().encode(text))


def detect_language(text):
    Language = _get_language_enums()
    lang = _get_language_detector().detect_language_of(text)
    return 'pt' if lang == Language.PORTUGUESE else 'en'


def compute_and_remove_repeated_ngrams(text, ngram_size=3, threshold=3):
    words = text.split()
    n = len(words)
    if n < ngram_size:
        return text

    ngram_tuples = [tuple(words[i:i + ngram_size]) for i in range(n - ngram_size + 1)]
    counter = Counter(ngram_tuples)
    repeated = [ng for ng, count in counter.items() if count > threshold]
    if not repeated:
        return text

    for ng in repeated:
        first = True
        i = 0
        while i <= len(words) - ngram_size:
            if tuple(words[i:i + ngram_size]) == ng:
                if first:
                    first = False
                    i += ngram_size
                else:
                    del words[i:i + ngram_size]
            else:
                i += 1
    return ' '.join(words)


def calculate_similarity(embed1, embed2):
    from sklearn.metrics.pairwise import cosine_similarity
    return cosine_similarity([embed1], [embed2])[0][0]


def _get_stopwords(lang):
    if lang == 'pt':
        return _get_portuguese_stopwords()
    return _get_english_stopwords()


def semantic_compress_text(full_text, compression_rate=0.7, num_topics=5, reference_text: str = None, perform_cleaning: bool = True):
    import warnings
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import TruncatedSVD
    import numpy as np
    import traceback

    try:
        if perform_cleaning:
            full_text = clean_text(full_text)

        sent_tokenize = _get_sent_tokenize()
        sentences = sent_tokenize(full_text)

        final_sentences = []
        for s in sentences:
            final_sentences.extend(s.split('\n'))
        sentences = final_sentences
        n_sentences = len(sentences)

        text_lang = detect_language(full_text)
        stopwords = _get_stopwords(text_lang)

        if n_sentences >= 3:
            n_topics = min(num_topics, max(2, n_sentences // 5))
            max_features = min(3000, max(500, n_sentences * 10))

            vectorizer = TfidfVectorizer(stop_words=stopwords, max_features=max_features)
            doc_term_matrix = vectorizer.fit_transform(sentences)
            svd = TruncatedSVD(n_components=n_topics, random_state=42)
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', category=RuntimeWarning, message='.*divide by zero.*')
                svd.fit(doc_term_matrix)
            topic_scores = np.abs(svd.transform(vectorizer.transform(sentences)))
        else:
            topic_scores = np.ones((n_sentences, 1)) * 0.5

        doc_embedding = extract_semantic_embeddings(full_text)

        if reference_text is not None:
            reference_text_embedding = extract_semantic_embeddings(reference_text)
            doc_embedding = 0.6 * doc_embedding + 0.4 * reference_text_embedding

        sentence_embeddings = _get_embedding_model().encode(sentences)

        sentence_scores = []
        for i, sentence in enumerate(sentences):
            sentence_embedding = sentence_embeddings[i]
            semantic_similarity = calculate_similarity(doc_embedding, sentence_embedding)

            topic_importance = float(np.max(topic_scores[i]))

            words = sentence.split()
            unique_words = set(w.lower() for w in words if w.lower() not in stopwords)
            lexical_diversity = len(unique_words) / len(words) if words else 0

            importance = 0.6 * semantic_similarity + 0.3 * topic_importance + 0.2 * lexical_diversity
            sentence_scores.append((sentence, importance))

        sorted_sentences = sorted(sentence_scores, key=lambda x: x[1], reverse=True)

        total_words = sum(len(s.split()) for s in sentences)
        target_words = int(total_words * compression_rate)

        compressed_text = []
        current_words = 0
        for sentence, _ in sorted_sentences:
            sentence_words = len(sentence.split())
            if current_words + sentence_words <= target_words:
                compressed_text.append(sentence)
                current_words += sentence_words
            else:
                break

        if not compressed_text:
            compressed_text = [sentences[0]]

        compressed_text.sort(key=lambda x: sentences.index(x))
        compressed_text = [s.capitalize() for s in compressed_text]

        cleaned_compressed_text = ' '.join(compressed_text).replace('  ', ' ').strip()
        cleaned_compressed_text = compute_and_remove_repeated_ngrams(cleaned_compressed_text)
        return cleaned_compressed_text
    except Exception:
        traceback.print_exc()
    return full_text


def compress_text(text, *, target_token_count=None, compression_rate=0.7, reference_text_steering=None, perform_cleaning=True):
    import traceback
    try:
        if target_token_count is None:
            compression_rate = 1 - compression_rate
        else:
            original_token_count = count_tokens(text)
            if original_token_count <= target_token_count:
                return text
            compression_rate = target_token_count / original_token_count

        return semantic_compress_text(
            full_text=text,
            compression_rate=compression_rate,
            reference_text=reference_text_steering,
            perform_cleaning=perform_cleaning
        )
    except Exception:
        traceback.print_exc()
    return text


def stem_text(text, lang='en'):
    if lang == 'en':
        stemmer = _get_english_stemmer()
    else:
        stemmer = _get_portuguese_stemmer()
    return ' '.join(stemmer.stem(word) for word in text.split())


def correct_spelling(sentence, detected_lang="pt"):
    from spellchecker import SpellChecker
    spell = SpellChecker(language=detected_lang)
    words = sentence.split()
    fixed = [spell.correction(word) for word in words]
    result = []
    for original, fixed_word in zip(words, fixed):
        result.append(fixed_word if fixed_word is not None else original)
    return ' '.join(result)


def preprocess_and_extract_textual_embedding(block, use_stemming, lang):
    processed_block = block.lower() if not use_stemming else stem_text(block.lower(), lang)
    return extract_textual_embeddings(processed_block)


def find_needle_in_haystack(
        *, haystack: str, needle: str, block_size=300,
        embedding_mode: str = 'both',
        semantic_embeddings_weight: float = 0.3,
        textual_embeddings_weight: float = 0.7,
        use_stemming: bool = False,
        correct_spelling_needle: bool = False
    ):
    import traceback
    try:
        if embedding_mode not in {'semantic', 'textual', 'both'}:
            raise ValueError("Invalid embedding_mode. Choose 'semantic', 'textual', or 'both'.")

        blocks = structurize_text(haystack, tokens_per_chunk=block_size)

        lang = detect_language(f"{needle}\n\n{haystack}")

        if correct_spelling_needle:
            needle = correct_spelling(needle, lang)

        needle_semantic_embedding = None
        needle_textual_embedding = None

        if embedding_mode in {'semantic', 'both'}:
            needle_semantic_embedding = extract_semantic_embeddings(needle)

        if embedding_mode in {'textual', 'both'}:
            needle_textual_embedding = extract_textual_embeddings(
                needle.lower() if not use_stemming else stem_text(needle, lang)
            )

        haystack_semantic_embeddings = []
        haystack_textual_embeddings = []

        if embedding_mode in {'semantic', 'both'}:
            if len(blocks) == 1:
                haystack_semantic_embeddings = [extract_semantic_embeddings(blocks[0])]
            else:
                from concurrent.futures import ProcessPoolExecutor
                with ProcessPoolExecutor() as executor:
                    haystack_semantic_embeddings = list(executor.map(extract_semantic_embeddings, blocks))

        if embedding_mode in {'textual', 'both'}:
            if len(blocks) == 1:
                haystack_textual_embeddings = [preprocess_and_extract_textual_embedding(blocks[0], use_stemming, lang)]
            else:
                from concurrent.futures import ProcessPoolExecutor
                from multiprocessing import cpu_count
                with ProcessPoolExecutor(max_workers=int(cpu_count() // 1.5)) as executor:
                    haystack_textual_embeddings = list(
                        executor.map(preprocess_and_extract_textual_embedding, blocks, [use_stemming] * len(blocks), [lang] * len(blocks))
                    )

        semantic_similarities = []
        textual_similarities = []

        if embedding_mode in {'semantic', 'both'}:
            semantic_similarities = [
                calculate_similarity(needle_semantic_embedding, be)
                for be in haystack_semantic_embeddings
            ]

        if embedding_mode in {'textual', 'both'}:
            textual_similarities = [
                calculate_similarity(needle_textual_embedding, be)
                for be in haystack_textual_embeddings
            ]

        if embedding_mode == 'semantic':
            sorted_blocks = sorted(zip(blocks, semantic_similarities), key=lambda x: x[1], reverse=True)
        elif embedding_mode == 'textual':
            sorted_blocks = sorted(zip(blocks, textual_similarities), key=lambda x: x[1], reverse=True)
        else:
            sorted_blocks = sorted(
                zip(blocks, semantic_similarities, textual_similarities),
                key=lambda x: x[1] * semantic_embeddings_weight + x[2] * textual_embeddings_weight,
                reverse=True
            )

        most_similar_block = sorted_blocks[0][0]
        most_similar_block_index = blocks.index(most_similar_block)
        start_index = most_similar_block_index - 1 if most_similar_block_index > 0 else 0
        needle_region = blocks[start_index:most_similar_block_index + 2]
        return ''.join(needle_region).strip()
    except Exception:
        traceback.print_exc()
    return haystack
