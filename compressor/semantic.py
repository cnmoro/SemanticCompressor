import os, sys, importlib.resources, functools, re

_NLTK_DATA_PATH = None
try:
    _NLTK_DATA_PATH = str(importlib.resources.files('compressor').joinpath('resources/nltk_data'))
except Exception:
    _NLTK_DATA_PATH = os.path.join(os.path.dirname(__file__), 'resources', 'nltk_data')

os.environ['NLTK_DATA'] = _NLTK_DATA_PATH

from collections import Counter
from itertools import cycle
from sklearn.metrics.pairwise import cosine_similarity

_PUNCT_REATTACH = re.compile(r'\s+([.!,\?;:])')
_PUNCT_BOUNDARY = re.compile(r'([.!,\?;:])(?=\S)')
_HYPHENATION = re.compile(r'(\w)-\s*\n\s*(\w)')
_NOISE_CHARS = re.compile(r'[\|\•\*“”]')
_LEADING_HYPHEN = re.compile(r'(?m)^\s*-\s*')
_STRAY_HYPHEN = re.compile(r'(?<!\w)-(?!\w)')
_REPEATED_PUNCT = re.compile(r'([!?.,;:]){2,}')
_MULTI_SPACE = re.compile(r'[ \t]+')
_MULTI_NEWLINE = re.compile(r'\n{2,}')
_AGGRESSIVE_CLEAN = re.compile(r'[^A-Za-zÀ-ÿ0-9\s\.\,\;\:\?\!\"\'%\\\$_\{\}\[\]\(\)\#\@\<\>\-\+\=\/\^]')
_MULTI_SPACE2 = re.compile(r'\s{2,}')
_URLS = re.compile(r'https?://\S+')
_HTML_TAGS = re.compile(r'<[^>]+>')
_MARKDOWN_LINKS = re.compile(r'\[([^\]]*)\]\([^)]+\)')
_EMAILS = re.compile(r'\S+@\S+\.\S+')

_EN_STOPWORDS_PATH = str(importlib.resources.files('compressor').joinpath('resources/en_stopwords.pkl'))
_PT_STOPWORDS_PATH = str(importlib.resources.files('compressor').joinpath('resources/pt_stopwords.pkl'))

_DANGLING_PRONOUNS = {
    'en': frozenset({'he', 'she', 'it', 'they', 'this', 'these', 'that', 'those'}),
    'pt': frozenset({'ele', 'ela', 'eles', 'elas', 'isto', 'isso', 'aquilo',
                     'este', 'esta', 'estes', 'estas', 'esse', 'essa', 'esses', 'essas',
                     'aquele', 'aquela', 'aqueles', 'aquelas'}),
}
_CONJUNCTION_STARTERS = {
    'en': frozenset({'and', 'but', 'so', 'or', 'nor', 'yet'}),
    'pt': frozenset({'e', 'mas', 'ou', 'nem', 'porém', 'contudo', 'todavia'}),
}
_STITCH_SKIP_FIRST = {
    'en': frozenset({
        'however', 'but', 'yet', 'although', 'though', 'while',
        'nevertheless', 'nonetheless', 'instead', 'rather', 'today',
        'meanwhile', 'furthermore', 'moreover', 'additionally', 'also',
        'besides', 'consequently', 'therefore', 'thus', 'hence',
        'which', 'that', 'who', 'whom', 'whose', 'where', 'when', 'if',
    }),
    'pt': frozenset({
        'embora', 'mas', 'porém', 'contudo', 'todavia', 'apesar',
        'entretanto', 'hoje', 'assim', 'também', 'ainda', 'logo',
        'afinal', 'portanto', 'consequentemente',
        'que', 'quem', 'cujo', 'cuja', 'onde', 'quando',
    }),
}
_STITCH_SKIP_FIRST_TWO = {
    'en': frozenset({('even', 'though'), ('even', 'if'), ('in', 'addition')}),
    'pt': frozenset({('no', 'entanto'), ('por', 'isso'), ('por', 'outro'),
                     ('apesar', 'de'), ('mesmo', 'assim'), ('ainda', 'assim')}),
}
_CONNECTORS = {
    'en': {
        'small': ['Additionally, ', 'Moreover, ', 'Further, ', 'Besides, '],
        'medium': ['Furthermore, ', 'In addition, ', 'On top of that, ', 'What is more, '],
        'large': ['Meanwhile, ', 'On the other hand, ', 'Separately, ', 'In other developments, '],
    },
    'pt': {
        'small': [
            'Além disso, ',
            'Também, ',
            'Ainda, ',
            'Assim, ',
            'Porém, ',
            'Contudo, ',
            'Todavia, ',
            'Logo, ',
            'Afinal, ',
        ],
        'medium': [
            'Do mesmo modo, ',
            'Nesse sentido, ',
            'Somado a isso, ',
            'Da mesma forma, ',
            'Nesse contexto, ',
            'Diante disso, ',
            'Com isso, ',
            'Sendo assim, ',
            'Desse modo, ',
            'Dessa forma, ',
            'Em seguida, ',
            'Ao mesmo tempo, ',
            'Por isso, ',
            'Ainda assim, ',
        ],
        'large': [
            'Por outro lado, ',
            'Enquanto isso, ',
            'Por sua vez, ',
            'Já, ',
            'Em contrapartida, ',
            'No mesmo sentido, ',
            'Além do mais, ',
            'Em complemento, ',
            'No mesmo contexto, ',
            'Em consonância com isso, ',
            'Nesse mesmo cenário, ',
            'Sob essa perspectiva, ',
            'Ao longo disso, ',
            'Em linha com isso, ',
        ]
    }
}


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


def _filter_noise_lines(text):
    lines = text.split('\n')
    filtered = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        words = stripped.split()
        if not words:
            continue
        first_word = words[0].lower().strip(':,;.!?')
        if first_word.startswith('image') or first_word.startswith('imagem'):
            continue
        if first_word in ('title', 'url', 'published', 'markdown', 'home', 'busca', 'bate', 'email', 'anuncie', 'colunistas', 'expediente', 'parcerias', 'cookie', 'subscribe', 'newsletter', 'copyright', 'direitos'):
            continue
        if first_word == 'prompt' or (len(words) >= 2 and words[1].lower() == 'prompt'):
            continue
        if len(words) <= 2 and any(c.isdigit() for c in stripped):
            continue
        alpha = sum(1 for c in stripped if c.isalpha())
        non_alpha = len(stripped) - alpha
        if non_alpha > alpha * 2:
            continue
        filtered.append(stripped)
    return '\n'.join(filtered)


def clean_text(text: str) -> str:
    text = _MARKDOWN_LINKS.sub(r'\1', text)
    text = _EMAILS.sub(' ', text)
    text = _URLS.sub(' ', text)
    text = _HTML_TAGS.sub(' ', text)
    text = _filter_noise_lines(text)
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
    text = re.sub(r'\b(e)\s*\.\s*(g)\s*[.,]', 'e.g.', text)
    text = re.sub(r'\b(i)\s*\.\s*(e)\s*[.,]', 'i.e.', text)
    text = re.sub(r'(\d)\s*\.\s*(\d)', r'\1.\2', text)
    text = re.sub(r'(\d)\s*:\s*(\d)', r'\1:\2', text)
    text = re.sub(r'(?<![A-Za-z])\.(?:\s+\.)+', '.', text)
    text = re.sub(r':\s*\.', ':', text)
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
        new_words = []
        i = 0
        while i < n:
            if i <= n - ngram_size and ngram_tuples[i] == ng:
                if first:
                    new_words.extend(words[i:i + ngram_size])
                    first = False
                i += ngram_size
            else:
                new_words.append(words[i])
                i += 1
        words = new_words
        n = len(words)
        ngram_tuples = [tuple(words[i:i + ngram_size]) for i in range(n - ngram_size + 1)]

    return ' '.join(words)


def calculate_similarity(embed1, embed2):
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
        code_blocks = re.findall(r'```(\w*)\n(.*?)```', full_text, re.DOTALL)
        full_text = re.sub(r'```(\w*)\n.*?```', '', full_text, flags=re.DOTALL)

        if perform_cleaning:
            full_text = clean_text(full_text)

        sent_tokenize = _get_sent_tokenize()
        sentences = sent_tokenize(full_text)

        final_sentences = []
        for s in sentences:
            final_sentences.extend(s.split('\n'))
        _FRAGMENT_STARTERS = frozenset({
            'which', 'that', 'who', 'whom', 'whose', 'where', 'when', 'while',
            'although', 'though', 'because', 'unless', 'until', 'after', 'before',
            'if', 'whether', 'whereas', 'whereby',
            'que', 'se', 'quando', 'enquanto', 'embora', 'caso', 'mesmo',
        })
        sentences = []
        for s in final_sentences:
            words = s.split()
            if len(words) < 4:
                continue
            if not s[0].isalnum():
                continue
            first_word = words[0].lower().strip(',;:.!?')
            if s[0].islower() and first_word in _FRAGMENT_STARTERS:
                continue
            caps_words = [w for w in words if len(w) > 1 and w.isupper()]
            if len(caps_words) >= 3:
                avg_len = sum(len(w) for w in caps_words) / len(caps_words)
                if avg_len < 4.5 and len(caps_words) / max(len(words), 1) > 0.3:
                    continue
            sentences.append(s)
        if not sentences:
            sentences = final_sentences
        n_sentences = len(sentences)
        if n_sentences == 0:
            return full_text

        text_lang = detect_language(full_text)
        stopwords = _get_stopwords(text_lang)

        dangling_pronouns = _DANGLING_PRONOUNS.get(text_lang, _DANGLING_PRONOUNS['en'])
        conjunction_starters = _CONJUNCTION_STARTERS.get(text_lang, _CONJUNCTION_STARTERS['en'])
        connector_pool = _CONNECTORS.get(text_lang, _CONNECTORS['en'])
        connector_iters = {k: cycle(v) for k, v in connector_pool.items()}
        stitch_skip_first = _STITCH_SKIP_FIRST.get(text_lang, _STITCH_SKIP_FIRST['en'])
        stitch_skip_first_two = _STITCH_SKIP_FIRST_TWO.get(text_lang, _STITCH_SKIP_FIRST_TWO['en'])

        sentence_words = [s.split() for s in sentences]
        sentence_word_counts = [len(w) for w in sentence_words]

        if n_sentences >= 6:
            n_topics = min(num_topics, max(2, n_sentences // 5))
            max_features = min(3000, max(500, n_sentences * 10))

            vectorizer = TfidfVectorizer(stop_words=stopwords, max_features=max_features)
            doc_term_matrix = vectorizer.fit_transform(sentences)
            svd = TruncatedSVD(n_components=n_topics, random_state=42)
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', category=RuntimeWarning, message='.*divide by zero.*')
                svd.fit(doc_term_matrix)
            topic_scores = np.abs(svd.transform(doc_term_matrix))
        else:
            topic_scores = np.ones((n_sentences, 1)) * 0.5

        doc_embedding = extract_semantic_embeddings(full_text)

        if reference_text is not None:
            reference_text_embedding = extract_semantic_embeddings(reference_text)
            doc_embedding = 0.6 * doc_embedding + 0.4 * reference_text_embedding

        sentence_embeddings = _get_embedding_model().encode(sentences)

        sentence_scores = []
        for i in range(n_sentences):
            words = sentence_words[i]
            sentence_embedding = sentence_embeddings[i]
            semantic_similarity = calculate_similarity(doc_embedding, sentence_embedding)

            topic_importance = float(np.max(topic_scores[i]))

            unique_non_stop = set(w.lower() for w in words if w.lower() not in stopwords)
            lexical_diversity = len(unique_non_stop) / max(len(words), 1)

            position_weight = 1.0 + 0.05 * (1 - i / max(n_sentences - 1, 1))
            importance = (0.6 * semantic_similarity + 0.3 * topic_importance + 0.2 * lexical_diversity) * position_weight

            noise_penalty = 1.0
            first_word = words[0].lower() if words else ''
            if first_word.startswith('image') or first_word.startswith('imagem'):
                noise_penalty = 0.7
            importance *= noise_penalty

            sentence_scores.append((sentences[i], importance, sentence_word_counts[i], i))

        sorted_sentences = sorted(sentence_scores, key=lambda x: x[1], reverse=True)

        total_words = sum(sentence_word_counts)
        target_words = int(total_words * compression_rate)

        kept_indices = set()
        result_pairs = []
        current_words = 0
        for sentence, _, word_count, idx in sorted_sentences:
            if current_words + word_count <= target_words:
                result_pairs.append((sentence, idx))
                kept_indices.add(idx)
                current_words += word_count
            elif current_words < target_words:
                remaining = target_words - current_words
                if remaining >= 2:
                    truncated = ' '.join(sentence_words[idx][:remaining])
                    if truncated:
                        result_pairs.append((truncated, idx))
                        kept_indices.add(idx)
                        current_words += remaining
                break
            else:
                break

        if not result_pairs:
            result_pairs = [(sentences[0], 0)]
            kept_indices.add(0)

        result_pairs.sort(key=lambda x: x[1])
        result_texts = [p[0] for p in result_pairs]
        result_indices = [p[1] for p in result_pairs]

        deduped_texts = []
        deduped_indices = []
        for s, idx in zip(result_texts, result_indices):
            words_s = set(w.lower() for w in s.split() if w.lower() not in stopwords)
            is_dup = False
            for existing in deduped_texts:
                words_e = set(w.lower() for w in existing.split() if w.lower() not in stopwords)
                if words_s and words_e:
                    overlap = len(words_s & words_e) / max(len(words_s), len(words_e))
                    if overlap > 0.55:
                        is_dup = True
                        break
            if not is_dup:
                deduped_texts.append(s)
                deduped_indices.append(idx)
        result_texts = deduped_texts
        result_indices = deduped_indices

        unused_candidates = [x for x in sorted_sentences if x[3] not in kept_indices]
        filtered_texts = []
        filtered_indices = []
        for s, idx in zip(result_texts, result_indices):
            first_word = s.split()[0].lower() if s.split() else ''
            if first_word in dangling_pronouns and idx > 0 and (idx - 1) not in kept_indices:
                for alt in unused_candidates:
                    alt_s, _, _, alt_idx = alt
                    if alt_s.split()[0].lower() not in dangling_pronouns:
                        filtered_texts.append(alt_s)
                        filtered_indices.append(alt_idx)
                        kept_indices.add(alt_idx)
                        unused_candidates = [x for x in unused_candidates if x[3] != alt_idx]
                        break
                else:
                    filtered_texts.append(s)
                    filtered_indices.append(idx)
            else:
                filtered_texts.append(s)
                filtered_indices.append(idx)
        result_texts = filtered_texts
        result_indices = filtered_indices

        if result_texts and result_texts[0]:
            first_word = result_texts[0].split()[0].lower() if result_texts[0].split() else ''
            if first_word not in {'def', 'class', 'import', 'from', 'return', 'if', 'for', 'while'}:
                result_texts[0] = result_texts[0][0].upper() + result_texts[0][1:]

        fused_texts = []
        fused_indices = []
        i = 0
        while i < len(result_texts):
            curr_s = result_texts[i]
            curr_idx = result_indices[i]
            curr_words = curr_s.split()
            if (i > 0 and curr_words and curr_words[0].lower() in conjunction_starters
                    and curr_idx - result_indices[i - 1] == 1):
                prev_s = fused_texts.pop()
                prev_idx = fused_indices.pop()
                if curr_words[0][0].isupper():
                    curr_words[0] = curr_words[0][0].lower() + curr_words[0][1:]
                merged = prev_s.rstrip('.!?') + ', ' + ' '.join(curr_words)
                fused_texts.append(merged)
                fused_indices.append(prev_idx)
            else:
                fused_texts.append(curr_s)
                fused_indices.append(curr_idx)
            i += 1
        result_texts = fused_texts
        result_indices = fused_indices

        stitched_texts = [result_texts[0]]
        for i in range(1, len(result_texts)):
            gap = result_indices[i] - result_indices[i - 1]
            if gap >= 4 and result_texts[i][0].isalpha():
                curr_words = result_texts[i].split()
                if curr_words:
                    first_lower = curr_words[0].lower().strip(',;:.!?')
                    first_two = (first_lower, curr_words[1].lower().strip(',;:.!?')) if len(curr_words) > 1 else None
                    if first_lower in stitch_skip_first or (first_two and first_two in stitch_skip_first_two):
                        stitched_texts.append(result_texts[i])
                    else:
                        if gap <= 6:
                            connector = next(connector_iters['small'])
                        elif gap <= 10:
                            connector = next(connector_iters['medium'])
                        else:
                            connector = next(connector_iters['large'])
                        curr_words[0] = curr_words[0][0].lower() + curr_words[0][1:]
                        stitched_texts.append(connector + ' '.join(curr_words))
                else:
                    stitched_texts.append(result_texts[i])
            else:
                stitched_texts.append(result_texts[i])
        result_texts = stitched_texts

        cleaned = ' '.join(result_texts).replace('  ', ' ').strip()
        cleaned = compute_and_remove_repeated_ngrams(cleaned)
        cleaned = compute_and_remove_repeated_ngrams(cleaned, ngram_size=5, threshold=1)
        cleaned = compute_and_remove_repeated_ngrams(cleaned, ngram_size=2, threshold=2)
        cleaned = re.sub(r',\s*,', ',', cleaned)
        cleaned = re.sub(r'\s+([,;:.!?])', r'\1', cleaned)
        if cleaned and cleaned[-1] not in '.!?':
            cleaned += '.'
        if code_blocks:
            code_section = '\n\n'.join(
                f'```{lang}\n{block.strip()}\n```' for lang, block in code_blocks
            )
            cleaned += '\n\n' + code_section
        return cleaned
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
