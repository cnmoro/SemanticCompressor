import numpy as np, pickle, fasttext, os, traceback, importlib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.metrics.pairwise import cosine_similarity
from onnxruntime_extensions import get_library_path
from compressor.minbpe.regex import RegexTokenizer
from nltk.tokenize import sent_tokenize
from multiprocessing import cpu_count
import onnxruntime as ort

tokenizer = RegexTokenizer()
nltk_data_path = str(importlib.resources.files('compressor').joinpath('resources/nltk_data'))

os.environ['NLTK_DATA'] = nltk_data_path

english_stopwords_path = str(importlib.resources.files('compressor').joinpath('resources/en_stopwords.pkl'))
portuguese_stopwords_path = str(importlib.resources.files('compressor').joinpath('resources/pt_stopwords.pkl'))
fasttext_model_path = str(importlib.resources.files('compressor').joinpath('resources/lid.176.ftz'))
english_stopwords = pickle.load(open(english_stopwords_path, "rb"))
portuguese_stopwords = pickle.load(open(portuguese_stopwords_path, "rb"))
langdetect_model = fasttext.load_model(fasttext_model_path)

embedding_model_cpu_count = os.environ.get('EMBEDDING_MODEL_CPU_COUNT', cpu_count() - 1)

_options = ort.SessionOptions()
_options.inter_op_num_threads, _options.intra_op_num_threads = embedding_model_cpu_count, embedding_model_cpu_count
_options.register_custom_ops_library(get_library_path())
_providers = ["CPUExecutionProvider"]

embedding_model = ort.InferenceSession(
    path_or_bytes = str(importlib.resources.files('compressor').joinpath('resources/embedding_model.onnx')),
    sess_options=_options,
    providers=_providers
)

def extract_embeddings(text):
    return embedding_model.run(output_names=["outputs"], input_feed={"inputs": [text]})[0][0]

def structurize_text(full_text, tokens_per_chunk=300, chunk_overlap=0):
    chunks = []
    current_chunk = []
    current_chunk_length = 0
    tokens = tokenizer.encode(full_text)
    for i, token in enumerate(tokens):
        if current_chunk_length + 1 > tokens_per_chunk:
            chunks.append(current_chunk)
            current_chunk = tokens[i-chunk_overlap:i] if i > chunk_overlap else []
            current_chunk_length = len(current_chunk)
        current_chunk.append(token)
        current_chunk_length += 1
    chunks.append(current_chunk)
    chunks = [tokenizer.decode(chunk) for chunk in chunks]
    return chunks

def count_tokens(text):
    return len(tokenizer.encode(text))

def detect_language(text):
    detected_lang = langdetect_model.predict(text.replace('\n', ' '), k=1)[0][0]
    return 'pt' if (str(detected_lang) == '__label__pt' or str(detected_lang) == 'portuguese') else 'en'

def semantic_compress_text(full_text, compression_rate=0.7, num_topics=5):
    def calculate_similarity(embed1, embed2):
        return cosine_similarity([embed1], [embed2])[0][0]

    def create_lda_model(texts, stopwords):
        vectorizer = CountVectorizer(stop_words=stopwords)
        doc_term_matrix = vectorizer.fit_transform(texts)
        lda = LatentDirichletAllocation(n_components=num_topics, random_state=42)
        lda.fit(doc_term_matrix)
        return lda, vectorizer

    def get_topic_distribution(text, lda, vectorizer):
        vec = vectorizer.transform([text])
        return lda.transform(vec)[0]

    def sentence_importance(sentence, doc_embedding, lda_model, vectorizer, stopwords):
        sentence_embedding = extract_embeddings(sentence)
        semantic_similarity = calculate_similarity(doc_embedding, sentence_embedding)
        
        topic_dist = get_topic_distribution(sentence, lda_model, vectorizer)
        topic_importance = np.max(topic_dist)
        
        # Calculate lexical diversity
        words = sentence.split()
        unique_words = set([word.lower() for word in words if word.lower() not in stopwords])
        lexical_diversity = len(unique_words) / len(words) if words else 0
        
        # Combine factors
        importance = (0.6 * semantic_similarity) + (0.3 * topic_importance) + (0.2 * lexical_diversity)
        return importance

    try:
        # Split the text into sentences
        sentences = sent_tokenize(full_text)

        final_sentences = []
        for s in sentences:
            broken_sentences = s.split('\n')
            final_sentences.extend(broken_sentences)
        sentences = final_sentences

        text_lang = detect_language(full_text)

        # Create LDA model
        lda_model, vectorizer = create_lda_model(sentences, portuguese_stopwords if text_lang == 'pt' else english_stopwords)

        # Get document-level embedding
        doc_embedding = extract_embeddings(full_text)

        # Calculate importance for each sentence
        sentence_scores = [(sentence, sentence_importance(sentence, doc_embedding, lda_model, vectorizer, portuguese_stopwords if text_lang == 'pt' else english_stopwords)) 
                        for sentence in sentences]

        # Sort sentences by importance
        sorted_sentences = sorted(sentence_scores, key=lambda x: x[1], reverse=True)

        # Determine how many words to keep
        total_words = sum(len(sentence.split()) for sentence in sentences)
        target_words = int(total_words * compression_rate)

        # Reconstruct the compressed text
        compressed_text = []
        current_words = 0
        for sentence, _ in sorted_sentences:
            sentence_words = len(sentence.split())
            if current_words + sentence_words <= target_words:
                compressed_text.append(sentence)
                current_words += sentence_words
            else:
                break
        
        if len(compressed_text) == 0:
            # Pick the first sentence if no compression is possible
            compressed_text = [sentences[0]]

        # Reorder sentences to maintain original flow
        compressed_text.sort(key=lambda x: sentences.index(x))

        return ' '.join(compressed_text)
    except Exception:
        traceback.print_exc()
    
    return full_text

def compress_text(text, *, target_token_count=None, compression_rate=0.7):
    """
    Compress text using either a compression rate or a target token count.
    If both are provided, the compression rate will be used.

    Args:
        text (str): The text to be compressed.
        target_token_count (int, optional): The target token count for compression. Defaults to None.
        compression_rate (float, optional): The compression rate as a percentage. Defaults to 0.7. Example: 0.7 means 70% reduction.

    Returns:
        str: The compressed text.
    """

    if target_token_count is None:
        compression_rate = 1 - compression_rate
        original_token_count = count_tokens(text)
        target_token_count = int(original_token_count * compression_rate)
    else:
        original_token_count = count_tokens(text)
        if original_token_count <= target_token_count:
            return text
        # Get the compression rate
        compression_rate = target_token_count / original_token_count

    return semantic_compress_text(text, compression_rate)
