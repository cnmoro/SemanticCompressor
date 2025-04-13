import os, importlib
os.environ['NLTK_DATA'] = str(importlib.resources.files('compressor').joinpath('resources/nltk_data'))

from sklearn.feature_extraction.text import CountVectorizer, HashingVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.metrics.pairwise import cosine_similarity
from compressor.minbpe.regex import RegexTokenizer
from concurrent.futures import ProcessPoolExecutor
import numpy as np, pickle, fasttext, traceback
from nltk.tokenize import sent_tokenize
from multiprocessing import cpu_count
from spellchecker import SpellChecker
from nltk.stem import PorterStemmer
from nltk.stem import RSLPStemmer
from collections import Counter
from model2vec import StaticModel

tokenizer = RegexTokenizer()

# Inicializando os stemmers
stemmer_english = PorterStemmer()
stemmer_portuguese = RSLPStemmer()

english_stopwords_path = str(importlib.resources.files('compressor').joinpath('resources/en_stopwords.pkl'))
portuguese_stopwords_path = str(importlib.resources.files('compressor').joinpath('resources/pt_stopwords.pkl'))
fasttext_model_path = str(importlib.resources.files('compressor').joinpath('resources/lid.176.ftz'))
english_stopwords = pickle.load(open(english_stopwords_path, "rb"))
portuguese_stopwords = pickle.load(open(portuguese_stopwords_path, "rb"))
langdetect_model = fasttext.load_model(fasttext_model_path)

embedding_model = StaticModel.from_pretrained("minishlab/potion-base-2M")

hashing_vectorizer = HashingVectorizer(ngram_range=(1, 6), analyzer='char', n_features=512)

def extract_textual_embeddings(text):
    X = hashing_vectorizer.fit_transform([text])
    dense_matrix = X.toarray()
    fixed_size_matrix = np.sum(dense_matrix, axis=0)
    return fixed_size_matrix.tolist()

def extract_semantic_embeddings(text):
    return embedding_model.encode([text])[0]

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

def compute_and_remove_repeated_ngrams(text, ngram_size=3, threshold=3):
    words = text.split()

    ngrams = [' '.join(words[i:i+ngram_size]) for i in range(len(words)-ngram_size+1)]

    counter = Counter(ngrams)

    repeated_ngrams = [ngram for ngram, count in counter.items() if count > threshold]

    # Iterate through each repeated n-gram and remove the duplicates
    for ngram in repeated_ngrams:
        # Track if it's the first occurrence
        first_occurrence = True
        i = 0
        
        while i <= len(words) - ngram_size:
            # Form a sliding window n-gram from the current position
            current_ngram = ' '.join(words[i:i+ngram_size])
            
            if current_ngram == ngram:
                if first_occurrence:
                    # Mark the first occurrence and skip
                    first_occurrence = False
                    i += ngram_size  # Move ahead by the size of the n-gram
                else:
                    # Remove the n-gram by removing the words that make up this n-gram
                    del words[i:i+ngram_size]
            else:
                i += 1  # Move forward

    # Rejoin the words back into a single string
    return ' '.join(words)

def calculate_similarity(embed1, embed2):
    return cosine_similarity([embed1], [embed2])[0][0]

def semantic_compress_text(full_text, compression_rate=0.7, num_topics=5, reference_text: str = None):
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
        sentence_embedding = extract_semantic_embeddings(sentence)
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
        doc_embedding = extract_semantic_embeddings(full_text)

        if reference_text is not None:
            reference_text_embedding = extract_semantic_embeddings(reference_text)

            # Compute an weighted average of the two embeddings (60% document and 40% reference)
            doc_embedding = 0.6 * doc_embedding + 0.4 * reference_text_embedding

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

        # Capitalize the first letter of each sentence
        compressed_text = [sentence.capitalize() for sentence in compressed_text]

        cleaned_compressed_text = ' '.join(compressed_text).replace('  ', ' ').strip()
        cleaned_compressed_text = compute_and_remove_repeated_ngrams(cleaned_compressed_text)
        return cleaned_compressed_text
    except Exception:
        traceback.print_exc()
    
    return full_text

def compress_text(text, *, target_token_count=None, compression_rate=0.7, reference_text_steering=None):
    """
    Compress text using either a compression rate or a target token count.
    If both are provided, the compression rate will be used.

    Args:
        text (str): The text to be compressed.
        target_token_count (int, optional): The target token count for compression. Defaults to None.
        compression_rate (float, optional): The compression rate as a percentage. Defaults to 0.7. Example: 0.7 means 70% reduction.
        reference_text_steering (str, optional): The reference text to steer the compression. Defaults to None.
        
    Returns:
        str: The compressed text.
    """
    try:
        if target_token_count is None:
            compression_rate = 1 - compression_rate
        else:
            original_token_count = count_tokens(text)
            if original_token_count <= target_token_count:
                return text
            # Get the compression rate
            compression_rate = target_token_count / original_token_count

        return semantic_compress_text(
            full_text = text,
            compression_rate = compression_rate,
            reference_text = reference_text_steering
        )
    except Exception:
        traceback.print_exc()

    return text

def stem_text(text, lang='en'):
    if lang == 'en':
        stems = [stemmer_english.stem(word) for word in text.split()]
        stemmed_text = " ".join(stems)
    else:
        stems = [stemmer_portuguese.stem(word) for word in text.split()]
        stemmed_text = " ".join(stems)

    return stemmed_text

def correct_spelling(sentence, detected_lang="pt"):
    spell = SpellChecker(language=detected_lang)
    words = sentence.split()
    fixed = [spell.correction(word) for word in words]

    final_words = []

    # Interpolate original words with fixed words (each word could be "None" in "fixed" when no correction is needed)
    for original, fixed_word in zip(words, fixed):
        final_words.append(fixed_word if fixed_word is not None else original)

    return " ".join(final_words)

def preprocess_and_extract_textual_embedding(block, use_stemming, lang):
    """
    Preprocesses a block (lowercasing and stemming if required) and extracts textual embeddings.

    Args:
        block (str): The text block to process.
        use_stemming (bool): Whether to apply stemming.
        lang (str): Language of the text for stemming.

    Returns:
        np.array: The textual embedding of the processed block.
    """
    processed_block = block.lower() if not use_stemming else stem_text(block.lower(), lang)
    return extract_textual_embeddings(processed_block)


def find_needle_in_haystack(
        *, haystack: str, needle: str, block_size=300,
        embedding_mode: str = 'both',  # 'semantic', 'textual', or 'both'
        semantic_embeddings_weight: float = 0.3,
        textual_embeddings_weight: float = 0.7,
        use_stemming: bool = False,
        correct_spelling_needle: bool = False
    ):
    """
    Finds the string block in the haystack that contains the needle.

    Args:
        haystack (str): The haystack string.
        needle (str): The needle string.
        block_size (int, optional): The size of each string block. The needle will be searched in each block. Defaults to 350.
        embedding_mode (str, optional): The embedding type to use: 'semantic', 'textual', or 'both'. Defaults to 'both'.
        semantic_embeddings_weight (float, optional): The weight of the semantic embeddings in the similarity calculation. Defaults to 0.3.
        textual_embeddings_weight (float, optional): The weight of the textual embeddings in the similarity calculation. Defaults to 0.7.
        use_stemming (bool, optional): Whether to use stemming for the text. Defaults to False.
        correct_spelling_needle (bool, optional): Whether to correct the spelling of the needle. Defaults to False.

    Returns:
        str: The string block in the haystack that contains the needle. The size of the needle will be less than or equal to the block size.
    """
    
    try:
        # Validate embedding_mode
        if embedding_mode not in {'semantic', 'textual', 'both'}:
            raise ValueError("Invalid embedding_mode. Choose 'semantic', 'textual', or 'both'.")
        
        # Split the haystack into blocks
        blocks = structurize_text(haystack, tokens_per_chunk=block_size)

        lang = detect_language(f"{needle}\n\n{haystack}")
        
        if correct_spelling_needle:
            needle = correct_spelling(needle, lang)
        
        # Compute the embeddings of the needle based on the embedding mode
        needle_semantic_embedding = None
        needle_textual_embedding = None

        if embedding_mode in {'semantic', 'both'}:
            needle_semantic_embedding = extract_semantic_embeddings(needle)
        
        if embedding_mode in {'textual', 'both'}:
            needle_textual_embedding = extract_textual_embeddings(
                needle.lower() if not use_stemming else stem_text(needle, lang)
            )
        
        # Compute the embeddings of the haystack (each block)
        haystack_semantic_embeddings = []
        haystack_textual_embeddings = []

        if embedding_mode in {'semantic', 'both'}:
            with ProcessPoolExecutor() as executor:
                haystack_semantic_embeddings = list(executor.map(extract_semantic_embeddings, blocks))
        
        if embedding_mode in {'textual', 'both'}:
            with ProcessPoolExecutor(max_workers=int(cpu_count()//1.5)) as executor:
                haystack_textual_embeddings = list(
                    executor.map(preprocess_and_extract_textual_embedding, blocks, [use_stemming]*len(blocks), [lang]*len(blocks))
                )
        
        # Compute similarities based on the embedding mode
        semantic_similarities = []
        textual_similarities = []

        if embedding_mode in {'semantic', 'both'}:
            semantic_similarities = [
                calculate_similarity(needle_semantic_embedding, block_embedding)
                for block_embedding in haystack_semantic_embeddings
            ]
        
        if embedding_mode in {'textual', 'both'}:
            textual_similarities = [
                calculate_similarity(needle_textual_embedding, block_embedding)
                for block_embedding in haystack_textual_embeddings
            ]

        # Calculate the overall similarity score
        if embedding_mode == 'semantic':
            sorted_blocks = sorted(zip(blocks, semantic_similarities), key=lambda x: x[1], reverse=True)
        elif embedding_mode == 'textual':
            sorted_blocks = sorted(zip(blocks, textual_similarities), key=lambda x: x[1], reverse=True)
        else:  # both
            sorted_blocks = sorted(
                zip(blocks, semantic_similarities, textual_similarities),
                key=lambda x: x[1] * semantic_embeddings_weight + x[2] * textual_embeddings_weight,
                reverse=True
            )
        
        # The most similar block is the one that contains the needle
        most_similar_block = sorted_blocks[0][0]

        # Find the index of the needle in all the blocks
        most_similar_block_index = blocks.index(most_similar_block)

        start_index = most_similar_block_index - 1 if most_similar_block_index > 0 else 0

        needle_region = blocks[start_index:most_similar_block_index + 2]

        return ''.join(needle_region).strip()
    except Exception:
        traceback.print_exc()
    
    return haystack
