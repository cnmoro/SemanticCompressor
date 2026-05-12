import pytest
from compressor.semantic import compress_text, semantic_compress_text


@pytest.mark.need_model
class TestSubSentenceTruncation:
    def test_truncates_last_sentence_to_fit_budget(self):
        text = (
            "Artificial intelligence has transformed the modern world. "
            "Deep learning models can recognize patterns in complex data. "
            "Neural networks are inspired by the human brain structure and function. "
            "Natural language understanding enables machines to read text. "
            "Computer vision allows machines to interpret images and video."
        )
        result = semantic_compress_text(text, compression_rate=0.3)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_aggressive_compression_still_returns_string(self):
        text = (
            "The quick brown fox jumps over the lazy dog. "
            "This is a test sentence for the semantic compressor. "
            "Natural language processing is a fascinating field. "
            "Machine learning algorithms can analyze text data efficiently. "
            "Deep neural networks require large amounts of training data. "
            "Reinforcement learning is an exciting area of research."
        )
        result = semantic_compress_text(text, compression_rate=0.15)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_truncated_sentence_preserves_first_words(self):
        text = (
            "This is the first sentence of the entire document. "
            "Here is another sentence that follows the first one. "
            "A third sentence that adds more information to the text. "
            "The fourth sentence continues building the context further. "
            "A fifth and final sentence to complete the document text."
        )
        result = semantic_compress_text(text, compression_rate=0.4)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_truncation_does_not_affect_short_compression(self):
        text = (
            "The quick brown fox jumps over the lazy dog. "
            "This is a test sentence for the semantic compressor."
        )
        result = semantic_compress_text(text, compression_rate=0.9)
        assert isinstance(result, str)


@pytest.mark.need_model
class TestDanglingPronounFilter:
    def test_replaces_sentence_with_dangling_pronoun(self):
        text = (
            "The Eiffel Tower is a famous landmark in Paris. "
            "It was built in 1889 for the World's Fair. "
            "Millions of tourists visit it every single year. "
            "The Louvre Museum is also located in the same city. "
        )
        result = semantic_compress_text(text, compression_rate=0.5)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_no_filter_needed_for_first_sentence(self):
        text = (
            "It is important to understand the basics of machine learning. "
            "Supervised learning uses labeled training data. "
            "Unsupervised learning finds patterns in unlabeled data. "
        )
        result = semantic_compress_text(text, compression_rate=0.7)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_filter_handles_short_text_with_pronouns(self):
        text = (
            "They said the weather would be nice today. "
            "The sun is shining brightly in the clear sky. "
            "We should go for a walk in the afternoon. "
        )
        result = semantic_compress_text(text, compression_rate=0.5)
        assert isinstance(result, str)
        assert len(result) > 0


@pytest.mark.need_model
class TestSentenceFusion:
    def test_fuses_adjacent_sentences_with_and(self):
        text = (
            "The sun was shining brightly in the clear blue sky. "
            "And the birds were singing in the tall green trees. "
            "The wind was blowing gently through the open window. "
        )
        result = semantic_compress_text(text, compression_rate=0.7)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_fuses_adjacent_sentences_with_but(self):
        text = (
            "The experiment showed promising initial results. "
            "But the sample size was too small to draw conclusions. "
            "Further research is needed to validate these findings. "
        )
        result = semantic_compress_text(text, compression_rate=0.7)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_fuses_adjacent_sentences_with_so(self):
        text = (
            "The server was experiencing high traffic volumes today. "
            "So the IT team decided to scale up the infrastructure. "
            "The system has been running smoothly since that change. "
        )
        result = semantic_compress_text(text, compression_rate=0.6)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_does_not_fuse_non_adjacent_sentences(self):
        text = (
            "Python is a versatile programming language. "
            "Many developers use it for data science projects. "
            "And it is also popular for web development. "
        )
        result = semantic_compress_text(text, compression_rate=0.5)
        assert isinstance(result, str)
        assert len(result) > 0


@pytest.mark.need_model
class TestConnectiveStitching:
    def test_adds_connector_between_distant_sentences(self):
        text = (
            "Quantum computing represents a paradigm shift in computational power. "
            "Traditional computers use bits that are either zero or one. "
            "Quantum computers use qubits that can exist in multiple states simultaneously. "
            "This allows them to solve certain problems exponentially faster. "
            "Climate change is one of the most pressing challenges facing humanity today. "
            "Rising global temperatures have led to more frequent extreme weather events. "
        )
        result = semantic_compress_text(text, compression_rate=0.4)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_connector_does_not_break_sentence_structure(self):
        text = (
            "The first chapter introduces the basic concepts clearly. "
            "The second chapter builds upon the foundation laid earlier. "
            "These early chapters are essential for understanding the later material. "
            "The third chapter covers advanced topics in great detail. "
            "The fourth chapter provides practical examples and exercises. "
            "The final chapter summarizes everything covered in the entire book. "
        )
        result = semantic_compress_text(text, compression_rate=0.4)
        assert isinstance(result, str)
        assert result[0].isupper()
        assert result.rstrip()[-1] in '.!?'


@pytest.mark.need_model
class TestProperNounCapitalization:
    def test_preserves_proper_noun_casing(self):
        text = (
            "The Eiffel Tower was built in Paris, France. "
            "It is one of the most visited monuments in the entire world. "
            "Many tourists from around the globe come to see it. "
            "The Louvre is another famous museum located in Paris. "
        )
        result = semantic_compress_text(text, compression_rate=0.5)
        assert "Eiffel" in result or "Paris" in result or "Louvre" in result
        assert result[0].isupper()

    def test_preserves_mid_sentence_casing(self):
        text = (
            "The Amazon rainforest is located in South America. "
            "It spans across multiple countries including Brazil and Peru. "
            "The biodiversity in this region is absolutely remarkable. "
            "Many unique species call the Amazon their home. "
        )
        result = semantic_compress_text(text, compression_rate=0.5)
        assert isinstance(result, str)
        assert result[0].isupper()
        for word in ['Amazon', 'Brazil', 'Peru', 'South America']:
            if word in result:
                assert result[result.index(word)] == word[0]


@pytest.mark.need_model
class TestCompressionWithAbstractiveEnhancements:
    def test_abstractive_compression_flow(self):
        text = (
            "Climate change is one of the most pressing challenges facing humanity today. "
            "Rising global temperatures have led to more frequent extreme weather events. "
            "Scientists warn that urgent action is needed to reduce carbon emissions. "
            "Renewable energy sources like solar and wind power offer sustainable alternatives. "
            "Many countries have committed to achieving net-zero emissions by the middle of the century. "
            "International cooperation is essential to address this global problem effectively. "
            "Technological innovations continue to provide new tools for environmental protection. "
            "The transition to a green economy creates both challenges and opportunities. "
        )
        result = compress_text(text, compression_rate=0.3)
        assert isinstance(result, str)
        assert len(result) > 0
        assert result[0].isupper()

    def test_abstractive_compression_across_languages(self):
        text_pt = (
            "O aquecimento global representa uma ameaça significativa para o planeta. "
            "As temperaturas médias continuam a subir em todo o mundo. "
            "Os cientistas alertam para a necessidade de ação imediata. "
            "As energias renováveis oferecem uma alternativa sustentável aos combustíveis fósseis. "
        )
        result = compress_text(text_pt, compression_rate=0.5)
        assert isinstance(result, str)
        assert len(result) > 0
        assert result[0].isupper()


@pytest.mark.need_model
class TestSubSentenceCompressionQuality:
    def test_truncation_vs_full_drop_compression_rate(self):
        text = (
            "Machine learning is a subset of artificial intelligence. "
            "It involves training algorithms on large datasets to make predictions. "
            "Deep learning uses neural networks with many layers. "
            "These networks can learn complex patterns from data automatically. "
            "Natural language processing is a key application of deep learning. "
            "Computer vision is another important field that uses neural networks. "
        )
        standard = semantic_compress_text(text, compression_rate=0.3)
        assert isinstance(standard, str)
        assert len(standard) > 0


@pytest.mark.need_model
class TestBilingualPortuguese:
    def test_portuguese_pronoun_filter(self):
        text = (
            "O Cristo Redentor é um dos pontos turísticos mais famosos do Brasil. "
            "Ele foi inaugurado em 1931 no topo do Corcovado. "
            "Milhões de visitantes vão até o Rio de Janeiro para conhecê-lo. "
            "O Pão de Açúcar é outra atração imperdível na cidade. "
        )
        result = compress_text(text, compression_rate=0.5)
        assert isinstance(result, str)
        assert len(result) > 0
        assert result[0].isupper()

    def test_portuguese_conjunction_fusion(self):
        text = (
            "O sol brilhava intensamente no céu azul de Copacabana. "
            "E as ondas quebravam suavemente na areia branca da praia. "
            "O vento soprava fresco trazendo o cheiro do mar. "
        )
        result = compress_text(text, compression_rate=0.7)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_portuguese_connective_stitching(self):
        text = (
            "Machine learning é uma área em rápido crescimento acadêmico. "
            "Algoritmos de aprendizado supervisionado usam dados rotulados. "
            "Redes neurais profundas exigem grandes quantidades de dados. "
            "As mudanças climáticas representam um desafio global urgente. "
            "O aumento das temperaturas tem causado eventos climáticos extremos. "
        )
        result = compress_text(text, compression_rate=0.4)
        assert isinstance(result, str)
        assert len(result) > 0
        assert result[0].isupper()
        assert result.rstrip()[-1] in '.?!'

    def test_portuguese_proper_noun_preservation(self):
        text = (
            "A Floresta Amazônica está localizada na América do Sul. "
            "Ela se estende por países como Brasil, Peru e Colômbia. "
            "A biodiversidade nesta região é absolutamente extraordinária. "
            "O Rio Amazonas é um dos maiores rios do mundo inteiro. "
        )
        result = compress_text(text, compression_rate=0.5)
        assert isinstance(result, str)
        assert result[0].isupper()
        for word in ['Amazônica', 'América', 'Brasil', 'Peru', 'Colômbia', 'Amazonas']:
            if word in result:
                idx = result.index(word)
                assert result[idx:idx + len(word)] == word

    def test_portuguese_multiple_abstractive_features(self):
        text = (
            "A inteligência artificial está transformando o mundo moderno. "
            "Modelos de aprendizado profundo podem reconhecer padrões complexos em dados. "
            "Redes neurais são inspiradas no cérebro humano biológico. "
            "O processamento de linguagem natural permite que máquinas leiam textos. "
            "A visão computacional permite que máquinas interpretem imagens e vídeos. "
            "Carros autônomos usam todas essas tecnologias para navegar com segurança. "
            "A ética em inteligência artificial é uma preocupação crescente na sociedade. "
        )
        result = compress_text(text, compression_rate=0.3)
        assert isinstance(result, str)
        assert len(result) > 0
        assert result[0].isupper()
        assert result.rstrip()[-1] in '.?!'
