import pytest
from compressor.semantic import detect_language, stem_text


class TestDetectLanguage:
    def test_detect_english(self, sample_text_en):
        lang = detect_language(sample_text_en)
        assert lang == 'en'

    def test_detect_portuguese(self, sample_text_pt):
        lang = detect_language(sample_text_pt)
        assert lang == 'pt'

    def test_detect_english_short_text(self):
        lang = detect_language("Hello world")
        assert lang == 'en'

    def test_detect_portuguese_short_text(self):
        lang = detect_language("Olá mundo")
        assert lang == 'pt'

    def test_mixed_text_favors_majority(self):
        lang = detect_language(
            "This is mostly English. "
            "But also tem um pouco de português. "
            "Still mostly English here."
        )
        assert lang == 'en'

    def test_empty_text_fallback(self):
        lang = detect_language("")
        assert lang == 'en'


class TestStemText:
    def test_stem_english_basic(self):
        result = stem_text("running runner ran", 'en')
        assert result == "run runner ran"

    def test_stem_english_regular_verbs(self):
        result = stem_text("jumps jumped jumping", 'en')
        assert result == "jump jump jump"

    def test_stem_english_plurals(self):
        result = stem_text("cats dogs horses", 'en')
        assert result == "cat dog hors"

    def test_stem_english_adverbs(self):
        result = stem_text("quickly nicely", 'en')
        assert "quick" in result

    def test_stem_portuguese_basic(self, sample_text_pt):
        result = stem_text("correndo correu correr", 'pt')
        assert result == "corr corr corr"

    def test_stem_portuguese_plural(self):
        result = stem_text("gatos casas flores", 'pt')
        assert "gat" in result
        assert "cas" in result

    def test_stem_portuguese_verbs(self):
        result = stem_text("falando falou falar", 'pt')
        assert result == "fal fal fal"

    def test_stem_empty_string(self):
        assert stem_text("", 'en') == ""
        assert stem_text("", 'pt') == ""

    def test_stem_single_word(self):
        assert stem_text("walking", 'en') == "walk"

    def test_stem_portuguese_articles(self):
        result = stem_text("o a os as", 'pt')
        assert isinstance(result, str)
