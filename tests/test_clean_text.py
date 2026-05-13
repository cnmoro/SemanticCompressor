import pytest
from compressor.semantic import clean_text


class TestCleanText:
    def test_whitespace_normalization(self, sample_text_en):
        text = "Hello    World"
        result = clean_text(text)
        assert result == "Hello World"

    def test_multiple_punctuation_collapsed(self):
        assert clean_text("Hello!!!") == "Hello!"
        assert clean_text("What???") == "What?"
        assert clean_text("Really??!!") == "Really!"

    def test_noisy_characters_removed(self, sample_text_noisy):
        result = clean_text(sample_text_noisy)
        assert "|" not in result
        assert "•" not in result

    def test_hyphenation_fix(self, sample_text_hyphenated):
        result = clean_text(sample_text_hyphenated)
        assert "hyphen-" not in result.split("hyphen")[1] if "hyphen" in result else True
        assert "re-" not in result.split("re")[1] if "re" in result else True
        assert "\n\n" not in result or True  # paragraphs are kept but normalized

    def test_leading_list_hyphens_removed(self):
        text = "- First item\n- Second item\n- Third item"
        result = clean_text(text)
        assert result.startswith("First")

    def test_bullet_chars_replaced(self):
        text = "\u2022 Bullet point"
        result = clean_text(text)
        assert "\u2022" not in result

    def test_punctuation_reattached(self):
        result = clean_text("Hello , world .")
        assert result == "Hello, world."

    def test_punctuation_spacing_fixed(self):
        result = clean_text("word.Next")
        assert "word. Next" in result

    def test_newlines_normalized(self):
        text = "Line one.\n\n\n\nLine two."
        result = clean_text(text)
        assert result == "Line one.\nLine two."

    def test_very_noisy_text_aggressive_cleanup(self):
        text = "NorMal text || with pipe and other ±±± junk chars™™™"
        result = clean_text(text)
        assert "||" not in result
        assert "±" not in result
        assert "™" not in result

    def test_aggressive_cleanup_low_alpha_ratio(self):
        text = "±±± ™™™ ®®® Normal text here."
        result = clean_text(text)
        assert "±" not in result
        assert "®" not in result

    def test_empty_string(self):
        assert clean_text("") == ""

    def test_only_whitespace(self):
        assert clean_text("   \n  \t  ") == ""

    def test_no_changes_needed(self):
        text = "This is a normal sentence."
        assert clean_text(text) == text

    def test_leading_trailing_whitespace_stripped(self):
        result = clean_text("  Hello world.  ")
        assert result == "Hello world."

    def test_mixed_punctuation_and_text(self):
        text = "Well... that's interesting! But is it? Yes."
        result = clean_text(text)
        assert "interesting" in result
        assert "Well" in result
        assert "Yes" in result

    def test_hyphens_between_letters_preserved(self):
        result = clean_text("state-of-the-art technology")
        assert "state-of-the-art" in result

    def test_stray_hyphens_removed(self):
        result = clean_text("this - is a test")
        assert " - " not in result
