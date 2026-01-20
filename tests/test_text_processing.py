"""Tests for text processing utilities."""

import pytest

from mindforge_dankoe.utils.text_processing import TextProcessor, TextStats


class TestTextProcessor:
    """Test suite for TextProcessor."""

    @pytest.fixture
    def processor(self):
        """Create a TextProcessor instance."""
        return TextProcessor()

    def test_detect_language_english(self, processor):
        """Test English language detection."""
        text = "This is a sample English text for testing purposes."
        assert processor.detect_language(text) == "en"

    def test_detect_language_chinese(self, processor):
        """Test Chinese language detection."""
        text = "这是一个中文测试文本，用于语言检测。"
        assert processor.detect_language(text) == "zh"

    def test_detect_language_mixed(self, processor):
        """Test mixed language detection."""
        text = "This text contains 一些中文 mixed in."
        result = processor.detect_language(text)
        assert result in ["mixed", "en"]

    def test_split_sentences_english(self, processor):
        """Test English sentence splitting."""
        text = "First sentence. Second sentence! Third sentence?"
        sentences = processor.split_sentences(text, "en")
        assert len(sentences) >= 2

    def test_split_paragraphs(self, processor):
        """Test paragraph splitting."""
        text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
        paragraphs = processor.split_paragraphs(text)
        assert len(paragraphs) == 3

    def test_tokenize_english(self, processor):
        """Test English tokenization."""
        text = "Hello world, this is a test."
        tokens = processor.tokenize(text, "en")
        assert "hello" in tokens
        assert "world" in tokens

    def test_calculate_stats(self, processor):
        """Test text statistics calculation."""
        text = """First sentence here. Second sentence there.

        Another paragraph with more content. And another sentence."""

        stats = processor.calculate_stats(text)

        assert isinstance(stats, TextStats)
        assert stats.word_count > 0
        assert stats.sentence_count > 0
        assert stats.paragraph_count > 0
        assert stats.avg_sentence_length > 0

    def test_clean_text(self, processor):
        """Test text cleaning."""
        text = "  Too   many    spaces   here.  \n\n\n\n  Extra lines.  "
        cleaned = processor.clean_text(text)

        # Should normalize whitespace
        assert "   " not in cleaned
        assert "\n\n\n" not in cleaned

    def test_find_transition_words(self, processor):
        """Test transition word detection."""
        text = "First point. However, there's another side. Therefore, we conclude."
        transitions = processor.find_transition_words(text)

        assert "however" in transitions or "Therefore" in transitions.keys()

    def test_extract_key_phrases(self, processor):
        """Test key phrase extraction."""
        text = """
        Artificial intelligence is transforming the technology landscape.
        Machine learning algorithms are becoming more sophisticated.
        AI and machine learning are related fields.
        """
        phrases = processor.extract_key_phrases(text, top_n=5)

        assert len(phrases) <= 5
        assert all(isinstance(p, tuple) and len(p) == 2 for p in phrases)

    def test_empty_text_handling(self, processor):
        """Test handling of empty text."""
        assert processor.detect_language("") == "en"
        assert processor.split_sentences("") == []
        assert processor.split_paragraphs("") == []
        assert processor.tokenize("") == []

        stats = processor.calculate_stats("")
        assert stats.word_count == 0
