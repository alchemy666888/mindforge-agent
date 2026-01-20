"""Tests for style analyzer."""

import pytest

from mindforge_dankoe.analysis.style_analyzer import (
    StyleAnalyzer,
    StyleFingerprint,
    VocabularyMetrics,
    SentenceMetrics,
)
from mindforge_dankoe.crawler.article_processor import ProcessedArticle


class TestStyleAnalyzer:
    """Test suite for StyleAnalyzer."""

    @pytest.fixture
    def analyzer(self):
        """Create a StyleAnalyzer instance."""
        return StyleAnalyzer()

    @pytest.fixture
    def sample_article(self):
        """Create a sample processed article."""
        return ProcessedArticle(
            url="https://example.com/test",
            title="Test Article",
            content_clean="""
            This is the first paragraph. It has multiple sentences.
            The content discusses technology and innovation.

            Here is the second paragraph. Technology is changing everything.
            We need to adapt to these changes. The future is coming fast.

            Finally, the conclusion. Take action today. Start learning now.
            You need to prepare for what's coming next.
            """,
            paragraphs=[
                "This is the first paragraph. It has multiple sentences. The content discusses technology and innovation.",
                "Here is the second paragraph. Technology is changing everything. We need to adapt to these changes. The future is coming fast.",
                "Finally, the conclusion. Take action today. Start learning now. You need to prepare for what's coming next.",
            ],
            sentences=[
                "This is the first paragraph.",
                "It has multiple sentences.",
                "The content discusses technology and innovation.",
                "Here is the second paragraph.",
                "Technology is changing everything.",
                "We need to adapt to these changes.",
                "The future is coming fast.",
                "Finally, the conclusion.",
                "Take action today.",
                "Start learning now.",
                "You need to prepare for what's coming next.",
            ],
        )

    def test_analyze_article(self, analyzer, sample_article):
        """Test single article analysis."""
        metrics = analyzer.analyze_article(sample_article)

        assert "word_count" in metrics
        assert "unique_words" in metrics
        assert "avg_word_length" in metrics
        assert metrics["word_count"] > 0

    def test_create_fingerprint(self, analyzer, sample_article):
        """Test fingerprint creation."""
        fingerprint = analyzer.create_fingerprint([sample_article])

        assert isinstance(fingerprint, StyleFingerprint)
        assert fingerprint.articles_analyzed == 1
        assert fingerprint.vocabulary.total_unique_words > 0

    def test_vocabulary_metrics(self, analyzer, sample_article):
        """Test vocabulary metrics calculation."""
        fingerprint = analyzer.create_fingerprint([sample_article])

        assert isinstance(fingerprint.vocabulary, VocabularyMetrics)
        assert fingerprint.vocabulary.vocabulary_richness > 0
        assert fingerprint.vocabulary.vocabulary_richness <= 1.0

    def test_sentence_metrics(self, analyzer, sample_article):
        """Test sentence metrics calculation."""
        fingerprint = analyzer.create_fingerprint([sample_article])

        assert isinstance(fingerprint.sentences, SentenceMetrics)
        assert fingerprint.sentences.avg_sentence_length > 0
        assert fingerprint.sentences.min_sentence_length >= 0

    def test_fingerprint_serialization(self, analyzer, sample_article):
        """Test fingerprint serialization and deserialization."""
        fingerprint = analyzer.create_fingerprint([sample_article])

        # Convert to dict and back
        data = fingerprint.to_dict()
        restored = StyleFingerprint.from_dict(data)

        assert restored.articles_analyzed == fingerprint.articles_analyzed
        assert restored.vocabulary.total_unique_words == fingerprint.vocabulary.total_unique_words

    def test_get_style_summary(self, analyzer, sample_article):
        """Test style summary generation."""
        analyzer.create_fingerprint([sample_article])
        summary = analyzer.get_style_summary()

        assert isinstance(summary, str)
        assert "Style Analysis" in summary or "Vocabulary" in summary

    def test_compare_to_fingerprint(self, analyzer, sample_article):
        """Test text comparison to fingerprint."""
        analyzer.create_fingerprint([sample_article])

        new_text = "This is a test text. It should be compared to the fingerprint."
        scores = analyzer.compare_to_fingerprint(new_text)

        assert "overall_similarity" in scores
        assert 0 <= scores["overall_similarity"] <= 1

    def test_empty_articles(self, analyzer):
        """Test handling of empty article list."""
        fingerprint = analyzer.create_fingerprint([])

        assert fingerprint.articles_analyzed == 0
