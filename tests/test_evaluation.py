"""Tests for evaluation modules."""

import pytest

from mindforge_dankoe.evaluation.creativity_scorer import CreativityScorer, CreativityReport
from mindforge_dankoe.evaluation.human_likeness_tests import HumanLikenessEvaluator, HumanLikenessReport
from mindforge_dankoe.evaluation.similarity_metrics import SimilarityMetrics, SimilarityScores
from mindforge_dankoe.generation.style_generator import GeneratedArticle


class TestCreativityScorer:
    """Test suite for CreativityScorer."""

    @pytest.fixture
    def scorer(self):
        """Create a CreativityScorer instance."""
        return CreativityScorer()

    @pytest.fixture
    def sample_article(self):
        """Create a sample generated article."""
        return GeneratedArticle(
            title="The Future of AI Agents",
            full_content="""
            Here's the thing about AI agents: they're changing everything.

            You should start learning about AI today. The reality is that most people
            are missing the opportunity. Just like the internet revolution, this will
            create winners and losers.

            Contrary to popular belief, AI won't replace all jobs. Actually, it will
            create new ones. The key is to adapt.

            Step 1: Learn the basics of machine learning.
            Step 2: Experiment with AI tools daily.
            Step 3: Build projects that solve real problems.

            For example, I've seen companies transform their productivity by 300%
            using AI assistants. The data is clear: early adopters win.

            Remember: the future belongs to those who prepare for it today.
            """,
            word_count=150,
        )

    def test_score_article(self, scorer, sample_article):
        """Test article creativity scoring."""
        report = scorer.score_article(sample_article)

        assert isinstance(report, CreativityReport)
        assert 0 <= report.overall_score <= 1
        assert 0 <= report.novelty.score <= 1
        assert 0 <= report.practicality.score <= 1

    def test_novelty_scoring(self, scorer, sample_article):
        """Test novelty dimension scoring."""
        report = scorer.score_article(sample_article)

        assert report.novelty.explanation != ""

    def test_practicality_scoring(self, scorer, sample_article):
        """Test practicality dimension scoring."""
        report = scorer.score_article(sample_article)

        # Article has clear action items, should score higher
        assert report.practicality.score > 0.3
        assert len(report.practicality.evidence) > 0

    def test_surprise_scoring(self, scorer, sample_article):
        """Test surprise dimension scoring."""
        report = scorer.score_article(sample_article)

        # Article has contrarian elements
        assert report.surprise.score > 0

    def test_interpretation_generation(self, scorer, sample_article):
        """Test interpretation generation."""
        report = scorer.score_article(sample_article)

        assert report.interpretation != ""
        assert "creativity" in report.interpretation.lower() or "score" in report.interpretation.lower()

    def test_recommendations(self, scorer, sample_article):
        """Test recommendation generation."""
        report = scorer.score_article(sample_article)

        assert isinstance(report.recommendations, list)

    def test_baseline_comparison(self, scorer, sample_article):
        """Test comparison to baseline."""
        report = scorer.score_article(sample_article)
        comparison = scorer.compare_to_baseline(report)

        assert "overall" in comparison
        assert "baseline" in comparison["overall"]
        assert "diff" in comparison["overall"]


class TestHumanLikenessEvaluator:
    """Test suite for HumanLikenessEvaluator."""

    @pytest.fixture
    def evaluator(self):
        """Create a HumanLikenessEvaluator instance."""
        return HumanLikenessEvaluator()

    @pytest.fixture
    def natural_article(self):
        """Create a natural-sounding article."""
        return GeneratedArticle(
            title="My Thoughts on Tech",
            full_content="""
            Look, I've been thinking about this a lot lately.

            Here's the thing - most people get this wrong. They think AI is just
            about automation. But honestly, it's so much more than that.

            I believe we're at a turning point. In my experience, the companies
            that adapt fastest win. It's not about being perfect, it's about
            starting.

            So what should you do? Try this: pick one AI tool and use it daily
            for a week. Just one. See what happens.

            The results might surprise you.
            """,
            word_count=100,
        )

    @pytest.fixture
    def ai_like_article(self):
        """Create an AI-sounding article."""
        return GeneratedArticle(
            title="The Comprehensive Guide",
            full_content="""
            In today's fast-paced digital world, it is paramount to understand
            the implications of artificial intelligence.

            Furthermore, one must delve into the various aspects of this
            transformative technology. Moreover, the synergies between AI and
            business processes cannot be overstated.

            It's worth noting that organizations should leverage AI to unlock
            their full potential. Additionally, stakeholders must be aligned
            with the holistic vision.

            In conclusion, the journey towards AI adoption is multifaceted.
            Ultimately, success depends on seamless integration.
            """,
            word_count=100,
        )

    def test_evaluate_natural(self, evaluator, natural_article):
        """Test evaluation of natural-sounding article."""
        report = evaluator.evaluate(natural_article)

        assert isinstance(report, HumanLikenessReport)
        assert report.overall_score > 0.4  # Should be reasonably human-like
        assert report.ai_detection_risk < 0.7

    def test_evaluate_ai_like(self, evaluator, ai_like_article):
        """Test evaluation of AI-sounding article."""
        report = evaluator.evaluate(ai_like_article)

        # Should detect AI patterns
        assert report.ai_detection_risk > 0.2
        assert len(report.red_flags) > 0

    def test_voice_consistency(self, evaluator, natural_article):
        """Test voice consistency calculation."""
        report = evaluator.evaluate(natural_article)

        assert 0 <= report.voice_consistency <= 1

    def test_natural_flow(self, evaluator, natural_article):
        """Test natural flow calculation."""
        report = evaluator.evaluate(natural_article)

        assert 0 <= report.natural_flow <= 1

    def test_red_flags_detection(self, evaluator, ai_like_article):
        """Test red flag detection."""
        report = evaluator.evaluate(ai_like_article)

        assert isinstance(report.red_flags, list)
        # Should find some AI-typical patterns
        assert len(report.red_flags) > 0

    def test_positive_indicators(self, evaluator, natural_article):
        """Test positive indicator detection."""
        report = evaluator.evaluate(natural_article)

        assert isinstance(report.positive_indicators, list)

    def test_recommendations(self, evaluator, ai_like_article):
        """Test recommendation generation."""
        report = evaluator.evaluate(ai_like_article)

        assert len(report.recommendations) > 0


class TestSimilarityMetrics:
    """Test suite for SimilarityMetrics."""

    @pytest.fixture
    def metrics(self):
        """Create a SimilarityMetrics instance."""
        return SimilarityMetrics()

    @pytest.fixture
    def sample_article(self):
        """Create a sample generated article."""
        return GeneratedArticle(
            title="Tech Analysis Article",
            full_content="""
            The technology landscape is evolving rapidly. You need to understand
            these changes to stay competitive.

            Here's the problem: most companies are not adapting fast enough.
            The solution is to embrace AI tools today.

            For example, consider how automation has transformed manufacturing.
            The same transformation is happening in knowledge work.

            Here's what you should do: Start small. Pick one process to automate.
            Learn from the results. Scale what works.

            The future belongs to those who prepare for it today.
            """,
            word_count=100,
        )

    def test_calculate_similarity(self, metrics, sample_article):
        """Test similarity calculation."""
        scores = metrics.calculate_similarity(sample_article)

        assert isinstance(scores, SimilarityScores)
        assert 0 <= scores.overall_score <= 1

    def test_thinking_pattern_similarity(self, metrics, sample_article):
        """Test thinking pattern similarity."""
        scores = metrics.calculate_similarity(sample_article)

        # Article follows problem-solution pattern
        assert scores.thinking_pattern_similarity >= 0

    def test_tone_consistency(self, metrics, sample_article):
        """Test tone consistency calculation."""
        scores = metrics.calculate_similarity(sample_article)

        assert 0 <= scores.tone_consistency <= 1

    def test_interpretation(self, metrics, sample_article):
        """Test interpretation generation."""
        scores = metrics.calculate_similarity(sample_article)

        assert scores.interpretation != ""
        assert "match" in scores.interpretation.lower() or "similarity" in scores.interpretation.lower()

    def test_details_collection(self, metrics, sample_article):
        """Test details collection."""
        scores = metrics.calculate_similarity(sample_article)

        assert "word_count" in scores.details
        assert "sentence_count" in scores.details
