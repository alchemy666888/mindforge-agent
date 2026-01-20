"""
Creativity Scorer - Quantify creative aspects of generated content.

Measures:
- Novelty: New insights not in source material
- Connection: Unusual but valid connections
- Practicality: Actionability of insights
- Surprise: Counter-intuitive but true insights
"""

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from mindforge_dankoe.config.settings import settings
from mindforge_dankoe.generation.research_engine import ResearchResult
from mindforge_dankoe.generation.style_generator import GeneratedArticle
from mindforge_dankoe.utils.logger import get_logger
from mindforge_dankoe.utils.text_processing import TextProcessor

logger = get_logger(__name__)


@dataclass
class CreativityDimension:
    """Score for a single creativity dimension."""

    score: float = 0.0
    evidence: List[str] = field(default_factory=list)
    explanation: str = ""


@dataclass
class CreativityReport:
    """Comprehensive creativity evaluation report."""

    novelty: CreativityDimension = field(default_factory=CreativityDimension)
    connection: CreativityDimension = field(default_factory=CreativityDimension)
    practicality: CreativityDimension = field(default_factory=CreativityDimension)
    surprise: CreativityDimension = field(default_factory=CreativityDimension)
    overall_score: float = 0.0
    percentile: float = 0.0  # Compared to baseline
    interpretation: str = ""
    recommendations: List[str] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "novelty": asdict(self.novelty),
            "connection": asdict(self.connection),
            "practicality": asdict(self.practicality),
            "surprise": asdict(self.surprise),
            "overall_score": self.overall_score,
            "percentile": self.percentile,
            "interpretation": self.interpretation,
            "recommendations": self.recommendations,
            "timestamp": self.timestamp,
        }


@dataclass
class BaselineMetrics:
    """Baseline creativity metrics from Dan Koe's articles."""

    avg_novelty: float = 0.35
    avg_connection: float = 0.45
    avg_practicality: float = 0.65
    avg_surprise: float = 0.25
    avg_overall: float = 0.42


class CreativityScorer:
    """
    Score creativity of generated content.

    Evaluates four dimensions:
    1. Novelty: Ideas not directly from sources
    2. Connection: Linking disparate concepts
    3. Practicality: Actionable insights
    4. Surprise: Counter-intuitive conclusions
    """

    # Action verbs indicating practical recommendations
    ACTION_VERBS = {
        "start", "stop", "try", "build", "create", "use", "find", "learn",
        "avoid", "embrace", "focus", "invest", "develop", "implement",
        "consider", "explore", "experiment", "adopt", "eliminate", "optimize",
    }

    # Indicators of surprising/counter-intuitive content
    SURPRISE_INDICATORS = [
        "contrary to popular belief",
        "surprisingly",
        "counterintuitively",
        "actually",
        "the truth is",
        "what most people miss",
        "the real",
        "but here's the thing",
        "the paradox",
        "opposite",
        "unconventional",
    ]

    # Connection indicators
    CONNECTION_INDICATORS = [
        "just like",
        "similar to",
        "connects to",
        "relates to",
        "the same way",
        "reminds me of",
        "pattern",
        "parallel",
        "intersection",
        "combines",
        "bridges",
    ]

    def __init__(self, baseline: Optional[BaselineMetrics] = None):
        """
        Initialize the creativity scorer.

        Args:
            baseline: Baseline metrics for comparison
        """
        self.baseline = baseline or BaselineMetrics()
        self.text_processor = TextProcessor()
        self._llm_client = None

    def _get_llm_response(self, prompt: str) -> str:
        """Get response from LLM."""
        try:
            import litellm

            model = settings.get_available_llm_model()
            response = litellm.completion(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=1000,
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"LLM error: {e}")
            return ""

    def score_article(
        self,
        article: GeneratedArticle,
        research: Optional[ResearchResult] = None,
    ) -> CreativityReport:
        """
        Score creativity of an article.

        Args:
            article: Generated article to evaluate
            research: Optional research sources for novelty comparison

        Returns:
            CreativityReport with detailed scores
        """
        logger.info(f"Scoring creativity for: {article.title}")

        report = CreativityReport()

        # Score each dimension
        report.novelty = self._score_novelty(article, research)
        report.connection = self._score_connection(article)
        report.practicality = self._score_practicality(article)
        report.surprise = self._score_surprise(article)

        # Calculate overall score (weighted average)
        report.overall_score = self._calculate_overall(report)

        # Calculate percentile against baseline
        report.percentile = self._calculate_percentile(report.overall_score)

        # Generate interpretation and recommendations
        report.interpretation = self._generate_interpretation(report)
        report.recommendations = self._generate_recommendations(report)

        return report

    def _score_novelty(
        self,
        article: GeneratedArticle,
        research: Optional[ResearchResult],
    ) -> CreativityDimension:
        """Score novelty - ideas not directly from sources."""
        dim = CreativityDimension()

        content = article.full_content.lower()
        sentences = self.text_processor.split_sentences(article.full_content)

        if research and research.sources:
            # Compare against source content
            source_text = " ".join(s.content.lower() for s in research.sources)
            source_words = set(source_text.split())

            # Find sentences with substantial new content
            novel_sentences = []
            for sentence in sentences:
                sentence_words = set(sentence.lower().split())
                # If less than 50% of words are from sources, consider it novel
                overlap = len(sentence_words & source_words) / max(len(sentence_words), 1)
                if overlap < 0.5 and len(sentence_words) > 5:
                    novel_sentences.append(sentence)

            dim.score = len(novel_sentences) / max(len(sentences), 1)
            dim.evidence = novel_sentences[:3]
            dim.explanation = f"Found {len(novel_sentences)} novel sentences out of {len(sentences)}"
        else:
            # Without sources, use heuristics
            # Check for original framings, predictions, recommendations
            novel_indicators = ["i believe", "my view", "predict", "my recommendation"]
            novel_count = sum(content.count(ind) for ind in novel_indicators)
            dim.score = min(1.0, novel_count / 5)
            dim.explanation = "Scored based on original framing indicators"

        return dim

    def _score_connection(self, article: GeneratedArticle) -> CreativityDimension:
        """Score connection - linking disparate concepts."""
        dim = CreativityDimension()

        content = article.full_content.lower()
        sentences = self.text_processor.split_sentences(article.full_content)

        # Find sentences with connection indicators
        connection_sentences = []
        for sentence in sentences:
            sentence_lower = sentence.lower()
            if any(ind in sentence_lower for ind in self.CONNECTION_INDICATORS):
                connection_sentences.append(sentence)

        # Also check for analogies (comparing unlike things)
        analogy_patterns = ["is like", "similar to", "just as", "the same way"]
        analogy_count = sum(content.count(p) for p in analogy_patterns)

        # Score based on connections found
        connection_count = len(connection_sentences) + analogy_count
        dim.score = min(1.0, connection_count / 5)
        dim.evidence = connection_sentences[:3]
        dim.explanation = f"Found {connection_count} connection/analogy instances"

        return dim

    def _score_practicality(self, article: GeneratedArticle) -> CreativityDimension:
        """Score practicality - actionable insights."""
        dim = CreativityDimension()

        content = article.full_content
        sentences = self.text_processor.split_sentences(content)

        # Find action-oriented sentences
        action_sentences = []
        for sentence in sentences:
            words = sentence.lower().split()
            if words and words[0] in self.ACTION_VERBS:
                action_sentences.append(sentence)
            elif any(f" {verb} " in sentence.lower() for verb in self.ACTION_VERBS):
                # Verb in sentence (not just at start)
                if "you should" in sentence.lower() or "you can" in sentence.lower():
                    action_sentences.append(sentence)

        # Check for specific vs vague recommendations
        specific_indicators = ["step 1", "first,", "specifically", "for example", "try this"]
        specificity_count = sum(content.lower().count(ind) for ind in specific_indicators)

        # Combined score
        action_ratio = len(action_sentences) / max(len(sentences), 1)
        specificity_bonus = min(0.3, specificity_count / 10)

        dim.score = min(1.0, action_ratio + specificity_bonus)
        dim.evidence = action_sentences[:5]
        dim.explanation = f"Found {len(action_sentences)} actionable recommendations"

        return dim

    def _score_surprise(self, article: GeneratedArticle) -> CreativityDimension:
        """Score surprise - counter-intuitive insights."""
        dim = CreativityDimension()

        content = article.full_content.lower()
        sentences = self.text_processor.split_sentences(article.full_content)

        # Find surprising/contrarian sentences
        surprise_sentences = []
        for sentence in sentences:
            sentence_lower = sentence.lower()
            if any(ind in sentence_lower for ind in self.SURPRISE_INDICATORS):
                surprise_sentences.append(sentence)

        # Check for "myth busting" or challenging assumptions
        challenge_indicators = ["myth", "wrong", "mistake", "misconception", "not true"]
        challenge_count = sum(content.count(ind) for ind in challenge_indicators)

        # Score
        surprise_count = len(surprise_sentences) + challenge_count
        dim.score = min(1.0, surprise_count / 4)
        dim.evidence = surprise_sentences[:3]
        dim.explanation = f"Found {surprise_count} surprising/contrarian elements"

        return dim

    def _calculate_overall(self, report: CreativityReport) -> float:
        """Calculate overall creativity score."""
        # Weighted average with practicality weighted highest
        weights = {
            "novelty": 0.25,
            "connection": 0.20,
            "practicality": 0.35,
            "surprise": 0.20,
        }

        return (
            report.novelty.score * weights["novelty"]
            + report.connection.score * weights["connection"]
            + report.practicality.score * weights["practicality"]
            + report.surprise.score * weights["surprise"]
        )

    def _calculate_percentile(self, overall_score: float) -> float:
        """Calculate percentile against baseline."""
        baseline_overall = self.baseline.avg_overall

        if overall_score >= baseline_overall * 1.3:
            return 0.90
        elif overall_score >= baseline_overall * 1.15:
            return 0.75
        elif overall_score >= baseline_overall:
            return 0.50
        elif overall_score >= baseline_overall * 0.85:
            return 0.35
        else:
            return 0.20

    def _generate_interpretation(self, report: CreativityReport) -> str:
        """Generate human-readable interpretation."""
        overall = report.overall_score

        if overall >= 0.7:
            quality = "Highly creative"
        elif overall >= 0.5:
            quality = "Good creativity"
        elif overall >= 0.35:
            quality = "Moderate creativity"
        else:
            quality = "Room for improvement"

        # Find strongest and weakest dimensions
        dimensions = [
            ("Novelty", report.novelty.score),
            ("Connection", report.connection.score),
            ("Practicality", report.practicality.score),
            ("Surprise", report.surprise.score),
        ]

        sorted_dims = sorted(dimensions, key=lambda x: x[1], reverse=True)
        strongest = sorted_dims[0][0]
        weakest = sorted_dims[-1][0]

        interpretation = (
            f"{quality} (score: {overall:.2f}, {report.percentile:.0%} percentile). "
            f"Strongest: {strongest}. Area to develop: {weakest}."
        )

        return interpretation

    def _generate_recommendations(self, report: CreativityReport) -> List[str]:
        """Generate actionable recommendations for improvement."""
        recommendations = []

        if report.novelty.score < 0.4:
            recommendations.append(
                "Add more original perspectives - try making bold predictions or sharing unique observations"
            )

        if report.connection.score < 0.4:
            recommendations.append(
                "Include more analogies - connect the topic to other fields or everyday experiences"
            )

        if report.practicality.score < 0.5:
            recommendations.append(
                "Make content more actionable - add specific steps, examples, or tools readers can use today"
            )

        if report.surprise.score < 0.3:
            recommendations.append(
                "Challenge assumptions - identify and counter common misconceptions about the topic"
            )

        if report.overall_score >= 0.6:
            recommendations.append(
                "Strong foundation - consider deepening one dimension for even more impact"
            )

        return recommendations

    def compare_to_baseline(
        self, report: CreativityReport
    ) -> Dict[str, Dict[str, float]]:
        """Compare report scores to Dan Koe baseline."""
        return {
            "novelty": {
                "score": report.novelty.score,
                "baseline": self.baseline.avg_novelty,
                "diff": report.novelty.score - self.baseline.avg_novelty,
            },
            "connection": {
                "score": report.connection.score,
                "baseline": self.baseline.avg_connection,
                "diff": report.connection.score - self.baseline.avg_connection,
            },
            "practicality": {
                "score": report.practicality.score,
                "baseline": self.baseline.avg_practicality,
                "diff": report.practicality.score - self.baseline.avg_practicality,
            },
            "surprise": {
                "score": report.surprise.score,
                "baseline": self.baseline.avg_surprise,
                "diff": report.surprise.score - self.baseline.avg_surprise,
            },
            "overall": {
                "score": report.overall_score,
                "baseline": self.baseline.avg_overall,
                "diff": report.overall_score - self.baseline.avg_overall,
            },
        }

    def save_report(
        self,
        report: CreativityReport,
        article_title: str,
        output_path: Optional[Path] = None,
    ) -> Path:
        """Save creativity report to disk."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = output_path or settings.get_output_path() / f"creativity_{timestamp}.json"

        data = {
            "article_title": article_title,
            "report": report.to_dict(),
            "baseline_comparison": self.compare_to_baseline(report),
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        logger.info(f"Saved creativity report to: {output_path}")
        return output_path
