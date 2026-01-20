"""
Similarity Metrics - Measure how closely generated content matches Dan Koe's style.

Implements multi-dimensional similarity scoring:
- Thinking pattern similarity
- Stylistic similarity
- Thematic alignment
- Argument structure
- Tone consistency
"""

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from mindforge_dankoe.analysis.style_analyzer import StyleAnalyzer, StyleFingerprint
from mindforge_dankoe.analysis.thinking_extractor import ThinkingExtractor
from mindforge_dankoe.config.settings import settings
from mindforge_dankoe.crawler.article_processor import ProcessedArticle
from mindforge_dankoe.generation.style_generator import GeneratedArticle
from mindforge_dankoe.utils.logger import get_logger
from mindforge_dankoe.utils.text_processing import TextProcessor

logger = get_logger(__name__)


@dataclass
class SimilarityScores:
    """Multi-dimensional similarity scores."""

    thinking_pattern_similarity: float = 0.0
    stylistic_similarity: float = 0.0
    thematic_alignment: float = 0.0
    argument_structure_similarity: float = 0.0
    tone_consistency: float = 0.0
    vocabulary_similarity: float = 0.0
    sentence_structure_similarity: float = 0.0
    overall_score: float = 0.0
    interpretation: str = ""
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


class SimilarityMetrics:
    """
    Calculate similarity between generated content and Dan Koe's style.

    Provides multi-dimensional analysis:
    - Thinking pattern matching
    - Style fingerprint comparison
    - Thematic consistency
    - Structural analysis
    """

    # Weights for overall score calculation
    DIMENSION_WEIGHTS = {
        "thinking_pattern_similarity": 0.20,
        "stylistic_similarity": 0.25,
        "thematic_alignment": 0.15,
        "argument_structure_similarity": 0.15,
        "tone_consistency": 0.15,
        "vocabulary_similarity": 0.05,
        "sentence_structure_similarity": 0.05,
    }

    def __init__(
        self,
        style_fingerprint: Optional[StyleFingerprint] = None,
        reference_articles: Optional[List[ProcessedArticle]] = None,
    ):
        """
        Initialize similarity metrics calculator.

        Args:
            style_fingerprint: Dan Koe's style fingerprint
            reference_articles: Reference articles for comparison
        """
        self.fingerprint = style_fingerprint
        self.reference_articles = reference_articles or []
        self.text_processor = TextProcessor()
        self.style_analyzer = StyleAnalyzer()
        self.thinking_extractor = ThinkingExtractor()

        # Cache reference data
        self._reference_tokens: List[str] = []
        self._reference_themes: List[str] = []

        if reference_articles:
            self._build_reference_cache()

    def _build_reference_cache(self):
        """Build cache of reference article data."""
        for article in self.reference_articles:
            tokens = self.text_processor.tokenize(article.content_clean)
            self._reference_tokens.extend(tokens)

            # Extract themes from key phrases
            self._reference_themes.extend(article.key_phrases[:10])

    def calculate_similarity(
        self, generated: GeneratedArticle
    ) -> SimilarityScores:
        """
        Calculate comprehensive similarity scores.

        Args:
            generated: Generated article to evaluate

        Returns:
            SimilarityScores with all dimensions
        """
        logger.info(f"Calculating similarity for: {generated.title}")

        scores = SimilarityScores()

        # Calculate each dimension
        scores.thinking_pattern_similarity = self._calc_thinking_pattern_similarity(
            generated
        )
        scores.stylistic_similarity = self._calc_stylistic_similarity(generated)
        scores.thematic_alignment = self._calc_thematic_alignment(generated)
        scores.argument_structure_similarity = self._calc_argument_structure_similarity(
            generated
        )
        scores.tone_consistency = self._calc_tone_consistency(generated)
        scores.vocabulary_similarity = self._calc_vocabulary_similarity(generated)
        scores.sentence_structure_similarity = self._calc_sentence_structure_similarity(
            generated
        )

        # Calculate weighted overall score
        scores.overall_score = self._calc_overall_score(scores)

        # Generate interpretation
        scores.interpretation = self._generate_interpretation(scores)

        # Add details
        scores.details = self._collect_details(generated)

        return scores

    def _calc_thinking_pattern_similarity(
        self, generated: GeneratedArticle
    ) -> float:
        """Calculate thinking pattern similarity."""
        # Create a pseudo-processed article for analysis
        content = generated.full_content

        # Check for thinking pattern indicators
        pattern_scores = []

        for pattern_name, pattern in self.thinking_extractor.patterns.items():
            step_matches = 0
            for step in pattern.steps:
                if any(indicator.lower() in content.lower() for indicator in step.indicators):
                    step_matches += 1

            if pattern.steps:
                pattern_score = step_matches / len(pattern.steps)
                pattern_scores.append(pattern_score)

        return np.mean(pattern_scores) if pattern_scores else 0.5

    def _calc_stylistic_similarity(self, generated: GeneratedArticle) -> float:
        """Calculate stylistic similarity using fingerprint."""
        if not self.fingerprint:
            return 0.5

        content = generated.full_content
        tokens = self.text_processor.tokenize(content)
        sentences = self.text_processor.split_sentences(content)

        if not tokens or not sentences:
            return 0.5

        # Compare word length
        avg_word_len = np.mean([len(w) for w in tokens])
        word_len_diff = abs(avg_word_len - self.fingerprint.vocabulary.avg_word_length)
        word_len_score = max(0, 1 - word_len_diff / 5)

        # Compare sentence length
        sent_lengths = [len(s.split()) for s in sentences]
        avg_sent_len = np.mean(sent_lengths)
        sent_len_diff = abs(avg_sent_len - self.fingerprint.sentences.avg_sentence_length)
        sent_len_score = max(0, 1 - sent_len_diff / 15)

        # Compare vocabulary richness
        vocab_richness = len(set(tokens)) / len(tokens)
        richness_diff = abs(vocab_richness - self.fingerprint.vocabulary.vocabulary_richness)
        richness_score = max(0, 1 - richness_diff / 0.3)

        return (word_len_score + sent_len_score + richness_score) / 3

    def _calc_thematic_alignment(self, generated: GeneratedArticle) -> float:
        """Calculate thematic alignment with reference content."""
        if not self._reference_themes:
            return 0.5

        content_lower = generated.full_content.lower()

        # Check how many reference themes appear in generated content
        theme_matches = sum(
            1 for theme in self._reference_themes if theme.lower() in content_lower
        )

        if self._reference_themes:
            return min(1.0, theme_matches / (len(self._reference_themes) * 0.3))

        return 0.5

    def _calc_argument_structure_similarity(
        self, generated: GeneratedArticle
    ) -> float:
        """Calculate argument structure similarity."""
        content = generated.full_content
        paragraphs = self.text_processor.split_paragraphs(content)
        sentences = self.text_processor.split_sentences(content)

        if not sentences:
            return 0.5

        scores = []

        # Check for problem-solution structure
        problem_indicators = ["problem", "challenge", "struggle", "difficult"]
        solution_indicators = ["solution", "answer", "here's how", "the key"]

        has_problem = any(ind in content.lower() for ind in problem_indicators)
        has_solution = any(ind in content.lower() for ind in solution_indicators)

        if has_problem and has_solution:
            scores.append(0.8)
        elif has_problem or has_solution:
            scores.append(0.5)
        else:
            scores.append(0.3)

        # Check for evidence/example usage
        evidence_indicators = ["for example", "research shows", "data", "study", "evidence"]
        has_evidence = any(ind in content.lower() for ind in evidence_indicators)
        scores.append(0.8 if has_evidence else 0.4)

        # Check for clear conclusion
        conclusion_indicators = ["in conclusion", "to summarize", "the key takeaway", "remember"]
        has_conclusion = any(ind in content.lower() for ind in conclusion_indicators)
        scores.append(0.8 if has_conclusion else 0.5)

        return np.mean(scores)

    def _calc_tone_consistency(self, generated: GeneratedArticle) -> float:
        """Calculate tone consistency."""
        content = generated.full_content.lower()
        words = content.split()

        if not words:
            return 0.5

        # Check directness (use of "you", imperative verbs)
        you_count = content.count(" you ")
        directness = min(1.0, you_count / (len(words) / 50))

        # Check confidence (absence of hedging)
        hedge_words = ["maybe", "perhaps", "might", "could be", "possibly"]
        hedge_count = sum(content.count(w) for w in hedge_words)
        confidence = max(0, 1 - hedge_count / (len(words) / 200))

        # Check reader engagement
        question_count = content.count("?")
        engagement = min(1.0, question_count / (len(words) / 300))

        # Dan Koe targets: high directness, high confidence, moderate engagement
        directness_score = 1 - abs(directness - 0.8)
        confidence_score = 1 - abs(confidence - 0.85)
        engagement_score = 1 - abs(engagement - 0.5)

        return (directness_score + confidence_score + engagement_score) / 3

    def _calc_vocabulary_similarity(self, generated: GeneratedArticle) -> float:
        """Calculate vocabulary similarity."""
        if not self._reference_tokens:
            return 0.5

        gen_tokens = set(self.text_processor.tokenize(generated.full_content.lower()))
        ref_tokens = set(t.lower() for t in self._reference_tokens)

        if not gen_tokens:
            return 0.5

        # Jaccard similarity
        intersection = len(gen_tokens & ref_tokens)
        union = len(gen_tokens | ref_tokens)

        return intersection / union if union > 0 else 0.5

    def _calc_sentence_structure_similarity(
        self, generated: GeneratedArticle
    ) -> float:
        """Calculate sentence structure similarity."""
        if not self.fingerprint:
            return 0.5

        sentences = self.text_processor.split_sentences(generated.full_content)

        if not sentences:
            return 0.5

        # Calculate sentence length distribution
        lengths = [len(s.split()) for s in sentences]
        short_ratio = sum(1 for l in lengths if l < 10) / len(lengths)
        long_ratio = sum(1 for l in lengths if l > 25) / len(lengths)

        # Compare with fingerprint
        short_diff = abs(short_ratio - self.fingerprint.sentences.short_sentence_ratio)
        long_diff = abs(long_ratio - self.fingerprint.sentences.long_sentence_ratio)

        short_score = max(0, 1 - short_diff / 0.3)
        long_score = max(0, 1 - long_diff / 0.3)

        return (short_score + long_score) / 2

    def _calc_overall_score(self, scores: SimilarityScores) -> float:
        """Calculate weighted overall similarity score."""
        weighted_sum = 0
        total_weight = 0

        for dimension, weight in self.DIMENSION_WEIGHTS.items():
            value = getattr(scores, dimension, 0)
            weighted_sum += value * weight
            total_weight += weight

        return weighted_sum / total_weight if total_weight > 0 else 0.5

    def _generate_interpretation(self, scores: SimilarityScores) -> str:
        """Generate human-readable interpretation of scores."""
        overall = scores.overall_score

        if overall >= 0.8:
            quality = "Excellent"
            description = "Very close to Dan Koe's authentic style"
        elif overall >= 0.7:
            quality = "Good"
            description = "Captures most of Dan Koe's key style elements"
        elif overall >= 0.6:
            quality = "Moderate"
            description = "Shows Dan Koe influence but has room for improvement"
        elif overall >= 0.5:
            quality = "Fair"
            description = "Some similarities but diverges in key areas"
        else:
            quality = "Needs improvement"
            description = "Significant style differences from reference"

        # Identify strengths and weaknesses
        dimensions = [
            ("Thinking patterns", scores.thinking_pattern_similarity),
            ("Style", scores.stylistic_similarity),
            ("Themes", scores.thematic_alignment),
            ("Argument structure", scores.argument_structure_similarity),
            ("Tone", scores.tone_consistency),
        ]

        sorted_dims = sorted(dimensions, key=lambda x: x[1], reverse=True)
        strengths = [d[0] for d in sorted_dims[:2] if d[1] >= 0.6]
        weaknesses = [d[0] for d in sorted_dims[-2:] if d[1] < 0.6]

        interpretation = f"{quality} match ({overall:.0%}): {description}."

        if strengths:
            interpretation += f" Strengths: {', '.join(strengths)}."
        if weaknesses:
            interpretation += f" Areas to improve: {', '.join(weaknesses)}."

        return interpretation

    def _collect_details(self, generated: GeneratedArticle) -> Dict[str, Any]:
        """Collect detailed analysis information."""
        content = generated.full_content
        stats = self.text_processor.calculate_stats(content)

        return {
            "word_count": stats.word_count,
            "sentence_count": stats.sentence_count,
            "paragraph_count": stats.paragraph_count,
            "avg_sentence_length": stats.avg_sentence_length,
            "vocabulary_richness": stats.vocabulary_richness,
            "question_count": stats.question_count,
            "language": stats.language,
        }

    def compare_articles(
        self,
        article1: GeneratedArticle,
        article2: GeneratedArticle,
    ) -> Dict[str, float]:
        """
        Compare two generated articles against the reference.

        Args:
            article1: First article
            article2: Second article

        Returns:
            Dictionary with comparison results
        """
        scores1 = self.calculate_similarity(article1)
        scores2 = self.calculate_similarity(article2)

        return {
            "article1_overall": scores1.overall_score,
            "article2_overall": scores2.overall_score,
            "winner": "article1" if scores1.overall_score > scores2.overall_score else "article2",
            "difference": abs(scores1.overall_score - scores2.overall_score),
        }

    def save_evaluation(
        self,
        scores: SimilarityScores,
        generated: GeneratedArticle,
        output_path: Optional[Path] = None,
    ) -> Path:
        """Save evaluation results to disk."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = output_path or settings.get_output_path() / f"evaluation_{timestamp}.json"

        data = {
            "article_title": generated.title,
            "scores": scores.to_dict(),
            "timestamp": datetime.now().isoformat(),
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        logger.info(f"Saved evaluation to: {output_path}")
        return output_path
