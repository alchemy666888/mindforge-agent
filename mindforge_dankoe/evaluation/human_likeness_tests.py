"""
Human Likeness Evaluator - Test if content feels authentically human.

Implements:
- AI detection resistance scoring
- Voice consistency analysis
- Factual coherence checks
- Natural flow evaluation
"""

import re
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from mindforge_dankoe.config.settings import settings
from mindforge_dankoe.generation.style_generator import GeneratedArticle
from mindforge_dankoe.utils.logger import get_logger
from mindforge_dankoe.utils.text_processing import TextProcessor

logger = get_logger(__name__)


@dataclass
class HumanLikenessReport:
    """Report on human-likeness of generated content."""

    overall_score: float = 0.0
    ai_detection_risk: float = 0.0  # 0=likely human, 1=likely AI
    voice_consistency: float = 0.0
    natural_flow: float = 0.0
    specificity: float = 0.0
    personality_markers: float = 0.0
    interpretation: str = ""
    red_flags: List[str] = field(default_factory=list)
    positive_indicators: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


class HumanLikenessEvaluator:
    """
    Evaluate how human-like generated content appears.

    Tests for:
    - AI-typical patterns (that make text detectable)
    - Voice consistency throughout
    - Natural language flow
    - Specificity vs generic statements
    - Personality and authenticity markers
    """

    # Common AI-generated text patterns to avoid
    AI_PATTERNS = [
        r"\bdelve\b",
        r"\bin today's (fast-paced|digital|modern) (world|age|era)\b",
        r"\bultimately\b",
        r"\bfurthermore\b",
        r"\bmoreover\b",
        r"\bnevertheless\b",
        r"\bit's worth noting\b",
        r"\bit's important to note\b",
        r"\bin conclusion\b",
        r"\bto summarize\b",
        r"\blet's dive in\b",
        r"\bwithout further ado\b",
        r"\bparamount\b",
        r"\bseamlessly\b",
        r"\bleverage\b.*\bsynergies?\b",
        r"\bholistic\b",
        r"\bunlock\b.*\bpotential\b",
        r"\bgame-?changer\b",
    ]

    # Patterns that indicate natural, human writing
    HUMAN_PATTERNS = [
        r"\bi think\b",
        r"\bi believe\b",
        r"\bhonestly\b",
        r"\bfrankly\b",
        r"\bto be (honest|fair)\b",
        r"\bin my experience\b",
        r"\bhere's the thing\b",
        r"\blook,\b",
        r"\bthe reality is\b",
        r"(?:^|\. )(but|and|so|yet) ",  # Starting sentences with conjunctions
        r"\bi've (seen|found|noticed|learned)\b",
        r"\bwe all know\b",
        r"\blet me (tell you|explain|share)\b",
    ]

    # Overly formal/corporate phrases that reduce authenticity
    CORPORATE_PATTERNS = [
        r"\boptimize\b.*\bworkflow\b",
        r"\bstakeholders\b",
        r"\bsynergy\b",
        r"\bactionable insights\b",
        r"\bbest practices\b",
        r"\bkey takeaways\b",
        r"\brobust solution\b",
        r"\bscalable\b",
        r"\bpivot\b",
    ]

    def __init__(self):
        """Initialize the evaluator."""
        self.text_processor = TextProcessor()

    def evaluate(self, article: GeneratedArticle) -> HumanLikenessReport:
        """
        Evaluate human-likeness of an article.

        Args:
            article: Generated article to evaluate

        Returns:
            HumanLikenessReport with detailed analysis
        """
        logger.info(f"Evaluating human-likeness: {article.title}")

        report = HumanLikenessReport()

        content = article.full_content

        # Evaluate each dimension
        report.ai_detection_risk = self._calc_ai_detection_risk(content)
        report.voice_consistency = self._calc_voice_consistency(content)
        report.natural_flow = self._calc_natural_flow(content)
        report.specificity = self._calc_specificity(content)
        report.personality_markers = self._calc_personality_markers(content)

        # Calculate overall score (inverted AI risk + other factors)
        report.overall_score = self._calc_overall_score(report)

        # Collect red flags and positive indicators
        report.red_flags = self._find_red_flags(content)
        report.positive_indicators = self._find_positive_indicators(content)

        # Generate interpretation and recommendations
        report.interpretation = self._generate_interpretation(report)
        report.recommendations = self._generate_recommendations(report)

        return report

    def _calc_ai_detection_risk(self, content: str) -> float:
        """Calculate risk of being detected as AI-generated."""
        content_lower = content.lower()
        word_count = len(content.split())

        if word_count == 0:
            return 0.5

        # Count AI-typical patterns
        ai_pattern_count = 0
        for pattern in self.AI_PATTERNS:
            matches = re.findall(pattern, content_lower, re.IGNORECASE)
            ai_pattern_count += len(matches)

        # Count corporate buzzwords
        corporate_count = 0
        for pattern in self.CORPORATE_PATTERNS:
            matches = re.findall(pattern, content_lower, re.IGNORECASE)
            corporate_count += len(matches)

        # Normalize by word count
        ai_density = ai_pattern_count / (word_count / 100)
        corporate_density = corporate_count / (word_count / 100)

        # Calculate risk (more patterns = higher risk)
        risk = min(1.0, (ai_density * 0.15) + (corporate_density * 0.1))

        return risk

    def _calc_voice_consistency(self, content: str) -> float:
        """Calculate voice consistency throughout the text."""
        paragraphs = self.text_processor.split_paragraphs(content)

        if len(paragraphs) < 2:
            return 0.7

        # Check consistency of various style markers across paragraphs
        consistency_scores = []

        # First person usage consistency
        first_person_usage = []
        for para in paragraphs:
            para_lower = para.lower()
            count = para_lower.count(" i ") + para_lower.count("i'm") + para_lower.count("i've")
            first_person_usage.append(count > 0)

        # Should be consistent - either use first person throughout or not
        fp_consistency = 1 - np.std([1 if fp else 0 for fp in first_person_usage])
        consistency_scores.append(fp_consistency)

        # Reader address consistency ("you")
        you_usage = []
        for para in paragraphs:
            count = para.lower().count(" you ")
            words = len(para.split())
            you_usage.append(count / max(words, 1))

        you_consistency = 1 - min(1.0, np.std(you_usage) * 10)
        consistency_scores.append(you_consistency)

        # Sentence length consistency
        para_sent_lengths = []
        for para in paragraphs:
            sentences = self.text_processor.split_sentences(para)
            if sentences:
                avg_len = np.mean([len(s.split()) for s in sentences])
                para_sent_lengths.append(avg_len)

        if len(para_sent_lengths) > 1:
            sent_consistency = 1 - min(1.0, np.std(para_sent_lengths) / 10)
            consistency_scores.append(sent_consistency)

        return np.mean(consistency_scores) if consistency_scores else 0.7

    def _calc_natural_flow(self, content: str) -> float:
        """Calculate how naturally the text flows."""
        sentences = self.text_processor.split_sentences(content)

        if len(sentences) < 3:
            return 0.7

        flow_scores = []

        # Check for varied sentence starters
        starters = []
        for sentence in sentences:
            words = sentence.split()
            if words:
                starters.append(words[0].lower())

        # Good flow = varied starters
        unique_starters = len(set(starters))
        starter_variety = unique_starters / len(starters)
        flow_scores.append(min(1.0, starter_variety * 1.5))

        # Check for varied sentence lengths
        lengths = [len(s.split()) for s in sentences]
        length_variety = np.std(lengths) / max(np.mean(lengths), 1)
        # Some variety is good, too much is chaotic
        length_score = 1 - abs(length_variety - 0.4)
        flow_scores.append(max(0, length_score))

        # Check for natural transitions
        transition_words = ["but", "however", "so", "and", "yet", "still", "though"]
        transition_count = 0
        for word in transition_words:
            transition_count += content.lower().count(f" {word} ")

        # Good transition usage (not too few, not too many)
        expected_transitions = len(sentences) / 5
        transition_ratio = transition_count / max(expected_transitions, 1)
        transition_score = 1 - abs(transition_ratio - 1) * 0.5
        flow_scores.append(max(0, min(1, transition_score)))

        return np.mean(flow_scores)

    def _calc_specificity(self, content: str) -> float:
        """Calculate specificity vs generic statements."""
        sentences = self.text_processor.split_sentences(content)

        if not sentences:
            return 0.5

        specific_indicators = 0
        generic_indicators = 0

        for sentence in sentences:
            sentence_lower = sentence.lower()

            # Specific indicators
            if re.search(r"\d+", sentence):  # Contains numbers
                specific_indicators += 1
            if re.search(r"(for example|such as|like \w+)", sentence_lower):
                specific_indicators += 1
            if re.search(r'"[^"]{5,}"', sentence):  # Contains quotes
                specific_indicators += 1
            if re.search(r"\b(specifically|exactly|precisely)\b", sentence_lower):
                specific_indicators += 1

            # Generic indicators
            if re.search(r"\b(things|stuff|aspects|elements|factors)\b", sentence_lower):
                generic_indicators += 1
            if re.search(r"\b(various|numerous|many|several)\b", sentence_lower):
                generic_indicators += 1
            if re.search(r"\b(etc|and so on|and more)\b", sentence_lower):
                generic_indicators += 1

        total = specific_indicators + generic_indicators + 1
        score = specific_indicators / total

        return min(1.0, score * 1.5)

    def _calc_personality_markers(self, content: str) -> float:
        """Calculate presence of personality/authenticity markers."""
        content_lower = content.lower()

        personality_score = 0

        # Human patterns add to personality
        for pattern in self.HUMAN_PATTERNS:
            matches = re.findall(pattern, content_lower, re.IGNORECASE | re.MULTILINE)
            personality_score += len(matches) * 0.1

        # Check for conversational elements
        conversational = [
            "?",  # Questions
            "!",  # Exclamations
            "...",  # Ellipses
            "—",  # Em dashes
            "-",  # Casual dashes
        ]
        for marker in conversational:
            personality_score += content.count(marker) * 0.02

        # Check for personal opinions/stance
        opinion_markers = ["i think", "i believe", "in my view", "from my perspective"]
        for marker in opinion_markers:
            if marker in content_lower:
                personality_score += 0.15

        # Check for humor/wit indicators
        humor_markers = ["joke", "funny", "laugh", "ironic", "amusing"]
        for marker in humor_markers:
            if marker in content_lower:
                personality_score += 0.1

        return min(1.0, personality_score)

    def _calc_overall_score(self, report: HumanLikenessReport) -> float:
        """Calculate overall human-likeness score."""
        # Invert AI detection risk (lower risk = higher score)
        ai_score = 1 - report.ai_detection_risk

        return (
            ai_score * 0.30
            + report.voice_consistency * 0.20
            + report.natural_flow * 0.20
            + report.specificity * 0.15
            + report.personality_markers * 0.15
        )

    def _find_red_flags(self, content: str) -> List[str]:
        """Find specific red flags in the content."""
        flags = []
        content_lower = content.lower()

        for pattern in self.AI_PATTERNS[:10]:  # Check top patterns
            matches = re.findall(pattern, content_lower, re.IGNORECASE)
            if matches:
                flags.append(f"AI-typical phrase: '{matches[0]}'")

        for pattern in self.CORPORATE_PATTERNS[:5]:
            matches = re.findall(pattern, content_lower, re.IGNORECASE)
            if matches:
                flags.append(f"Corporate jargon: '{matches[0]}'")

        return flags[:5]  # Return top 5

    def _find_positive_indicators(self, content: str) -> List[str]:
        """Find positive human-like indicators."""
        indicators = []
        content_lower = content.lower()

        for pattern in self.HUMAN_PATTERNS:
            matches = re.findall(pattern, content_lower, re.IGNORECASE | re.MULTILINE)
            if matches:
                indicators.append(f"Natural expression: '{matches[0]}'")

        # Check for specific examples
        example_matches = re.findall(r"for example[^.]{10,50}", content_lower)
        if example_matches:
            indicators.append("Contains specific examples")

        # Check for personal anecdotes
        if "my experience" in content_lower or "i remember" in content_lower:
            indicators.append("Includes personal experience")

        return indicators[:5]

    def _generate_interpretation(self, report: HumanLikenessReport) -> str:
        """Generate interpretation of results."""
        overall = report.overall_score

        if overall >= 0.8:
            quality = "Highly authentic"
            detail = "Content reads naturally with strong personality"
        elif overall >= 0.65:
            quality = "Good authenticity"
            detail = "Generally natural with minor areas for improvement"
        elif overall >= 0.5:
            quality = "Moderate authenticity"
            detail = "Some AI-like patterns detected"
        else:
            quality = "Needs improvement"
            detail = "Contains noticeable AI-like patterns"

        risk_level = "low" if report.ai_detection_risk < 0.3 else (
            "moderate" if report.ai_detection_risk < 0.6 else "high"
        )

        return f"{quality} (score: {overall:.2f}). {detail}. AI detection risk: {risk_level}."

    def _generate_recommendations(self, report: HumanLikenessReport) -> List[str]:
        """Generate recommendations for improvement."""
        recommendations = []

        if report.ai_detection_risk > 0.4:
            recommendations.append(
                "Reduce AI-typical phrases like 'delve', 'furthermore', 'in today's world'"
            )

        if report.voice_consistency < 0.6:
            recommendations.append(
                "Maintain consistent voice throughout - pick a style and stick with it"
            )

        if report.natural_flow < 0.6:
            recommendations.append(
                "Vary sentence structure more - mix short punchy sentences with longer ones"
            )

        if report.specificity < 0.5:
            recommendations.append(
                "Add more specific examples, numbers, and concrete details"
            )

        if report.personality_markers < 0.4:
            recommendations.append(
                "Add more personal voice - use 'I think', share opinions, be conversational"
            )

        if report.red_flags:
            recommendations.append(
                f"Consider removing or rephrasing: {report.red_flags[0]}"
            )

        return recommendations

    def save_report(
        self,
        report: HumanLikenessReport,
        article_title: str,
        output_path: Optional[Path] = None,
    ) -> Path:
        """Save human-likeness report to disk."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = output_path or settings.get_output_path() / f"humanlikeness_{timestamp}.json"

        data = {
            "article_title": article_title,
            "report": report.to_dict(),
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        logger.info(f"Saved human-likeness report to: {output_path}")
        return output_path
