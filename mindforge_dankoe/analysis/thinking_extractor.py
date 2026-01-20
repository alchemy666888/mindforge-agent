"""
Thinking Pattern Extractor - Identify and codify Dan Koe's thinking frameworks.

This module uses LLM analysis to extract:
- Problem-solving approaches
- Mental models used
- Argument structures
- Value judgments
- Prediction patterns
"""

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from mindforge_dankoe.config.settings import settings
from mindforge_dankoe.crawler.article_processor import ProcessedArticle
from mindforge_dankoe.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ThinkingStep:
    """A single step in a thinking framework."""

    step_number: int
    description: str
    purpose: str
    example: str = ""
    indicators: List[str] = field(default_factory=list)


@dataclass
class ThinkingPattern:
    """A complete thinking pattern/framework."""

    name: str
    description: str
    category: str  # e.g., "problem_analysis", "future_projection", "practical_guide"
    steps: List[ThinkingStep] = field(default_factory=list)
    use_cases: List[str] = field(default_factory=list)
    key_questions: List[str] = field(default_factory=list)
    mental_models: List[str] = field(default_factory=list)
    example_article: str = ""
    frequency: float = 0.0  # How often this pattern appears (0-1)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ThinkingPattern":
        """Create from dictionary."""
        steps_data = data.pop("steps", [])
        pattern = cls(**data)
        pattern.steps = [ThinkingStep(**s) for s in steps_data]
        return pattern


@dataclass
class ArgumentStructure:
    """Structure of an argument or article."""

    opening_type: str  # "hook", "question", "statement", "story"
    opening_content: str
    main_thesis: str
    supporting_points: List[str] = field(default_factory=list)
    evidence_types: List[str] = field(default_factory=list)  # "data", "anecdote", "analogy"
    counter_arguments: List[str] = field(default_factory=list)
    resolution: str = ""
    call_to_action: str = ""
    closing_type: str = ""  # "summary", "question", "call_to_action", "prediction"


@dataclass
class ExtractedThinking:
    """Complete thinking analysis for an article."""

    article_url: str
    article_title: str
    patterns_used: List[str] = field(default_factory=list)
    argument_structure: Optional[ArgumentStructure] = None
    mental_models: List[str] = field(default_factory=list)
    value_judgments: List[Dict[str, str]] = field(default_factory=list)
    predictions: List[str] = field(default_factory=list)
    key_insights: List[str] = field(default_factory=list)
    problem_framing: str = ""
    solution_approach: str = ""


class ThinkingExtractor:
    """
    Extract thinking patterns from Dan Koe's articles.

    Uses LLM analysis to identify:
    - How problems are broken down
    - Mental models employed
    - Argument construction patterns
    - Value systems and judgments
    - Prediction methodologies
    """

    # Known Dan Koe thinking patterns (to be refined through analysis)
    KNOWN_PATTERNS = {
        "technology_impact_analysis": ThinkingPattern(
            name="Technology Impact Analysis",
            description="Framework for analyzing how technology affects society and individuals",
            category="analysis",
            steps=[
                ThinkingStep(
                    step_number=1,
                    description="Identify the core capability of the technology",
                    purpose="Understand what the technology fundamentally enables",
                    indicators=["allows", "enables", "makes possible", "core function"],
                ),
                ThinkingStep(
                    step_number=2,
                    description="Map current applications",
                    purpose="Ground the analysis in real-world usage",
                    indicators=["currently used", "companies like", "examples include"],
                ),
                ThinkingStep(
                    step_number=3,
                    description="Project second and third order effects",
                    purpose="Think beyond immediate implications",
                    indicators=["this leads to", "consequently", "as a result", "down the line"],
                ),
                ThinkingStep(
                    step_number=4,
                    description="Identify winners and losers",
                    purpose="Understand who benefits and who is disrupted",
                    indicators=["benefit from", "disrupted by", "winners", "losers", "at risk"],
                ),
                ThinkingStep(
                    step_number=5,
                    description="Recommend action for readers",
                    purpose="Make insights actionable",
                    indicators=["you should", "consider", "prepare for", "take action"],
                ),
            ],
            use_cases=["AI developments", "new platforms", "industry changes"],
            mental_models=["systems thinking", "second-order effects", "disruption theory"],
        ),
        "problem_solution_framework": ThinkingPattern(
            name="Problem-Solution Framework",
            description="Classic problem identification and solution presentation",
            category="practical_guide",
            steps=[
                ThinkingStep(
                    step_number=1,
                    description="Present a relatable problem",
                    purpose="Connect with reader's experience",
                    indicators=["struggle with", "most people", "common problem", "you've probably"],
                ),
                ThinkingStep(
                    step_number=2,
                    description="Explain why common solutions fail",
                    purpose="Build credibility by acknowledging failed approaches",
                    indicators=["doesn't work", "the problem is", "what most people miss"],
                ),
                ThinkingStep(
                    step_number=3,
                    description="Introduce the insight or framework",
                    purpose="Present the core solution concept",
                    indicators=["the key is", "here's what works", "the solution"],
                ),
                ThinkingStep(
                    step_number=4,
                    description="Break down implementation steps",
                    purpose="Make the solution actionable",
                    indicators=["step 1", "first", "start by", "then"],
                ),
                ThinkingStep(
                    step_number=5,
                    description="Show expected results",
                    purpose="Motivate action with outcomes",
                    indicators=["you'll find", "the result", "expect to see"],
                ),
            ],
            use_cases=["productivity", "business strategies", "personal development"],
            mental_models=["root cause analysis", "implementation thinking"],
        ),
        "future_projection": ThinkingPattern(
            name="Future Projection",
            description="Framework for making predictions about trends and developments",
            category="prediction",
            steps=[
                ThinkingStep(
                    step_number=1,
                    description="Identify current trends and signals",
                    purpose="Ground predictions in observable reality",
                    indicators=["we're seeing", "trend", "signals", "emerging"],
                ),
                ThinkingStep(
                    step_number=2,
                    description="Extrapolate trajectory",
                    purpose="Project where trends are heading",
                    indicators=["in 5 years", "eventually", "this means", "trajectory"],
                ),
                ThinkingStep(
                    step_number=3,
                    description="Consider accelerating/decelerating factors",
                    purpose="Account for variables that affect speed of change",
                    indicators=["accelerate", "slow down", "depends on", "factors"],
                ),
                ThinkingStep(
                    step_number=4,
                    description="Paint the future scenario",
                    purpose="Make the prediction vivid and tangible",
                    indicators=["imagine", "picture", "will look like", "scenario"],
                ),
                ThinkingStep(
                    step_number=5,
                    description="Recommend preparation strategies",
                    purpose="Help readers position themselves",
                    indicators=["prepare", "position yourself", "get ahead", "adapt"],
                ),
            ],
            use_cases=["technology forecasting", "industry predictions", "career advice"],
            mental_models=["trend extrapolation", "scenario planning"],
        ),
        "contrarian_analysis": ThinkingPattern(
            name="Contrarian Analysis",
            description="Challenge conventional wisdom with alternative perspective",
            category="analysis",
            steps=[
                ThinkingStep(
                    step_number=1,
                    description="Present the conventional view",
                    purpose="Establish what most people believe",
                    indicators=["conventional wisdom", "most people think", "the narrative"],
                ),
                ThinkingStep(
                    step_number=2,
                    description="Identify flaws in conventional thinking",
                    purpose="Create doubt about accepted beliefs",
                    indicators=["but", "however", "the problem", "what they miss"],
                ),
                ThinkingStep(
                    step_number=3,
                    description="Present alternative perspective",
                    purpose="Offer a new way of seeing",
                    indicators=["actually", "the truth is", "here's what's really happening"],
                ),
                ThinkingStep(
                    step_number=4,
                    description="Provide supporting evidence",
                    purpose="Validate the contrarian view",
                    indicators=["evidence", "data shows", "for example", "case in point"],
                ),
                ThinkingStep(
                    step_number=5,
                    description="Explain implications of new perspective",
                    purpose="Show why this matters",
                    indicators=["this means", "implication", "therefore", "so what"],
                ),
            ],
            use_cases=["market analysis", "trend challenges", "rethinking strategies"],
            mental_models=["contrarian thinking", "first principles"],
        ),
    }

    def __init__(self, llm_client=None):
        """
        Initialize the extractor.

        Args:
            llm_client: Optional LLM client for analysis (uses LiteLLM if not provided)
        """
        self.llm_client = llm_client
        self.patterns: Dict[str, ThinkingPattern] = dict(self.KNOWN_PATTERNS)
        self.extracted_analyses: List[ExtractedThinking] = []

    def _get_llm_response(self, prompt: str, system_prompt: str = "") -> str:
        """Get response from LLM."""
        try:
            import litellm

            model = settings.get_available_llm_model()

            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            response = litellm.completion(
                model=model,
                messages=messages,
                temperature=0.3,  # Lower temperature for analysis
                max_tokens=2000,
            )

            return response.choices[0].message.content

        except Exception as e:
            logger.error(f"LLM error: {e}")
            return ""

    def analyze_article(self, article: ProcessedArticle) -> ExtractedThinking:
        """
        Analyze a single article for thinking patterns.

        Args:
            article: Processed article to analyze

        Returns:
            ExtractedThinking with identified patterns
        """
        logger.info(f"Analyzing thinking patterns: {article.title}")

        analysis = ExtractedThinking(
            article_url=article.url,
            article_title=article.title,
        )

        # Analyze argument structure
        analysis.argument_structure = self._analyze_argument_structure(article)

        # Identify patterns used
        analysis.patterns_used = self._identify_patterns(article)

        # Extract mental models
        analysis.mental_models = self._extract_mental_models(article)

        # Extract value judgments
        analysis.value_judgments = self._extract_value_judgments(article)

        # Extract predictions
        analysis.predictions = self._extract_predictions(article)

        # Extract key insights
        analysis.key_insights = self._extract_key_insights(article)

        # Analyze problem framing
        analysis.problem_framing = self._analyze_problem_framing(article)

        # Analyze solution approach
        analysis.solution_approach = self._analyze_solution_approach(article)

        self.extracted_analyses.append(analysis)
        return analysis

    def _analyze_argument_structure(self, article: ProcessedArticle) -> ArgumentStructure:
        """Analyze the article's argument structure."""
        structure = ArgumentStructure(
            opening_type="unknown",
            opening_content=article.intro_paragraph[:200] if article.intro_paragraph else "",
            main_thesis="",
        )

        # Determine opening type
        intro = article.intro_paragraph.lower() if article.intro_paragraph else ""
        if "?" in intro[:100]:
            structure.opening_type = "question"
        elif any(word in intro[:50] for word in ["i ", "my ", "when i"]):
            structure.opening_type = "story"
        elif any(word in intro[:100] for word in ["imagine", "picture", "think about"]):
            structure.opening_type = "hook"
        else:
            structure.opening_type = "statement"

        # Determine closing type
        conclusion = article.conclusion_paragraph.lower() if article.conclusion_paragraph else ""
        if "?" in conclusion[-100:]:
            structure.closing_type = "question"
        elif any(word in conclusion for word in ["should", "try", "start", "do"]):
            structure.closing_type = "call_to_action"
        elif any(word in conclusion for word in ["will", "future", "expect"]):
            structure.closing_type = "prediction"
        else:
            structure.closing_type = "summary"

        # Extract evidence types used
        content = article.content_clean.lower()
        if any(word in content for word in ["study", "research", "data", "%", "percent"]):
            structure.evidence_types.append("data")
        if any(word in content for word in ["for example", "case", "instance"]):
            structure.evidence_types.append("anecdote")
        if any(word in content for word in ["like", "similar to", "as if", "imagine"]):
            structure.evidence_types.append("analogy")

        return structure

    def _identify_patterns(self, article: ProcessedArticle) -> List[str]:
        """Identify which thinking patterns are used in the article."""
        patterns_found = []
        content = article.content_clean.lower()

        for pattern_name, pattern in self.patterns.items():
            # Check for pattern indicators in steps
            step_matches = 0
            for step in pattern.steps:
                if any(indicator.lower() in content for indicator in step.indicators):
                    step_matches += 1

            # If majority of steps are present, pattern is likely used
            if step_matches >= len(pattern.steps) * 0.5:
                patterns_found.append(pattern_name)

        return patterns_found

    def _extract_mental_models(self, article: ProcessedArticle) -> List[str]:
        """Extract mental models referenced or applied."""
        mental_models = []
        content = article.content_clean.lower()

        # Common mental models to look for
        model_indicators = {
            "first_principles": ["first principles", "from scratch", "fundamental", "basic truth"],
            "systems_thinking": ["system", "interconnected", "feedback loop", "emergent"],
            "second_order_effects": ["second order", "downstream", "ripple effect", "consequences"],
            "opportunity_cost": ["opportunity cost", "trade-off", "instead of", "at the expense"],
            "leverage": ["leverage", "multiply", "scale", "compound"],
            "network_effects": ["network effect", "viral", "exponential growth"],
            "inversion": ["invert", "avoid", "what not to do", "opposite"],
            "probabilistic_thinking": ["probability", "likely", "odds", "chance"],
        }

        for model, indicators in model_indicators.items():
            if any(indicator in content for indicator in indicators):
                mental_models.append(model)

        return mental_models

    def _extract_value_judgments(self, article: ProcessedArticle) -> List[Dict[str, str]]:
        """Extract value judgments and opinions."""
        judgments = []
        sentences = article.sentences

        value_indicators = [
            ("positive", ["good", "great", "excellent", "important", "valuable", "essential"]),
            ("negative", ["bad", "wrong", "mistake", "problem", "dangerous", "harmful"]),
            ("should", ["should", "must", "need to", "have to", "ought to"]),
            ("preference", ["better", "best", "prefer", "ideal", "optimal"]),
        ]

        for sentence in sentences[:50]:  # Check first 50 sentences
            sentence_lower = sentence.lower()
            for judgment_type, indicators in value_indicators:
                if any(indicator in sentence_lower for indicator in indicators):
                    judgments.append({"type": judgment_type, "statement": sentence[:200]})
                    break

        return judgments[:20]  # Return top 20 judgments

    def _extract_predictions(self, article: ProcessedArticle) -> List[str]:
        """Extract predictions and forecasts."""
        predictions = []
        sentences = article.sentences

        prediction_indicators = ["will", "going to", "expect", "predict", "in the future", "eventually", "soon"]

        for sentence in sentences:
            sentence_lower = sentence.lower()
            if any(indicator in sentence_lower for indicator in prediction_indicators):
                # Filter out questions
                if "?" not in sentence:
                    predictions.append(sentence)

        return predictions[:10]

    def _extract_key_insights(self, article: ProcessedArticle) -> List[str]:
        """Extract key insights and takeaways."""
        insights = []
        sentences = article.sentences

        insight_indicators = [
            "the key",
            "the truth",
            "the reality",
            "what matters",
            "the important",
            "the insight",
            "this means",
            "the takeaway",
            "remember",
            "don't forget",
        ]

        for sentence in sentences:
            sentence_lower = sentence.lower()
            if any(indicator in sentence_lower for indicator in insight_indicators):
                insights.append(sentence)

        return insights[:10]

    def _analyze_problem_framing(self, article: ProcessedArticle) -> str:
        """Analyze how problems are framed in the article."""
        # Look at the first few paragraphs for problem statements
        intro_content = " ".join(article.paragraphs[:3]) if article.paragraphs else ""

        problem_patterns = [
            "struggle with",
            "problem is",
            "challenge",
            "difficult",
            "most people",
            "common mistake",
        ]

        for pattern in problem_patterns:
            if pattern in intro_content.lower():
                # Find the sentence containing the problem
                for sentence in article.sentences[:15]:
                    if pattern in sentence.lower():
                        return sentence

        return ""

    def _analyze_solution_approach(self, article: ProcessedArticle) -> str:
        """Analyze the solution approach used."""
        content = article.content_clean.lower()

        approaches = {
            "framework": "Presents a structured framework or system",
            "step_by_step": "Provides step-by-step instructions",
            "mindset_shift": "Focuses on changing perspective or mindset",
            "tool_recommendation": "Recommends specific tools or methods",
            "principle_based": "Teaches underlying principles to apply",
        }

        if any(word in content for word in ["framework", "system", "model"]):
            return approaches["framework"]
        elif any(word in content for word in ["step 1", "first,", "then,", "finally"]):
            return approaches["step_by_step"]
        elif any(word in content for word in ["mindset", "perspective", "think differently"]):
            return approaches["mindset_shift"]
        elif any(word in content for word in ["tool", "app", "software", "use this"]):
            return approaches["tool_recommendation"]
        elif any(word in content for word in ["principle", "fundamental", "core idea"]):
            return approaches["principle_based"]

        return "mixed_approach"

    def synthesize_patterns(self, analyses: List[ExtractedThinking]) -> Dict[str, Any]:
        """
        Synthesize patterns across multiple articles.

        Args:
            analyses: List of extracted thinking analyses

        Returns:
            Synthesized pattern summary
        """
        if not analyses:
            return {}

        # Count pattern usage
        pattern_counts = {}
        for analysis in analyses:
            for pattern in analysis.patterns_used:
                pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1

        # Aggregate mental models
        all_models = []
        for analysis in analyses:
            all_models.extend(analysis.mental_models)
        model_counts = {m: all_models.count(m) for m in set(all_models)}

        # Analyze argument structures
        opening_types = {}
        closing_types = {}
        for analysis in analyses:
            if analysis.argument_structure:
                ot = analysis.argument_structure.opening_type
                ct = analysis.argument_structure.closing_type
                opening_types[ot] = opening_types.get(ot, 0) + 1
                closing_types[ct] = closing_types.get(ct, 0) + 1

        return {
            "total_articles_analyzed": len(analyses),
            "pattern_frequency": pattern_counts,
            "mental_model_frequency": model_counts,
            "opening_type_distribution": opening_types,
            "closing_type_distribution": closing_types,
            "common_patterns": [
                p for p, c in sorted(pattern_counts.items(), key=lambda x: -x[1])[:5]
            ],
            "common_mental_models": [
                m for m, c in sorted(model_counts.items(), key=lambda x: -x[1])[:5]
            ],
        }

    def save_patterns(self, output_path: Optional[Path] = None) -> Path:
        """Save extracted patterns to disk."""
        output_path = output_path or settings.get_processed_data_path() / "thinking_patterns.json"

        data = {
            "patterns": {name: pattern.to_dict() for name, pattern in self.patterns.items()},
            "extracted_analyses": [asdict(a) for a in self.extracted_analyses],
            "synthesis": self.synthesize_patterns(self.extracted_analyses),
            "timestamp": datetime.now().isoformat(),
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        logger.info(f"Saved thinking patterns to: {output_path}")
        return output_path

    def load_patterns(self, input_path: Optional[Path] = None) -> None:
        """Load patterns from disk."""
        input_path = input_path or settings.get_processed_data_path() / "thinking_patterns.json"

        if not input_path.exists():
            logger.warning(f"Pattern file not found: {input_path}")
            return

        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.patterns = {
            name: ThinkingPattern.from_dict(p) for name, p in data.get("patterns", {}).items()
        }

        logger.info(f"Loaded {len(self.patterns)} thinking patterns")
