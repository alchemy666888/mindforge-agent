"""
Cognitive Processor - Apply Dan Koe's thinking patterns to generate insights.

Implements:
- Thinking template application
- Creativity quantification
- Bilingual thought generation
"""

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from mindforge_dankoe.analysis.thinking_extractor import ThinkingExtractor, ThinkingPattern
from mindforge_dankoe.config.settings import settings
from mindforge_dankoe.generation.research_engine import ResearchResult
from mindforge_dankoe.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ProcessedInsight:
    """A processed insight ready for article generation."""

    content: str
    insight_type: str  # "finding", "prediction", "recommendation", "analysis"
    confidence: float = 0.7
    supporting_evidence: List[str] = field(default_factory=list)
    novelty_score: float = 0.5  # How novel compared to source material
    source_refs: List[str] = field(default_factory=list)


@dataclass
class CognitiveOutput:
    """Output from cognitive processing."""

    topic: str
    template_used: str
    insights: List[ProcessedInsight] = field(default_factory=list)
    main_thesis: str = ""
    supporting_arguments: List[str] = field(default_factory=list)
    counter_arguments: List[str] = field(default_factory=list)
    predictions: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    questions_raised: List[str] = field(default_factory=list)
    creativity_scores: Dict[str, float] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)
        data["insights"] = [asdict(i) for i in self.insights]
        return data


@dataclass
class CreativityScore:
    """Detailed creativity scoring."""

    novelty_score: float = 0.0  # % of insights not in source material
    connection_score: float = 0.0  # Unusual but valid connections made
    practicality_score: float = 0.0  # Actionability of insights
    surprise_score: float = 0.0  # Counter-intuitive but true insights
    overall_score: float = 0.0
    interpretation: str = ""


class CognitiveProcessor:
    """
    Process research through Dan Koe's thinking frameworks.

    Applies:
    - Thinking templates to structure analysis
    - Creativity quantification
    - Insight generation and validation
    """

    def __init__(
        self,
        thinking_extractor: Optional[ThinkingExtractor] = None,
        llm_client=None,
    ):
        """
        Initialize the cognitive processor.

        Args:
            thinking_extractor: Optional ThinkingExtractor with loaded patterns
            llm_client: Optional LLM client
        """
        self.thinking_extractor = thinking_extractor or ThinkingExtractor()
        self.llm_client = llm_client
        self._creativity_baseline: Dict[str, float] = {
            "novelty_score": 0.3,
            "connection_score": 0.4,
            "practicality_score": 0.6,
            "surprise_score": 0.2,
        }

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
                temperature=0.7,
                max_tokens=2500,
            )

            return response.choices[0].message.content

        except Exception as e:
            logger.error(f"LLM error: {e}")
            return ""

    def select_thinking_template(
        self, topic: str, research: ResearchResult
    ) -> ThinkingPattern:
        """
        Select the most appropriate thinking template for a topic.

        Args:
            topic: Topic being analyzed
            research: Research results

        Returns:
            Most appropriate ThinkingPattern
        """
        topic_lower = topic.lower()

        # Simple keyword-based selection
        if any(word in topic_lower for word in ["future", "predict", "trend", "next"]):
            return self.thinking_extractor.patterns.get(
                "future_projection",
                list(self.thinking_extractor.patterns.values())[0],
            )

        if any(word in topic_lower for word in ["problem", "solve", "how to", "guide"]):
            return self.thinking_extractor.patterns.get(
                "problem_solution_framework",
                list(self.thinking_extractor.patterns.values())[0],
            )

        if any(word in topic_lower for word in ["wrong", "myth", "actually", "truth"]):
            return self.thinking_extractor.patterns.get(
                "contrarian_analysis",
                list(self.thinking_extractor.patterns.values())[0],
            )

        # Default to technology impact analysis for AI topics
        return self.thinking_extractor.patterns.get(
            "technology_impact_analysis",
            list(self.thinking_extractor.patterns.values())[0],
        )

    def apply_thinking_template(
        self,
        topic: str,
        research: ResearchResult,
        template: Optional[ThinkingPattern] = None,
    ) -> CognitiveOutput:
        """
        Apply a thinking template to generate structured insights.

        Args:
            topic: Topic being analyzed
            research: Research results
            template: Optional specific template to use

        Returns:
            CognitiveOutput with structured insights
        """
        logger.info(f"Applying thinking template to: {topic}")

        # Select template if not provided
        if template is None:
            template = self.select_thinking_template(topic, research)

        output = CognitiveOutput(
            topic=topic,
            template_used=template.name,
        )

        # Prepare research context
        research_context = self._prepare_research_context(research)

        # Apply each step of the template
        for step in template.steps:
            insight = self._process_thinking_step(
                step, topic, research_context, template
            )
            if insight:
                output.insights.append(insight)

        # Generate main thesis
        output.main_thesis = self._generate_thesis(topic, output.insights, template)

        # Generate supporting and counter arguments
        output.supporting_arguments = self._generate_arguments(
            output.main_thesis, research, "supporting"
        )
        output.counter_arguments = self._generate_arguments(
            output.main_thesis, research, "counter"
        )

        # Generate predictions
        output.predictions = self._generate_predictions(topic, research)

        # Generate recommendations
        output.recommendations = self._generate_recommendations(
            topic, research, output.insights
        )

        # Generate questions
        output.questions_raised = research.unanswered_questions

        # Calculate creativity scores
        output.creativity_scores = self._calculate_creativity_scores(
            output, research
        )

        return output

    def _prepare_research_context(self, research: ResearchResult) -> str:
        """Prepare research context for LLM prompts."""
        context_parts = []

        # Key findings
        if research.key_findings:
            context_parts.append("Key Findings:")
            for finding in research.key_findings[:5]:
                context_parts.append(f"- {finding}")

        # Source summaries
        if research.sources:
            context_parts.append("\nSource Excerpts:")
            for source in research.sources[:3]:
                excerpt = source.content[:500] if source.content else ""
                context_parts.append(f"From {source.title}:\n{excerpt}\n")

        # Conflicting perspectives
        if research.conflicting_perspectives:
            context_parts.append("\nDiffering Viewpoints:")
            for persp in research.conflicting_perspectives[:3]:
                context_parts.append(f"- {persp.get('perspective', '')}")

        return "\n".join(context_parts)

    def _process_thinking_step(
        self,
        step,
        topic: str,
        research_context: str,
        template: ThinkingPattern,
    ) -> Optional[ProcessedInsight]:
        """Process a single thinking step."""
        prompt = f"""You are applying Dan Koe's "{template.name}" thinking framework.

Current step: {step.step_number}. {step.description}
Purpose: {step.purpose}

Topic: {topic}

Research context:
{research_context}

Based on this step of the framework, generate a key insight about {topic}.
The insight should:
1. Directly address the purpose of this step
2. Be grounded in the research
3. Provide value to the reader

Respond with:
INSIGHT: [Your insight]
EVIDENCE: [Supporting evidence from research]
CONFIDENCE: [High/Medium/Low]"""

        response = self._get_llm_response(prompt)

        if not response:
            return None

        # Parse response
        insight_content = ""
        evidence = []
        confidence = 0.7

        for line in response.split("\n"):
            if line.startswith("INSIGHT:"):
                insight_content = line.replace("INSIGHT:", "").strip()
            elif line.startswith("EVIDENCE:"):
                evidence.append(line.replace("EVIDENCE:", "").strip())
            elif line.startswith("CONFIDENCE:"):
                conf_text = line.replace("CONFIDENCE:", "").strip().lower()
                confidence = {"high": 0.9, "medium": 0.7, "low": 0.5}.get(conf_text, 0.7)

        if insight_content:
            return ProcessedInsight(
                content=insight_content,
                insight_type="analysis",
                confidence=confidence,
                supporting_evidence=evidence,
            )

        return None

    def _generate_thesis(
        self, topic: str, insights: List[ProcessedInsight], template: ThinkingPattern
    ) -> str:
        """Generate main thesis from insights."""
        if not insights:
            return f"Understanding {topic} requires examining multiple perspectives."

        insight_texts = [i.content for i in insights[:5]]

        prompt = f"""Based on these insights about "{topic}":

{chr(10).join('- ' + i for i in insight_texts)}

Generate a clear, compelling main thesis statement that:
1. Synthesizes the key insights
2. Takes a clear position
3. Is specific and actionable
4. Follows Dan Koe's direct, confident style

Respond with just the thesis statement (1-2 sentences)."""

        response = self._get_llm_response(prompt)
        return response.strip() if response else f"The future of {topic} will reshape how we work and live."

    def _generate_arguments(
        self, thesis: str, research: ResearchResult, argument_type: str
    ) -> List[str]:
        """Generate supporting or counter arguments."""
        direction = "support" if argument_type == "supporting" else "challenge"

        prompt = f"""Thesis: {thesis}

Generate 3 arguments that {direction} this thesis.
Base arguments on this research:
{chr(10).join('- ' + f for f in research.key_findings[:5])}

Each argument should be:
1. Specific and evidence-based
2. Concise (1-2 sentences)
3. Logically sound

Respond with numbered arguments:
1. [argument]
2. [argument]
3. [argument]"""

        response = self._get_llm_response(prompt)

        if not response:
            return []

        arguments = []
        for line in response.split("\n"):
            line = line.strip()
            if line and line[0].isdigit():
                # Remove numbering
                arg = line.lstrip("0123456789.").strip()
                if arg:
                    arguments.append(arg)

        return arguments[:3]

    def _generate_predictions(self, topic: str, research: ResearchResult) -> List[str]:
        """Generate future predictions."""
        prompt = f"""Topic: {topic}

Based on current trends and research:
{chr(10).join('- ' + f for f in research.key_findings[:5])}

Generate 3 specific, time-bound predictions about {topic}:
1. A near-term prediction (6-12 months)
2. A medium-term prediction (1-3 years)
3. A long-term prediction (5+ years)

Each prediction should be:
- Specific and falsifiable
- Based on logical extrapolation
- Bold but defensible

Format:
NEAR-TERM: [prediction]
MEDIUM-TERM: [prediction]
LONG-TERM: [prediction]"""

        response = self._get_llm_response(prompt)

        if not response:
            return []

        predictions = []
        for line in response.split("\n"):
            line = line.strip()
            for prefix in ["NEAR-TERM:", "MEDIUM-TERM:", "LONG-TERM:"]:
                if line.startswith(prefix):
                    pred = line.replace(prefix, "").strip()
                    if pred:
                        predictions.append(pred)

        return predictions

    def _generate_recommendations(
        self,
        topic: str,
        research: ResearchResult,
        insights: List[ProcessedInsight],
    ) -> List[str]:
        """Generate actionable recommendations."""
        if research.actionable_takeaways:
            return research.actionable_takeaways[:5]

        insight_texts = [i.content for i in insights[:3]]

        prompt = f"""Topic: {topic}

Key insights:
{chr(10).join('- ' + i for i in insight_texts)}

Generate 5 specific, actionable recommendations for someone wanting to leverage {topic}.

Each recommendation should:
1. Be immediately actionable
2. Be specific (not generic advice)
3. Start with an action verb
4. Be achievable by an individual

Format as numbered list."""

        response = self._get_llm_response(prompt)

        if not response:
            return [f"Start learning about {topic} today"]

        recommendations = []
        for line in response.split("\n"):
            line = line.strip()
            if line and line[0].isdigit():
                rec = line.lstrip("0123456789.").strip()
                if rec:
                    recommendations.append(rec)

        return recommendations[:5]

    def _calculate_creativity_scores(
        self, output: CognitiveOutput, research: ResearchResult
    ) -> Dict[str, float]:
        """Calculate creativity scores for the output."""
        scores = CreativityScore()

        # Novelty: How much new content vs just summarizing research
        if output.insights and research.key_findings:
            research_text = " ".join(research.key_findings).lower()
            novel_insights = sum(
                1
                for i in output.insights
                if i.content.lower() not in research_text
            )
            scores.novelty_score = novel_insights / len(output.insights)

        # Connection: Unusual but valid connections
        if output.supporting_arguments:
            # Simple heuristic: longer arguments often make more connections
            avg_length = sum(len(a.split()) for a in output.supporting_arguments) / len(
                output.supporting_arguments
            )
            scores.connection_score = min(1.0, avg_length / 30)

        # Practicality: Actionable recommendations
        if output.recommendations:
            action_verbs = ["start", "stop", "try", "build", "create", "use", "find", "learn"]
            actionable = sum(
                1
                for r in output.recommendations
                if r.split()[0].lower() in action_verbs
            )
            scores.practicality_score = actionable / len(output.recommendations)

        # Surprise: Counter-intuitive insights
        if output.counter_arguments:
            scores.surprise_score = min(1.0, len(output.counter_arguments) / 3)

        # Overall score
        scores.overall_score = (
            scores.novelty_score * 0.3
            + scores.connection_score * 0.2
            + scores.practicality_score * 0.3
            + scores.surprise_score * 0.2
        )

        # Interpretation
        if scores.overall_score >= 0.7:
            scores.interpretation = "Highly creative output with novel insights"
        elif scores.overall_score >= 0.5:
            scores.interpretation = "Good creative output with solid analysis"
        else:
            scores.interpretation = "Standard analysis, could benefit from more original thinking"

        return {
            "novelty_score": round(scores.novelty_score, 2),
            "connection_score": round(scores.connection_score, 2),
            "practicality_score": round(scores.practicality_score, 2),
            "surprise_score": round(scores.surprise_score, 2),
            "overall_score": round(scores.overall_score, 2),
            "interpretation": scores.interpretation,
        }

    def save_output(self, output: CognitiveOutput, output_path: Optional[Path] = None) -> Path:
        """Save cognitive output to disk."""
        output_path = output_path or settings.get_output_path() / f"cognitive_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output.to_dict(), f, indent=2, ensure_ascii=False)

        logger.info(f"Saved cognitive output to: {output_path}")
        return output_path
