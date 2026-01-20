"""
Style Generator - Generate content in Dan Koe's signature style.

Implements:
- Sentence-level style transfer
- Chinese adaptation
- Personal voice blending
"""

import json
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from mindforge_dankoe.analysis.bilingual_handler import BilingualContent, BilingualHandler
from mindforge_dankoe.analysis.style_analyzer import StyleAnalyzer, StyleFingerprint
from mindforge_dankoe.config.settings import settings
from mindforge_dankoe.generation.cognitive_processor import CognitiveOutput
from mindforge_dankoe.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ArticleSection:
    """A section of the generated article."""

    heading: str = ""
    content: str = ""
    section_type: str = ""  # "intro", "body", "conclusion", "cta"


@dataclass
class GeneratedArticle:
    """A fully generated article."""

    title: str
    subtitle: str = ""
    sections: List[ArticleSection] = field(default_factory=list)
    full_content: str = ""
    word_count: int = 0
    reading_time: int = 0
    language: str = "en"
    style_similarity_score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "title": self.title,
            "subtitle": self.subtitle,
            "sections": [asdict(s) for s in self.sections],
            "full_content": self.full_content,
            "word_count": self.word_count,
            "reading_time": self.reading_time,
            "language": self.language,
            "style_similarity_score": self.style_similarity_score,
            "metadata": self.metadata,
            "timestamp": self.timestamp,
        }

    def to_markdown(self) -> str:
        """Convert to Markdown format."""
        parts = []

        parts.append(f"# {self.title}")
        if self.subtitle:
            parts.append(f"\n*{self.subtitle}*")
        parts.append("")

        for section in self.sections:
            if section.heading:
                parts.append(f"\n## {section.heading}\n")
            parts.append(section.content)
            parts.append("")

        return "\n".join(parts)


@dataclass
class VoiceSettings:
    """Configurable voice settings for generation."""

    directness: float = 0.8  # 0=subtle, 1=direct
    data_density: float = 0.6  # How much data vs narrative
    practicality_bias: float = 0.9  # Theory vs actionable
    optimism_level: float = 0.7  # Skeptical vs optimistic
    formality: float = 0.5  # Casual vs formal
    reader_engagement: float = 0.8  # How often to address reader

    @classmethod
    def dankoe_default(cls) -> "VoiceSettings":
        """Get Dan Koe's typical voice settings."""
        return cls(
            directness=0.8,
            data_density=0.6,
            practicality_bias=0.9,
            optimism_level=0.7,
            formality=0.5,
            reader_engagement=0.8,
        )


class StyleGenerator:
    """
    Generate articles in Dan Koe's style.

    Features:
    - Style fingerprint-based generation
    - Sentence-level style transfer
    - Chinese adaptation
    - Voice blending for personalization
    """

    # Dan Koe's typical opening patterns
    OPENING_PATTERNS = [
        "hook_question",  # Start with a thought-provoking question
        "bold_statement",  # Start with a bold claim
        "problem_setup",  # Start by describing a common problem
        "future_vision",  # Start with a vision of the future
        "personal_story",  # Start with a brief personal anecdote
    ]

    # Dan Koe's typical closing patterns
    CLOSING_PATTERNS = [
        "call_to_action",  # End with specific action to take
        "future_outlook",  # End with prediction or outlook
        "reflection_question",  # End with question for reader to ponder
        "key_takeaway",  # End with the single most important point
    ]

    def __init__(
        self,
        style_fingerprint: Optional[StyleFingerprint] = None,
        voice_settings: Optional[VoiceSettings] = None,
        bilingual_handler: Optional[BilingualHandler] = None,
    ):
        """
        Initialize the style generator.

        Args:
            style_fingerprint: Dan Koe style fingerprint
            voice_settings: Voice configuration
            bilingual_handler: Handler for Chinese adaptation
        """
        self.fingerprint = style_fingerprint
        self.voice = voice_settings or VoiceSettings.dankoe_default()
        self.bilingual = bilingual_handler or BilingualHandler()
        self._style_analyzer = StyleAnalyzer()

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
                max_tokens=3000,
            )

            return response.choices[0].message.content

        except Exception as e:
            logger.error(f"LLM error: {e}")
            return ""

    def generate_article(
        self,
        cognitive_output: CognitiveOutput,
        target_language: str = "en",
        word_count_target: int = 1500,
    ) -> GeneratedArticle:
        """
        Generate a full article from cognitive output.

        Args:
            cognitive_output: Processed insights and structure
            target_language: Target language ('en' or 'zh')
            word_count_target: Target word count

        Returns:
            GeneratedArticle
        """
        logger.info(f"Generating {target_language} article on: {cognitive_output.topic}")

        article = GeneratedArticle(
            title="",
            language=target_language,
        )

        # Generate title
        article.title = self._generate_title(cognitive_output)
        article.subtitle = self._generate_subtitle(cognitive_output)

        # Generate sections
        article.sections = self._generate_sections(cognitive_output, word_count_target)

        # Combine into full content
        article.full_content = self._combine_sections(article.sections)

        # Calculate metrics
        article.word_count = len(article.full_content.split())
        article.reading_time = max(1, article.word_count // 200)

        # Apply style transfer if fingerprint available
        if self.fingerprint:
            article.full_content = self._apply_style_transfer(article.full_content)
            article.style_similarity_score = self._calculate_style_similarity(
                article.full_content
            )

        # Translate to Chinese if needed
        if target_language == "zh":
            article = self._adapt_to_chinese(article, cognitive_output)

        article.metadata = {
            "template_used": cognitive_output.template_used,
            "creativity_scores": cognitive_output.creativity_scores,
            "source_topic": cognitive_output.topic,
        }

        return article

    def _generate_title(self, cognitive_output: CognitiveOutput) -> str:
        """Generate an engaging title."""
        prompt = f"""Generate a compelling article title about "{cognitive_output.topic}".

Main thesis: {cognitive_output.main_thesis}

The title should:
1. Be attention-grabbing but not clickbait
2. Hint at the value proposition
3. Be concise (under 10 words)
4. Follow Dan Koe's style: direct, insightful, slightly provocative

Generate 3 options:"""

        response = self._get_llm_response(prompt)

        if response:
            # Take first option
            lines = [l.strip() for l in response.split("\n") if l.strip()]
            for line in lines:
                clean = line.lstrip("0123456789.)-").strip()
                if clean and len(clean) > 5:
                    return clean

        return f"The Future of {cognitive_output.topic}: What You Need to Know"

    def _generate_subtitle(self, cognitive_output: CognitiveOutput) -> str:
        """Generate a subtitle."""
        if cognitive_output.predictions:
            return cognitive_output.predictions[0][:100]
        return ""

    def _generate_sections(
        self, cognitive_output: CognitiveOutput, word_count_target: int
    ) -> List[ArticleSection]:
        """Generate article sections."""
        sections = []

        # Introduction
        intro = self._generate_introduction(cognitive_output)
        sections.append(intro)

        # Main body sections based on insights
        body_sections = self._generate_body_sections(cognitive_output)
        sections.extend(body_sections)

        # Conclusion
        conclusion = self._generate_conclusion(cognitive_output)
        sections.append(conclusion)

        return sections

    def _generate_introduction(self, cognitive_output: CognitiveOutput) -> ArticleSection:
        """Generate the introduction section."""
        system_prompt = """You are Dan Koe, a tech analyst known for:
- Direct, confident writing style
- Making complex topics accessible
- Balancing data with narrative
- Engaging readers with questions and bold claims
- Being practical and actionable"""

        prompt = f"""Write an engaging introduction for an article about "{cognitive_output.topic}".

Main thesis: {cognitive_output.main_thesis}

Key points to set up:
{chr(10).join('- ' + arg for arg in cognitive_output.supporting_arguments[:3])}

The introduction should:
1. Hook the reader immediately (question, bold statement, or relatable problem)
2. Establish why this topic matters NOW
3. Preview the value they'll get
4. Be 150-200 words
5. Use "you" to directly address the reader

Write only the introduction paragraph(s):"""

        content = self._get_llm_response(prompt, system_prompt)

        return ArticleSection(
            heading="",
            content=content or f"Let's talk about {cognitive_output.topic}.",
            section_type="intro",
        )

    def _generate_body_sections(
        self, cognitive_output: CognitiveOutput
    ) -> List[ArticleSection]:
        """Generate body sections."""
        sections = []

        system_prompt = """You are Dan Koe, writing in your signature style:
- Short paragraphs (2-4 sentences)
- Mix of data and narrative
- Direct address to reader ("you")
- Occasional questions to engage
- Clear, confident statements
- Practical examples"""

        # Section for main insights
        if cognitive_output.insights:
            insight_texts = [i.content for i in cognitive_output.insights[:3]]

            prompt = f"""Write a section exploring these key insights about "{cognitive_output.topic}":

{chr(10).join('- ' + i for i in insight_texts)}

Guidelines:
1. Develop each insight with explanation and examples
2. Connect insights to show the bigger picture
3. Use data or evidence where relevant
4. Keep paragraphs short and punchy
5. About 300-400 words total

Write the section (no heading needed):"""

            content = self._get_llm_response(prompt, system_prompt)

            sections.append(
                ArticleSection(
                    heading="The Key Insights",
                    content=content or "Here's what the data tells us...",
                    section_type="body",
                )
            )

        # Section for predictions/future outlook
        if cognitive_output.predictions:
            prompt = f"""Write a section about future predictions for "{cognitive_output.topic}":

Predictions:
{chr(10).join('- ' + p for p in cognitive_output.predictions)}

Guidelines:
1. Present predictions with confidence but acknowledge uncertainty
2. Explain the reasoning behind each prediction
3. Connect to current trends
4. About 200-300 words

Write the section:"""

            content = self._get_llm_response(prompt, system_prompt)

            sections.append(
                ArticleSection(
                    heading="What's Coming Next",
                    content=content or "Here's where things are heading...",
                    section_type="body",
                )
            )

        # Section for recommendations
        if cognitive_output.recommendations:
            prompt = f"""Write a section with actionable recommendations about "{cognitive_output.topic}":

Recommendations:
{chr(10).join('- ' + r for r in cognitive_output.recommendations)}

Guidelines:
1. Make each recommendation specific and actionable
2. Explain WHY each matters
3. Start with the most impactful action
4. Use imperative voice ("Start...", "Focus on...")
5. About 200-300 words

Write the section:"""

            content = self._get_llm_response(prompt, system_prompt)

            sections.append(
                ArticleSection(
                    heading="What You Should Do Now",
                    content=content or "Here's how to take action...",
                    section_type="body",
                )
            )

        return sections

    def _generate_conclusion(self, cognitive_output: CognitiveOutput) -> ArticleSection:
        """Generate the conclusion section."""
        system_prompt = """You are Dan Koe. End articles with:
- A powerful summary of the main point
- A forward-looking statement
- A clear call to action or thought to ponder
- Confidence and momentum"""

        prompt = f"""Write a compelling conclusion for an article about "{cognitive_output.topic}".

Main thesis: {cognitive_output.main_thesis}

Key takeaway to reinforce:
{cognitive_output.recommendations[0] if cognitive_output.recommendations else "Take action now"}

The conclusion should:
1. Reinforce the main message without just summarizing
2. Leave the reader motivated to act
3. End on a strong, memorable note
4. Be 100-150 words

Write only the conclusion:"""

        content = self._get_llm_response(prompt, system_prompt)

        return ArticleSection(
            heading="",
            content=content or "The future belongs to those who prepare for it today.",
            section_type="conclusion",
        )

    def _combine_sections(self, sections: List[ArticleSection]) -> str:
        """Combine sections into full article content."""
        parts = []

        for section in sections:
            if section.heading:
                parts.append(f"\n## {section.heading}\n")
            parts.append(section.content)
            parts.append("")

        return "\n".join(parts)

    def _apply_style_transfer(self, content: str) -> str:
        """Apply Dan Koe's style characteristics to content."""
        if not self.fingerprint:
            return content

        # Apply sentence length adjustments
        sentences = content.split(". ")
        adjusted = []

        for sentence in sentences:
            words = sentence.split()

            # If sentence is too long, try to break it
            if len(words) > 30:
                # Find a good break point
                mid = len(words) // 2
                break_words = ["and", "but", "however", "while", "because"]
                for i, word in enumerate(words[mid - 5 : mid + 5]):
                    if word.lower() in break_words:
                        break_point = mid - 5 + i
                        first_part = " ".join(words[:break_point])
                        second_part = " ".join(words[break_point:])
                        adjusted.append(first_part)
                        adjusted.append(second_part.capitalize())
                        break
                else:
                    adjusted.append(sentence)
            else:
                adjusted.append(sentence)

        return ". ".join(adjusted)

    def _calculate_style_similarity(self, content: str) -> float:
        """Calculate style similarity to Dan Koe's fingerprint."""
        if not self.fingerprint:
            return 0.5

        # Compare key metrics
        scores = self._style_analyzer.compare_to_fingerprint(content)
        return scores.get("overall_similarity", 0.5)

    def _adapt_to_chinese(
        self, article: GeneratedArticle, cognitive_output: CognitiveOutput
    ) -> GeneratedArticle:
        """Adapt article to Chinese."""
        logger.info("Adapting article to Chinese")

        # Translate title
        chinese_title = self.bilingual.translate_tech_terms(article.title, to_chinese=True)

        # Translate and adapt content
        chinese_content = self.bilingual.translate_tech_terms(
            article.full_content, to_chinese=True
        )
        chinese_content = self.bilingual.adapt_examples(chinese_content)
        chinese_content, _ = self.bilingual.apply_style_adaptations(chinese_content)
        chinese_content = self.bilingual.add_chinese_punctuation(chinese_content)

        # Create Chinese version
        chinese_article = GeneratedArticle(
            title=chinese_title,
            subtitle=self.bilingual.translate_tech_terms(article.subtitle, to_chinese=True),
            full_content=chinese_content,
            word_count=len(chinese_content),
            reading_time=max(1, len(chinese_content) // 400),  # Chinese reads faster
            language="zh",
            style_similarity_score=article.style_similarity_score,
            metadata=article.metadata,
        )

        return chinese_article

    def generate_bilingual_article(
        self,
        cognitive_output: CognitiveOutput,
        word_count_target: int = 1500,
    ) -> Tuple[GeneratedArticle, GeneratedArticle]:
        """
        Generate both English and Chinese versions.

        Args:
            cognitive_output: Processed insights
            word_count_target: Target word count

        Returns:
            Tuple of (English article, Chinese article)
        """
        english = self.generate_article(
            cognitive_output, target_language="en", word_count_target=word_count_target
        )
        chinese = self._adapt_to_chinese(english, cognitive_output)

        return english, chinese

    def adjust_voice(self, new_settings: VoiceSettings) -> None:
        """Update voice settings for generation."""
        self.voice = new_settings
        logger.info("Voice settings updated")

    def save_article(self, article: GeneratedArticle, output_path: Optional[Path] = None) -> Path:
        """Save generated article to disk."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_title = re.sub(r"[^\w\s-]", "", article.title)[:30]
        safe_title = re.sub(r"\s+", "_", safe_title).lower()

        filename = f"{timestamp}_{safe_title}_{article.language}.json"
        output_path = output_path or settings.get_output_path() / filename

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(article.to_dict(), f, indent=2, ensure_ascii=False)

        # Also save Markdown version
        md_path = output_path.with_suffix(".md")
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(article.to_markdown())

        logger.info(f"Saved article to: {output_path}")
        return output_path
