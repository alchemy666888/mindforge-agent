"""
Bilingual Handler - Manage English-Chinese content processing.

Handles:
- Language detection and separation
- Translation mapping
- Cultural adaptation rules
- Chinese-specific text processing
"""

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import jieba

from mindforge_dankoe.config.settings import settings
from mindforge_dankoe.utils.logger import get_logger
from mindforge_dankoe.utils.text_processing import TextProcessor

logger = get_logger(__name__)


@dataclass
class CulturalMapping:
    """Mapping between Western and Chinese cultural concepts."""

    western_concept: str
    chinese_equivalent: str
    explanation: str
    usage_context: str
    examples: List[str] = field(default_factory=list)


@dataclass
class TranslationRule:
    """Rule for adapting content between languages."""

    category: str  # e.g., "tech_term", "idiom", "example"
    english_pattern: str
    chinese_adaptation: str
    notes: str = ""


@dataclass
class BilingualContent:
    """Content with both English and Chinese versions."""

    english: str
    chinese: str
    is_adapted: bool = False  # True if Chinese is culturally adapted, not just translated
    adaptation_notes: List[str] = field(default_factory=list)


class BilingualHandler:
    """
    Handle bilingual content processing for English and Chinese.

    Provides:
    - Language detection and separation
    - Translation assistance
    - Cultural adaptation rules
    - Chinese text analysis
    """

    # Tech term translations (English -> Simplified Chinese)
    TECH_TRANSLATIONS = {
        "artificial intelligence": "人工智能",
        "machine learning": "机器学习",
        "deep learning": "深度学习",
        "neural network": "神经网络",
        "algorithm": "算法",
        "startup": "创业公司/初创企业",
        "venture capital": "风险投资",
        "productivity": "生产力/效率",
        "automation": "自动化",
        "blockchain": "区块链",
        "cryptocurrency": "加密货币",
        "cloud computing": "云计算",
        "big data": "大数据",
        "internet of things": "物联网",
        "software as a service": "软件即服务 (SaaS)",
        "platform": "平台",
        "ecosystem": "生态系统",
        "monetize": "变现",
        "scale": "规模化",
        "disrupt": "颠覆",
        "innovation": "创新",
        "framework": "框架",
        "methodology": "方法论",
        "mindset": "思维方式/心态",
        "leverage": "杠杆/利用",
    }

    # Cultural adaptation mappings
    CULTURAL_MAPPINGS = [
        CulturalMapping(
            western_concept="Silicon Valley",
            chinese_equivalent="中关村/深圳",
            explanation="Chinese tech hubs as local equivalent",
            usage_context="When discussing tech startup culture",
            examples=["Silicon Valley startups", "中关村的创业公司"],
        ),
        CulturalMapping(
            western_concept="American Dream",
            chinese_equivalent="中国梦/个人奋斗",
            explanation="Cultural aspiration equivalent",
            usage_context="When discussing success narratives",
            examples=["pursuing the American Dream", "追求个人价值实现"],
        ),
        CulturalMapping(
            western_concept="work-life balance",
            chinese_equivalent="工作生活平衡/996问题",
            explanation="Work culture discussion",
            usage_context="When discussing productivity and life",
            examples=["achieving work-life balance", "逃离996，追求生活品质"],
        ),
        CulturalMapping(
            western_concept="side hustle",
            chinese_equivalent="副业/斜杠青年",
            explanation="Additional income concept",
            usage_context="When discussing entrepreneurship",
            examples=["starting a side hustle", "发展副业/成为斜杠青年"],
        ),
        CulturalMapping(
            western_concept="FAANG companies",
            chinese_equivalent="BAT/大厂",
            explanation="Major tech companies",
            usage_context="When discussing tech industry",
            examples=["working at FAANG", "在BAT/大厂工作"],
        ),
    ]

    # Writing style adaptations
    STYLE_ADAPTATIONS = [
        TranslationRule(
            category="directness",
            english_pattern="You should...",
            chinese_adaptation="建议您可以考虑...",
            notes="Chinese prefers softer suggestions",
        ),
        TranslationRule(
            category="directness",
            english_pattern="You need to...",
            chinese_adaptation="或许需要...",
            notes="Less imperative tone",
        ),
        TranslationRule(
            category="evidence",
            english_pattern="Studies show...",
            chinese_adaptation="研究表明...",
            notes="Similar but often add more context",
        ),
        TranslationRule(
            category="collectivism",
            english_pattern="Individual success",
            chinese_adaptation="个人与集体的共同发展",
            notes="Balance individual and collective",
        ),
    ]

    def __init__(self):
        """Initialize the bilingual handler."""
        self.text_processor = TextProcessor()
        self._llm_client = None

    @property
    def llm_client(self):
        """Lazy load LLM client."""
        if self._llm_client is None:
            try:
                import litellm

                self._llm_client = litellm
            except ImportError:
                logger.warning("LiteLLM not available for translation")
        return self._llm_client

    def detect_language(self, text: str) -> str:
        """
        Detect the primary language of text.

        Args:
            text: Text to analyze

        Returns:
            Language code ('en', 'zh', 'mixed')
        """
        return self.text_processor.detect_language(text)

    def separate_languages(self, text: str) -> Dict[str, str]:
        """
        Separate mixed-language content.

        Args:
            text: Mixed language text

        Returns:
            Dictionary with 'english' and 'chinese' content
        """
        chinese_pattern = re.compile(r"[\u4e00-\u9fff]+")

        chinese_parts = chinese_pattern.findall(text)
        english_parts = chinese_pattern.split(text)

        return {
            "english": " ".join([p.strip() for p in english_parts if p.strip()]),
            "chinese": "".join(chinese_parts),
            "is_mixed": bool(chinese_parts) and bool([p for p in english_parts if p.strip()]),
        }

    def translate_tech_terms(self, text: str, to_chinese: bool = True) -> str:
        """
        Translate technical terms in text.

        Args:
            text: Text with tech terms
            to_chinese: True to translate to Chinese, False for English

        Returns:
            Text with translated terms
        """
        if to_chinese:
            for eng, chn in self.TECH_TRANSLATIONS.items():
                # Case-insensitive replacement
                pattern = re.compile(re.escape(eng), re.IGNORECASE)
                text = pattern.sub(chn, text)
        else:
            for eng, chn in self.TECH_TRANSLATIONS.items():
                # Take first Chinese term if multiple options
                chn_primary = chn.split("/")[0]
                text = text.replace(chn_primary, eng)

        return text

    def get_cultural_mapping(self, concept: str) -> Optional[CulturalMapping]:
        """
        Get cultural mapping for a Western concept.

        Args:
            concept: Western concept to map

        Returns:
            CulturalMapping if found, None otherwise
        """
        concept_lower = concept.lower()
        for mapping in self.CULTURAL_MAPPINGS:
            if concept_lower in mapping.western_concept.lower():
                return mapping
        return None

    def adapt_examples(self, text: str) -> str:
        """
        Adapt Western examples to Chinese equivalents.

        Args:
            text: Text with Western examples

        Returns:
            Text with adapted examples
        """
        for mapping in self.CULTURAL_MAPPINGS:
            if mapping.western_concept.lower() in text.lower():
                # Simple replacement - in production, use LLM for context-aware adaptation
                pattern = re.compile(re.escape(mapping.western_concept), re.IGNORECASE)
                text = pattern.sub(mapping.chinese_equivalent, text)

        return text

    def apply_style_adaptations(self, text: str) -> Tuple[str, List[str]]:
        """
        Apply Chinese writing style adaptations.

        Args:
            text: Text to adapt

        Returns:
            Tuple of (adapted text, list of changes made)
        """
        changes = []

        for rule in self.STYLE_ADAPTATIONS:
            if rule.english_pattern.lower() in text.lower():
                pattern = re.compile(re.escape(rule.english_pattern), re.IGNORECASE)
                text = pattern.sub(rule.chinese_adaptation, text)
                changes.append(f"{rule.category}: '{rule.english_pattern}' -> '{rule.chinese_adaptation}'")

        return text, changes

    def segment_chinese(self, text: str) -> List[str]:
        """
        Segment Chinese text into words.

        Args:
            text: Chinese text

        Returns:
            List of segmented words
        """
        return list(jieba.cut(text))

    def add_chinese_punctuation(self, text: str) -> str:
        """
        Convert English punctuation to Chinese equivalents.

        Args:
            text: Text with English punctuation

        Returns:
            Text with Chinese punctuation
        """
        replacements = {
            ",": "，",
            ".": "。",
            "?": "？",
            "!": "！",
            ":": "：",
            ";": "；",
            "(": "（",
            ")": "）",
            '"': """,  # Opening quote
        }

        for eng, chn in replacements.items():
            text = text.replace(eng, chn)

        return text

    async def translate_with_llm(
        self, text: str, to_chinese: bool = True, adapt_culturally: bool = True
    ) -> BilingualContent:
        """
        Translate text using LLM with optional cultural adaptation.

        Args:
            text: Text to translate
            to_chinese: True to translate to Chinese
            adapt_culturally: Whether to apply cultural adaptations

        Returns:
            BilingualContent with both versions
        """
        if not self.llm_client:
            logger.warning("LLM client not available for translation")
            return BilingualContent(
                english=text if not to_chinese else "",
                chinese=text if to_chinese else "",
                is_adapted=False,
            )

        target_lang = "Simplified Chinese" if to_chinese else "English"
        source_lang = "English" if to_chinese else "Chinese"

        adaptation_instruction = ""
        if adapt_culturally and to_chinese:
            adaptation_instruction = """
            Also apply these cultural adaptations:
            - Replace Western examples with Chinese equivalents (e.g., Silicon Valley -> 中关村/深圳)
            - Adjust directness level (Chinese prefers softer suggestions)
            - Use appropriate Chinese tech terminology
            - Consider Chinese reader's cultural context
            """

        prompt = f"""Translate the following {source_lang} text to {target_lang}.
        Maintain the original meaning and tone while making it natural in the target language.
        {adaptation_instruction}

        Text to translate:
        {text}

        Provide only the translation, no explanations."""

        try:
            model = settings.get_available_llm_model()
            response = self.llm_client.completion(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
            )

            translated = response.choices[0].message.content.strip()

            if to_chinese:
                return BilingualContent(
                    english=text,
                    chinese=translated,
                    is_adapted=adapt_culturally,
                    adaptation_notes=["LLM translation with cultural adaptation"] if adapt_culturally else [],
                )
            else:
                return BilingualContent(
                    english=translated,
                    chinese=text,
                    is_adapted=adapt_culturally,
                )

        except Exception as e:
            logger.error(f"Translation error: {e}")
            return BilingualContent(
                english=text if not to_chinese else "",
                chinese=text if to_chinese else "",
                is_adapted=False,
            )

    def analyze_chinese_text(self, text: str) -> Dict[str, Any]:
        """
        Analyze Chinese text characteristics.

        Args:
            text: Chinese text to analyze

        Returns:
            Dictionary of analysis results
        """
        words = self.segment_chinese(text)
        chars = list(text)

        # Character analysis
        char_count = len([c for c in chars if "\u4e00" <= c <= "\u9fff"])

        # Sentence analysis
        sentences = re.split(r"[。！？]", text)
        sentences = [s.strip() for s in sentences if s.strip()]

        # Common measure words and particles
        measure_words = ["个", "只", "条", "件", "本", "张", "把", "位"]
        measure_count = sum(text.count(mw) for mw in measure_words)

        return {
            "character_count": char_count,
            "word_count": len(words),
            "sentence_count": len(sentences),
            "avg_sentence_length": char_count / len(sentences) if sentences else 0,
            "unique_words": len(set(words)),
            "measure_word_usage": measure_count,
            "vocabulary_richness": len(set(words)) / len(words) if words else 0,
        }

    def create_bilingual_article(
        self, english_text: str, chinese_text: Optional[str] = None
    ) -> BilingualContent:
        """
        Create a bilingual article structure.

        Args:
            english_text: Original English text
            chinese_text: Optional pre-translated Chinese text

        Returns:
            BilingualContent structure
        """
        if chinese_text:
            return BilingualContent(
                english=english_text,
                chinese=chinese_text,
                is_adapted=False,
            )

        # Apply basic transformations for a quick version
        chinese = self.translate_tech_terms(english_text, to_chinese=True)
        chinese = self.adapt_examples(chinese)
        chinese, changes = self.apply_style_adaptations(chinese)

        return BilingualContent(
            english=english_text,
            chinese=chinese,
            is_adapted=bool(changes),
            adaptation_notes=changes,
        )
