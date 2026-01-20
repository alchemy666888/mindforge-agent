"""
Text processing utilities for content analysis and manipulation.
"""

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import jieba


@dataclass
class TextStats:
    """Statistics about a text document."""

    word_count: int = 0
    sentence_count: int = 0
    paragraph_count: int = 0
    avg_sentence_length: float = 0.0
    avg_paragraph_length: float = 0.0
    unique_words: int = 0
    vocabulary_richness: float = 0.0
    question_count: int = 0
    exclamation_count: int = 0
    language: str = "en"


@dataclass
class ExtractedContent:
    """Extracted and cleaned content from an article."""

    title: str = ""
    content: str = ""
    paragraphs: List[str] = field(default_factory=list)
    sentences: List[str] = field(default_factory=list)
    headings: List[str] = field(default_factory=list)
    links: List[Dict[str, str]] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)


class TextProcessor:
    """
    Comprehensive text processing utilities for English and Chinese content.
    """

    # Common English sentence endings
    SENTENCE_ENDINGS = re.compile(r"[.!?]+(?=\s|$)")

    # Chinese sentence endings
    CHINESE_SENTENCE_ENDINGS = re.compile(r"[。！？]+")

    # Paragraph separator patterns
    PARAGRAPH_SEPARATORS = re.compile(r"\n\s*\n")

    # Chinese character detection
    CHINESE_CHARS = re.compile(r"[\u4e00-\u9fff]")

    def __init__(self):
        """Initialize the text processor."""
        self._spacy_en = None
        self._spacy_zh = None

    @property
    def spacy_en(self):
        """Lazy load English spaCy model."""
        if self._spacy_en is None:
            try:
                import spacy

                self._spacy_en = spacy.load("en_core_web_sm")
            except OSError:
                from mindforge_dankoe.utils.logger import get_logger

                logger = get_logger(__name__)
                logger.warning("English spaCy model not found. Run: python -m spacy download en_core_web_sm")
                self._spacy_en = None
        return self._spacy_en

    @property
    def spacy_zh(self):
        """Lazy load Chinese spaCy model."""
        if self._spacy_zh is None:
            try:
                import spacy

                self._spacy_zh = spacy.load("zh_core_web_sm")
            except OSError:
                from mindforge_dankoe.utils.logger import get_logger

                logger = get_logger(__name__)
                logger.warning("Chinese spaCy model not found. Run: python -m spacy download zh_core_web_sm")
                self._spacy_zh = None
        return self._spacy_zh

    def detect_language(self, text: str) -> str:
        """
        Detect if text is primarily English or Chinese.

        Args:
            text: Text to analyze

        Returns:
            Language code ('en', 'zh', or 'mixed')
        """
        if not text:
            return "en"

        chinese_chars = len(self.CHINESE_CHARS.findall(text))
        total_chars = len(text.replace(" ", ""))

        if total_chars == 0:
            return "en"

        chinese_ratio = chinese_chars / total_chars

        if chinese_ratio > 0.5:
            return "zh"
        elif chinese_ratio > 0.1:
            return "mixed"
        else:
            return "en"

    def split_sentences(self, text: str, language: Optional[str] = None) -> List[str]:
        """
        Split text into sentences, handling both English and Chinese.

        Args:
            text: Text to split
            language: Language code ('en' or 'zh'), auto-detected if None

        Returns:
            List of sentences
        """
        if not text:
            return []

        lang = language or self.detect_language(text)

        if lang == "zh":
            # Chinese sentence splitting
            sentences = self.CHINESE_SENTENCE_ENDINGS.split(text)
        else:
            # English sentence splitting
            sentences = self.SENTENCE_ENDINGS.split(text)

        # Clean and filter
        sentences = [s.strip() for s in sentences if s.strip()]
        return sentences

    def split_paragraphs(self, text: str) -> List[str]:
        """
        Split text into paragraphs.

        Args:
            text: Text to split

        Returns:
            List of paragraphs
        """
        if not text:
            return []

        paragraphs = self.PARAGRAPH_SEPARATORS.split(text)
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        return paragraphs

    def tokenize(self, text: str, language: Optional[str] = None) -> List[str]:
        """
        Tokenize text into words.

        Args:
            text: Text to tokenize
            language: Language code, auto-detected if None

        Returns:
            List of tokens
        """
        if not text:
            return []

        lang = language or self.detect_language(text)

        if lang == "zh":
            # Use jieba for Chinese
            tokens = list(jieba.cut(text))
        else:
            # Simple word tokenization for English
            tokens = re.findall(r"\b\w+\b", text.lower())

        return [t for t in tokens if t.strip()]

    def calculate_stats(self, text: str) -> TextStats:
        """
        Calculate comprehensive statistics for a text.

        Args:
            text: Text to analyze

        Returns:
            TextStats object with computed metrics
        """
        if not text:
            return TextStats()

        language = self.detect_language(text)
        paragraphs = self.split_paragraphs(text)
        sentences = self.split_sentences(text, language)
        tokens = self.tokenize(text, language)

        word_count = len(tokens)
        sentence_count = len(sentences)
        paragraph_count = len(paragraphs)
        unique_words = len(set(tokens))

        # Calculate averages
        avg_sentence_length = word_count / sentence_count if sentence_count > 0 else 0
        avg_paragraph_length = sentence_count / paragraph_count if paragraph_count > 0 else 0
        vocabulary_richness = unique_words / word_count if word_count > 0 else 0

        # Count questions and exclamations
        question_count = text.count("?") + text.count("？")
        exclamation_count = text.count("!") + text.count("！")

        return TextStats(
            word_count=word_count,
            sentence_count=sentence_count,
            paragraph_count=paragraph_count,
            avg_sentence_length=avg_sentence_length,
            avg_paragraph_length=avg_paragraph_length,
            unique_words=unique_words,
            vocabulary_richness=vocabulary_richness,
            question_count=question_count,
            exclamation_count=exclamation_count,
            language=language,
        )

    def clean_text(self, text: str) -> str:
        """
        Clean and normalize text content.

        Args:
            text: Text to clean

        Returns:
            Cleaned text
        """
        if not text:
            return ""

        # Normalize whitespace
        text = re.sub(r"\s+", " ", text)

        # Remove multiple newlines but preserve paragraph breaks
        text = re.sub(r"\n{3,}", "\n\n", text)

        # Strip leading/trailing whitespace
        text = text.strip()

        return text

    def extract_key_phrases(
        self, text: str, language: Optional[str] = None, top_n: int = 10
    ) -> List[Tuple[str, float]]:
        """
        Extract key phrases from text using TF-IDF-like scoring.

        Args:
            text: Text to analyze
            language: Language code
            top_n: Number of top phrases to return

        Returns:
            List of (phrase, score) tuples
        """
        lang = language or self.detect_language(text)
        tokens = self.tokenize(text, lang)

        if not tokens:
            return []

        # Simple frequency-based extraction
        freq = {}
        for token in tokens:
            if len(token) > 2:  # Filter short words
                freq[token] = freq.get(token, 0) + 1

        # Score by frequency normalized by position (earlier = more important)
        scored = []
        for token, count in freq.items():
            first_pos = tokens.index(token)
            position_weight = 1 / (1 + first_pos / len(tokens))
            score = count * position_weight
            scored.append((token, score))

        # Sort by score and return top N
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_n]

    def find_transition_words(self, text: str) -> Dict[str, int]:
        """
        Find and count transition words/phrases in text.

        Args:
            text: Text to analyze

        Returns:
            Dictionary of transition word -> count
        """
        transition_words = [
            # English
            "however",
            "therefore",
            "furthermore",
            "moreover",
            "consequently",
            "nevertheless",
            "meanwhile",
            "subsequently",
            "ultimately",
            "essentially",
            "specifically",
            "additionally",
            "alternatively",
            "accordingly",
            "in conclusion",
            "for example",
            "in other words",
            "on the other hand",
            "as a result",
            "in fact",
            "first",
            "second",
            "third",
            "finally",
            "next",
            "then",
            "but",
            "and",
            "so",
            "yet",
            # Chinese
            "然而",
            "因此",
            "此外",
            "而且",
            "不过",
            "但是",
            "所以",
            "总之",
            "例如",
            "换句话说",
            "另一方面",
            "事实上",
            "首先",
            "其次",
            "最后",
            "接着",
            "然后",
        ]

        text_lower = text.lower()
        found = {}

        for word in transition_words:
            count = text_lower.count(word.lower())
            if count > 0:
                found[word] = count

        return found
