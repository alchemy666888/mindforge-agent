"""
Style Analyzer - Extract and quantify Dan Koe's writing style.

This module analyzes:
- Vocabulary patterns and preferences
- Sentence structure and rhythm
- Rhetorical devices
- Tone and voice characteristics
"""

import json
import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from mindforge_dankoe.config.settings import settings
from mindforge_dankoe.crawler.article_processor import ProcessedArticle
from mindforge_dankoe.utils.logger import get_logger
from mindforge_dankoe.utils.text_processing import TextProcessor

logger = get_logger(__name__)


@dataclass
class VocabularyMetrics:
    """Vocabulary-related style metrics."""

    total_unique_words: int = 0
    vocabulary_richness: float = 0.0  # unique/total
    avg_word_length: float = 0.0
    favorite_words: List[Tuple[str, int]] = field(default_factory=list)
    favorite_adjectives: List[str] = field(default_factory=list)
    favorite_verbs: List[str] = field(default_factory=list)
    tech_jargon_frequency: float = 0.0
    transition_words: Dict[str, int] = field(default_factory=dict)
    power_words: List[str] = field(default_factory=list)


@dataclass
class SentenceMetrics:
    """Sentence structure metrics."""

    avg_sentence_length: float = 0.0
    sentence_length_std: float = 0.0
    min_sentence_length: int = 0
    max_sentence_length: int = 0
    short_sentence_ratio: float = 0.0  # < 10 words
    long_sentence_ratio: float = 0.0  # > 25 words
    question_frequency: float = 0.0  # questions per 100 sentences
    exclamation_frequency: float = 0.0
    imperative_frequency: float = 0.0


@dataclass
class ParagraphMetrics:
    """Paragraph structure metrics."""

    avg_paragraph_length: float = 0.0  # sentences per paragraph
    avg_paragraph_words: float = 0.0
    paragraph_length_std: float = 0.0
    single_sentence_paragraph_ratio: float = 0.0


@dataclass
class RhetoricalMetrics:
    """Rhetorical device usage metrics."""

    analogy_frequency: float = 0.0  # per article
    metaphor_frequency: float = 0.0
    rhetorical_question_frequency: float = 0.0
    repetition_patterns: List[str] = field(default_factory=list)
    contrast_usage: float = 0.0  # but, however, etc.
    list_usage: float = 0.0  # bulleted/numbered lists


@dataclass
class ToneMetrics:
    """Tone and voice metrics."""

    formality_score: float = 0.5  # 0=casual, 1=formal
    directness_score: float = 0.5  # 0=indirect, 1=direct
    confidence_score: float = 0.5  # 0=hedged, 1=confident
    reader_address_frequency: float = 0.0  # "you" usage
    first_person_frequency: float = 0.0  # "I" usage
    inclusive_language_frequency: float = 0.0  # "we" usage


@dataclass
class StyleFingerprint:
    """Complete style fingerprint for an author."""

    author: str = "Dan Koe"
    vocabulary: VocabularyMetrics = field(default_factory=VocabularyMetrics)
    sentences: SentenceMetrics = field(default_factory=SentenceMetrics)
    paragraphs: ParagraphMetrics = field(default_factory=ParagraphMetrics)
    rhetorical: RhetoricalMetrics = field(default_factory=RhetoricalMetrics)
    tone: ToneMetrics = field(default_factory=ToneMetrics)
    articles_analyzed: int = 0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StyleFingerprint":
        """Create from dictionary."""
        fp = cls()
        fp.author = data.get("author", "Dan Koe")
        fp.articles_analyzed = data.get("articles_analyzed", 0)
        fp.timestamp = data.get("timestamp", "")

        if "vocabulary" in data:
            fp.vocabulary = VocabularyMetrics(**data["vocabulary"])
        if "sentences" in data:
            fp.sentences = SentenceMetrics(**data["sentences"])
        if "paragraphs" in data:
            fp.paragraphs = ParagraphMetrics(**data["paragraphs"])
        if "rhetorical" in data:
            fp.rhetorical = RhetoricalMetrics(**data["rhetorical"])
        if "tone" in data:
            fp.tone = ToneMetrics(**data["tone"])

        return fp


class StyleAnalyzer:
    """
    Analyze and quantify writing style characteristics.

    Extracts comprehensive metrics about:
    - Word choice and vocabulary patterns
    - Sentence structure and rhythm
    - Paragraph organization
    - Rhetorical devices
    - Tone and voice
    """

    # Common tech jargon words
    TECH_JARGON = {
        "ai", "ml", "algorithm", "data", "automation", "platform", "api",
        "digital", "tech", "software", "hardware", "cloud", "blockchain",
        "crypto", "saas", "startup", "scale", "leverage", "optimize",
        "iterate", "disrupt", "innovation", "ecosystem", "paradigm",
    }

    # Power words commonly used in persuasive writing
    POWER_WORDS = {
        "revolutionary", "essential", "critical", "ultimate", "proven",
        "powerful", "guaranteed", "exclusive", "secret", "instant",
        "breakthrough", "transform", "discover", "unlock", "master",
    }

    # Hedging words that reduce confidence
    HEDGE_WORDS = {
        "maybe", "perhaps", "might", "could", "possibly", "somewhat",
        "sort of", "kind of", "seems", "appears", "suggests",
    }

    # Confident assertion words
    CONFIDENT_WORDS = {
        "will", "definitely", "certainly", "clearly", "obviously",
        "absolutely", "undoubtedly", "must", "always", "never",
    }

    def __init__(self):
        """Initialize the analyzer."""
        self.text_processor = TextProcessor()
        self.fingerprint: Optional[StyleFingerprint] = None
        self._word_frequencies: Counter = Counter()
        self._all_sentences: List[str] = []
        self._all_paragraphs: List[str] = []

    def analyze_article(self, article: ProcessedArticle) -> Dict[str, Any]:
        """
        Analyze style metrics for a single article.

        Args:
            article: Processed article to analyze

        Returns:
            Dictionary of style metrics
        """
        metrics = {}

        # Vocabulary analysis
        tokens = self.text_processor.tokenize(article.content_clean)
        self._word_frequencies.update(tokens)

        # Store for aggregate analysis
        self._all_sentences.extend(article.sentences)
        self._all_paragraphs.extend(article.paragraphs)

        # Per-article metrics
        metrics["word_count"] = len(tokens)
        metrics["unique_words"] = len(set(tokens))
        metrics["avg_word_length"] = np.mean([len(w) for w in tokens]) if tokens else 0

        # Sentence metrics
        sentence_lengths = [len(s.split()) for s in article.sentences]
        if sentence_lengths:
            metrics["avg_sentence_length"] = np.mean(sentence_lengths)
            metrics["sentence_length_std"] = np.std(sentence_lengths)
            metrics["question_count"] = sum(1 for s in article.sentences if "?" in s)

        # Paragraph metrics
        para_lengths = [len(self.text_processor.split_sentences(p)) for p in article.paragraphs]
        if para_lengths:
            metrics["avg_paragraph_sentences"] = np.mean(para_lengths)

        return metrics

    def create_fingerprint(self, articles: List[ProcessedArticle]) -> StyleFingerprint:
        """
        Create a comprehensive style fingerprint from multiple articles.

        Args:
            articles: List of processed articles

        Returns:
            StyleFingerprint with aggregated metrics
        """
        logger.info(f"Creating style fingerprint from {len(articles)} articles")

        # Reset aggregation
        self._word_frequencies = Counter()
        self._all_sentences = []
        self._all_paragraphs = []

        # Analyze each article
        for article in articles:
            self.analyze_article(article)

        # Create fingerprint
        fp = StyleFingerprint(articles_analyzed=len(articles))

        # Vocabulary metrics
        fp.vocabulary = self._compute_vocabulary_metrics()

        # Sentence metrics
        fp.sentences = self._compute_sentence_metrics()

        # Paragraph metrics
        fp.paragraphs = self._compute_paragraph_metrics()

        # Rhetorical metrics
        fp.rhetorical = self._compute_rhetorical_metrics(articles)

        # Tone metrics
        fp.tone = self._compute_tone_metrics()

        self.fingerprint = fp
        return fp

    def _compute_vocabulary_metrics(self) -> VocabularyMetrics:
        """Compute vocabulary-related metrics."""
        metrics = VocabularyMetrics()

        all_words = list(self._word_frequencies.elements())
        total_words = sum(self._word_frequencies.values())

        if not total_words:
            return metrics

        metrics.total_unique_words = len(self._word_frequencies)
        metrics.vocabulary_richness = metrics.total_unique_words / total_words

        # Average word length
        metrics.avg_word_length = np.mean([len(w) for w in all_words])

        # Most common words (excluding very common ones)
        common_stop = {"the", "a", "an", "is", "are", "was", "were", "to", "of", "and", "in", "for", "on", "with", "that", "this", "it"}
        filtered_freq = {w: c for w, c in self._word_frequencies.items() if w.lower() not in common_stop and len(w) > 2}
        metrics.favorite_words = Counter(filtered_freq).most_common(20)

        # Tech jargon frequency
        tech_count = sum(self._word_frequencies.get(word, 0) for word in self.TECH_JARGON)
        metrics.tech_jargon_frequency = tech_count / total_words

        # Power words
        metrics.power_words = [w for w in self.POWER_WORDS if self._word_frequencies.get(w, 0) > 0]

        # Transition words
        all_text = " ".join(self._all_paragraphs)
        metrics.transition_words = self.text_processor.find_transition_words(all_text)

        return metrics

    def _compute_sentence_metrics(self) -> SentenceMetrics:
        """Compute sentence-related metrics."""
        metrics = SentenceMetrics()

        if not self._all_sentences:
            return metrics

        # Sentence lengths
        lengths = [len(s.split()) for s in self._all_sentences]
        metrics.avg_sentence_length = np.mean(lengths)
        metrics.sentence_length_std = np.std(lengths)
        metrics.min_sentence_length = min(lengths)
        metrics.max_sentence_length = max(lengths)

        # Length distribution
        total = len(lengths)
        metrics.short_sentence_ratio = sum(1 for l in lengths if l < 10) / total
        metrics.long_sentence_ratio = sum(1 for l in lengths if l > 25) / total

        # Question and exclamation frequency (per 100 sentences)
        questions = sum(1 for s in self._all_sentences if "?" in s)
        exclamations = sum(1 for s in self._all_sentences if "!" in s)
        metrics.question_frequency = (questions / total) * 100
        metrics.exclamation_frequency = (exclamations / total) * 100

        # Imperative frequency (sentences starting with verbs)
        imperative_starters = {"start", "stop", "try", "make", "get", "do", "be", "take", "use", "find", "learn", "build", "create", "focus", "think", "remember", "consider", "avoid", "embrace"}
        imperatives = sum(1 for s in self._all_sentences if s.split()[0].lower() in imperative_starters) if self._all_sentences else 0
        metrics.imperative_frequency = (imperatives / total) * 100

        return metrics

    def _compute_paragraph_metrics(self) -> ParagraphMetrics:
        """Compute paragraph-related metrics."""
        metrics = ParagraphMetrics()

        if not self._all_paragraphs:
            return metrics

        # Paragraph lengths (in sentences)
        para_sentence_counts = [
            len(self.text_processor.split_sentences(p)) for p in self._all_paragraphs
        ]
        para_word_counts = [len(p.split()) for p in self._all_paragraphs]

        metrics.avg_paragraph_length = np.mean(para_sentence_counts)
        metrics.avg_paragraph_words = np.mean(para_word_counts)
        metrics.paragraph_length_std = np.std(para_sentence_counts)

        # Single sentence paragraphs
        single_sentence = sum(1 for c in para_sentence_counts if c == 1)
        metrics.single_sentence_paragraph_ratio = single_sentence / len(self._all_paragraphs)

        return metrics

    def _compute_rhetorical_metrics(self, articles: List[ProcessedArticle]) -> RhetoricalMetrics:
        """Compute rhetorical device metrics."""
        metrics = RhetoricalMetrics()

        if not articles:
            return metrics

        total_articles = len(articles)
        all_text = " ".join(a.content_clean for a in articles).lower()

        # Analogy patterns
        analogy_patterns = ["like a", "similar to", "as if", "just as", "imagine", "think of it as"]
        analogy_count = sum(all_text.count(p) for p in analogy_patterns)
        metrics.analogy_frequency = analogy_count / total_articles

        # Metaphor indicators (harder to detect precisely)
        metaphor_patterns = ["is a", "are the", "becomes a"]
        metaphor_count = sum(all_text.count(p) for p in metaphor_patterns)
        metrics.metaphor_frequency = metaphor_count / total_articles

        # Rhetorical questions (questions not seeking literal answers)
        rhetorical_indicators = ["isn't it", "don't you", "right?", "wouldn't you agree"]
        rhetorical_count = sum(all_text.count(p) for p in rhetorical_indicators)
        metrics.rhetorical_question_frequency = rhetorical_count / total_articles

        # Contrast usage
        contrast_words = ["but", "however", "although", "yet", "nevertheless", "on the other hand"]
        contrast_count = sum(all_text.count(f" {w} ") for w in contrast_words)
        metrics.contrast_usage = contrast_count / total_articles

        return metrics

    def _compute_tone_metrics(self) -> ToneMetrics:
        """Compute tone and voice metrics."""
        metrics = ToneMetrics()

        all_text = " ".join(self._all_sentences).lower()
        total_words = sum(self._word_frequencies.values())

        if not total_words:
            return metrics

        # Pronoun frequencies
        you_count = all_text.count(" you ")
        i_count = all_text.count(" i ")
        we_count = all_text.count(" we ")

        metrics.reader_address_frequency = you_count / total_words * 100
        metrics.first_person_frequency = i_count / total_words * 100
        metrics.inclusive_language_frequency = we_count / total_words * 100

        # Confidence score (confident words vs hedge words)
        confident_count = sum(self._word_frequencies.get(w, 0) for w in self.CONFIDENT_WORDS)
        hedge_count = sum(self._word_frequencies.get(w, 0) for w in self.HEDGE_WORDS)

        if confident_count + hedge_count > 0:
            metrics.confidence_score = confident_count / (confident_count + hedge_count)
        else:
            metrics.confidence_score = 0.5

        # Directness (imperative sentences, short sentences)
        metrics.directness_score = min(1.0, self.fingerprint.sentences.short_sentence_ratio * 2 if hasattr(self, 'fingerprint') and self.fingerprint else 0.5)

        # Formality (contractions, casual words reduce formality)
        contractions = ["'s", "'t", "'re", "'ve", "'ll", "'d"]
        contraction_count = sum(all_text.count(c) for c in contractions)
        metrics.formality_score = max(0, 1 - (contraction_count / total_words * 20))

        return metrics

    def save_fingerprint(self, output_path: Optional[Path] = None) -> Path:
        """Save the style fingerprint to disk."""
        if not self.fingerprint:
            raise ValueError("No fingerprint created yet. Call create_fingerprint first.")

        output_path = output_path or settings.get_processed_data_path() / "style_fingerprint.json"

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(self.fingerprint.to_dict(), f, indent=2, ensure_ascii=False)

        logger.info(f"Saved style fingerprint to: {output_path}")
        return output_path

    def load_fingerprint(self, input_path: Optional[Path] = None) -> StyleFingerprint:
        """Load a style fingerprint from disk."""
        input_path = input_path or settings.get_processed_data_path() / "style_fingerprint.json"

        if not input_path.exists():
            raise FileNotFoundError(f"Fingerprint file not found: {input_path}")

        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.fingerprint = StyleFingerprint.from_dict(data)
        logger.info(f"Loaded style fingerprint from: {input_path}")
        return self.fingerprint

    def compare_to_fingerprint(self, text: str) -> Dict[str, float]:
        """
        Compare a text to the stored fingerprint.

        Args:
            text: Text to compare

        Returns:
            Dictionary of similarity scores (0-1)
        """
        if not self.fingerprint:
            raise ValueError("No fingerprint loaded")

        # Analyze the text
        tokens = self.text_processor.tokenize(text)
        sentences = self.text_processor.split_sentences(text)
        paragraphs = self.text_processor.split_paragraphs(text)

        scores = {}

        # Vocabulary similarity
        if tokens:
            word_lengths = [len(w) for w in tokens]
            avg_word_len = np.mean(word_lengths)
            vocab_richness = len(set(tokens)) / len(tokens)

            scores["vocabulary_similarity"] = 1 - abs(avg_word_len - self.fingerprint.vocabulary.avg_word_length) / 10

        # Sentence length similarity
        if sentences:
            sent_lengths = [len(s.split()) for s in sentences]
            avg_sent_len = np.mean(sent_lengths)

            scores["sentence_similarity"] = 1 - abs(avg_sent_len - self.fingerprint.sentences.avg_sentence_length) / 20

        # Overall similarity
        scores["overall_similarity"] = np.mean(list(scores.values()))

        return scores

    def get_style_summary(self) -> str:
        """Get a human-readable summary of the style fingerprint."""
        if not self.fingerprint:
            return "No fingerprint available"

        fp = self.fingerprint
        summary = []

        summary.append(f"Style Analysis Summary for {fp.author}")
        summary.append(f"Based on {fp.articles_analyzed} articles\n")

        summary.append("Vocabulary:")
        summary.append(f"  - Vocabulary richness: {fp.vocabulary.vocabulary_richness:.2%}")
        summary.append(f"  - Average word length: {fp.vocabulary.avg_word_length:.1f} characters")
        summary.append(f"  - Tech jargon frequency: {fp.vocabulary.tech_jargon_frequency:.2%}")

        summary.append("\nSentence Structure:")
        summary.append(f"  - Average sentence length: {fp.sentences.avg_sentence_length:.1f} words")
        summary.append(f"  - Short sentences (<10 words): {fp.sentences.short_sentence_ratio:.1%}")
        summary.append(f"  - Questions per 100 sentences: {fp.sentences.question_frequency:.1f}")

        summary.append("\nTone:")
        summary.append(f"  - Formality: {fp.tone.formality_score:.0%}")
        summary.append(f"  - Confidence: {fp.tone.confidence_score:.0%}")
        summary.append(f"  - Reader address ('you'): {fp.tone.reader_address_frequency:.1f}%")

        return "\n".join(summary)
