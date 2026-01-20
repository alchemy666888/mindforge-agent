"""
Article Processor - Clean and structure crawled content for analysis.

This module handles:
- Text cleaning and normalization
- Content structure extraction
- Metadata enrichment
- Export to various formats
"""

import json
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from bs4 import BeautifulSoup

from mindforge_dankoe.config.settings import settings
from mindforge_dankoe.crawler.dankoe_spider import CrawledArticle
from mindforge_dankoe.utils.logger import get_logger
from mindforge_dankoe.utils.text_processing import TextProcessor, TextStats

logger = get_logger(__name__)


@dataclass
class ProcessedArticle:
    """Cleaned and enriched article data."""

    # Basic info
    url: str
    title: str
    author: str = "Dan Koe"

    # Content
    content_clean: str = ""
    paragraphs: List[str] = field(default_factory=list)
    sentences: List[str] = field(default_factory=list)
    headings: List[Dict[str, Any]] = field(default_factory=list)

    # Structure analysis
    intro_paragraph: str = ""
    conclusion_paragraph: str = ""
    main_sections: List[Dict[str, Any]] = field(default_factory=list)

    # Statistics
    stats: Optional[TextStats] = None
    word_count: int = 0
    reading_time: int = 0

    # Metadata
    date_published: Optional[str] = None
    date_processed: str = field(default_factory=lambda: datetime.now().isoformat())
    tags: List[str] = field(default_factory=list)
    excerpt: str = ""

    # Extracted elements
    key_phrases: List[str] = field(default_factory=list)
    questions: List[str] = field(default_factory=list)
    action_items: List[str] = field(default_factory=list)
    quotes: List[str] = field(default_factory=list)
    links: List[Dict[str, str]] = field(default_factory=list)

    # Language
    language: str = "en"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)
        if self.stats:
            data["stats"] = asdict(self.stats)
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ProcessedArticle":
        """Create from dictionary."""
        stats_data = data.pop("stats", None)
        article = cls(**data)
        if stats_data:
            article.stats = TextStats(**stats_data)
        return article


class ArticleProcessor:
    """
    Process and enrich crawled articles for analysis.

    Performs:
    - HTML cleanup and text extraction
    - Structural analysis (headings, sections, paragraphs)
    - Content extraction (questions, quotes, action items)
    - Statistical analysis
    - Language detection
    """

    def __init__(self, output_dir: Optional[Path] = None):
        """
        Initialize the processor.

        Args:
            output_dir: Directory to save processed articles
        """
        self.output_dir = output_dir or settings.get_processed_data_path() / "dankoe"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.text_processor = TextProcessor()

    def process_article(self, crawled: CrawledArticle) -> ProcessedArticle:
        """
        Process a single crawled article.

        Args:
            crawled: Raw crawled article

        Returns:
            ProcessedArticle with cleaned and enriched content
        """
        logger.info(f"Processing: {crawled.title}")

        # Initialize processed article
        processed = ProcessedArticle(
            url=crawled.url,
            title=crawled.title,
            author=crawled.author,
            date_published=crawled.date_published,
            tags=crawled.tags,
            excerpt=crawled.excerpt,
            links=crawled.links,
        )

        # Clean content
        content = self._clean_content(crawled.content, crawled.raw_html)
        processed.content_clean = content

        # Detect language
        processed.language = self.text_processor.detect_language(content)

        # Split into paragraphs and sentences
        processed.paragraphs = self.text_processor.split_paragraphs(content)
        processed.sentences = self.text_processor.split_sentences(content, processed.language)

        # Extract structural elements
        if crawled.raw_html:
            processed.headings = self._extract_headings(crawled.raw_html)
            processed.main_sections = self._extract_sections(crawled.raw_html)

        # Identify intro and conclusion
        if processed.paragraphs:
            processed.intro_paragraph = processed.paragraphs[0]
            processed.conclusion_paragraph = processed.paragraphs[-1]

        # Calculate statistics
        processed.stats = self.text_processor.calculate_stats(content)
        processed.word_count = processed.stats.word_count
        processed.reading_time = max(1, processed.word_count // 200)

        # Extract special elements
        processed.questions = self._extract_questions(content)
        processed.action_items = self._extract_action_items(content)
        processed.quotes = self._extract_quotes(content)

        # Extract key phrases
        key_phrases = self.text_processor.extract_key_phrases(content, processed.language)
        processed.key_phrases = [phrase for phrase, _ in key_phrases]

        return processed

    def _clean_content(self, content: str, raw_html: str = "") -> str:
        """
        Clean and normalize article content.

        Args:
            content: Extracted text content
            raw_html: Raw HTML for fallback processing

        Returns:
            Cleaned text content
        """
        if not content:
            if raw_html:
                soup = BeautifulSoup(raw_html, "html.parser")
                # Remove script and style elements
                for element in soup(["script", "style", "nav", "footer", "header"]):
                    element.decompose()
                content = soup.get_text(separator="\n", strip=True)
            else:
                return ""

        # Remove excessive whitespace
        content = re.sub(r"\n{3,}", "\n\n", content)
        content = re.sub(r"[ \t]+", " ", content)

        # Remove common newsletter artifacts
        artifacts = [
            r"Subscribe to.*",
            r"Share this.*",
            r"Forward this.*",
            r"Unsubscribe.*",
            r"View in browser.*",
            r"Click here to.*",
        ]
        for pattern in artifacts:
            content = re.sub(pattern, "", content, flags=re.IGNORECASE)

        return content.strip()

    def _extract_headings(self, raw_html: str) -> List[Dict[str, Any]]:
        """
        Extract headings hierarchy from HTML.

        Args:
            raw_html: Raw HTML content

        Returns:
            List of heading dictionaries with level and text
        """
        soup = BeautifulSoup(raw_html, "html.parser")
        headings = []

        for tag in soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6"]):
            level = int(tag.name[1])
            text = tag.get_text(strip=True)
            if text:
                headings.append({"level": level, "text": text})

        return headings

    def _extract_sections(self, raw_html: str) -> List[Dict[str, Any]]:
        """
        Extract main content sections based on headings.

        Args:
            raw_html: Raw HTML content

        Returns:
            List of sections with heading and content
        """
        soup = BeautifulSoup(raw_html, "html.parser")
        sections = []
        current_section = {"heading": "", "content": [], "level": 0}

        # Find main content area
        content_area = soup.find("article") or soup.find("main") or soup.find("body")
        if not content_area:
            return sections

        for element in content_area.children:
            if not hasattr(element, "name"):
                continue

            if element.name in ["h1", "h2", "h3"]:
                # Save previous section if it has content
                if current_section["content"]:
                    current_section["content"] = " ".join(current_section["content"])
                    sections.append(current_section)

                # Start new section
                current_section = {
                    "heading": element.get_text(strip=True),
                    "content": [],
                    "level": int(element.name[1]),
                }
            elif element.name in ["p", "div", "ul", "ol"]:
                text = element.get_text(strip=True)
                if text:
                    current_section["content"].append(text)

        # Don't forget the last section
        if current_section["content"]:
            current_section["content"] = " ".join(current_section["content"])
            sections.append(current_section)

        return sections

    def _extract_questions(self, content: str) -> List[str]:
        """
        Extract questions from content.

        Args:
            content: Text content

        Returns:
            List of questions
        """
        # Match sentences ending with ?
        questions = re.findall(r"[^.!?]*\?", content)
        questions = [q.strip() for q in questions if q.strip() and len(q) > 10]
        return questions

    def _extract_action_items(self, content: str) -> List[str]:
        """
        Extract actionable items/recommendations.

        Args:
            content: Text content

        Returns:
            List of action items
        """
        action_patterns = [
            r"(?:You should|You need to|Try to|Make sure|Consider|Start|Stop|Remember to)[^.!?]+[.!]",
            r"(?:Step \d+:?|First,|Next,|Finally,|Then,)[^.!?]+[.!]",
            r"(?:Here's how|Here is how|To do this)[^.!?]+[.!]",
        ]

        actions = []
        for pattern in action_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            actions.extend([m.strip() for m in matches])

        return list(set(actions))

    def _extract_quotes(self, content: str) -> List[str]:
        """
        Extract quoted text from content.

        Args:
            content: Text content

        Returns:
            List of quotes
        """
        # Match text in quotes
        quotes = re.findall(r'"([^"]{20,})"', content)
        quotes += re.findall(r'"([^"]{20,})"', content)  # Curly quotes
        quotes += re.findall(r"'([^']{20,})'", content)

        return list(set(quotes))

    def save_processed(self, article: ProcessedArticle) -> Path:
        """
        Save processed article to disk.

        Args:
            article: Processed article

        Returns:
            Path to saved file
        """
        # Create safe filename
        safe_title = re.sub(r"[^\w\s-]", "", article.title)[:50]
        safe_title = re.sub(r"\s+", "_", safe_title).lower()

        timestamp = datetime.now().strftime("%Y%m%d")
        filename = f"{timestamp}_{safe_title}.json"
        filepath = self.output_dir / filename

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(article.to_dict(), f, indent=2, ensure_ascii=False)

        logger.info(f"Saved processed article: {filepath}")
        return filepath

    def process_all(self, crawled_articles: List[CrawledArticle]) -> List[ProcessedArticle]:
        """
        Process multiple crawled articles.

        Args:
            crawled_articles: List of crawled articles

        Returns:
            List of processed articles
        """
        processed = []
        for article in crawled_articles:
            try:
                proc = self.process_article(article)
                self.save_processed(proc)
                processed.append(proc)
            except Exception as e:
                logger.error(f"Failed to process {article.url}: {e}")

        logger.info(f"Processed {len(processed)} articles")
        return processed

    def load_processed_articles(self) -> List[ProcessedArticle]:
        """
        Load previously processed articles from disk.

        Returns:
            List of processed articles
        """
        articles = []
        for filepath in self.output_dir.glob("*.json"):
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    articles.append(ProcessedArticle.from_dict(data))
            except Exception as e:
                logger.warning(f"Failed to load {filepath}: {e}")

        logger.info(f"Loaded {len(articles)} processed articles")
        return articles

    def export_corpus(self, articles: List[ProcessedArticle], format: str = "jsonl") -> Path:
        """
        Export all articles as a corpus file.

        Args:
            articles: List of processed articles
            format: Export format ('jsonl', 'txt', 'csv')

        Returns:
            Path to exported file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"dankoe_corpus_{timestamp}.{format}"
        filepath = self.output_dir / filename

        if format == "jsonl":
            with open(filepath, "w", encoding="utf-8") as f:
                for article in articles:
                    f.write(json.dumps(article.to_dict(), ensure_ascii=False) + "\n")

        elif format == "txt":
            with open(filepath, "w", encoding="utf-8") as f:
                for article in articles:
                    f.write(f"# {article.title}\n\n")
                    f.write(article.content_clean)
                    f.write("\n\n---\n\n")

        elif format == "csv":
            import csv

            with open(filepath, "w", encoding="utf-8", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["title", "url", "date", "word_count", "excerpt", "content"])
                for article in articles:
                    writer.writerow(
                        [
                            article.title,
                            article.url,
                            article.date_published,
                            article.word_count,
                            article.excerpt,
                            article.content_clean[:500],
                        ]
                    )

        logger.info(f"Exported corpus to: {filepath}")
        return filepath
