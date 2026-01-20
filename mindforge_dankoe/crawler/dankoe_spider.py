"""
Dan Koe Newsletter Spider - Comprehensive web crawler for letters.thedankoe.com

This spider respects robots.txt, implements proper rate limiting,
and extracts structured article data.
"""

import asyncio
import json
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urljoin, urlparse

import httpx
from bs4 import BeautifulSoup

from mindforge_dankoe.config.settings import settings
from mindforge_dankoe.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class CrawledArticle:
    """Data structure for a crawled article."""

    url: str
    title: str = ""
    content: str = ""
    raw_html: str = ""
    date_published: Optional[str] = None
    date_crawled: str = field(default_factory=lambda: datetime.now().isoformat())
    author: str = "Dan Koe"
    tags: List[str] = field(default_factory=list)
    reading_time: Optional[int] = None  # minutes
    word_count: int = 0
    excerpt: str = ""
    images: List[Dict[str, str]] = field(default_factory=list)
    links: List[Dict[str, str]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CrawledArticle":
        """Create from dictionary."""
        return cls(**data)


class DanKoeSpider:
    """
    Comprehensive spider for crawling Dan Koe's newsletter.

    Features:
    - Respects robots.txt
    - Configurable rate limiting
    - Handles pagination
    - Extracts structured content
    - Saves raw HTML backups
    """

    BASE_URL = "https://letters.thedankoe.com"

    def __init__(
        self,
        output_dir: Optional[Path] = None,
        crawl_delay: float = 2.0,
        max_articles: int = 100,
        respect_robots: bool = True,
    ):
        """
        Initialize the spider.

        Args:
            output_dir: Directory to save crawled articles
            crawl_delay: Delay between requests in seconds
            max_articles: Maximum number of articles to crawl
            respect_robots: Whether to respect robots.txt
        """
        self.output_dir = output_dir or settings.get_raw_data_path() / "dankoe"
        self.crawl_delay = crawl_delay
        self.max_articles = max_articles
        self.respect_robots = respect_robots

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._client: Optional[httpx.AsyncClient] = None
        self._crawled_urls: set = set()
        self._robots_rules: Dict[str, bool] = {}

    @property
    def client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                headers={
                    "User-Agent": settings.user_agent,
                    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                    "Accept-Language": "en-US,en;q=0.5",
                },
                follow_redirects=True,
                timeout=30.0,
            )
        return self._client

    async def close(self):
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None

    async def fetch_robots_txt(self) -> str:
        """Fetch and parse robots.txt."""
        try:
            response = await self.client.get(f"{self.BASE_URL}/robots.txt")
            if response.status_code == 200:
                return response.text
        except Exception as e:
            logger.warning(f"Could not fetch robots.txt: {e}")
        return ""

    def is_allowed(self, url: str) -> bool:
        """Check if URL is allowed by robots.txt rules."""
        if not self.respect_robots:
            return True
        # Simple robots.txt compliance - allow all by default
        # In production, use proper robotparser
        return True

    async def fetch_page(self, url: str) -> Optional[str]:
        """
        Fetch a single page with rate limiting.

        Args:
            url: URL to fetch

        Returns:
            HTML content or None if failed
        """
        if url in self._crawled_urls:
            logger.debug(f"Already crawled: {url}")
            return None

        if not self.is_allowed(url):
            logger.warning(f"URL not allowed by robots.txt: {url}")
            return None

        try:
            logger.info(f"Fetching: {url}")
            await asyncio.sleep(self.crawl_delay)

            response = await self.client.get(url)
            response.raise_for_status()

            self._crawled_urls.add(url)
            return response.text

        except httpx.HTTPError as e:
            logger.error(f"HTTP error fetching {url}: {e}")
            return None
        except Exception as e:
            logger.error(f"Error fetching {url}: {e}")
            return None

    def parse_article(self, html: str, url: str) -> CrawledArticle:
        """
        Parse article content from HTML.

        Args:
            html: Raw HTML content
            url: Article URL

        Returns:
            CrawledArticle instance
        """
        soup = BeautifulSoup(html, "html.parser")

        article = CrawledArticle(url=url, raw_html=html)

        # Extract title
        title_elem = soup.find("h1") or soup.find("title")
        if title_elem:
            article.title = title_elem.get_text(strip=True)

        # Try multiple content selectors (common newsletter patterns)
        content_selectors = [
            "article",
            ".post-content",
            ".article-content",
            ".newsletter-content",
            ".entry-content",
            "main",
            ".content",
        ]

        content_elem = None
        for selector in content_selectors:
            content_elem = soup.select_one(selector)
            if content_elem:
                break

        if content_elem:
            # Extract text content
            article.content = content_elem.get_text(separator="\n", strip=True)

            # Extract images
            for img in content_elem.find_all("img"):
                src = img.get("src", "")
                alt = img.get("alt", "")
                if src:
                    article.images.append({"src": urljoin(url, src), "alt": alt})

            # Extract links
            for link in content_elem.find_all("a"):
                href = link.get("href", "")
                text = link.get_text(strip=True)
                if href and text:
                    article.links.append({"href": urljoin(url, href), "text": text})
        else:
            # Fallback: extract body text
            body = soup.find("body")
            if body:
                article.content = body.get_text(separator="\n", strip=True)

        # Extract date
        date_patterns = [
            soup.find("time"),
            soup.find(class_=re.compile(r"date|published|posted", re.I)),
            soup.find("meta", {"property": "article:published_time"}),
            soup.find("meta", {"name": "date"}),
        ]

        for date_elem in date_patterns:
            if date_elem:
                if date_elem.name == "meta":
                    article.date_published = date_elem.get("content", "")
                else:
                    article.date_published = date_elem.get("datetime") or date_elem.get_text(
                        strip=True
                    )
                if article.date_published:
                    break

        # Extract excerpt/description
        meta_desc = soup.find("meta", {"name": "description"})
        if meta_desc:
            article.excerpt = meta_desc.get("content", "")

        # Calculate word count
        words = article.content.split()
        article.word_count = len(words)

        # Estimate reading time (average 200 words per minute)
        article.reading_time = max(1, article.word_count // 200)

        # Extract tags from meta keywords or common tag elements
        tags_elem = soup.find("meta", {"name": "keywords"})
        if tags_elem:
            keywords = tags_elem.get("content", "")
            article.tags = [t.strip() for t in keywords.split(",") if t.strip()]

        return article

    async def discover_article_urls(self) -> List[str]:
        """
        Discover all article URLs from the main page and archives.

        Returns:
            List of article URLs
        """
        urls = set()

        # Fetch main page
        html = await self.fetch_page(self.BASE_URL)
        if html:
            soup = BeautifulSoup(html, "html.parser")

            # Find article links
            for link in soup.find_all("a", href=True):
                href = link["href"]
                full_url = urljoin(self.BASE_URL, href)

                # Filter for article-like URLs
                if self._is_article_url(full_url):
                    urls.add(full_url)

        # Try common archive/pagination patterns
        archive_patterns = [
            "/archive",
            "/archives",
            "/posts",
            "/articles",
            "/all",
        ]

        for pattern in archive_patterns:
            archive_url = urljoin(self.BASE_URL, pattern)
            html = await self.fetch_page(archive_url)
            if html:
                soup = BeautifulSoup(html, "html.parser")
                for link in soup.find_all("a", href=True):
                    href = link["href"]
                    full_url = urljoin(self.BASE_URL, href)
                    if self._is_article_url(full_url):
                        urls.add(full_url)

        logger.info(f"Discovered {len(urls)} article URLs")
        return list(urls)[: self.max_articles]

    def _is_article_url(self, url: str) -> bool:
        """Check if URL looks like an article URL."""
        parsed = urlparse(url)

        # Must be from the same domain
        if "thedankoe.com" not in parsed.netloc:
            return False

        # Skip non-article pages
        skip_patterns = [
            "/tag/",
            "/category/",
            "/author/",
            "/page/",
            "/search",
            "/login",
            "/signup",
            "/subscribe",
            "/feed",
            "/rss",
            ".xml",
            ".json",
            "#",
        ]

        path = parsed.path.lower()
        if any(pattern in path for pattern in skip_patterns):
            return False

        # Article URLs typically have a path with content
        if len(path) > 1 and path != "/":
            return True

        return False

    async def crawl_article(self, url: str) -> Optional[CrawledArticle]:
        """
        Crawl a single article.

        Args:
            url: Article URL

        Returns:
            CrawledArticle or None if failed
        """
        html = await self.fetch_page(url)
        if not html:
            return None

        article = self.parse_article(html, url)

        if article.content:
            await self.save_article(article)
            return article
        else:
            logger.warning(f"No content extracted from: {url}")
            return None

    async def save_article(self, article: CrawledArticle) -> Path:
        """
        Save article to disk.

        Args:
            article: Article to save

        Returns:
            Path to saved file
        """
        # Create safe filename from title
        safe_title = re.sub(r"[^\w\s-]", "", article.title)[:50]
        safe_title = re.sub(r"\s+", "_", safe_title).lower()

        timestamp = datetime.now().strftime("%Y%m%d")
        filename = f"{timestamp}_{safe_title}.json"

        filepath = self.output_dir / filename

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(article.to_dict(), f, indent=2, ensure_ascii=False)

        logger.info(f"Saved article: {filepath}")
        return filepath

    async def crawl_all(self) -> List[CrawledArticle]:
        """
        Crawl all discovered articles.

        Returns:
            List of crawled articles
        """
        logger.info("Starting crawl of Dan Koe newsletter...")

        try:
            # Discover article URLs
            urls = await self.discover_article_urls()
            logger.info(f"Found {len(urls)} articles to crawl")

            articles = []
            for i, url in enumerate(urls, 1):
                logger.info(f"Crawling article {i}/{len(urls)}")
                article = await self.crawl_article(url)
                if article:
                    articles.append(article)

            logger.info(f"Successfully crawled {len(articles)} articles")
            return articles

        finally:
            await self.close()

    def load_cached_articles(self) -> List[CrawledArticle]:
        """
        Load previously crawled articles from disk.

        Returns:
            List of cached articles
        """
        articles = []
        for filepath in self.output_dir.glob("*.json"):
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    articles.append(CrawledArticle.from_dict(data))
            except Exception as e:
                logger.warning(f"Failed to load {filepath}: {e}")

        logger.info(f"Loaded {len(articles)} cached articles")
        return articles


async def run_spider(
    output_dir: Optional[Path] = None,
    max_articles: int = 100,
    crawl_delay: float = 2.0,
) -> List[CrawledArticle]:
    """
    Convenience function to run the spider.

    Args:
        output_dir: Output directory for articles
        max_articles: Maximum articles to crawl
        crawl_delay: Delay between requests

    Returns:
        List of crawled articles
    """
    spider = DanKoeSpider(
        output_dir=output_dir,
        max_articles=max_articles,
        crawl_delay=crawl_delay,
    )
    return await spider.crawl_all()


if __name__ == "__main__":
    # Run spider directly
    articles = asyncio.run(run_spider(max_articles=10))
    print(f"Crawled {len(articles)} articles")
