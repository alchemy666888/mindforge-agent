"""
Research Engine - Web search and research synthesis for AI topics.

Implements:
- Topic-specific search strategies
- Credibility-weighted scraping
- Research synthesis like Dan Koe
"""

import asyncio
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import httpx
from bs4 import BeautifulSoup

from mindforge_dankoe.config.settings import settings
from mindforge_dankoe.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class SearchResult:
    """A single search result."""

    title: str
    url: str
    snippet: str
    source_domain: str = ""
    credibility_score: float = 0.5
    relevance_score: float = 0.5


@dataclass
class ScrapedContent:
    """Scraped content from a URL."""

    url: str
    title: str
    content: str
    date_published: Optional[str] = None
    author: Optional[str] = None
    source_credibility: float = 0.5
    word_count: int = 0


@dataclass
class ResearchResult:
    """Complete research result for a topic."""

    topic: str
    search_queries: List[str] = field(default_factory=list)
    sources: List[ScrapedContent] = field(default_factory=list)
    key_findings: List[str] = field(default_factory=list)
    conflicting_perspectives: List[Dict[str, str]] = field(default_factory=list)
    unanswered_questions: List[str] = field(default_factory=list)
    actionable_takeaways: List[str] = field(default_factory=list)
    synthesis: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "topic": self.topic,
            "search_queries": self.search_queries,
            "sources": [
                {
                    "url": s.url,
                    "title": s.title,
                    "content": s.content[:500],
                    "credibility": s.source_credibility,
                }
                for s in self.sources
            ],
            "key_findings": self.key_findings,
            "conflicting_perspectives": self.conflicting_perspectives,
            "unanswered_questions": self.unanswered_questions,
            "actionable_takeaways": self.actionable_takeaways,
            "synthesis": self.synthesis,
            "timestamp": self.timestamp,
        }


class ResearchEngine:
    """
    Web research engine for AI and tech topics.

    Features:
    - Topic-specific search strategy generation
    - Credibility-weighted source selection
    - Content extraction and synthesis
    - Dan Koe-style research approach
    """

    # High-credibility sources for tech/AI content
    PRIORITY_SOURCES = {
        # Research and academic
        "arxiv.org": 0.95,
        "nature.com": 0.95,
        "science.org": 0.95,
        "ieee.org": 0.90,
        # Tech companies
        "openai.com": 0.90,
        "deepmind.google": 0.90,
        "anthropic.com": 0.90,
        "ai.meta.com": 0.85,
        "microsoft.com": 0.85,
        "google.com": 0.85,
        # Tech media
        "techcrunch.com": 0.75,
        "wired.com": 0.75,
        "arstechnica.com": 0.75,
        "theverge.com": 0.70,
        "venturebeat.com": 0.70,
        # Chinese tech sources
        "36kr.com": 0.75,
        "pingwest.com": 0.70,
        "zhihu.com": 0.65,
        "36kr.com": 0.70,
    }

    # Low credibility patterns to avoid
    LOW_CREDIBILITY_PATTERNS = [
        "clickbait",
        "affiliate",
        "sponsored",
        "ad",
        "promo",
    ]

    def __init__(self, llm_client=None):
        """
        Initialize the research engine.

        Args:
            llm_client: Optional LLM client for synthesis
        """
        self.llm_client = llm_client
        self._http_client: Optional[httpx.AsyncClient] = None

    @property
    def http_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._http_client is None:
            self._http_client = httpx.AsyncClient(
                headers={
                    "User-Agent": settings.user_agent,
                    "Accept": "text/html,application/xhtml+xml",
                },
                follow_redirects=True,
                timeout=30.0,
            )
        return self._http_client

    async def close(self):
        """Close HTTP client."""
        if self._http_client:
            await self._http_client.aclose()
            self._http_client = None

    def generate_search_queries(
        self, topic: str, thinking_style: Optional[Dict[str, Any]] = None
    ) -> List[str]:
        """
        Generate Dan Koe-style search queries for a topic.

        Args:
            topic: Main topic to research
            thinking_style: Optional thinking style preferences

        Returns:
            List of search queries
        """
        queries = []

        # Core topic search
        queries.append(f"{topic} explained")
        queries.append(f"{topic} how it works")

        # Impact analysis style (Dan Koe pattern)
        queries.append(f"{topic} real-world applications case studies")
        queries.append(f"{topic} industry adoption {datetime.now().year}")
        queries.append(f"{topic} impact on business")

        # Future projection style
        queries.append(f"{topic} future predictions expert opinions")
        queries.append(f"{topic} trends {datetime.now().year}")
        queries.append(f"{topic} what's next")

        # Practical guide style
        queries.append(f"{topic} how to get started tutorial")
        queries.append(f"{topic} best practices")
        queries.append(f"{topic} tools and frameworks {datetime.now().year}")

        # Contrarian/critical views
        queries.append(f"{topic} criticism problems")
        queries.append(f"{topic} challenges limitations")
        queries.append(f"{topic} debate controversy")

        # Chinese language searches for bilingual output
        queries.append(f"{topic} 人工智能 应用 案例")
        queries.append(f"{topic} 发展 趋势 {datetime.now().year}")
        queries.append(f"{topic} 中国 应用")

        return queries

    def get_credibility_score(self, url: str) -> float:
        """
        Calculate credibility score for a URL.

        Args:
            url: URL to score

        Returns:
            Credibility score (0-1)
        """
        parsed = urlparse(url)
        domain = parsed.netloc.lower()

        # Remove www prefix
        if domain.startswith("www."):
            domain = domain[4:]

        # Check priority sources
        for source, score in self.PRIORITY_SOURCES.items():
            if source in domain:
                return score

        # Check for low credibility patterns
        url_lower = url.lower()
        for pattern in self.LOW_CREDIBILITY_PATTERNS:
            if pattern in url_lower:
                return 0.3

        # Default credibility
        return 0.5

    async def search_serper(self, query: str, num_results: int = 10) -> List[SearchResult]:
        """
        Search using Serper API.

        Args:
            query: Search query
            num_results: Number of results to return

        Returns:
            List of search results
        """
        if not settings.serper_api_key:
            logger.warning("Serper API key not configured")
            return []

        try:
            response = await self.http_client.post(
                "https://google.serper.dev/search",
                headers={"X-API-KEY": settings.serper_api_key},
                json={"q": query, "num": num_results},
            )
            response.raise_for_status()
            data = response.json()

            results = []
            for item in data.get("organic", []):
                result = SearchResult(
                    title=item.get("title", ""),
                    url=item.get("link", ""),
                    snippet=item.get("snippet", ""),
                    source_domain=urlparse(item.get("link", "")).netloc,
                )
                result.credibility_score = self.get_credibility_score(result.url)
                results.append(result)

            return results

        except Exception as e:
            logger.error(f"Serper search error: {e}")
            return []

    async def search_tavily(self, query: str, num_results: int = 10) -> List[SearchResult]:
        """
        Search using Tavily API.

        Args:
            query: Search query
            num_results: Number of results to return

        Returns:
            List of search results
        """
        if not settings.tavily_api_key:
            logger.warning("Tavily API key not configured")
            return []

        try:
            response = await self.http_client.post(
                "https://api.tavily.com/search",
                json={
                    "api_key": settings.tavily_api_key,
                    "query": query,
                    "max_results": num_results,
                    "search_depth": "advanced",
                },
            )
            response.raise_for_status()
            data = response.json()

            results = []
            for item in data.get("results", []):
                result = SearchResult(
                    title=item.get("title", ""),
                    url=item.get("url", ""),
                    snippet=item.get("content", ""),
                    source_domain=urlparse(item.get("url", "")).netloc,
                )
                result.credibility_score = self.get_credibility_score(result.url)
                results.append(result)

            return results

        except Exception as e:
            logger.error(f"Tavily search error: {e}")
            return []

    async def search(self, query: str, num_results: int = 10) -> List[SearchResult]:
        """
        Search using available search API.

        Args:
            query: Search query
            num_results: Number of results

        Returns:
            List of search results
        """
        # Try Serper first
        results = await self.search_serper(query, num_results)
        if results:
            return results

        # Try Tavily
        results = await self.search_tavily(query, num_results)
        if results:
            return results

        logger.warning("No search API available")
        return []

    async def scrape_content(self, url: str) -> Optional[ScrapedContent]:
        """
        Scrape content from a URL.

        Args:
            url: URL to scrape

        Returns:
            ScrapedContent or None if failed
        """
        try:
            logger.debug(f"Scraping: {url}")
            await asyncio.sleep(1)  # Rate limiting

            response = await self.http_client.get(url)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, "html.parser")

            # Remove unwanted elements
            for element in soup(["script", "style", "nav", "footer", "header", "aside"]):
                element.decompose()

            # Extract title
            title = ""
            title_elem = soup.find("h1") or soup.find("title")
            if title_elem:
                title = title_elem.get_text(strip=True)

            # Extract content
            content_selectors = ["article", "main", ".content", ".post-content", ".entry-content"]
            content = ""

            for selector in content_selectors:
                content_elem = soup.select_one(selector)
                if content_elem:
                    content = content_elem.get_text(separator="\n", strip=True)
                    break

            if not content:
                body = soup.find("body")
                if body:
                    content = body.get_text(separator="\n", strip=True)

            # Clean content
            content = re.sub(r"\n{3,}", "\n\n", content)
            content = re.sub(r"[ \t]+", " ", content)

            return ScrapedContent(
                url=url,
                title=title,
                content=content[:10000],  # Limit content length
                source_credibility=self.get_credibility_score(url),
                word_count=len(content.split()),
            )

        except Exception as e:
            logger.error(f"Scraping error for {url}: {e}")
            return None

    async def research_topic(
        self,
        topic: str,
        max_sources: int = 10,
        thinking_style: Optional[Dict[str, Any]] = None,
    ) -> ResearchResult:
        """
        Conduct comprehensive research on a topic.

        Args:
            topic: Topic to research
            max_sources: Maximum number of sources to use
            thinking_style: Optional Dan Koe thinking style

        Returns:
            ResearchResult with synthesized findings
        """
        logger.info(f"Researching topic: {topic}")

        result = ResearchResult(topic=topic)

        # Generate search queries
        queries = self.generate_search_queries(topic, thinking_style)
        result.search_queries = queries[:5]  # Use top 5 queries

        # Collect search results
        all_search_results = []
        for query in result.search_queries:
            search_results = await self.search(query, num_results=5)
            all_search_results.extend(search_results)

        # Deduplicate by URL
        seen_urls = set()
        unique_results = []
        for sr in all_search_results:
            if sr.url not in seen_urls:
                seen_urls.add(sr.url)
                unique_results.append(sr)

        # Sort by credibility and take top sources
        unique_results.sort(key=lambda x: x.credibility_score, reverse=True)
        top_results = unique_results[:max_sources]

        # Scrape content from top sources
        for sr in top_results:
            content = await self.scrape_content(sr.url)
            if content and content.word_count > 100:
                result.sources.append(content)

        # Synthesize findings
        if result.sources:
            result = await self._synthesize_research(result)

        logger.info(f"Research complete: {len(result.sources)} sources, {len(result.key_findings)} findings")
        return result

    async def _synthesize_research(self, result: ResearchResult) -> ResearchResult:
        """
        Synthesize research findings using LLM.

        Args:
            result: ResearchResult with raw sources

        Returns:
            ResearchResult with synthesized findings
        """
        try:
            import litellm

            # Prepare source summaries
            source_summaries = []
            for source in result.sources[:5]:  # Use top 5 sources
                summary = f"Source: {source.title}\nContent: {source.content[:1000]}"
                source_summaries.append(summary)

            source_text = "\n\n---\n\n".join(source_summaries)

            prompt = f"""Analyze these research sources about "{result.topic}" and provide:

1. KEY FINDINGS: List 5-7 key findings from the research
2. CONFLICTING PERSPECTIVES: Identify 2-3 areas where sources disagree
3. UNANSWERED QUESTIONS: List 2-3 questions that remain unanswered
4. ACTIONABLE TAKEAWAYS: List 3-5 actionable insights for readers

Sources:
{source_text}

Respond in this exact format:
KEY FINDINGS:
- [finding 1]
- [finding 2]
...

CONFLICTING PERSPECTIVES:
- [perspective 1]
- [perspective 2]
...

UNANSWERED QUESTIONS:
- [question 1]
- [question 2]
...

ACTIONABLE TAKEAWAYS:
- [takeaway 1]
- [takeaway 2]
..."""

            model = settings.get_available_llm_model()
            response = litellm.completion(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=2000,
            )

            output = response.choices[0].message.content

            # Parse the structured output
            result.key_findings = self._extract_list_items(output, "KEY FINDINGS")
            result.actionable_takeaways = self._extract_list_items(output, "ACTIONABLE TAKEAWAYS")
            result.unanswered_questions = self._extract_list_items(output, "UNANSWERED QUESTIONS")

            # Parse conflicting perspectives
            perspectives_text = self._extract_section(output, "CONFLICTING PERSPECTIVES")
            if perspectives_text:
                for item in perspectives_text:
                    result.conflicting_perspectives.append({"perspective": item})

            result.synthesis = output

        except Exception as e:
            logger.error(f"Synthesis error: {e}")
            # Fallback: extract key points from content
            result.key_findings = [s.title for s in result.sources[:5]]

        return result

    def _extract_list_items(self, text: str, section_name: str) -> List[str]:
        """Extract list items from a section."""
        section_text = self._extract_section(text, section_name)
        return section_text

    def _extract_section(self, text: str, section_name: str) -> List[str]:
        """Extract items from a named section."""
        items = []
        lines = text.split("\n")
        in_section = False

        for line in lines:
            if section_name.upper() in line.upper():
                in_section = True
                continue
            if in_section:
                if line.strip().startswith("-") or line.strip().startswith("•"):
                    items.append(line.strip().lstrip("-•").strip())
                elif line.strip() and ":" in line and line.strip()[0].isupper():
                    # New section started
                    break

        return items


async def research_ai_topic(
    topic: str, max_sources: int = 10
) -> ResearchResult:
    """
    Convenience function to research an AI topic.

    Args:
        topic: Topic to research
        max_sources: Maximum sources to use

    Returns:
        ResearchResult
    """
    engine = ResearchEngine()
    try:
        return await engine.research_topic(topic, max_sources)
    finally:
        await engine.close()
