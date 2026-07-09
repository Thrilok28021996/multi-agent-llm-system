"""Web scraper for fetching content from various sources."""

import asyncio
import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

import aiohttp
from aiohttp import ClientTimeout
from bs4 import BeautifulSoup

from utils.rate_limiter import RateLimiter, get_global_rate_limiter

logger = logging.getLogger(__name__)


@dataclass
class ScrapedContent:
    """Content scraped from a source."""

    url: str
    title: str
    content: str
    source_type: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    success: bool = True
    error: Optional[str] = None


class WebScraper:
    """
    Web scraper for fetching content from various sources.
    Supports Reddit API, HTML pages, and JSON APIs.

    Features:
    - Token bucket rate limiting with per-domain configuration
    - Automatic retry with exponential backoff
    - 429 (Too Many Requests) handling
    - Concurrent request limiting
    """

    def __init__(
        self,
        timeout: int = 30,
        max_concurrent: int = 5,
        rate_limiter: Optional[RateLimiter] = None,
        max_retries: int = 3
    ):
        self.timeout = ClientTimeout(total=timeout)
        self.max_concurrent = max_concurrent
        self.max_retries = max_retries
        self.semaphore = asyncio.Semaphore(max_concurrent)

        # Use provided rate limiter or global instance
        self.rate_limiter = rate_limiter or get_global_rate_limiter()

        self.headers = {
            "User-Agent": "CompanyAGI/1.0 (Autonomous Research Agent)",
            "Accept": "text/html,application/json",
        }

    async def _rate_limit(self, domain: str) -> None:
        """Apply rate limiting per domain using token bucket algorithm."""
        await self.rate_limiter.wait_async(domain)

    def _extract_domain(self, url: str) -> str:
        """Extract domain from URL."""
        match = re.search(r"https?://([^/]+)", url)
        return match.group(1) if match else "unknown"

    async def _handle_response_error(
        self,
        domain: str,
        status_code: int,
        headers: Dict
    ) -> Optional[float]:
        """
        Handle HTTP error responses.

        Returns:
            Wait time if should retry, None otherwise
        """
        if status_code == 429:
            # Too Many Requests - respect Retry-After header if present
            retry_after = headers.get("Retry-After")
            if retry_after:
                try:
                    wait_time = int(retry_after)
                except ValueError:
                    wait_time = None
            else:
                wait_time = None

            return self.rate_limiter.record_429(domain, wait_time)

        elif status_code >= 500:
            # Server error - use backoff
            return self.rate_limiter.record_error(domain, status_code)

        return None

    async def fetch_url(self, url: str) -> ScrapedContent:
        """
        Fetch content from a URL with retry logic and rate limiting.

        Args:
            url: URL to fetch

        Returns:
            ScrapedContent with the fetched data
        """
        domain = self._extract_domain(url)
        last_error = None

        for attempt in range(self.max_retries + 1):
            async with self.semaphore:
                await self._rate_limit(domain)

                try:
                    async with aiohttp.ClientSession(timeout=self.timeout) as session:
                        async with session.get(url, headers=self.headers) as response:
                            # Handle error responses
                            if response.status != 200:
                                retry_wait = await self._handle_response_error(
                                    domain, response.status, dict(response.headers)
                                )

                                if retry_wait and attempt < self.max_retries:
                                    logger.debug("Rate limited, waiting %.1fs before retry...", retry_wait)
                                    await asyncio.sleep(retry_wait)
                                    continue

                                return ScrapedContent(
                                    url=url,
                                    title="",
                                    content="",
                                    source_type="error",
                                    success=False,
                                    error=f"HTTP {response.status}"
                                )

                            # Record successful request
                            self.rate_limiter.record_success(domain)

                            content_type = response.headers.get("Content-Type", "")

                            if "application/json" in content_type:
                                data = await response.json()
                                return ScrapedContent(
                                    url=url,
                                    title="JSON Data",
                                    content=json.dumps(data, indent=2),
                                    source_type="json",
                                    metadata={"response_headers": dict(response.headers)}
                                )
                            else:
                                html = await response.text()
                                soup = BeautifulSoup(html, "html.parser")

                                # Extract title
                                title_tag = soup.find("title")
                                title = title_tag.get_text(strip=True) if title_tag else ""

                                # Extract main content
                                content = self._extract_main_content(soup)

                                return ScrapedContent(
                                    url=url,
                                    title=title,
                                    content=content,
                                    source_type="html",
                                    metadata={"response_headers": dict(response.headers)}
                                )

                except asyncio.TimeoutError as e:
                    last_error = "Request timed out"
                    if attempt < self.max_retries:
                        backoff = self.rate_limiter.record_error(domain)
                        logger.debug("Timeout, retrying in %.1fs...", backoff)
                        await asyncio.sleep(backoff)
                        continue

                except aiohttp.ClientError as e:
                    last_error = str(e)
                    if attempt < self.max_retries:
                        backoff = self.rate_limiter.record_error(domain)
                        logger.debug("Client error, retrying in %.1fs...", backoff)
                        await asyncio.sleep(backoff)
                        continue

                except Exception as e:
                    last_error = str(e)
                    # Don't retry on unknown errors
                    break

        # All retries exhausted
        return ScrapedContent(
            url=url,
            title="",
            content="",
            source_type="error",
            success=False,
            error=last_error or "Unknown error"
        )

    def _extract_main_content(self, soup: BeautifulSoup) -> str:
        """Extract main content from HTML."""
        # Remove unwanted elements
        for tag in soup(["script", "style", "nav", "header", "footer", "aside"]):
            tag.decompose()

        # Try to find main content area
        main_content = (
            soup.find("main") or
            soup.find("article") or
            soup.find(class_=re.compile(r"content|post|article", re.I)) or
            soup.find("body")
        )

        if main_content:
            # Get text with some structure preserved
            text = main_content.get_text(separator="\n", strip=True)
            # Clean up multiple newlines
            text = re.sub(r"\n{3,}", "\n\n", text)
            return text

        return soup.get_text(separator="\n", strip=True)

    async def fetch_reddit_subreddit(
        self,
        subreddit: str,
        sort: str = "top",
        time_period: str = "week",
        limit: int = 25
    ) -> List[Dict[str, Any]]:
        """
        Fetch posts from a Reddit subreddit using JSON API.

        Args:
            subreddit: Subreddit name (without r/)
            sort: Sort method (hot, new, top, rising)
            time_period: Time period for top (hour, day, week, month, year, all)
            limit: Number of posts to fetch
        """
        url = f"https://www.reddit.com/r/{subreddit}/{sort}.json?t={time_period}&limit={limit}"

        async with self.semaphore:
            await self._rate_limit("reddit.com")

            try:
                async with aiohttp.ClientSession(timeout=self.timeout) as session:
                    async with session.get(url, headers=self.headers) as response:
                        if response.status != 200:
                            return []

                        data = await response.json()
                        posts = []

                        for child in data.get("data", {}).get("children", []):
                            post = child.get("data", {})
                            posts.append({
                                "id": post.get("id"),
                                "title": post.get("title", ""),
                                "text": post.get("selftext", ""),
                                "score": post.get("score", 0),
                                "comments": post.get("num_comments", 0),
                                "url": post.get("url", ""),
                                "permalink": f"https://reddit.com{post.get('permalink', '')}",
                                "created_utc": post.get("created_utc"),
                                "subreddit": subreddit,
                                "author": post.get("author", ""),
                            })

                        return posts

            except Exception as e:
                logger.warning("Error fetching Reddit: %s", e)
                return []

    async def fetch_reddit_comments(
        self,
        post_id: str,
        subreddit: str,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Fetch comments from a Reddit post.

        Args:
            post_id: Reddit post ID
            subreddit: Subreddit name
            limit: Number of comments to fetch
        """
        url = f"https://www.reddit.com/r/{subreddit}/comments/{post_id}.json?limit={limit}"

        async with self.semaphore:
            await self._rate_limit("reddit.com")

            try:
                async with aiohttp.ClientSession(timeout=self.timeout) as session:
                    async with session.get(url, headers=self.headers) as response:
                        if response.status != 200:
                            return []

                        data = await response.json()
                        comments = []

                        if len(data) > 1:
                            self._extract_comments(
                                data[1].get("data", {}).get("children", []),
                                comments
                            )

                        return comments

            except Exception as e:
                logger.warning("Error fetching Reddit comments: %s", e)
                return []

    def _extract_comments(
        self,
        children: List[Dict],
        comments: List[Dict],
        depth: int = 0
    ) -> None:
        """Recursively extract comments from Reddit response."""
        for child in children:
            if child.get("kind") != "t1":
                continue

            comment_data = child.get("data", {})
            comments.append({
                "id": comment_data.get("id"),
                "body": comment_data.get("body", ""),
                "score": comment_data.get("score", 0),
                "author": comment_data.get("author", ""),
                "depth": depth,
            })

            # Get replies
            replies = comment_data.get("replies")
            if isinstance(replies, dict):
                self._extract_comments(
                    replies.get("data", {}).get("children", []),
                    comments,
                    depth + 1
                )

    async def fetch_hacker_news_top(self, limit: int = 30) -> List[Dict[str, Any]]:
        """Fetch top stories from Hacker News."""
        top_url = "https://hacker-news.firebaseio.com/v0/topstories.json"

        async with self.semaphore:
            try:
                async with aiohttp.ClientSession(timeout=self.timeout) as session:
                    async with session.get(top_url) as response:
                        if response.status != 200:
                            return []

                        story_ids = await response.json()
                        story_ids = story_ids[:limit]

                        # Fetch story details
                        stories = []
                        for story_id in story_ids:
                            story_url = f"https://hacker-news.firebaseio.com/v0/item/{story_id}.json"
                            async with session.get(story_url) as story_response:
                                if story_response.status == 200:
                                    story = await story_response.json()
                                    stories.append({
                                        "id": story.get("id"),
                                        "title": story.get("title", ""),
                                        "url": story.get("url", ""),
                                        "score": story.get("score", 0),
                                        "comments": story.get("descendants", 0),
                                        "author": story.get("by", ""),
                                        "text": story.get("text", ""),
                                        "type": story.get("type", ""),
                                        "time": story.get("time"),
                                    })

                        return stories

            except Exception as e:
                logger.warning("Error fetching HackerNews: %s", e)
                return []

    async def fetch_multiple(self, urls: List[str]) -> List[ScrapedContent]:
        """
        Fetch multiple URLs concurrently.

        Args:
            urls: List of URLs to fetch

        Returns:
            List of ScrapedContent objects
        """
        tasks = [self.fetch_url(url) for url in urls]
        return await asyncio.gather(*tasks)

    async def search_reddit(
        self,
        query: str,
        subreddit: str = None,
        sort: str = "relevance",
        limit: int = 25
    ) -> List[Dict[str, Any]]:
        """
        Search Reddit for posts matching a query.

        Args:
            query: Search query
            subreddit: Optional subreddit to limit search
            sort: Sort method (relevance, hot, top, new, comments)
            limit: Number of results
        """
        if subreddit:
            url = f"https://www.reddit.com/r/{subreddit}/search.json?q={query}&restrict_sr=1&sort={sort}&limit={limit}"
        else:
            url = f"https://www.reddit.com/search.json?q={query}&sort={sort}&limit={limit}"

        async with self.semaphore:
            await self._rate_limit("reddit.com")

            try:
                async with aiohttp.ClientSession(timeout=self.timeout) as session:
                    async with session.get(url, headers=self.headers) as response:
                        if response.status != 200:
                            return []

                        data = await response.json()
                        results = []

                        for child in data.get("data", {}).get("children", []):
                            post = child.get("data", {})
                            results.append({
                                "id": post.get("id"),
                                "title": post.get("title", ""),
                                "text": post.get("selftext", ""),
                                "score": post.get("score", 0),
                                "comments": post.get("num_comments", 0),
                                "subreddit": post.get("subreddit", ""),
                                "url": post.get("url", ""),
                                "permalink": f"https://reddit.com{post.get('permalink', '')}",
                            })

                        return results

            except Exception as e:
                logger.warning("Error searching Reddit: %s", e)
                return []
