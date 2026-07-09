"""Web search without APIs - DuckDuckGo and other free sources."""

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional
from urllib.parse import quote_plus

import aiohttp
from aiohttp import ClientTimeout
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """A single search result."""
    title: str
    url: str
    snippet: str
    source: str
    published_date: Optional[datetime] = None
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


class WebSearch:
    """
    Web search without paid APIs.
    Uses DuckDuckGo HTML search and other free sources.
    """

    def __init__(self, timeout: int = 30):
        self.timeout = ClientTimeout(total=timeout)
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
        }

    async def search_duckduckgo(
        self,
        query: str,
        max_results: int = 10,
        recency_days: int = 30
    ) -> List[SearchResult]:
        """
        Search using DuckDuckGo HTML (no API needed).

        Args:
            query: Search query
            max_results: Maximum results to return
            recency_days: Filter results by recency. <=1 = past day,
                <=7 = past week, <=30 = past month, >30 = no filter.
        """
        url = f"https://html.duckduckgo.com/html/?q={quote_plus(query)}"

        # Apply date filter via DuckDuckGo's &df= parameter
        if recency_days <= 1:
            url += "&df=d"
        elif recency_days <= 7:
            url += "&df=w"
        elif recency_days <= 30:
            url += "&df=m"

        try:
            async with aiohttp.ClientSession(timeout=self.timeout) as session:
                async with session.get(url, headers=self.headers) as response:
                    if response.status != 200:
                        return []

                    html = await response.text()
                    soup = BeautifulSoup(html, "html.parser")

                    results = []
                    for result in soup.select(".result"):
                        title_elem = result.select_one(".result__title")
                        snippet_elem = result.select_one(".result__snippet")
                        link_elem = result.select_one(".result__url")

                        if title_elem and snippet_elem:
                            # Extract URL from the link
                            url_text = ""
                            if link_elem:
                                url_text = link_elem.get_text(strip=True)
                                if not url_text.startswith("http"):
                                    url_text = "https://" + url_text

                            results.append(SearchResult(
                                title=title_elem.get_text(strip=True),
                                url=url_text,
                                snippet=snippet_elem.get_text(strip=True),
                                source="duckduckgo"
                            ))

                            if len(results) >= max_results:
                                break

                    return results

        except Exception as e:
            logger.warning("DuckDuckGo search error: %s", e)
            return []

    async def search_for_problems(
        self,
        domain: str,
        keywords: List[str] = None
    ) -> List[SearchResult]:
        """
        Search for user problems and pain points.

        Args:
            domain: Domain to search (e.g., "programming", "small business")
            keywords: Additional keywords
        """
        pain_keywords = [
            "frustrated with", "hate when", "wish there was",
            "struggling with", "annoying", "problem with"
        ]

        base_keywords = keywords or []
        all_results = []

        # Search with different pain point keywords
        for pain_kw in pain_keywords[:3]:
            query = f"{domain} {pain_kw} {' '.join(base_keywords)}"
            results = await self.search_duckduckgo(query, max_results=5)
            all_results.extend(results)

            # Rate limiting
            await asyncio.sleep(1)

        # Deduplicate by URL
        seen_urls = set()
        unique_results = []
        for r in all_results:
            if r.url not in seen_urls:
                seen_urls.add(r.url)
                unique_results.append(r)

        return unique_results

    async def search_product_hunt(self, category: str = "") -> List[Dict[str, Any]]:
        """
        Scrape ProductHunt for product discussions.
        Note: Basic scraping, may need adjustment if site changes.
        """
        url = "https://www.producthunt.com/topics/developer-tools"
        if category:
            url = f"https://www.producthunt.com/topics/{category}"

        try:
            async with aiohttp.ClientSession(timeout=self.timeout) as session:
                async with session.get(url, headers=self.headers) as response:
                    if response.status != 200:
                        return []

                    html = await response.text()
                    soup = BeautifulSoup(html, "html.parser")

                    products = []
                    for item in soup.select("[data-test='product-item']")[:10]:
                        title = item.select_one("h3")
                        tagline = item.select_one("p")

                        if title:
                            products.append({
                                "title": title.get_text(strip=True),
                                "tagline": tagline.get_text(strip=True) if tagline else "",
                                "source": "producthunt"
                            })

                    return products

        except Exception as e:
            logger.warning("ProductHunt scrape error: %s", e)
            return []

    async def search_indie_hackers(self, query: str = "") -> List[Dict[str, Any]]:
        """
        Search IndieHackers discussions.
        """
        url = "https://www.indiehackers.com/posts"
        if query:
            url = f"https://www.indiehackers.com/search?q={quote_plus(query)}"

        try:
            async with aiohttp.ClientSession(timeout=self.timeout) as session:
                async with session.get(url, headers=self.headers) as response:
                    if response.status != 200:
                        return []

                    html = await response.text()
                    soup = BeautifulSoup(html, "html.parser")

                    posts = []
                    for item in soup.select(".feed-item, .post-preview")[:15]:
                        title_elem = item.select_one("h2, .post-title")
                        content_elem = item.select_one("p, .post-body")

                        if title_elem:
                            posts.append({
                                "title": title_elem.get_text(strip=True),
                                "content": content_elem.get_text(strip=True) if content_elem else "",
                                "source": "indiehackers"
                            })

                    return posts

        except Exception as e:
            logger.warning("IndieHackers scrape error: %s", e)
            return []

    async def search_stackoverflow_questions(
        self,
        query: str,
        tag: str = ""
    ) -> List[Dict[str, Any]]:
        """
        Search StackOverflow for questions (problems people have).
        Uses the public questions page, no API needed.
        """
        url = f"https://stackoverflow.com/search?q={quote_plus(query)}"
        if tag:
            url += f"+[{tag}]"

        try:
            async with aiohttp.ClientSession(timeout=self.timeout) as session:
                async with session.get(url, headers=self.headers) as response:
                    if response.status != 200:
                        return []

                    html = await response.text()
                    soup = BeautifulSoup(html, "html.parser")

                    questions = []
                    for item in soup.select(".s-post-summary")[:15]:
                        title_elem = item.select_one(".s-post-summary--content-title a")
                        excerpt_elem = item.select_one(".s-post-summary--content-excerpt")
                        votes_elem = item.select_one(".s-post-summary--stats-item-number")

                        if title_elem:
                            questions.append({
                                "title": title_elem.get_text(strip=True),
                                "excerpt": excerpt_elem.get_text(strip=True) if excerpt_elem else "",
                                "votes": votes_elem.get_text(strip=True) if votes_elem else "0",
                                "url": "https://stackoverflow.com" + title_elem.get("href", ""),
                                "source": "stackoverflow"
                            })

                    return questions

        except Exception as e:
            logger.warning("StackOverflow search error: %s", e)
            return []

    async def search_github_issues(
        self,
        query: str,
        language: str = ""
    ) -> List[Dict[str, Any]]:
        """
        Search GitHub issues for problems/bugs people face.
        """
        search_query = f"{query} is:issue is:open"
        if language:
            search_query += f" language:{language}"

        url = f"https://github.com/search?q={quote_plus(search_query)}&type=issues"

        try:
            async with aiohttp.ClientSession(timeout=self.timeout) as session:
                async with session.get(url, headers=self.headers) as response:
                    if response.status != 200:
                        return []

                    html = await response.text()
                    soup = BeautifulSoup(html, "html.parser")

                    issues = []
                    for item in soup.select(".search-title a")[:15]:
                        title = item.get_text(strip=True)
                        href = item.get("href", "")

                        if title:
                            issues.append({
                                "title": title,
                                "url": "https://github.com" + href if href.startswith("/") else href,
                                "source": "github"
                            })

                    return issues

        except Exception as e:
            logger.warning("GitHub search error: %s", e)
            return []

    async def comprehensive_search(
        self,
        query: str,
        sources: List[str] = None
    ) -> Dict[str, List[Any]]:
        """
        Search across multiple sources.

        Args:
            query: Search query
            sources: List of sources to search. Options:
                     ["duckduckgo", "stackoverflow", "github", "indiehackers"]
        """
        sources = sources or ["duckduckgo", "stackoverflow"]
        results = {}

        tasks = []

        if "duckduckgo" in sources:
            tasks.append(("duckduckgo", self.search_duckduckgo(query)))

        if "stackoverflow" in sources:
            tasks.append(("stackoverflow", self.search_stackoverflow_questions(query)))

        if "github" in sources:
            tasks.append(("github", self.search_github_issues(query)))

        if "indiehackers" in sources:
            tasks.append(("indiehackers", self.search_indie_hackers(query)))

        # Run searches in parallel
        for source_name, task in tasks:
            try:
                results[source_name] = await task
            except Exception as e:
                logger.warning("Error searching %s: %s", source_name, e)
                results[source_name] = []

            # Small delay between sources
            await asyncio.sleep(0.5)

        return results


    async def search_github_repos(
        self,
        query: str,
        language: str = "",
        min_stars: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Search GitHub repositories by stars.

        Args:
            query: Search query
            language: Optional language filter
            min_stars: Minimum star count to include
        """
        search_query = f"{query} stars:>={min_stars}"
        if language:
            search_query += f" language:{language}"

        url = f"https://github.com/search?q={quote_plus(search_query)}&type=repositories&s=stars"

        try:
            async with aiohttp.ClientSession(timeout=self.timeout) as session:
                async with session.get(url, headers=self.headers) as response:
                    if response.status != 200:
                        return []

                    html = await response.text()
                    soup = BeautifulSoup(html, "html.parser")

                    repos = []
                    for item in soup.select("[data-testid='results-list'] > div, .repo-list-item, .search-title"):
                        # Try multiple selector patterns for repo info
                        name_elem = item.select_one("a.v-align-middle, .search-title a, a[href*='/']")
                        desc_elem = item.select_one("p, .mb-1, .search-match")
                        stars_elem = item.select_one("a[href*='stargazers'], .stars, [aria-label*='star']")

                        if name_elem:
                            name = name_elem.get_text(strip=True)
                            href = name_elem.get("href", "")
                            description = desc_elem.get_text(strip=True) if desc_elem else ""

                            # Parse star count
                            stars = 0
                            if stars_elem:
                                stars_text = stars_elem.get_text(strip=True).replace(",", "")
                                try:
                                    if "k" in stars_text.lower():
                                        stars = int(float(stars_text.lower().replace("k", "")) * 1000)
                                    else:
                                        stars = int(stars_text)
                                except (ValueError, TypeError):
                                    pass

                            repo_url = "https://github.com" + href if href.startswith("/") else href

                            repos.append({
                                "name": name,
                                "stars": stars,
                                "description": description,
                                "url": repo_url,
                            })

                            if len(repos) >= 10:
                                break

                    return repos

        except Exception as e:
            logger.warning("GitHub repo search error: %s", e)
            return []

    async def search_pypi(
        self,
        query: str,
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Search PyPI for packages.

        Args:
            query: Search query
            limit: Maximum results to return
        """
        url = f"https://pypi.org/search/?q={quote_plus(query)}"

        try:
            async with aiohttp.ClientSession(timeout=self.timeout) as session:
                async with session.get(url, headers=self.headers) as response:
                    if response.status != 200:
                        return []

                    html = await response.text()
                    soup = BeautifulSoup(html, "html.parser")

                    packages = []
                    for item in soup.select(".package-snippet")[:limit]:
                        name_elem = item.select_one(".package-snippet__name")
                        desc_elem = item.select_one(".package-snippet__description")
                        version_elem = item.select_one(".package-snippet__version")

                        if name_elem:
                            name = name_elem.get_text(strip=True)
                            packages.append({
                                "name": name,
                                "description": desc_elem.get_text(strip=True) if desc_elem else "",
                                "version": version_elem.get_text(strip=True) if version_elem else "",
                                "url": f"https://pypi.org/project/{name}/",
                            })

                    return packages

        except Exception as e:
            logger.warning("PyPI search error: %s", e)
            return []


# Convenience function
async def quick_search(query: str, max_results: int = 10) -> List[SearchResult]:
    """Quick DuckDuckGo search."""
    searcher = WebSearch()
    return await searcher.search_duckduckgo(query, max_results)
