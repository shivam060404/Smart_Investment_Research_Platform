import logging
import time
from datetime import datetime, timedelta
from typing import Any, Optional

import httpx
from pydantic import BaseModel, Field

from app.config import get_settings
from app.mcp.base import BaseMCPServer, ContentBlock, ContentType, HealthStatus
from app.mcp.errors import MCPError, MCPErrorCode, MCPInvalidParamsError

logger = logging.getLogger(__name__)
settings = get_settings()


# =============================================================================
# Response Models
# =============================================================================


class NewsArticle(BaseModel):
    """News article data."""

    title: str
    description: Optional[str] = None
    content: Optional[str] = None
    source: str
    url: str
    published_at: str
    ticker: Optional[str] = None


class Headline(BaseModel):
    """News headline (simplified article)."""

    title: str
    source: str
    published_at: str
    url: str


class SentimentScore(BaseModel):
    """Aggregate sentiment score from news articles."""

    ticker: str
    overall_score: float = Field(ge=-1.0, le=1.0)
    positive_pct: float = Field(ge=0.0, le=100.0)
    negative_pct: float = Field(ge=0.0, le=100.0)
    neutral_pct: float = Field(ge=0.0, le=100.0)
    article_count: int
    analyzed_at: datetime = Field(default_factory=datetime.utcnow)


class NewsError(BaseModel):
    """Structured error response for News API failures."""

    error_code: str
    message: str
    ticker: Optional[str] = None
    details: Optional[dict[str, Any]] = None


# =============================================================================
# Cache Implementation
# =============================================================================


class SimpleCache:
    """Simple in-memory cache with TTL support."""

    def __init__(self, ttl_seconds: int = 300):
        """Initialize cache with TTL.

        Args:
            ttl_seconds: Time-to-live for cached items (default 5 minutes).
        """
        self._cache: dict[str, tuple[Any, float]] = {}
        self._ttl_seconds = ttl_seconds

    def get(self, key: str) -> Optional[Any]:
        """Get cached value if not expired."""
        if key in self._cache:
            value, timestamp = self._cache[key]
            if time.time() - timestamp < self._ttl_seconds:
                return value
            del self._cache[key]
        return None

    def set(self, key: str, value: Any) -> None:
        """Set cached value with current timestamp."""
        self._cache[key] = (value, time.time())

    def clear(self) -> None:
        """Clear all cached values."""
        self._cache.clear()


# =============================================================================
# Sentiment Keywords for Basic Analysis
# =============================================================================

POSITIVE_KEYWORDS = [
    "surge", "soar", "jump", "gain", "rise", "rally", "boost", "growth",
    "profit", "beat", "exceed", "outperform", "upgrade", "bullish", "strong",
    "record", "breakthrough", "success", "positive", "optimistic", "expand",
    "innovation", "partnership", "acquisition", "dividend", "buyback"
]

NEGATIVE_KEYWORDS = [
    "fall", "drop", "decline", "plunge", "crash", "loss", "miss", "cut",
    "downgrade", "bearish", "weak", "concern", "risk", "warning", "lawsuit",
    "investigation", "scandal", "layoff", "restructure", "debt", "default",
    "bankruptcy", "recall", "fine", "penalty", "negative", "pessimistic"
]


# =============================================================================
# News MCP Server
# =============================================================================


class NewsMCPServer(BaseMCPServer):
    """MCP Server for News and Sentiment data.

    Provides tools for fetching news articles, headlines, and
    calculating sentiment scores through the MCP protocol.
    """

    def __init__(self, cache_ttl_seconds: int = 300):
        """Initialize the News MCP Server.

        Args:
            cache_ttl_seconds: Cache TTL in seconds (default 5 minutes).
        """
        super().__init__(name="news", version="1.0.0")
        self._cache = SimpleCache(ttl_seconds=cache_ttl_seconds)
        self._http_client: Optional[httpx.AsyncClient] = None
        self._register_tools()
        self._register_resources()

    @property
    def http_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._http_client is None:
            self._http_client = httpx.AsyncClient(timeout=30.0)
        return self._http_client

    async def close(self) -> None:
        """Close HTTP client connections."""
        if self._http_client:
            await self._http_client.aclose()
            self._http_client = None

    def _register_tools(self) -> None:
        """Register all available tools."""
        # News fetching tools
        self.register_tool(
            name="get_news",
            description="Fetch recent news articles for a given stock ticker",
            handler=self._get_news,
            input_schema={
                "type": "object",
                "properties": {
                    "ticker": {
                        "type": "string",
                        "description": "Stock ticker symbol (e.g., 'AAPL', 'MSFT')",
                    },
                    "days": {
                        "type": "integer",
                        "description": "Number of days to look back (default: 7)",
                        "default": 7,
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of articles to return (default: 10)",
                        "default": 10,
                    },
                },
                "required": ["ticker"],
            },
        )

        self.register_tool(
            name="get_headlines",
            description="Fetch news headlines for a ticker with configurable limit",
            handler=self._get_headlines,
            input_schema={
                "type": "object",
                "properties": {
                    "ticker": {
                        "type": "string",
                        "description": "Stock ticker symbol",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of headlines to return (default: 5)",
                        "default": 5,
                    },
                },
                "required": ["ticker"],
            },
        )

        # Sentiment analysis tools
        self.register_tool(
            name="calculate_sentiment",
            description="Calculate aggregate sentiment scores from news articles for a ticker",
            handler=self._calculate_sentiment,
            input_schema={
                "type": "object",
                "properties": {
                    "ticker": {
                        "type": "string",
                        "description": "Stock ticker symbol",
                    },
                    "days": {
                        "type": "integer",
                        "description": "Number of days to analyze (default: 7)",
                        "default": 7,
                    },
                },
                "required": ["ticker"],
            },
        )

    def _register_resources(self) -> None:
        """Register all available resources."""
        self.register_resource(
            uri="news://articles/{ticker}",
            name="News Articles",
            description="Recent news articles for a ticker",
        )
        self.register_resource(
            uri="news://sentiment/{ticker}",
            name="Sentiment Score",
            description="Aggregate sentiment score for a ticker",
        )

    # -------------------------------------------------------------------------
    # Tool Implementations
    # -------------------------------------------------------------------------

    async def _get_news(
        self,
        ticker: str,
        days: int = 7,
        limit: int = 10,
    ) -> dict[str, Any]:
        """Fetch news articles for a ticker.

        Args:
            ticker: Stock ticker symbol.
            days: Number of days to look back.
            limit: Maximum number of articles.

        Returns:
            Dictionary with articles list.

        Raises:
            MCPInvalidParamsError: If ticker is invalid.
            MCPError: If API call fails.
        """
        ticker = ticker.upper().strip()
        if not ticker:
            raise MCPInvalidParamsError("Ticker symbol is required", param="ticker")

        cache_key = f"news:{ticker}:{days}:{limit}"
        cached = self._cache.get(cache_key)
        if cached:
            logger.debug(f"Cache hit for news: {ticker}")
            return cached

        articles = await self._fetch_articles_from_api(ticker, days, limit)

        result = {
            "ticker": ticker,
            "article_count": len(articles),
            "articles": [a.model_dump() for a in articles],
        }

        # Convert datetime to ISO string for JSON serialization
        for article in result["articles"]:
            if "analyzed_at" in article and isinstance(article["analyzed_at"], datetime):
                article["analyzed_at"] = article["analyzed_at"].isoformat()

        self._cache.set(cache_key, result)
        logger.info(f"Fetched {len(articles)} news articles for {ticker}")
        return result

    async def _get_headlines(
        self,
        ticker: str,
        limit: int = 5,
    ) -> dict[str, Any]:
        """Fetch news headlines for a ticker.

        Args:
            ticker: Stock ticker symbol.
            limit: Maximum number of headlines.

        Returns:
            Dictionary with headlines list.

        Raises:
            MCPInvalidParamsError: If ticker is invalid.
            MCPError: If API call fails.
        """
        ticker = ticker.upper().strip()
        if not ticker:
            raise MCPInvalidParamsError("Ticker symbol is required", param="ticker")

        cache_key = f"headlines:{ticker}:{limit}"
        cached = self._cache.get(cache_key)
        if cached:
            logger.debug(f"Cache hit for headlines: {ticker}")
            return cached

        # Fetch articles and convert to headlines
        articles = await self._fetch_articles_from_api(ticker, days=7, limit=limit)

        headlines = [
            Headline(
                title=a.title,
                source=a.source,
                published_at=a.published_at,
                url=a.url,
            )
            for a in articles
        ]

        result = {
            "ticker": ticker,
            "headline_count": len(headlines),
            "headlines": [h.model_dump() for h in headlines],
        }

        self._cache.set(cache_key, result)
        logger.info(f"Fetched {len(headlines)} headlines for {ticker}")
        return result

    async def _calculate_sentiment(
        self,
        ticker: str,
        days: int = 7,
    ) -> dict[str, Any]:
        """Calculate aggregate sentiment from news articles.

        Args:
            ticker: Stock ticker symbol.
            days: Number of days to analyze.

        Returns:
            Sentiment score dictionary.

        Raises:
            MCPInvalidParamsError: If ticker is invalid.
            MCPError: If analysis fails.
        """
        ticker = ticker.upper().strip()
        if not ticker:
            raise MCPInvalidParamsError("Ticker symbol is required", param="ticker")

        cache_key = f"sentiment:{ticker}:{days}"
        cached = self._cache.get(cache_key)
        if cached:
            logger.debug(f"Cache hit for sentiment: {ticker}")
            return cached

        # Fetch articles for sentiment analysis
        articles = await self._fetch_articles_from_api(ticker, days=days, limit=20)

        # Handle empty results with zero sentiment scores (Requirement 3.4)
        if not articles:
            result = SentimentScore(
                ticker=ticker,
                overall_score=0.0,
                positive_pct=0.0,
                negative_pct=0.0,
                neutral_pct=0.0,
                article_count=0,
            ).model_dump()
            result["analyzed_at"] = result["analyzed_at"].isoformat()
            self._cache.set(cache_key, result)
            return result

        # Analyze sentiment for each article
        positive_count = 0
        negative_count = 0
        neutral_count = 0

        for article in articles:
            sentiment = self._analyze_article_sentiment(article)
            if sentiment > 0:
                positive_count += 1
            elif sentiment < 0:
                negative_count += 1
            else:
                neutral_count += 1

        total = len(articles)
        positive_pct = (positive_count / total) * 100 if total > 0 else 0.0
        negative_pct = (negative_count / total) * 100 if total > 0 else 0.0
        neutral_pct = (neutral_count / total) * 100 if total > 0 else 0.0

        # Calculate overall score (-1 to 1 scale)
        overall_score = (positive_count - negative_count) / total if total > 0 else 0.0

        result = SentimentScore(
            ticker=ticker,
            overall_score=round(overall_score, 3),
            positive_pct=round(positive_pct, 1),
            negative_pct=round(negative_pct, 1),
            neutral_pct=round(neutral_pct, 1),
            article_count=total,
        ).model_dump()

        # Convert datetime to ISO string for JSON serialization
        result["analyzed_at"] = result["analyzed_at"].isoformat()

        self._cache.set(cache_key, result)
        logger.info(f"Calculated sentiment for {ticker}: score={overall_score:.3f}")
        return result

    # -------------------------------------------------------------------------
    # Helper Methods
    # -------------------------------------------------------------------------

    async def _fetch_articles_from_api(
        self,
        ticker: str,
        days: int,
        limit: int,
    ) -> list[NewsArticle]:
        """Fetch articles from NewsAPI.

        Args:
            ticker: Stock ticker symbol.
            days: Number of days to look back.
            limit: Maximum number of articles.

        Returns:
            List of NewsArticle objects.

        Raises:
            MCPError: If API call fails.
        """
        if not settings.news_api_key:
            logger.warning("NewsAPI key not configured")
            return []

        try:
            from_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
            to_date = datetime.now().strftime("%Y-%m-%d")

            url = "https://newsapi.org/v2/everything"
            params = {
                "q": f"{ticker} stock",
                "from": from_date,
                "to": to_date,
                "sortBy": "relevancy",
                "pageSize": limit,
                "language": "en",
                "apiKey": settings.news_api_key,
            }

            response = await self.http_client.get(url, params=params)
            response.raise_for_status()
            data = response.json()

            if data.get("status") != "ok":
                error_msg = data.get("message", "Unknown NewsAPI error")
                raise MCPError(
                    message=f"NewsAPI error: {error_msg}",
                    code=MCPErrorCode.SERVER_ERROR,
                    details={"ticker": ticker},
                )

            articles = []
            for article in data.get("articles", []):
                articles.append(
                    NewsArticle(
                        title=article.get("title", ""),
                        description=article.get("description"),
                        content=article.get("content"),
                        source=article.get("source", {}).get("name", "Unknown"),
                        url=article.get("url", ""),
                        published_at=article.get("publishedAt", ""),
                        ticker=ticker,
                    )
                )

            return articles

        except httpx.HTTPStatusError as e:
            logger.error(f"NewsAPI HTTP error: {e}")
            raise MCPError(
                message=f"NewsAPI request failed: {str(e)}",
                code=MCPErrorCode.SERVER_ERROR,
                details={"ticker": ticker},
            )
        except MCPError:
            raise
        except Exception as e:
            logger.error(f"NewsAPI error for '{ticker}': {e}")
            raise MCPError(
                message=f"Failed to fetch news articles: {str(e)}",
                code=MCPErrorCode.SERVER_ERROR,
                details={"ticker": ticker},
            )

    def _analyze_article_sentiment(self, article: NewsArticle) -> int:
        """Analyze sentiment of a single article using keyword matching.

        Args:
            article: NewsArticle to analyze.

        Returns:
            1 for positive, -1 for negative, 0 for neutral.
        """
        # Combine title, description, and content for analysis
        text = " ".join(
            filter(None, [article.title, article.description, article.content])
        ).lower()

        positive_matches = sum(1 for kw in POSITIVE_KEYWORDS if kw in text)
        negative_matches = sum(1 for kw in NEGATIVE_KEYWORDS if kw in text)

        if positive_matches > negative_matches:
            return 1
        elif negative_matches > positive_matches:
            return -1
        return 0

    # -------------------------------------------------------------------------
    # Resource Implementation
    # -------------------------------------------------------------------------

    async def read_resource(self, uri: str) -> ContentBlock:
        """Read a resource by URI.

        Args:
            uri: Resource URI (e.g., 'news://articles/AAPL').

        Returns:
            Content block with resource data.

        Raises:
            MCPInvalidParamsError: If URI is invalid.
            MCPError: If resource fetch fails.
        """
        if not uri.startswith("news://"):
            raise MCPInvalidParamsError(f"Invalid URI scheme: {uri}")

        path = uri.replace("news://", "")
        parts = path.split("/")

        if len(parts) != 2:
            raise MCPInvalidParamsError(f"Invalid URI format: {uri}")

        resource_type, ticker = parts

        if resource_type == "articles":
            data = await self._get_news(ticker)
        elif resource_type == "sentiment":
            data = await self._calculate_sentiment(ticker)
        else:
            raise MCPInvalidParamsError(f"Unknown resource type: {resource_type}")

        return ContentBlock(type=ContentType.JSON, data=data)

    # -------------------------------------------------------------------------
    # Health Check
    # -------------------------------------------------------------------------

    async def health_check(self) -> HealthStatus:
        """Check server health by testing NewsAPI connectivity.

        Returns:
            Health status with latency information.
        """
        start_time = time.time()

        # Check if API key is configured
        if not settings.news_api_key:
            return HealthStatus(
                healthy=False,
                server_name=self.name,
                error_message="NewsAPI key not configured",
            )

        try:
            # Test with a simple query
            url = "https://newsapi.org/v2/everything"
            params = {
                "q": "stock market",
                "pageSize": 1,
                "apiKey": settings.news_api_key,
            }

            response = await self.http_client.get(url, params=params)
            response.raise_for_status()
            data = response.json()

            latency_ms = (time.time() - start_time) * 1000

            if data.get("status") == "ok":
                return HealthStatus(
                    healthy=True,
                    server_name=self.name,
                    latency_ms=latency_ms,
                )
            else:
                return HealthStatus(
                    healthy=False,
                    server_name=self.name,
                    latency_ms=latency_ms,
                    error_message=data.get("message", "Unknown error"),
                )

        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            return HealthStatus(
                healthy=False,
                server_name=self.name,
                latency_ms=latency_ms,
                error_message=str(e),
            )
