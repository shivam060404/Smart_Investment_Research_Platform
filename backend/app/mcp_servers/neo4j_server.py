import logging
import time
from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field

from app.mcp.base import BaseMCPServer, ContentBlock, ContentType, HealthStatus
from app.mcp.errors import MCPError, MCPErrorCode, MCPInvalidParamsError
from app.services.neo4j_service import Neo4jService, get_neo4j_service

logger = logging.getLogger(__name__)


# =============================================================================
# Response Models
# =============================================================================


class StockCorrelation(BaseModel):
    """Stock correlation data."""

    ticker: str
    correlated_ticker: str
    correlation: float
    name: Optional[str] = None


class SectorPerformance(BaseModel):
    """Sector performance metrics."""

    sector: str
    stock_count: int
    avg_confidence: Optional[float] = None
    bullish_count: int = 0
    bearish_count: int = 0
    neutral_count: int = 0


class AnalysisHistoryItem(BaseModel):
    """Historical analysis record."""

    id: str
    ticker: str
    timestamp: datetime
    recommendation: str
    confidence: float
    weighted_score: float
    reasoning: Optional[str] = None
    signals: list[dict[str, Any]] = Field(default_factory=list)


class SimilarStock(BaseModel):
    """Similar stock based on fundamental metrics."""

    ticker: str
    name: Optional[str] = None
    sector: Optional[str] = None
    industry: Optional[str] = None
    similarity_score: float
    matching_factors: list[str] = Field(default_factory=list)


# =============================================================================
# Cache Implementation
# =============================================================================


class SimpleCache:
    """Simple in-memory cache with TTL support for fallback."""

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
# Neo4j MCP Server
# =============================================================================


class Neo4jMCPServer(BaseMCPServer):
    """MCP Server for Neo4j knowledge graph queries.

    Provides tools for querying stock correlations, sector performance,
    analysis history, and finding similar stocks through the MCP protocol.
    """

    def __init__(
        self,
        neo4j_service: Optional[Neo4jService] = None,
        cache_ttl_seconds: int = 300,
    ):
        """Initialize the Neo4j MCP Server.

        Args:
            neo4j_service: Optional Neo4j service instance.
            cache_ttl_seconds: Cache TTL in seconds (default 5 minutes).
        """
        super().__init__(name="neo4j", version="1.0.0")
        self._neo4j_service = neo4j_service
        self._cache = SimpleCache(ttl_seconds=cache_ttl_seconds)
        self._register_tools()
        self._register_resources()

    def _get_neo4j_service(self) -> Neo4jService:
        """Get or create Neo4j service instance."""
        if self._neo4j_service is None:
            self._neo4j_service = get_neo4j_service()
        return self._neo4j_service

    def _register_tools(self) -> None:
        """Register all available tools."""
        # Correlation tools
        self.register_tool(
            name="get_correlations",
            description="Query stock correlations within a sector or for a specific ticker",
            handler=self._get_correlations,
            input_schema={
                "type": "object",
                "properties": {
                    "ticker": {
                        "type": "string",
                        "description": "Stock ticker symbol (e.g., 'AAPL', 'MSFT')",
                    },
                    "min_correlation": {
                        "type": "number",
                        "description": "Minimum correlation threshold (0.0 to 1.0)",
                        "default": 0.5,
                    },
                },
                "required": ["ticker"],
            },
        )

        # Sector performance tools
        self.register_tool(
            name="get_sector_performance",
            description="Get performance metrics for a specific sector",
            handler=self._get_sector_performance,
            input_schema={
                "type": "object",
                "properties": {
                    "sector": {
                        "type": "string",
                        "description": "Sector name (e.g., 'Technology', 'Healthcare')",
                    },
                },
                "required": ["sector"],
            },
        )

        # Analysis history tools
        self.register_tool(
            name="get_analysis_history",
            description="Retrieve historical analysis signals for a ticker",
            handler=self._get_analysis_history,
            input_schema={
                "type": "object",
                "properties": {
                    "ticker": {
                        "type": "string",
                        "description": "Stock ticker symbol",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of results",
                        "default": 10,
                    },
                },
                "required": ["ticker"],
            },
        )

        # Similar stocks tools
        self.register_tool(
            name="find_similar_stocks",
            description="Find stocks similar to the given ticker based on fundamental metrics",
            handler=self._find_similar_stocks,
            input_schema={
                "type": "object",
                "properties": {
                    "ticker": {
                        "type": "string",
                        "description": "Stock ticker symbol",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of similar stocks to return",
                        "default": 5,
                    },
                },
                "required": ["ticker"],
            },
        )

    def _register_resources(self) -> None:
        """Register all available resources."""
        self.register_resource(
            uri="neo4j://correlations/{ticker}",
            name="Stock Correlations",
            description="Correlated stocks for a given ticker",
        )
        self.register_resource(
            uri="neo4j://history/{ticker}",
            name="Analysis History",
            description="Historical analysis records for a ticker",
        )

    # -------------------------------------------------------------------------
    # Tool Implementations
    # -------------------------------------------------------------------------

    async def _get_correlations(
        self,
        ticker: str,
        min_correlation: float = 0.5,
    ) -> dict[str, Any]:
        """Query stock correlations for a ticker.

        Args:
            ticker: Stock ticker symbol.
            min_correlation: Minimum correlation threshold.

        Returns:
            Correlation data dictionary.

        Raises:
            MCPInvalidParamsError: If ticker is invalid.
            MCPError: If database query fails.
        """
        ticker = ticker.upper().strip()
        if not ticker:
            raise MCPInvalidParamsError("Ticker symbol is required", param="ticker")

        if not 0.0 <= min_correlation <= 1.0:
            raise MCPInvalidParamsError(
                "min_correlation must be between 0.0 and 1.0",
                param="min_correlation",
            )

        cache_key = f"correlations:{ticker}:{min_correlation}"
        cached = self._cache.get(cache_key)
        if cached:
            logger.debug(f"Cache hit for correlations: {ticker}")
            return cached

        try:
            neo4j = self._get_neo4j_service()
            correlations = await neo4j.get_correlated_stocks(
                ticker=ticker,
                min_correlation=min_correlation,
            )

            result = {
                "ticker": ticker,
                "min_correlation": min_correlation,
                "correlations": [
                    StockCorrelation(
                        ticker=ticker,
                        correlated_ticker=c["ticker"],
                        correlation=c["correlation"],
                        name=c.get("name"),
                    ).model_dump()
                    for c in correlations
                ],
                "count": len(correlations),
            }

            self._cache.set(cache_key, result)
            logger.info(f"Fetched {len(correlations)} correlations for {ticker}")
            return result

        except MCPInvalidParamsError:
            raise
        except Exception as e:
            logger.error(f"Neo4j query error for correlations {ticker}: {e}")
            # Try to return cached data as fallback
            cached_fallback = self._cache.get(f"correlations:{ticker}:0.0")
            if cached_fallback:
                logger.warning(f"Returning cached fallback for {ticker}")
                return cached_fallback
            raise MCPError(
                message=f"Failed to fetch correlations for {ticker}: {str(e)}",
                code=MCPErrorCode.SERVER_ERROR,
                details={"ticker": ticker},
            )

    async def _get_sector_performance(self, sector: str) -> dict[str, Any]:
        """Get performance metrics for a sector.

        Args:
            sector: Sector name.

        Returns:
            Sector performance dictionary.

        Raises:
            MCPInvalidParamsError: If sector is invalid.
            MCPError: If database query fails.
        """
        sector = sector.strip()
        if not sector:
            raise MCPInvalidParamsError("Sector name is required", param="sector")

        cache_key = f"sector_performance:{sector}"
        cached = self._cache.get(cache_key)
        if cached:
            logger.debug(f"Cache hit for sector performance: {sector}")
            return cached

        try:
            neo4j = self._get_neo4j_service()

            async with neo4j.get_session() as session:
                # Query sector performance metrics
                query = """
                MATCH (s:Stock)-[:BELONGS_TO]->(sec:Sector {name: $sector})
                OPTIONAL MATCH (s)-[:HAS_ANALYSIS]->(a:Analysis)
                WITH sec, s, a
                ORDER BY a.timestamp DESC
                WITH sec, s, collect(a)[0] as latest_analysis
                RETURN sec.name as sector,
                       count(DISTINCT s) as stock_count,
                       avg(latest_analysis.confidence) as avg_confidence,
                       sum(CASE WHEN latest_analysis.recommendation = 'BUY' 
                           OR latest_analysis.recommendation = 'STRONG_BUY' 
                           THEN 1 ELSE 0 END) as bullish_count,
                       sum(CASE WHEN latest_analysis.recommendation = 'SELL' 
                           OR latest_analysis.recommendation = 'STRONG_SELL' 
                           THEN 1 ELSE 0 END) as bearish_count,
                       sum(CASE WHEN latest_analysis.recommendation = 'HOLD' 
                           THEN 1 ELSE 0 END) as neutral_count
                """

                result = await session.run(query, {"sector": sector})
                record = await result.single()

                if not record or record["stock_count"] == 0:
                    # Return empty result for unknown sector
                    result_data = SectorPerformance(
                        sector=sector,
                        stock_count=0,
                    ).model_dump()
                else:
                    result_data = SectorPerformance(
                        sector=record["sector"],
                        stock_count=record["stock_count"],
                        avg_confidence=record["avg_confidence"],
                        bullish_count=record["bullish_count"] or 0,
                        bearish_count=record["bearish_count"] or 0,
                        neutral_count=record["neutral_count"] or 0,
                    ).model_dump()

            self._cache.set(cache_key, result_data)
            logger.info(f"Fetched sector performance for {sector}")
            return result_data

        except MCPInvalidParamsError:
            raise
        except Exception as e:
            logger.error(f"Neo4j query error for sector {sector}: {e}")
            # Try to return cached data as fallback
            cached_fallback = self._cache.get(cache_key)
            if cached_fallback:
                logger.warning(f"Returning cached fallback for sector {sector}")
                return cached_fallback
            raise MCPError(
                message=f"Failed to fetch sector performance for {sector}: {str(e)}",
                code=MCPErrorCode.SERVER_ERROR,
                details={"sector": sector},
            )

    async def _get_analysis_history(
        self,
        ticker: str,
        limit: int = 10,
    ) -> dict[str, Any]:
        """Retrieve historical analysis signals for a ticker.

        Args:
            ticker: Stock ticker symbol.
            limit: Maximum number of results.

        Returns:
            Analysis history dictionary.

        Raises:
            MCPInvalidParamsError: If ticker is invalid.
            MCPError: If database query fails.
        """
        ticker = ticker.upper().strip()
        if not ticker:
            raise MCPInvalidParamsError("Ticker symbol is required", param="ticker")

        if limit < 1 or limit > 100:
            raise MCPInvalidParamsError(
                "limit must be between 1 and 100",
                param="limit",
            )

        cache_key = f"analysis_history:{ticker}:{limit}"
        cached = self._cache.get(cache_key)
        if cached:
            logger.debug(f"Cache hit for analysis history: {ticker}")
            return cached

        try:
            neo4j = self._get_neo4j_service()
            analyses = await neo4j.get_analysis_history(ticker=ticker, limit=limit)

            result = {
                "ticker": ticker,
                "analyses": [
                    AnalysisHistoryItem(
                        id=a["id"],
                        ticker=ticker,
                        timestamp=a["timestamp"],
                        recommendation=a["recommendation"],
                        confidence=a["confidence"],
                        weighted_score=a["weighted_score"],
                        reasoning=a.get("reasoning"),
                        signals=a.get("signals", []),
                    ).model_dump()
                    for a in analyses
                ],
                "count": len(analyses),
            }

            # Convert datetime to ISO string for JSON serialization
            for analysis in result["analyses"]:
                if isinstance(analysis["timestamp"], datetime):
                    analysis["timestamp"] = analysis["timestamp"].isoformat()

            self._cache.set(cache_key, result)
            logger.info(f"Fetched {len(analyses)} analyses for {ticker}")
            return result

        except MCPInvalidParamsError:
            raise
        except Exception as e:
            logger.error(f"Neo4j query error for analysis history {ticker}: {e}")
            # Try to return cached data as fallback
            cached_fallback = self._cache.get(f"analysis_history:{ticker}:10")
            if cached_fallback:
                logger.warning(f"Returning cached fallback for {ticker}")
                return cached_fallback
            raise MCPError(
                message=f"Failed to fetch analysis history for {ticker}: {str(e)}",
                code=MCPErrorCode.SERVER_ERROR,
                details={"ticker": ticker},
            )

    async def _find_similar_stocks(
        self,
        ticker: str,
        limit: int = 5,
    ) -> dict[str, Any]:
        """Find stocks similar to the given ticker based on fundamental metrics.

        Args:
            ticker: Stock ticker symbol.
            limit: Maximum number of similar stocks to return.

        Returns:
            Similar stocks dictionary.

        Raises:
            MCPInvalidParamsError: If ticker is invalid.
            MCPError: If database query fails.
        """
        ticker = ticker.upper().strip()
        if not ticker:
            raise MCPInvalidParamsError("Ticker symbol is required", param="ticker")

        if limit < 1 or limit > 20:
            raise MCPInvalidParamsError(
                "limit must be between 1 and 20",
                param="limit",
            )

        cache_key = f"similar_stocks:{ticker}:{limit}"
        cached = self._cache.get(cache_key)
        if cached:
            logger.debug(f"Cache hit for similar stocks: {ticker}")
            return cached

        try:
            neo4j = self._get_neo4j_service()

            async with neo4j.get_session() as session:
                # First get the source stock's sector and industry
                source_query = """
                MATCH (s:Stock {ticker: $ticker})
                OPTIONAL MATCH (s)-[:BELONGS_TO]->(sec:Sector)
                RETURN s.ticker as ticker,
                       s.name as name,
                       s.sector as sector,
                       s.industry as industry,
                       sec.name as sector_name
                """
                source_result = await session.run(source_query, {"ticker": ticker})
                source_record = await source_result.single()

                if not source_record:
                    return {
                        "ticker": ticker,
                        "similar_stocks": [],
                        "count": 0,
                        "message": f"Stock {ticker} not found in knowledge graph",
                    }

                source_sector = source_record["sector"] or source_record["sector_name"]
                source_industry = source_record["industry"]

                # Find similar stocks in same sector/industry
                similar_query = """
                MATCH (source:Stock {ticker: $ticker})
                MATCH (s:Stock)
                WHERE s.ticker <> $ticker
                  AND (s.sector = $sector OR s.industry = $industry)
                OPTIONAL MATCH (s)-[:BELONGS_TO]->(sec:Sector)
                WITH s, sec,
                     CASE WHEN s.sector = $sector AND s.industry = $industry THEN 1.0
                          WHEN s.industry = $industry THEN 0.8
                          WHEN s.sector = $sector THEN 0.6
                          ELSE 0.3 END as similarity_score,
                     CASE WHEN s.sector = $sector THEN ['Same Sector'] ELSE [] END +
                     CASE WHEN s.industry = $industry THEN ['Same Industry'] ELSE [] END as factors
                RETURN s.ticker as ticker,
                       s.name as name,
                       COALESCE(s.sector, sec.name) as sector,
                       s.industry as industry,
                       similarity_score,
                       factors
                ORDER BY similarity_score DESC
                LIMIT $limit
                """

                similar_result = await session.run(
                    similar_query,
                    {
                        "ticker": ticker,
                        "sector": source_sector or "",
                        "industry": source_industry or "",
                        "limit": limit,
                    },
                )
                similar_records = await similar_result.data()

                similar_stocks = [
                    SimilarStock(
                        ticker=r["ticker"],
                        name=r.get("name"),
                        sector=r.get("sector"),
                        industry=r.get("industry"),
                        similarity_score=r["similarity_score"],
                        matching_factors=r.get("factors", []),
                    ).model_dump()
                    for r in similar_records
                ]

                result = {
                    "ticker": ticker,
                    "source_sector": source_sector,
                    "source_industry": source_industry,
                    "similar_stocks": similar_stocks,
                    "count": len(similar_stocks),
                }

            self._cache.set(cache_key, result)
            logger.info(f"Found {len(similar_stocks)} similar stocks for {ticker}")
            return result

        except MCPInvalidParamsError:
            raise
        except Exception as e:
            logger.error(f"Neo4j query error for similar stocks {ticker}: {e}")
            # Try to return cached data as fallback
            cached_fallback = self._cache.get(f"similar_stocks:{ticker}:5")
            if cached_fallback:
                logger.warning(f"Returning cached fallback for {ticker}")
                return cached_fallback
            raise MCPError(
                message=f"Failed to find similar stocks for {ticker}: {str(e)}",
                code=MCPErrorCode.SERVER_ERROR,
                details={"ticker": ticker},
            )

    # -------------------------------------------------------------------------
    # Resource Implementation
    # -------------------------------------------------------------------------

    async def read_resource(self, uri: str) -> ContentBlock:
        """Read a resource by URI.

        Args:
            uri: Resource URI (e.g., 'neo4j://correlations/AAPL').

        Returns:
            Content block with resource data.

        Raises:
            MCPInvalidParamsError: If URI is invalid.
            MCPError: If resource fetch fails.
        """
        if not uri.startswith("neo4j://"):
            raise MCPInvalidParamsError(f"Invalid URI scheme: {uri}")

        path = uri.replace("neo4j://", "")
        parts = path.split("/")

        if len(parts) != 2:
            raise MCPInvalidParamsError(f"Invalid URI format: {uri}")

        resource_type, ticker = parts

        if resource_type == "correlations":
            data = await self._get_correlations(ticker)
        elif resource_type == "history":
            data = await self._get_analysis_history(ticker)
        else:
            raise MCPInvalidParamsError(f"Unknown resource type: {resource_type}")

        return ContentBlock(type=ContentType.JSON, data=data)

    # -------------------------------------------------------------------------
    # Health Check
    # -------------------------------------------------------------------------

    async def health_check(self) -> HealthStatus:
        """Check server health by testing Neo4j connection.

        Returns:
            Health status with latency information.
        """
        start_time = time.time()
        try:
            neo4j = self._get_neo4j_service()
            health = await neo4j.health_check()
            latency_ms = (time.time() - start_time) * 1000

            return HealthStatus(
                healthy=health["status"] == "healthy",
                server_name=self.name,
                latency_ms=latency_ms,
                error_message=health.get("message") if health["status"] != "healthy" else None,
            )
        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            return HealthStatus(
                healthy=False,
                server_name=self.name,
                latency_ms=latency_ms,
                error_message=str(e),
            )
