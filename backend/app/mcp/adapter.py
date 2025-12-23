import logging
from typing import Any, Optional

from pydantic import BaseModel, Field

from app.mcp.client import MCPClient
from app.mcp.errors import MCPError, MCPServerUnavailableError
from app.models.stock import (
    FundamentalMetrics,
    SentimentData,
    StockData,
    TechnicalIndicators,
    MACDData,
    BollingerData,
)

logger = logging.getLogger(__name__)


class MCPAdapterError(Exception):
    """Base exception for MCP adapter errors."""
    pass


class MCPDataUnavailableError(MCPAdapterError):
    """Raised when MCP data cannot be retrieved."""
    pass


class MCPDataAdapter:
    """Adapter layer that maps existing data service calls to MCP servers.

    This adapter provides backward-compatible interfaces for agents while
    transparently routing requests to the appropriate MCP servers.

    Attributes:
        mcp_client: The MCP client for server communication.
    """

    def __init__(self, mcp_client: MCPClient):
        """Initialize the MCP data adapter.

        Args:
            mcp_client: MCPClient instance for server communication.
        """
        self.mcp_client = mcp_client
        logger.info("MCPDataAdapter initialized")

    async def get_stock_data(self, ticker: str) -> StockData:
        """Get stock data by mapping to YFinance MCP calls.

        Fetches price and profile data from the YFinance MCP server
        and combines them into a StockData object.

        Args:
            ticker: Stock ticker symbol (e.g., 'AAPL').

        Returns:
            StockData object with basic stock information.

        Raises:
            MCPDataUnavailableError: If MCP data cannot be retrieved.
        """
        ticker = ticker.upper().strip()
        logger.debug(f"MCPDataAdapter: Getting stock data for {ticker}")

        try:
            # Fetch price data from YFinance MCP server
            price_result = await self.mcp_client.call_tool(
                server="yfinance",
                tool="get_stock_price",
                params={"ticker": ticker},
            )
            price_data = self._extract_tool_result(price_result)

            # Fetch profile data from YFinance MCP server
            profile_result = await self.mcp_client.call_tool(
                server="yfinance",
                tool="get_company_profile",
                params={"ticker": ticker},
            )
            profile_data = self._extract_tool_result(profile_result)

            # Combine into StockData model
            return StockData(
                ticker=ticker,
                name=profile_data.get("name", ticker),
                sector=profile_data.get("sector", "Unknown"),
                industry=profile_data.get("industry", "Unknown"),
                current_price=price_data.get("current_price", 0.0),
                market_cap=profile_data.get("market_cap", 0.0),
                currency=price_data.get("currency", "USD"),
            )

        except MCPServerUnavailableError as e:
            logger.error(f"YFinance MCP server unavailable: {e}")
            raise MCPDataUnavailableError(f"YFinance MCP server unavailable: {e}")
        except MCPError as e:
            logger.error(f"MCP error fetching stock data for {ticker}: {e}")
            raise MCPDataUnavailableError(f"Failed to fetch stock data: {e}")
        except Exception as e:
            logger.error(f"Unexpected error fetching stock data for {ticker}: {e}")
            raise MCPDataUnavailableError(f"Unexpected error: {e}")

    async def get_fundamental_metrics(self, ticker: str) -> FundamentalMetrics:
        """Get fundamental metrics by mapping to YFinance MCP calls.

        Fetches fundamental data from the YFinance MCP server.

        Args:
            ticker: Stock ticker symbol.

        Returns:
            FundamentalMetrics object with financial metrics.

        Raises:
            MCPDataUnavailableError: If MCP data cannot be retrieved.
        """
        ticker = ticker.upper().strip()
        logger.debug(f"MCPDataAdapter: Getting fundamental metrics for {ticker}")

        try:
            result = await self.mcp_client.call_tool(
                server="yfinance",
                tool="get_fundamentals",
                params={"ticker": ticker},
            )
            data = self._extract_tool_result(result)

            return FundamentalMetrics(
                pe_ratio=data.get("pe_ratio"),
                pb_ratio=data.get("pb_ratio"),
                roe=data.get("roe"),
                roa=data.get("roa"),
                debt_to_equity=data.get("debt_to_equity"),
                revenue_growth=data.get("revenue_growth"),
                profit_margin=data.get("profit_margin"),
                dividend_yield=data.get("dividend_yield"),
            )

        except MCPServerUnavailableError as e:
            logger.error(f"YFinance MCP server unavailable: {e}")
            raise MCPDataUnavailableError(f"YFinance MCP server unavailable: {e}")
        except MCPError as e:
            logger.error(f"MCP error fetching fundamentals for {ticker}: {e}")
            raise MCPDataUnavailableError(f"Failed to fetch fundamentals: {e}")
        except Exception as e:
            logger.error(f"Unexpected error fetching fundamentals for {ticker}: {e}")
            raise MCPDataUnavailableError(f"Unexpected error: {e}")

    async def get_technical_indicators(self, ticker: str) -> TechnicalIndicators:
        """Get technical indicators by mapping to Technical MCP calls.

        Fetches aggregated technical signals from the Technical MCP server.

        Args:
            ticker: Stock ticker symbol.

        Returns:
            TechnicalIndicators object with technical analysis data.

        Raises:
            MCPDataUnavailableError: If MCP data cannot be retrieved.
        """
        ticker = ticker.upper().strip()
        logger.debug(f"MCPDataAdapter: Getting technical indicators for {ticker}")

        try:
            result = await self.mcp_client.call_tool(
                server="technical",
                tool="get_signals",
                params={"ticker": ticker},
            )
            data = self._extract_tool_result(result)

            # Map MCP response to TechnicalIndicators model
            macd_data = MACDData(
                macd_line=data.get("macd", 0.0) or 0.0,
                signal_line=data.get("macd_signal", 0.0) or 0.0,
                histogram=data.get("macd_histogram", 0.0) or 0.0,
            )

            # Calculate Bollinger Bands from SMA (simplified)
            sma_20 = data.get("sma_20", 0.0) or 0.0
            bollinger_data = BollingerData(
                upper_band=sma_20 * 1.02,  # Simplified approximation
                middle_band=sma_20,
                lower_band=sma_20 * 0.98,
            )

            # Determine trend from overall signal
            overall_signal = data.get("overall_signal", "neutral")
            current_trend = overall_signal if overall_signal else "neutral"

            return TechnicalIndicators(
                sma_50=data.get("sma_50", 0.0) or 0.0,
                sma_200=data.get("sma_200", 0.0) or 0.0,
                rsi_14=data.get("rsi", 50.0) or 50.0,
                macd=macd_data,
                bollinger_bands=bollinger_data,
                current_trend=current_trend,
            )

        except MCPServerUnavailableError as e:
            logger.error(f"Technical MCP server unavailable: {e}")
            raise MCPDataUnavailableError(f"Technical MCP server unavailable: {e}")
        except MCPError as e:
            logger.error(f"MCP error fetching technical indicators for {ticker}: {e}")
            raise MCPDataUnavailableError(f"Failed to fetch technical indicators: {e}")
        except Exception as e:
            logger.error(f"Unexpected error fetching technical indicators for {ticker}: {e}")
            raise MCPDataUnavailableError(f"Unexpected error: {e}")

    async def get_sentiment(self, ticker: str, days: int = 7) -> SentimentData:
        """Get sentiment data by mapping to News MCP calls.

        Fetches sentiment analysis from the News MCP server.

        Args:
            ticker: Stock ticker symbol.
            days: Number of days to analyze (default: 7).

        Returns:
            SentimentData object with sentiment analysis results.

        Raises:
            MCPDataUnavailableError: If MCP data cannot be retrieved.
        """
        ticker = ticker.upper().strip()
        logger.debug(f"MCPDataAdapter: Getting sentiment for {ticker}")

        try:
            result = await self.mcp_client.call_tool(
                server="news",
                tool="calculate_sentiment",
                params={"ticker": ticker, "days": days},
            )
            data = self._extract_tool_result(result)

            # Convert MCP sentiment score (-1 to 1) to model scale (-100 to 100)
            overall_score = data.get("overall_score", 0.0) * 100

            # Build source breakdown from percentages
            source_breakdown = {
                "positive": data.get("positive_pct", 0.0),
                "negative": data.get("negative_pct", 0.0),
                "neutral": data.get("neutral_pct", 0.0),
            }

            return SentimentData(
                overall_score=overall_score,
                article_count=data.get("article_count", 0),
                key_catalysts=[],  # Would need additional MCP call for catalysts
                source_breakdown=source_breakdown,
            )

        except MCPServerUnavailableError as e:
            logger.error(f"News MCP server unavailable: {e}")
            raise MCPDataUnavailableError(f"News MCP server unavailable: {e}")
        except MCPError as e:
            logger.error(f"MCP error fetching sentiment for {ticker}: {e}")
            raise MCPDataUnavailableError(f"Failed to fetch sentiment: {e}")
        except Exception as e:
            logger.error(f"Unexpected error fetching sentiment for {ticker}: {e}")
            raise MCPDataUnavailableError(f"Unexpected error: {e}")

    async def get_news_articles(
        self,
        ticker: str,
        days: int = 7,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Get news articles by mapping to News MCP calls.

        Fetches news articles from the News MCP server.

        Args:
            ticker: Stock ticker symbol.
            days: Number of days to look back.
            limit: Maximum number of articles.

        Returns:
            List of article dictionaries.

        Raises:
            MCPDataUnavailableError: If MCP data cannot be retrieved.
        """
        ticker = ticker.upper().strip()
        logger.debug(f"MCPDataAdapter: Getting news articles for {ticker}")

        try:
            result = await self.mcp_client.call_tool(
                server="news",
                tool="get_news",
                params={"ticker": ticker, "days": days, "limit": limit},
            )
            data = self._extract_tool_result(result)

            return data.get("articles", [])

        except MCPServerUnavailableError as e:
            logger.error(f"News MCP server unavailable: {e}")
            raise MCPDataUnavailableError(f"News MCP server unavailable: {e}")
        except MCPError as e:
            logger.error(f"MCP error fetching news for {ticker}: {e}")
            raise MCPDataUnavailableError(f"Failed to fetch news: {e}")
        except Exception as e:
            logger.error(f"Unexpected error fetching news for {ticker}: {e}")
            raise MCPDataUnavailableError(f"Unexpected error: {e}")

    async def get_correlations(
        self,
        ticker: str,
        sector: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        """Get stock correlations by mapping to Neo4j MCP calls.

        Fetches correlation data from the Neo4j MCP server.

        Args:
            ticker: Stock ticker symbol.
            sector: Optional sector filter.

        Returns:
            List of correlation dictionaries.

        Raises:
            MCPDataUnavailableError: If MCP data cannot be retrieved.
        """
        ticker = ticker.upper().strip()
        logger.debug(f"MCPDataAdapter: Getting correlations for {ticker}")

        try:
            params = {"ticker": ticker}
            if sector:
                params["sector"] = sector

            result = await self.mcp_client.call_tool(
                server="neo4j",
                tool="get_correlations",
                params=params,
            )
            data = self._extract_tool_result(result)

            return data.get("correlations", [])

        except MCPServerUnavailableError as e:
            logger.error(f"Neo4j MCP server unavailable: {e}")
            raise MCPDataUnavailableError(f"Neo4j MCP server unavailable: {e}")
        except MCPError as e:
            logger.error(f"MCP error fetching correlations for {ticker}: {e}")
            raise MCPDataUnavailableError(f"Failed to fetch correlations: {e}")
        except Exception as e:
            logger.error(f"Unexpected error fetching correlations for {ticker}: {e}")
            raise MCPDataUnavailableError(f"Unexpected error: {e}")

    async def run_backtest(
        self,
        ticker: str,
        strategy: str,
        period: str = "1y",
    ) -> dict[str, Any]:
        """Run backtest by mapping to Backtesting MCP calls.

        Executes a backtest through the Backtesting MCP server.

        Args:
            ticker: Stock ticker symbol.
            strategy: Strategy name to backtest.
            period: Backtest period (default: '1y').

        Returns:
            Backtest results dictionary.

        Raises:
            MCPDataUnavailableError: If MCP data cannot be retrieved.
        """
        ticker = ticker.upper().strip()
        logger.debug(f"MCPDataAdapter: Running backtest for {ticker}")

        try:
            result = await self.mcp_client.call_tool(
                server="backtesting",
                tool="run_backtest",
                params={"ticker": ticker, "strategy": strategy, "period": period},
            )
            return self._extract_tool_result(result)

        except MCPServerUnavailableError as e:
            logger.error(f"Backtesting MCP server unavailable: {e}")
            raise MCPDataUnavailableError(f"Backtesting MCP server unavailable: {e}")
        except MCPError as e:
            logger.error(f"MCP error running backtest for {ticker}: {e}")
            raise MCPDataUnavailableError(f"Failed to run backtest: {e}")
        except Exception as e:
            logger.error(f"Unexpected error running backtest for {ticker}: {e}")
            raise MCPDataUnavailableError(f"Unexpected error: {e}")

    def _extract_tool_result(self, tool_result: Any) -> dict[str, Any]:
        """Extract data from MCP tool result.

        Args:
            tool_result: ToolResult from MCP call.

        Returns:
            Extracted data dictionary.

        Raises:
            MCPDataUnavailableError: If result indicates an error.
        """
        if tool_result.is_error:
            error_msg = "Unknown error"
            if tool_result.content:
                error_msg = str(tool_result.content[0].data)
            raise MCPDataUnavailableError(error_msg)

        if not tool_result.content:
            return {}

        data = tool_result.content[0].data
        if isinstance(data, dict):
            return data
        return {"data": data}


# Singleton instance
_mcp_adapter: Optional[MCPDataAdapter] = None


def get_mcp_adapter(mcp_client: Optional[MCPClient] = None) -> Optional[MCPDataAdapter]:
    """Get or create MCPDataAdapter singleton instance.

    Args:
        mcp_client: Optional MCPClient to use for initialization.

    Returns:
        MCPDataAdapter instance or None if not initialized.
    """
    global _mcp_adapter
    if _mcp_adapter is None and mcp_client is not None:
        _mcp_adapter = MCPDataAdapter(mcp_client)
    return _mcp_adapter


def set_mcp_adapter(adapter: Optional[MCPDataAdapter]) -> None:
    """Set the global MCPDataAdapter instance.

    Args:
        adapter: MCPDataAdapter instance or None to clear.
    """
    global _mcp_adapter
    _mcp_adapter = adapter
