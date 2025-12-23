import logging
import time
from datetime import datetime, timedelta
from typing import Any, Optional

import yfinance as yf
from pydantic import BaseModel, Field

from app.mcp.base import BaseMCPServer, ContentBlock, ContentType, HealthStatus
from app.mcp.errors import MCPError, MCPErrorCode, MCPInvalidParamsError

logger = logging.getLogger(__name__)


# =============================================================================
# Response Models
# =============================================================================


class PriceData(BaseModel):
    """Stock price data (OHLCV)."""

    ticker: str
    current_price: float
    open: float
    high: float
    low: float
    close: float
    volume: int
    timestamp: datetime
    currency: str = "USD"


class CompanyProfile(BaseModel):
    """Company profile information."""

    ticker: str
    name: str
    sector: str
    industry: str
    market_cap: float
    description: Optional[str] = None
    website: Optional[str] = None
    employees: Optional[int] = None
    country: Optional[str] = None
    exchange: Optional[str] = None


class FundamentalData(BaseModel):
    """Fundamental financial metrics."""

    ticker: str
    pe_ratio: Optional[float] = None
    pb_ratio: Optional[float] = None
    roe: Optional[float] = None
    roa: Optional[float] = None
    profit_margin: Optional[float] = None
    revenue_growth: Optional[float] = None
    debt_to_equity: Optional[float] = None
    dividend_yield: Optional[float] = None
    eps: Optional[float] = None
    book_value: Optional[float] = None


class FinancialRatios(BaseModel):
    """Calculated financial ratios."""

    ticker: str
    current_ratio: Optional[float] = None
    quick_ratio: Optional[float] = None
    debt_ratio: Optional[float] = None
    interest_coverage: Optional[float] = None
    asset_turnover: Optional[float] = None
    inventory_turnover: Optional[float] = None
    receivables_turnover: Optional[float] = None
    gross_margin: Optional[float] = None
    operating_margin: Optional[float] = None
    net_margin: Optional[float] = None


class YFinanceError(BaseModel):
    """Structured error response for YFinance API failures."""

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
# YFinance MCP Server
# =============================================================================


class YFinanceMCPServer(BaseMCPServer):
    """MCP Server for Yahoo Finance data.

    Provides tools for fetching stock prices, company profiles,
    fundamental data, and financial ratios through the MCP protocol.
    """

    def __init__(self, cache_ttl_seconds: int = 300):
        """Initialize the YFinance MCP Server.

        Args:
            cache_ttl_seconds: Cache TTL in seconds (default 5 minutes).
        """
        super().__init__(name="yfinance", version="1.0.0")
        self._cache = SimpleCache(ttl_seconds=cache_ttl_seconds)
        self._register_tools()
        self._register_resources()

    def _register_tools(self) -> None:
        """Register all available tools."""
        # Price tools
        self.register_tool(
            name="get_stock_price",
            description="Fetch current stock price data including OHLCV (Open, High, Low, Close, Volume)",
            handler=self._get_stock_price,
            input_schema={
                "type": "object",
                "properties": {
                    "ticker": {
                        "type": "string",
                        "description": "Stock ticker symbol (e.g., 'AAPL', 'MSFT')",
                    },
                },
                "required": ["ticker"],
            },
        )

        # Profile tools
        self.register_tool(
            name="get_company_profile",
            description="Fetch company profile including sector, industry, and market cap",
            handler=self._get_company_profile,
            input_schema={
                "type": "object",
                "properties": {
                    "ticker": {
                        "type": "string",
                        "description": "Stock ticker symbol",
                    },
                },
                "required": ["ticker"],
            },
        )

        # Fundamental tools
        self.register_tool(
            name="get_fundamentals",
            description="Fetch fundamental financial metrics including P/E, ROE, ROA, profit margins",
            handler=self._get_fundamentals,
            input_schema={
                "type": "object",
                "properties": {
                    "ticker": {
                        "type": "string",
                        "description": "Stock ticker symbol",
                    },
                },
                "required": ["ticker"],
            },
        )

        # Ratio calculation tools
        self.register_tool(
            name="calculate_ratios",
            description="Calculate financial ratios from company financial statements",
            handler=self._calculate_ratios,
            input_schema={
                "type": "object",
                "properties": {
                    "ticker": {
                        "type": "string",
                        "description": "Stock ticker symbol",
                    },
                },
                "required": ["ticker"],
            },
        )

    def _register_resources(self) -> None:
        """Register all available resources."""
        self.register_resource(
            uri="yfinance://price/{ticker}",
            name="Stock Price",
            description="Current stock price data for a ticker",
        )
        self.register_resource(
            uri="yfinance://profile/{ticker}",
            name="Company Profile",
            description="Company profile information",
        )
        self.register_resource(
            uri="yfinance://fundamentals/{ticker}",
            name="Fundamental Metrics",
            description="Fundamental financial metrics",
        )

    # -------------------------------------------------------------------------
    # Tool Implementations
    # -------------------------------------------------------------------------

    async def _get_stock_price(self, ticker: str) -> dict[str, Any]:
        """Fetch current stock price data.

        Args:
            ticker: Stock ticker symbol.

        Returns:
            Price data dictionary.

        Raises:
            MCPInvalidParamsError: If ticker is invalid.
            MCPError: If API call fails.
        """
        ticker = ticker.upper().strip()
        if not ticker:
            raise MCPInvalidParamsError("Ticker symbol is required", param="ticker")

        cache_key = f"price:{ticker}"
        cached = self._cache.get(cache_key)
        if cached:
            logger.debug(f"Cache hit for price data: {ticker}")
            return cached

        try:
            stock = yf.Ticker(ticker)
            info = stock.info

            if not info or info.get("regularMarketPrice") is None:
                # Try fast_info as fallback
                try:
                    fast_info = stock.fast_info
                    if fast_info.last_price is None:
                        raise MCPInvalidParamsError(
                            f"Ticker '{ticker}' not found", param="ticker"
                        )
                    # Use fast_info data
                    result = PriceData(
                        ticker=ticker,
                        current_price=fast_info.last_price,
                        open=fast_info.open or 0.0,
                        high=fast_info.day_high or 0.0,
                        low=fast_info.day_low or 0.0,
                        close=fast_info.previous_close or 0.0,
                        volume=int(fast_info.last_volume or 0),
                        timestamp=datetime.utcnow(),
                        currency=info.get("currency", "USD"),
                    ).model_dump()
                    self._cache.set(cache_key, result)
                    return result
                except Exception:
                    raise MCPInvalidParamsError(
                        f"Ticker '{ticker}' not found", param="ticker"
                    )

            result = PriceData(
                ticker=ticker,
                current_price=info.get("regularMarketPrice") or info.get("currentPrice", 0.0),
                open=info.get("regularMarketOpen") or info.get("open", 0.0),
                high=info.get("regularMarketDayHigh") or info.get("dayHigh", 0.0),
                low=info.get("regularMarketDayLow") or info.get("dayLow", 0.0),
                close=info.get("regularMarketPreviousClose") or info.get("previousClose", 0.0),
                volume=int(info.get("regularMarketVolume") or info.get("volume", 0)),
                timestamp=datetime.utcnow(),
                currency=info.get("currency", "USD"),
            ).model_dump()

            # Convert datetime to ISO string for JSON serialization
            result["timestamp"] = result["timestamp"].isoformat()
            self._cache.set(cache_key, result)
            logger.info(f"Fetched price data for {ticker}")
            return result

        except MCPInvalidParamsError:
            raise
        except Exception as e:
            logger.error(f"YFinance API error for {ticker}: {e}")
            raise MCPError(
                message=f"Failed to fetch price data for {ticker}: {str(e)}",
                code=MCPErrorCode.SERVER_ERROR,
                details={"ticker": ticker},
            )

    async def _get_company_profile(self, ticker: str) -> dict[str, Any]:
        """Fetch company profile information.

        Args:
            ticker: Stock ticker symbol.

        Returns:
            Company profile dictionary.

        Raises:
            MCPInvalidParamsError: If ticker is invalid.
            MCPError: If API call fails.
        """
        ticker = ticker.upper().strip()
        if not ticker:
            raise MCPInvalidParamsError("Ticker symbol is required", param="ticker")

        cache_key = f"profile:{ticker}"
        cached = self._cache.get(cache_key)
        if cached:
            logger.debug(f"Cache hit for profile data: {ticker}")
            return cached

        try:
            stock = yf.Ticker(ticker)
            info = stock.info

            if not info or not info.get("longName"):
                raise MCPInvalidParamsError(
                    f"Ticker '{ticker}' not found", param="ticker"
                )

            result = CompanyProfile(
                ticker=ticker,
                name=info.get("longName") or info.get("shortName", ticker),
                sector=info.get("sector", "Unknown"),
                industry=info.get("industry", "Unknown"),
                market_cap=float(info.get("marketCap", 0)),
                description=info.get("longBusinessSummary"),
                website=info.get("website"),
                employees=info.get("fullTimeEmployees"),
                country=info.get("country"),
                exchange=info.get("exchange"),
            ).model_dump()

            self._cache.set(cache_key, result)
            logger.info(f"Fetched profile for {ticker}")
            return result

        except MCPInvalidParamsError:
            raise
        except Exception as e:
            logger.error(f"YFinance API error for {ticker}: {e}")
            raise MCPError(
                message=f"Failed to fetch company profile for {ticker}: {str(e)}",
                code=MCPErrorCode.SERVER_ERROR,
                details={"ticker": ticker},
            )


    async def _get_fundamentals(self, ticker: str) -> dict[str, Any]:
        """Fetch fundamental financial metrics.

        Args:
            ticker: Stock ticker symbol.

        Returns:
            Fundamental data dictionary.

        Raises:
            MCPInvalidParamsError: If ticker is invalid.
            MCPError: If API call fails.
        """
        ticker = ticker.upper().strip()
        if not ticker:
            raise MCPInvalidParamsError("Ticker symbol is required", param="ticker")

        cache_key = f"fundamentals:{ticker}"
        cached = self._cache.get(cache_key)
        if cached:
            logger.debug(f"Cache hit for fundamentals data: {ticker}")
            return cached

        try:
            stock = yf.Ticker(ticker)
            info = stock.info

            if not info:
                raise MCPInvalidParamsError(
                    f"Ticker '{ticker}' not found", param="ticker"
                )

            result = FundamentalData(
                ticker=ticker,
                pe_ratio=info.get("trailingPE") or info.get("forwardPE"),
                pb_ratio=info.get("priceToBook"),
                roe=info.get("returnOnEquity"),
                roa=info.get("returnOnAssets"),
                profit_margin=info.get("profitMargins"),
                revenue_growth=info.get("revenueGrowth"),
                debt_to_equity=info.get("debtToEquity"),
                dividend_yield=info.get("dividendYield"),
                eps=info.get("trailingEps") or info.get("forwardEps"),
                book_value=info.get("bookValue"),
            ).model_dump()

            self._cache.set(cache_key, result)
            logger.info(f"Fetched fundamentals for {ticker}")
            return result

        except MCPInvalidParamsError:
            raise
        except Exception as e:
            logger.error(f"YFinance API error for {ticker}: {e}")
            raise MCPError(
                message=f"Failed to fetch fundamentals for {ticker}: {str(e)}",
                code=MCPErrorCode.SERVER_ERROR,
                details={"ticker": ticker},
            )

    async def _calculate_ratios(self, ticker: str) -> dict[str, Any]:
        """Calculate financial ratios from company statements.

        Args:
            ticker: Stock ticker symbol.

        Returns:
            Financial ratios dictionary.

        Raises:
            MCPInvalidParamsError: If ticker is invalid.
            MCPError: If API call fails.
        """
        ticker = ticker.upper().strip()
        if not ticker:
            raise MCPInvalidParamsError("Ticker symbol is required", param="ticker")

        cache_key = f"ratios:{ticker}"
        cached = self._cache.get(cache_key)
        if cached:
            logger.debug(f"Cache hit for ratios data: {ticker}")
            return cached

        try:
            stock = yf.Ticker(ticker)
            info = stock.info

            if not info:
                raise MCPInvalidParamsError(
                    f"Ticker '{ticker}' not found", param="ticker"
                )

            # Get balance sheet and income statement for ratio calculations
            balance_sheet = stock.balance_sheet
            income_stmt = stock.income_stmt

            # Initialize ratios
            current_ratio = None
            quick_ratio = None
            debt_ratio = None
            interest_coverage = None
            asset_turnover = None
            inventory_turnover = None
            receivables_turnover = None

            # Calculate ratios from financial statements if available
            if not balance_sheet.empty:
                latest_bs = balance_sheet.iloc[:, 0]

                # Current Ratio = Current Assets / Current Liabilities
                current_assets = latest_bs.get("Current Assets")
                current_liabilities = latest_bs.get("Current Liabilities")
                if current_assets and current_liabilities and current_liabilities != 0:
                    current_ratio = float(current_assets / current_liabilities)

                # Quick Ratio = (Current Assets - Inventory) / Current Liabilities
                inventory = latest_bs.get("Inventory", 0) or 0
                if current_assets and current_liabilities and current_liabilities != 0:
                    quick_ratio = float((current_assets - inventory) / current_liabilities)

                # Debt Ratio = Total Debt / Total Assets
                total_debt = latest_bs.get("Total Debt")
                total_assets = latest_bs.get("Total Assets")
                if total_debt and total_assets and total_assets != 0:
                    debt_ratio = float(total_debt / total_assets)

            if not income_stmt.empty and not balance_sheet.empty:
                latest_is = income_stmt.iloc[:, 0]
                latest_bs = balance_sheet.iloc[:, 0]

                # Interest Coverage = EBIT / Interest Expense
                ebit = latest_is.get("EBIT")
                interest_expense = latest_is.get("Interest Expense")
                if ebit and interest_expense and interest_expense != 0:
                    interest_coverage = float(abs(ebit / interest_expense))

                # Asset Turnover = Revenue / Total Assets
                revenue = latest_is.get("Total Revenue")
                total_assets = latest_bs.get("Total Assets")
                if revenue and total_assets and total_assets != 0:
                    asset_turnover = float(revenue / total_assets)

                # Inventory Turnover = COGS / Average Inventory
                cogs = latest_is.get("Cost Of Revenue")
                inventory = latest_bs.get("Inventory")
                if cogs and inventory and inventory != 0:
                    inventory_turnover = float(cogs / inventory)

                # Receivables Turnover = Revenue / Accounts Receivable
                receivables = latest_bs.get("Accounts Receivable")
                if revenue and receivables and receivables != 0:
                    receivables_turnover = float(revenue / receivables)

            result = FinancialRatios(
                ticker=ticker,
                current_ratio=current_ratio,
                quick_ratio=quick_ratio,
                debt_ratio=debt_ratio,
                interest_coverage=interest_coverage,
                asset_turnover=asset_turnover,
                inventory_turnover=inventory_turnover,
                receivables_turnover=receivables_turnover,
                gross_margin=info.get("grossMargins"),
                operating_margin=info.get("operatingMargins"),
                net_margin=info.get("profitMargins"),
            ).model_dump()

            self._cache.set(cache_key, result)
            logger.info(f"Calculated ratios for {ticker}")
            return result

        except MCPInvalidParamsError:
            raise
        except Exception as e:
            logger.error(f"YFinance API error for {ticker}: {e}")
            raise MCPError(
                message=f"Failed to calculate ratios for {ticker}: {str(e)}",
                code=MCPErrorCode.SERVER_ERROR,
                details={"ticker": ticker},
            )

    # -------------------------------------------------------------------------
    # Resource Implementation
    # -------------------------------------------------------------------------

    async def read_resource(self, uri: str) -> ContentBlock:
        """Read a resource by URI.

        Args:
            uri: Resource URI (e.g., 'yfinance://price/AAPL').

        Returns:
            Content block with resource data.

        Raises:
            MCPInvalidParamsError: If URI is invalid.
            MCPError: If resource fetch fails.
        """
        if not uri.startswith("yfinance://"):
            raise MCPInvalidParamsError(f"Invalid URI scheme: {uri}")

        path = uri.replace("yfinance://", "")
        parts = path.split("/")

        if len(parts) != 2:
            raise MCPInvalidParamsError(f"Invalid URI format: {uri}")

        resource_type, ticker = parts

        if resource_type == "price":
            data = await self._get_stock_price(ticker)
        elif resource_type == "profile":
            data = await self._get_company_profile(ticker)
        elif resource_type == "fundamentals":
            data = await self._get_fundamentals(ticker)
        else:
            raise MCPInvalidParamsError(f"Unknown resource type: {resource_type}")

        return ContentBlock(type=ContentType.JSON, data=data)

    # -------------------------------------------------------------------------
    # Health Check
    # -------------------------------------------------------------------------

    async def health_check(self) -> HealthStatus:
        """Check server health by testing YFinance API.

        Returns:
            Health status with latency information.
        """
        start_time = time.time()
        try:
            # Test with a known ticker
            stock = yf.Ticker("AAPL")
            _ = stock.fast_info.last_price
            latency_ms = (time.time() - start_time) * 1000

            return HealthStatus(
                healthy=True,
                server_name=self.name,
                latency_ms=latency_ms,
            )
        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            return HealthStatus(
                healthy=False,
                server_name=self.name,
                latency_ms=latency_ms,
                error_message=str(e),
            )
