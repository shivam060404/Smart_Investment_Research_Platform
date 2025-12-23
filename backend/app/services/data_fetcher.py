import logging
import time
from datetime import datetime, timedelta
from typing import Optional

import httpx
import yfinance as yf
from circuitbreaker import circuit, CircuitBreakerError

from app.config import get_settings
from app.models.stock import (
    StockData,
    FundamentalMetrics,
)

logger = logging.getLogger(__name__)
settings = get_settings()


def _log_external_call(service: str, operation: str, ticker: Optional[str] = None) -> None:
    """Log external service call start."""
    logger.debug(
        f"Calling {service}: {operation}",
        extra={
            "event": "external_call_start",
            "service": service,
            "operation": operation,
            "ticker": ticker,
        }
    )


def _log_external_response(
    service: str,
    operation: str,
    success: bool,
    duration_ms: float,
    ticker: Optional[str] = None,
    error: Optional[str] = None,
) -> None:
    """Log external service response."""
    level = logging.DEBUG if success else logging.WARNING
    status = "success" if success else "failed"
    
    extra = {
        "event": "external_call_complete",
        "service": service,
        "operation": operation,
        "success": success,
        "duration_ms": duration_ms,
        "ticker": ticker,
    }
    if error:
        extra["error"] = error
    
    logger.log(
        level,
        f"{service} {operation}: {status} ({duration_ms:.0f}ms)",
        extra=extra,
    )


# Circuit breaker configuration
YFINANCE_FAILURE_THRESHOLD = 5
YFINANCE_RECOVERY_TIMEOUT = 60
NEWSAPI_FAILURE_THRESHOLD = 3
NEWSAPI_RECOVERY_TIMEOUT = 30
ALPHAVANTAGE_FAILURE_THRESHOLD = 3
ALPHAVANTAGE_RECOVERY_TIMEOUT = 60


class DataFetcherError(Exception):
    """Base exception for data fetcher errors."""
    pass


class TickerNotFoundError(DataFetcherError):
    """Raised when a ticker symbol is not found."""
    pass


class ExternalAPIError(DataFetcherError):
    """Raised when an external API call fails."""
    pass


class DataFetcher:
    """Service for fetching stock data from external APIs."""

    def __init__(self):
        """Initialize the data fetcher with HTTP client."""
        self._http_client: Optional[httpx.AsyncClient] = None

    @property
    def http_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._http_client is None:
            self._http_client = httpx.AsyncClient(timeout=30.0)
        return self._http_client

    async def close(self):
        """Close HTTP client connections."""
        if self._http_client:
            await self._http_client.aclose()
            self._http_client = None

    # -------------------------------------------------------------------------
    # Yahoo Finance Integration
    # -------------------------------------------------------------------------

    @circuit(
        failure_threshold=YFINANCE_FAILURE_THRESHOLD,
        recovery_timeout=YFINANCE_RECOVERY_TIMEOUT,
    )
    def get_stock_data(self, ticker: str) -> StockData:
        """Fetch basic stock information from Yahoo Finance.

        Args:
            ticker: Stock ticker symbol (e.g., 'AAPL')

        Returns:
            StockData: Basic stock information

        Raises:
            TickerNotFoundError: If ticker is not found
            ExternalAPIError: If Yahoo Finance API fails
        """
        start_time = time.perf_counter()
        _log_external_call("YahooFinance", "get_stock_data", ticker)
        
        try:
            stock = yf.Ticker(ticker)
            info = stock.info

            # Check if ticker is valid
            if not info or info.get("regularMarketPrice") is None:
                # Try fast_info as fallback
                try:
                    fast_info = stock.fast_info
                    if fast_info.last_price is None:
                        raise TickerNotFoundError(f"Ticker '{ticker}' not found")
                except Exception:
                    raise TickerNotFoundError(f"Ticker '{ticker}' not found")

            result = StockData(
                ticker=ticker.upper(),
                name=info.get("longName") or info.get("shortName") or ticker,
                sector=info.get("sector", "Unknown"),
                industry=info.get("industry", "Unknown"),
                current_price=info.get("regularMarketPrice") or info.get("currentPrice", 0.0),
                market_cap=info.get("marketCap", 0.0),
                currency=info.get("currency", "USD"),
            )
            
            duration_ms = (time.perf_counter() - start_time) * 1000
            _log_external_response("YahooFinance", "get_stock_data", True, duration_ms, ticker)
            return result

        except TickerNotFoundError:
            duration_ms = (time.perf_counter() - start_time) * 1000
            _log_external_response("YahooFinance", "get_stock_data", False, duration_ms, ticker, "Ticker not found")
            raise
        except Exception as e:
            duration_ms = (time.perf_counter() - start_time) * 1000
            _log_external_response("YahooFinance", "get_stock_data", False, duration_ms, ticker, str(e))
            logger.error(f"Yahoo Finance error for {ticker}: {e}")
            raise ExternalAPIError(f"Failed to fetch stock data: {e}")


    @circuit(
        failure_threshold=YFINANCE_FAILURE_THRESHOLD,
        recovery_timeout=YFINANCE_RECOVERY_TIMEOUT,
    )
    def get_financial_metrics(self, ticker: str) -> FundamentalMetrics:
        """Fetch fundamental financial metrics from Yahoo Finance.

        Args:
            ticker: Stock ticker symbol

        Returns:
            FundamentalMetrics: Financial metrics for fundamental analysis

        Raises:
            TickerNotFoundError: If ticker is not found
            ExternalAPIError: If Yahoo Finance API fails
        """
        try:
            stock = yf.Ticker(ticker)
            info = stock.info

            if not info:
                raise TickerNotFoundError(f"Ticker '{ticker}' not found")

            return FundamentalMetrics(
                pe_ratio=info.get("trailingPE") or info.get("forwardPE"),
                pb_ratio=info.get("priceToBook"),
                roe=info.get("returnOnEquity"),
                roa=info.get("returnOnAssets"),
                debt_to_equity=info.get("debtToEquity"),
                revenue_growth=info.get("revenueGrowth"),
                profit_margin=info.get("profitMargins"),
                dividend_yield=info.get("dividendYield"),
            )

        except TickerNotFoundError:
            raise
        except Exception as e:
            logger.error(f"Yahoo Finance metrics error for {ticker}: {e}")
            raise ExternalAPIError(f"Failed to fetch financial metrics: {e}")

    @circuit(
        failure_threshold=YFINANCE_FAILURE_THRESHOLD,
        recovery_timeout=YFINANCE_RECOVERY_TIMEOUT,
    )
    def get_price_history(
        self,
        ticker: str,
        period: str = "1y",
        interval: str = "1d",
    ) -> list[dict]:
        """Fetch historical price data from Yahoo Finance.

        Args:
            ticker: Stock ticker symbol
            period: Time period (e.g., '1y', '6mo', '3mo')
            interval: Data interval (e.g., '1d', '1wk')

        Returns:
            List of price data dictionaries with date, open, high, low, close, volume

        Raises:
            TickerNotFoundError: If ticker is not found
            ExternalAPIError: If Yahoo Finance API fails
        """
        try:
            stock = yf.Ticker(ticker)
            history = stock.history(period=period, interval=interval)

            if history.empty:
                raise TickerNotFoundError(f"No price history for ticker '{ticker}'")

            price_data = []
            for date, row in history.iterrows():
                price_data.append({
                    "date": date.strftime("%Y-%m-%d"),
                    "open": float(row["Open"]),
                    "high": float(row["High"]),
                    "low": float(row["Low"]),
                    "close": float(row["Close"]),
                    "volume": int(row["Volume"]),
                })

            return price_data

        except TickerNotFoundError:
            raise
        except Exception as e:
            logger.error(f"Yahoo Finance history error for {ticker}: {e}")
            raise ExternalAPIError(f"Failed to fetch price history: {e}")


    # -------------------------------------------------------------------------
    # NewsAPI Integration
    # -------------------------------------------------------------------------

    @circuit(
        failure_threshold=NEWSAPI_FAILURE_THRESHOLD,
        recovery_timeout=NEWSAPI_RECOVERY_TIMEOUT,
    )
    async def get_news_articles(
        self,
        query: str,
        days_back: int = 7,
        page_size: int = 10,
    ) -> list[dict]:
        """Fetch news articles from NewsAPI.

        Args:
            query: Search query (typically company name or ticker)
            days_back: Number of days to look back for articles
            page_size: Maximum number of articles to return

        Returns:
            List of article dictionaries with title, description, source, url, publishedAt

        Raises:
            ExternalAPIError: If NewsAPI call fails
        """
        if not settings.news_api_key:
            logger.warning("NewsAPI key not configured")
            return []

        try:
            from_date = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")
            to_date = datetime.now().strftime("%Y-%m-%d")

            url = "https://newsapi.org/v2/everything"
            params = {
                "q": query,
                "from": from_date,
                "to": to_date,
                "sortBy": "relevancy",
                "pageSize": page_size,
                "language": "en",
                "apiKey": settings.news_api_key,
            }

            response = await self.http_client.get(url, params=params)
            response.raise_for_status()
            data = response.json()

            if data.get("status") != "ok":
                error_msg = data.get("message", "Unknown NewsAPI error")
                raise ExternalAPIError(f"NewsAPI error: {error_msg}")

            articles = []
            for article in data.get("articles", []):
                articles.append({
                    "title": article.get("title", ""),
                    "description": article.get("description", ""),
                    "content": article.get("content", ""),
                    "source": article.get("source", {}).get("name", "Unknown"),
                    "url": article.get("url", ""),
                    "published_at": article.get("publishedAt", ""),
                })

            logger.info(f"Fetched {len(articles)} news articles for '{query}'")
            return articles

        except httpx.HTTPStatusError as e:
            logger.error(f"NewsAPI HTTP error: {e}")
            raise ExternalAPIError(f"NewsAPI request failed: {e}")
        except Exception as e:
            logger.error(f"NewsAPI error for '{query}': {e}")
            raise ExternalAPIError(f"Failed to fetch news articles: {e}")


    # -------------------------------------------------------------------------
    # AlphaVantage Integration (Fallback)
    # -------------------------------------------------------------------------

    @circuit(
        failure_threshold=ALPHAVANTAGE_FAILURE_THRESHOLD,
        recovery_timeout=ALPHAVANTAGE_RECOVERY_TIMEOUT,
    )
    async def get_stock_data_alphavantage(self, ticker: str) -> StockData:
        """Fetch stock data from AlphaVantage as fallback.

        Args:
            ticker: Stock ticker symbol

        Returns:
            StockData: Basic stock information

        Raises:
            ExternalAPIError: If AlphaVantage API fails
        """
        if not settings.alpha_vantage_key:
            raise ExternalAPIError("AlphaVantage API key not configured")

        try:
            url = "https://www.alphavantage.co/query"
            params = {
                "function": "GLOBAL_QUOTE",
                "symbol": ticker,
                "apikey": settings.alpha_vantage_key,
            }

            response = await self.http_client.get(url, params=params)
            response.raise_for_status()
            data = response.json()

            if "Error Message" in data:
                raise TickerNotFoundError(f"Ticker '{ticker}' not found on AlphaVantage")

            if "Note" in data:
                raise ExternalAPIError("AlphaVantage rate limit exceeded")

            quote = data.get("Global Quote", {})
            if not quote:
                raise TickerNotFoundError(f"No data for ticker '{ticker}'")

            price = float(quote.get("05. price", 0))

            return StockData(
                ticker=ticker.upper(),
                name=ticker.upper(),  # AlphaVantage doesn't provide company name in quote
                sector="Unknown",
                industry="Unknown",
                current_price=price,
                market_cap=0.0,
                currency="USD",
            )

        except (TickerNotFoundError, ExternalAPIError):
            raise
        except Exception as e:
            logger.error(f"AlphaVantage error for {ticker}: {e}")
            raise ExternalAPIError(f"AlphaVantage request failed: {e}")

    @circuit(
        failure_threshold=ALPHAVANTAGE_FAILURE_THRESHOLD,
        recovery_timeout=ALPHAVANTAGE_RECOVERY_TIMEOUT,
    )
    async def get_financial_metrics_alphavantage(self, ticker: str) -> FundamentalMetrics:
        """Fetch fundamental metrics from AlphaVantage as fallback.

        Args:
            ticker: Stock ticker symbol

        Returns:
            FundamentalMetrics: Financial metrics

        Raises:
            ExternalAPIError: If AlphaVantage API fails
        """
        if not settings.alpha_vantage_key:
            raise ExternalAPIError("AlphaVantage API key not configured")

        try:
            url = "https://www.alphavantage.co/query"
            params = {
                "function": "OVERVIEW",
                "symbol": ticker,
                "apikey": settings.alpha_vantage_key,
            }

            response = await self.http_client.get(url, params=params)
            response.raise_for_status()
            data = response.json()

            if "Error Message" in data:
                raise TickerNotFoundError(f"Ticker '{ticker}' not found on AlphaVantage")

            if "Note" in data:
                raise ExternalAPIError("AlphaVantage rate limit exceeded")

            if not data or "Symbol" not in data:
                raise TickerNotFoundError(f"No overview data for ticker '{ticker}'")

            def safe_float(value) -> Optional[float]:
                """Safely convert value to float."""
                if value is None or value == "None" or value == "-":
                    return None
                try:
                    return float(value)
                except (ValueError, TypeError):
                    return None

            return FundamentalMetrics(
                pe_ratio=safe_float(data.get("PERatio")),
                pb_ratio=safe_float(data.get("PriceToBookRatio")),
                roe=safe_float(data.get("ReturnOnEquityTTM")),
                roa=safe_float(data.get("ReturnOnAssetsTTM")),
                debt_to_equity=None,  # Not directly available
                revenue_growth=safe_float(data.get("QuarterlyRevenueGrowthYOY")),
                profit_margin=safe_float(data.get("ProfitMargin")),
                dividend_yield=safe_float(data.get("DividendYield")),
            )

        except (TickerNotFoundError, ExternalAPIError):
            raise
        except Exception as e:
            logger.error(f"AlphaVantage metrics error for {ticker}: {e}")
            raise ExternalAPIError(f"AlphaVantage request failed: {e}")


    # -------------------------------------------------------------------------
    # Failover Methods (Yahoo Finance with AlphaVantage fallback)
    # -------------------------------------------------------------------------

    async def get_stock_data_with_fallback(self, ticker: str) -> StockData:
        """Fetch stock data with automatic failover to AlphaVantage.

        Tries Yahoo Finance first, falls back to AlphaVantage if unavailable.

        Args:
            ticker: Stock ticker symbol

        Returns:
            StockData: Basic stock information

        Raises:
            TickerNotFoundError: If ticker not found on any source
            ExternalAPIError: If all data sources fail
        """
        try:
            return self.get_stock_data(ticker)
        except CircuitBreakerError:
            logger.warning(f"Yahoo Finance circuit open, trying AlphaVantage for {ticker}")
        except ExternalAPIError as e:
            logger.warning(f"Yahoo Finance failed for {ticker}: {e}, trying AlphaVantage")
        except TickerNotFoundError:
            raise

        # Fallback to AlphaVantage
        try:
            return await self.get_stock_data_alphavantage(ticker)
        except CircuitBreakerError:
            raise ExternalAPIError("All data sources unavailable (circuit breakers open)")
        except TickerNotFoundError:
            raise
        except ExternalAPIError:
            raise ExternalAPIError("Failed to fetch stock data from all sources")

    async def get_financial_metrics_with_fallback(self, ticker: str) -> FundamentalMetrics:
        """Fetch financial metrics with automatic failover to AlphaVantage.

        Tries Yahoo Finance first, falls back to AlphaVantage if unavailable.

        Args:
            ticker: Stock ticker symbol

        Returns:
            FundamentalMetrics: Financial metrics

        Raises:
            TickerNotFoundError: If ticker not found on any source
            ExternalAPIError: If all data sources fail
        """
        try:
            return self.get_financial_metrics(ticker)
        except CircuitBreakerError:
            logger.warning(f"Yahoo Finance circuit open, trying AlphaVantage for {ticker}")
        except ExternalAPIError as e:
            logger.warning(f"Yahoo Finance metrics failed for {ticker}: {e}, trying AlphaVantage")
        except TickerNotFoundError:
            raise

        # Fallback to AlphaVantage
        try:
            return await self.get_financial_metrics_alphavantage(ticker)
        except CircuitBreakerError:
            raise ExternalAPIError("All data sources unavailable (circuit breakers open)")
        except TickerNotFoundError:
            raise
        except ExternalAPIError:
            raise ExternalAPIError("Failed to fetch financial metrics from all sources")


# Singleton instance for dependency injection
_data_fetcher: Optional[DataFetcher] = None


def get_data_fetcher() -> DataFetcher:
    """Get or create DataFetcher singleton instance.

    Returns:
        DataFetcher: Singleton data fetcher instance
    """
    global _data_fetcher
    if _data_fetcher is None:
        _data_fetcher = DataFetcher()
    return _data_fetcher
