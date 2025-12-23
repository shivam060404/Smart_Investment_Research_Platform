import logging
import time
from datetime import datetime
from typing import Any, Optional

import yfinance as yf
import pandas as pd
import numpy as np
from pydantic import BaseModel, Field

from app.mcp.base import BaseMCPServer, ContentBlock, ContentType, HealthStatus
from app.mcp.errors import MCPError, MCPErrorCode, MCPInvalidParamsError

logger = logging.getLogger(__name__)


# =============================================================================
# Response Models
# =============================================================================


class SMAResult(BaseModel):
    """Simple Moving Average result."""

    ticker: str
    period: int
    sma_value: Optional[float] = None
    current_price: Optional[float] = None
    price_vs_sma: Optional[str] = None  # "above" or "below"
    calculated_at: datetime = Field(default_factory=datetime.utcnow)
    warning: Optional[str] = None


class RSIResult(BaseModel):
    """Relative Strength Index result."""

    ticker: str
    period: int
    rsi_value: Optional[float] = None
    signal: Optional[str] = None  # "overbought", "oversold", "neutral"
    calculated_at: datetime = Field(default_factory=datetime.utcnow)
    warning: Optional[str] = None


class MACDResult(BaseModel):
    """MACD (Moving Average Convergence Divergence) result."""

    ticker: str
    macd_line: Optional[float] = None
    signal_line: Optional[float] = None
    histogram: Optional[float] = None
    signal: Optional[str] = None  # "bullish", "bearish", "neutral"
    calculated_at: datetime = Field(default_factory=datetime.utcnow)
    warning: Optional[str] = None


class PatternResult(BaseModel):
    """Pattern detection result."""

    ticker: str
    pattern: str
    detected: bool
    description: str
    detected_at: Optional[datetime] = None
    warning: Optional[str] = None


class TechnicalSignals(BaseModel):
    """Aggregated technical signals."""

    ticker: str
    sma_20: Optional[float] = None
    sma_50: Optional[float] = None
    sma_200: Optional[float] = None
    rsi: Optional[float] = None
    macd: Optional[float] = None
    macd_signal: Optional[float] = None
    macd_histogram: Optional[float] = None
    golden_cross: bool = False
    death_cross: bool = False
    overall_signal: Optional[str] = None  # "bullish", "bearish", "neutral"
    calculated_at: datetime = Field(default_factory=datetime.utcnow)
    warnings: list[str] = Field(default_factory=list)


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
# Technical Indicators MCP Server
# =============================================================================


class TechnicalMCPServer(BaseMCPServer):
    """MCP Server for Technical Indicators.

    Provides tools for calculating technical analysis indicators
    including SMA, RSI, MACD, and pattern detection through the MCP protocol.
    """

    def __init__(self, cache_ttl_seconds: int = 300):
        """Initialize the Technical MCP Server.

        Args:
            cache_ttl_seconds: Cache TTL in seconds (default 5 minutes).
        """
        super().__init__(name="technical", version="1.0.0")
        self._cache = SimpleCache(ttl_seconds=cache_ttl_seconds)
        self._register_tools()
        self._register_resources()

    def _register_tools(self) -> None:
        """Register all available tools."""
        # SMA tool
        self.register_tool(
            name="calculate_sma",
            description="Calculate Simple Moving Average (SMA) with configurable periods (20, 50, 200)",
            handler=self._calculate_sma,
            input_schema={
                "type": "object",
                "properties": {
                    "ticker": {
                        "type": "string",
                        "description": "Stock ticker symbol (e.g., 'AAPL', 'MSFT')",
                    },
                    "period": {
                        "type": "integer",
                        "description": "SMA period (default: 20, common values: 20, 50, 200)",
                        "default": 20,
                    },
                },
                "required": ["ticker"],
            },
        )

        # RSI tool
        self.register_tool(
            name="calculate_rsi",
            description="Calculate Relative Strength Index (RSI) with configurable period",
            handler=self._calculate_rsi,
            input_schema={
                "type": "object",
                "properties": {
                    "ticker": {
                        "type": "string",
                        "description": "Stock ticker symbol",
                    },
                    "period": {
                        "type": "integer",
                        "description": "RSI period (default: 14)",
                        "default": 14,
                    },
                },
                "required": ["ticker"],
            },
        )

        # MACD tool
        self.register_tool(
            name="calculate_macd",
            description="Calculate MACD (Moving Average Convergence Divergence) returning MACD line, signal, and histogram",
            handler=self._calculate_macd,
            input_schema={
                "type": "object",
                "properties": {
                    "ticker": {
                        "type": "string",
                        "description": "Stock ticker symbol",
                    },
                    "fast_period": {
                        "type": "integer",
                        "description": "Fast EMA period (default: 12)",
                        "default": 12,
                    },
                    "slow_period": {
                        "type": "integer",
                        "description": "Slow EMA period (default: 26)",
                        "default": 26,
                    },
                    "signal_period": {
                        "type": "integer",
                        "description": "Signal line period (default: 9)",
                        "default": 9,
                    },
                },
                "required": ["ticker"],
            },
        )

        # Pattern detection tool
        self.register_tool(
            name="detect_patterns",
            description="Detect Golden Cross and Death Cross patterns",
            handler=self._detect_patterns,
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

        # Aggregated signals tool
        self.register_tool(
            name="get_signals",
            description="Get aggregated technical signals including all indicators and patterns",
            handler=self._get_signals,
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
            uri="technical://indicators/{ticker}",
            name="Technical Indicators",
            description="All technical indicators for a ticker",
        )
        self.register_resource(
            uri="technical://signals/{ticker}",
            name="Trading Signals",
            description="Aggregated trading signals for a ticker",
        )

    # -------------------------------------------------------------------------
    # Helper Methods
    # -------------------------------------------------------------------------

    def _fetch_historical_data(
        self, ticker: str, period: str = "1y"
    ) -> Optional[pd.DataFrame]:
        """Fetch historical price data from YFinance.

        Args:
            ticker: Stock ticker symbol.
            period: Data period (default: 1 year).

        Returns:
            DataFrame with historical data or None if fetch fails.
        """
        cache_key = f"history:{ticker}:{period}"
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        try:
            stock = yf.Ticker(ticker)
            df = stock.history(period=period)

            if df.empty:
                return None

            self._cache.set(cache_key, df)
            return df

        except Exception as e:
            logger.error(f"Failed to fetch historical data for {ticker}: {e}")
            return None

    def _calculate_ema(self, data: pd.Series, period: int) -> pd.Series:
        """Calculate Exponential Moving Average.

        Args:
            data: Price series.
            period: EMA period.

        Returns:
            EMA series.
        """
        return data.ewm(span=period, adjust=False).mean()

    # -------------------------------------------------------------------------
    # Tool Implementations
    # -------------------------------------------------------------------------

    async def _calculate_sma(
        self, ticker: str, period: int = 20
    ) -> dict[str, Any]:
        """Calculate Simple Moving Average.

        Args:
            ticker: Stock ticker symbol.
            period: SMA period (default: 20).

        Returns:
            SMA result dictionary.

        Raises:
            MCPInvalidParamsError: If ticker is invalid.
            MCPError: If calculation fails.
        """
        ticker = ticker.upper().strip()
        if not ticker:
            raise MCPInvalidParamsError("Ticker symbol is required", param="ticker")

        if period < 1:
            raise MCPInvalidParamsError("Period must be at least 1", param="period")

        cache_key = f"sma:{ticker}:{period}"
        cached = self._cache.get(cache_key)
        if cached:
            logger.debug(f"Cache hit for SMA: {ticker}")
            return cached

        try:
            df = self._fetch_historical_data(ticker)

            if df is None or df.empty:
                raise MCPInvalidParamsError(
                    f"Ticker '{ticker}' not found or no data available",
                    param="ticker",
                )

            warning = None
            if len(df) < period:
                warning = f"Insufficient data: only {len(df)} days available, {period} required for accurate SMA"

            # Calculate SMA
            close_prices = df["Close"]
            sma = close_prices.rolling(window=period).mean()

            current_price = float(close_prices.iloc[-1])
            sma_value = float(sma.iloc[-1]) if not pd.isna(sma.iloc[-1]) else None

            price_vs_sma = None
            if sma_value is not None:
                price_vs_sma = "above" if current_price > sma_value else "below"

            result = SMAResult(
                ticker=ticker,
                period=period,
                sma_value=round(sma_value, 4) if sma_value else None,
                current_price=round(current_price, 4),
                price_vs_sma=price_vs_sma,
                warning=warning,
            ).model_dump()

            result["calculated_at"] = result["calculated_at"].isoformat()
            self._cache.set(cache_key, result)
            logger.info(f"Calculated SMA({period}) for {ticker}: {sma_value}")
            return result

        except MCPInvalidParamsError:
            raise
        except Exception as e:
            logger.error(f"SMA calculation error for {ticker}: {e}")
            raise MCPError(
                message=f"Failed to calculate SMA for {ticker}: {str(e)}",
                code=MCPErrorCode.SERVER_ERROR,
                details={"ticker": ticker, "period": period},
            )


    async def _calculate_rsi(
        self, ticker: str, period: int = 14
    ) -> dict[str, Any]:
        """Calculate Relative Strength Index.

        Args:
            ticker: Stock ticker symbol.
            period: RSI period (default: 14).

        Returns:
            RSI result dictionary.

        Raises:
            MCPInvalidParamsError: If ticker is invalid.
            MCPError: If calculation fails.
        """
        ticker = ticker.upper().strip()
        if not ticker:
            raise MCPInvalidParamsError("Ticker symbol is required", param="ticker")

        if period < 1:
            raise MCPInvalidParamsError("Period must be at least 1", param="period")

        cache_key = f"rsi:{ticker}:{period}"
        cached = self._cache.get(cache_key)
        if cached:
            logger.debug(f"Cache hit for RSI: {ticker}")
            return cached

        try:
            df = self._fetch_historical_data(ticker)

            if df is None or df.empty:
                raise MCPInvalidParamsError(
                    f"Ticker '{ticker}' not found or no data available",
                    param="ticker",
                )

            warning = None
            if len(df) < period + 1:
                warning = f"Insufficient data: only {len(df)} days available, {period + 1} required for accurate RSI"

            # Calculate RSI
            close_prices = df["Close"]
            delta = close_prices.diff()

            gain = delta.where(delta > 0, 0.0)
            loss = (-delta).where(delta < 0, 0.0)

            avg_gain = gain.rolling(window=period).mean()
            avg_loss = loss.rolling(window=period).mean()

            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))

            rsi_value = float(rsi.iloc[-1]) if not pd.isna(rsi.iloc[-1]) else None

            # Determine signal
            signal = None
            if rsi_value is not None:
                if rsi_value >= 70:
                    signal = "overbought"
                elif rsi_value <= 30:
                    signal = "oversold"
                else:
                    signal = "neutral"

            result = RSIResult(
                ticker=ticker,
                period=period,
                rsi_value=round(rsi_value, 2) if rsi_value else None,
                signal=signal,
                warning=warning,
            ).model_dump()

            result["calculated_at"] = result["calculated_at"].isoformat()
            self._cache.set(cache_key, result)
            logger.info(f"Calculated RSI({period}) for {ticker}: {rsi_value}")
            return result

        except MCPInvalidParamsError:
            raise
        except Exception as e:
            logger.error(f"RSI calculation error for {ticker}: {e}")
            raise MCPError(
                message=f"Failed to calculate RSI for {ticker}: {str(e)}",
                code=MCPErrorCode.SERVER_ERROR,
                details={"ticker": ticker, "period": period},
            )

    async def _calculate_macd(
        self,
        ticker: str,
        fast_period: int = 12,
        slow_period: int = 26,
        signal_period: int = 9,
    ) -> dict[str, Any]:
        """Calculate MACD (Moving Average Convergence Divergence).

        Args:
            ticker: Stock ticker symbol.
            fast_period: Fast EMA period (default: 12).
            slow_period: Slow EMA period (default: 26).
            signal_period: Signal line period (default: 9).

        Returns:
            MACD result dictionary with MACD line, signal line, and histogram.

        Raises:
            MCPInvalidParamsError: If ticker is invalid.
            MCPError: If calculation fails.
        """
        ticker = ticker.upper().strip()
        if not ticker:
            raise MCPInvalidParamsError("Ticker symbol is required", param="ticker")

        if fast_period < 1 or slow_period < 1 or signal_period < 1:
            raise MCPInvalidParamsError("All periods must be at least 1", param="period")

        if fast_period >= slow_period:
            raise MCPInvalidParamsError(
                "Fast period must be less than slow period", param="fast_period"
            )

        cache_key = f"macd:{ticker}:{fast_period}:{slow_period}:{signal_period}"
        cached = self._cache.get(cache_key)
        if cached:
            logger.debug(f"Cache hit for MACD: {ticker}")
            return cached

        try:
            df = self._fetch_historical_data(ticker)

            if df is None or df.empty:
                raise MCPInvalidParamsError(
                    f"Ticker '{ticker}' not found or no data available",
                    param="ticker",
                )

            min_required = slow_period + signal_period
            warning = None
            if len(df) < min_required:
                warning = f"Insufficient data: only {len(df)} days available, {min_required} required for accurate MACD"

            # Calculate MACD
            close_prices = df["Close"]

            fast_ema = self._calculate_ema(close_prices, fast_period)
            slow_ema = self._calculate_ema(close_prices, slow_period)

            macd_line = fast_ema - slow_ema
            signal_line = self._calculate_ema(macd_line, signal_period)
            histogram = macd_line - signal_line

            macd_value = float(macd_line.iloc[-1]) if not pd.isna(macd_line.iloc[-1]) else None
            signal_value = float(signal_line.iloc[-1]) if not pd.isna(signal_line.iloc[-1]) else None
            histogram_value = float(histogram.iloc[-1]) if not pd.isna(histogram.iloc[-1]) else None

            # Determine signal
            signal = None
            if macd_value is not None and signal_value is not None:
                if macd_value > signal_value:
                    signal = "bullish"
                elif macd_value < signal_value:
                    signal = "bearish"
                else:
                    signal = "neutral"

            result = MACDResult(
                ticker=ticker,
                macd_line=round(macd_value, 4) if macd_value else None,
                signal_line=round(signal_value, 4) if signal_value else None,
                histogram=round(histogram_value, 4) if histogram_value else None,
                signal=signal,
                warning=warning,
            ).model_dump()

            result["calculated_at"] = result["calculated_at"].isoformat()
            self._cache.set(cache_key, result)
            logger.info(f"Calculated MACD for {ticker}: line={macd_value}, signal={signal_value}")
            return result

        except MCPInvalidParamsError:
            raise
        except Exception as e:
            logger.error(f"MACD calculation error for {ticker}: {e}")
            raise MCPError(
                message=f"Failed to calculate MACD for {ticker}: {str(e)}",
                code=MCPErrorCode.SERVER_ERROR,
                details={"ticker": ticker},
            )


    async def _detect_patterns(self, ticker: str) -> dict[str, Any]:
        """Detect Golden Cross and Death Cross patterns.

        Golden Cross: 50-day SMA crosses above 200-day SMA (bullish)
        Death Cross: 50-day SMA crosses below 200-day SMA (bearish)

        Args:
            ticker: Stock ticker symbol.

        Returns:
            Dictionary with detected patterns.

        Raises:
            MCPInvalidParamsError: If ticker is invalid.
            MCPError: If detection fails.
        """
        ticker = ticker.upper().strip()
        if not ticker:
            raise MCPInvalidParamsError("Ticker symbol is required", param="ticker")

        cache_key = f"patterns:{ticker}"
        cached = self._cache.get(cache_key)
        if cached:
            logger.debug(f"Cache hit for patterns: {ticker}")
            return cached

        try:
            df = self._fetch_historical_data(ticker)

            if df is None or df.empty:
                raise MCPInvalidParamsError(
                    f"Ticker '{ticker}' not found or no data available",
                    param="ticker",
                )

            patterns = []
            warnings = []

            # Check for sufficient data
            if len(df) < 200:
                warnings.append(
                    f"Insufficient data: only {len(df)} days available, 200 required for accurate pattern detection"
                )

            close_prices = df["Close"]

            # Calculate SMAs
            sma_50 = close_prices.rolling(window=50).mean()
            sma_200 = close_prices.rolling(window=200).mean()

            # Check for Golden Cross (50 SMA crosses above 200 SMA)
            golden_cross_detected = False
            death_cross_detected = False

            if len(df) >= 200 and not pd.isna(sma_50.iloc[-1]) and not pd.isna(sma_200.iloc[-1]):
                # Check recent crossover (within last 5 days)
                for i in range(-5, 0):
                    if i - 1 >= -len(sma_50):
                        prev_50 = sma_50.iloc[i - 1]
                        prev_200 = sma_200.iloc[i - 1]
                        curr_50 = sma_50.iloc[i]
                        curr_200 = sma_200.iloc[i]

                        if not any(pd.isna([prev_50, prev_200, curr_50, curr_200])):
                            # Golden Cross: 50 SMA was below 200 SMA, now above
                            if prev_50 <= prev_200 and curr_50 > curr_200:
                                golden_cross_detected = True
                            # Death Cross: 50 SMA was above 200 SMA, now below
                            if prev_50 >= prev_200 and curr_50 < curr_200:
                                death_cross_detected = True

            patterns.append(
                PatternResult(
                    ticker=ticker,
                    pattern="golden_cross",
                    detected=golden_cross_detected,
                    description="50-day SMA crossed above 200-day SMA (bullish signal)",
                    detected_at=datetime.utcnow() if golden_cross_detected else None,
                    warning=warnings[0] if warnings and not golden_cross_detected else None,
                ).model_dump()
            )

            patterns.append(
                PatternResult(
                    ticker=ticker,
                    pattern="death_cross",
                    detected=death_cross_detected,
                    description="50-day SMA crossed below 200-day SMA (bearish signal)",
                    detected_at=datetime.utcnow() if death_cross_detected else None,
                    warning=warnings[0] if warnings and not death_cross_detected else None,
                ).model_dump()
            )

            # Convert datetime to ISO string
            for pattern in patterns:
                if pattern.get("detected_at"):
                    pattern["detected_at"] = pattern["detected_at"].isoformat()

            result = {
                "ticker": ticker,
                "patterns": patterns,
                "warnings": warnings,
            }

            self._cache.set(cache_key, result)
            logger.info(
                f"Pattern detection for {ticker}: golden_cross={golden_cross_detected}, death_cross={death_cross_detected}"
            )
            return result

        except MCPInvalidParamsError:
            raise
        except Exception as e:
            logger.error(f"Pattern detection error for {ticker}: {e}")
            raise MCPError(
                message=f"Failed to detect patterns for {ticker}: {str(e)}",
                code=MCPErrorCode.SERVER_ERROR,
                details={"ticker": ticker},
            )

    async def _get_signals(self, ticker: str) -> dict[str, Any]:
        """Get aggregated technical signals.

        Combines all technical indicators and patterns into a single response.

        Args:
            ticker: Stock ticker symbol.

        Returns:
            Aggregated technical signals dictionary.

        Raises:
            MCPInvalidParamsError: If ticker is invalid.
            MCPError: If aggregation fails.
        """
        ticker = ticker.upper().strip()
        if not ticker:
            raise MCPInvalidParamsError("Ticker symbol is required", param="ticker")

        cache_key = f"signals:{ticker}"
        cached = self._cache.get(cache_key)
        if cached:
            logger.debug(f"Cache hit for signals: {ticker}")
            return cached

        try:
            warnings = []

            # Calculate all indicators
            sma_20_result = await self._calculate_sma(ticker, 20)
            sma_50_result = await self._calculate_sma(ticker, 50)
            sma_200_result = await self._calculate_sma(ticker, 200)
            rsi_result = await self._calculate_rsi(ticker, 14)
            macd_result = await self._calculate_macd(ticker)
            patterns_result = await self._detect_patterns(ticker)

            # Collect warnings
            if sma_20_result.get("warning"):
                warnings.append(sma_20_result["warning"])
            if sma_50_result.get("warning"):
                warnings.append(sma_50_result["warning"])
            if sma_200_result.get("warning"):
                warnings.append(sma_200_result["warning"])
            if rsi_result.get("warning"):
                warnings.append(rsi_result["warning"])
            if macd_result.get("warning"):
                warnings.append(macd_result["warning"])
            if patterns_result.get("warnings"):
                warnings.extend(patterns_result["warnings"])

            # Remove duplicates
            warnings = list(set(warnings))

            # Extract pattern detection results
            golden_cross = False
            death_cross = False
            for pattern in patterns_result.get("patterns", []):
                if pattern["pattern"] == "golden_cross":
                    golden_cross = pattern["detected"]
                elif pattern["pattern"] == "death_cross":
                    death_cross = pattern["detected"]

            # Determine overall signal
            bullish_signals = 0
            bearish_signals = 0

            # RSI signals
            if rsi_result.get("signal") == "oversold":
                bullish_signals += 1
            elif rsi_result.get("signal") == "overbought":
                bearish_signals += 1

            # MACD signals
            if macd_result.get("signal") == "bullish":
                bullish_signals += 1
            elif macd_result.get("signal") == "bearish":
                bearish_signals += 1

            # Pattern signals
            if golden_cross:
                bullish_signals += 2  # Weight patterns more heavily
            if death_cross:
                bearish_signals += 2

            # Price vs SMA signals
            if sma_200_result.get("price_vs_sma") == "above":
                bullish_signals += 1
            elif sma_200_result.get("price_vs_sma") == "below":
                bearish_signals += 1

            overall_signal = "neutral"
            if bullish_signals > bearish_signals:
                overall_signal = "bullish"
            elif bearish_signals > bullish_signals:
                overall_signal = "bearish"

            result = TechnicalSignals(
                ticker=ticker,
                sma_20=sma_20_result.get("sma_value"),
                sma_50=sma_50_result.get("sma_value"),
                sma_200=sma_200_result.get("sma_value"),
                rsi=rsi_result.get("rsi_value"),
                macd=macd_result.get("macd_line"),
                macd_signal=macd_result.get("signal_line"),
                macd_histogram=macd_result.get("histogram"),
                golden_cross=golden_cross,
                death_cross=death_cross,
                overall_signal=overall_signal,
                warnings=warnings,
            ).model_dump()

            result["calculated_at"] = result["calculated_at"].isoformat()
            self._cache.set(cache_key, result)
            logger.info(f"Aggregated signals for {ticker}: {overall_signal}")
            return result

        except MCPInvalidParamsError:
            raise
        except Exception as e:
            logger.error(f"Signal aggregation error for {ticker}: {e}")
            raise MCPError(
                message=f"Failed to aggregate signals for {ticker}: {str(e)}",
                code=MCPErrorCode.SERVER_ERROR,
                details={"ticker": ticker},
            )

    # -------------------------------------------------------------------------
    # Resource Implementation
    # -------------------------------------------------------------------------

    async def read_resource(self, uri: str) -> ContentBlock:
        """Read a resource by URI.

        Args:
            uri: Resource URI (e.g., 'technical://indicators/AAPL').

        Returns:
            Content block with resource data.

        Raises:
            MCPInvalidParamsError: If URI is invalid.
            MCPError: If resource fetch fails.
        """
        if not uri.startswith("technical://"):
            raise MCPInvalidParamsError(f"Invalid URI scheme: {uri}")

        path = uri.replace("technical://", "")
        parts = path.split("/")

        if len(parts) != 2:
            raise MCPInvalidParamsError(f"Invalid URI format: {uri}")

        resource_type, ticker = parts

        if resource_type == "indicators":
            data = await self._get_signals(ticker)
        elif resource_type == "signals":
            data = await self._get_signals(ticker)
        else:
            raise MCPInvalidParamsError(f"Unknown resource type: {resource_type}")

        return ContentBlock(type=ContentType.JSON, data=data)

    # -------------------------------------------------------------------------
    # Health Check
    # -------------------------------------------------------------------------

    async def health_check(self) -> HealthStatus:
        """Check server health by testing indicator calculation.

        Returns:
            Health status with latency information.
        """
        start_time = time.time()
        try:
            # Test with a known ticker
            df = self._fetch_historical_data("AAPL", period="5d")
            if df is None or df.empty:
                raise Exception("Failed to fetch test data")

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
