import logging
import time
import uuid
from datetime import datetime, timedelta
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


class EquityPoint(BaseModel):
    """Single point in an equity curve."""

    date: str
    portfolio_value: float
    daily_return: Optional[float] = None
    cumulative_return: float


class BacktestResult(BaseModel):
    """Result from a backtest execution."""

    backtest_id: str
    ticker: str
    strategy: str
    start_date: str
    end_date: str
    initial_capital: float
    final_value: float
    total_return: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    status: str = "completed"
    created_at: datetime = Field(default_factory=datetime.utcnow)


class PerformanceMetrics(BaseModel):
    """Performance metrics for a backtest."""

    backtest_id: str
    ticker: str
    total_return: float
    annualized_return: Optional[float] = None
    sharpe_ratio: Optional[float] = None
    max_drawdown: float
    max_drawdown_duration_days: Optional[int] = None
    volatility: Optional[float] = None
    win_rate: Optional[float] = None
    profit_factor: Optional[float] = None
    calculated_at: datetime = Field(default_factory=datetime.utcnow)


class EquityCurve(BaseModel):
    """Equity curve with daily portfolio values."""

    backtest_id: str
    ticker: str
    strategy: str
    points: list[EquityPoint]
    start_date: str
    end_date: str


class HistoricalRecommendation(BaseModel):
    """Historical recommendation record."""

    recommendation_id: str
    ticker: str
    recommendation: str  # "buy", "sell", "hold"
    date: str
    price_at_recommendation: float
    outcome: Optional[str] = None  # "profit", "loss", "pending"
    return_pct: Optional[float] = None


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
# Backtesting MCP Server
# =============================================================================


class BacktestingMCPServer(BaseMCPServer):
    """MCP Server for Backtesting.

    Provides tools for running strategy backtests, calculating performance
    metrics, and retrieving equity curves through the MCP protocol.
    """

    def __init__(self, cache_ttl_seconds: int = 300):
        """Initialize the Backtesting MCP Server.

        Args:
            cache_ttl_seconds: Cache TTL in seconds (default 5 minutes).
        """
        super().__init__(name="backtesting", version="1.0.0")
        self._cache = SimpleCache(ttl_seconds=cache_ttl_seconds)
        self._backtest_results: dict[str, dict[str, Any]] = {}
        self._register_tools()
        self._register_resources()

    def _register_tools(self) -> None:
        """Register all available tools."""
        # Run backtest tool
        self.register_tool(
            name="run_backtest",
            description="Execute a strategy backtest on historical price data",
            handler=self._run_backtest,
            input_schema={
                "type": "object",
                "properties": {
                    "ticker": {
                        "type": "string",
                        "description": "Stock ticker symbol (e.g., 'AAPL', 'MSFT')",
                    },
                    "strategy": {
                        "type": "string",
                        "description": "Strategy name: 'sma_crossover', 'rsi', 'buy_and_hold', 'macd'",
                        "enum": ["sma_crossover", "rsi", "buy_and_hold", "macd"],
                    },
                    "period": {
                        "type": "string",
                        "description": "Backtest period: '1y', '2y', '5y' (default: '1y')",
                        "default": "1y",
                    },
                    "initial_capital": {
                        "type": "number",
                        "description": "Initial capital for backtest (default: 10000)",
                        "default": 10000,
                    },
                },
                "required": ["ticker", "strategy"],
            },
        )

        # Get equity curve tool
        self.register_tool(
            name="get_equity_curve",
            description="Get the equity curve (daily portfolio values) for a completed backtest",
            handler=self._get_equity_curve,
            input_schema={
                "type": "object",
                "properties": {
                    "backtest_id": {
                        "type": "string",
                        "description": "Backtest ID from run_backtest result",
                    },
                },
                "required": ["backtest_id"],
            },
        )

        # Calculate metrics tool
        self.register_tool(
            name="calculate_metrics",
            description="Calculate performance metrics including total return, Sharpe ratio, and max drawdown",
            handler=self._calculate_metrics,
            input_schema={
                "type": "object",
                "properties": {
                    "backtest_id": {
                        "type": "string",
                        "description": "Backtest ID from run_backtest result",
                    },
                },
                "required": ["backtest_id"],
            },
        )

        # Get historical recommendations tool
        self.register_tool(
            name="get_historical_recommendations",
            description="Get historical recommendations and their outcomes for a ticker",
            handler=self._get_historical_recommendations,
            input_schema={
                "type": "object",
                "properties": {
                    "ticker": {
                        "type": "string",
                        "description": "Stock ticker symbol",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of recommendations to return (default: 10)",
                        "default": 10,
                    },
                },
                "required": ["ticker"],
            },
        )

    def _register_resources(self) -> None:
        """Register all available resources."""
        self.register_resource(
            uri="backtest://results/{id}",
            name="Backtest Results",
            description="Results from a completed backtest",
        )
        self.register_resource(
            uri="backtest://history/{ticker}",
            name="Historical Backtests",
            description="Historical backtest results for a ticker",
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

    def _calculate_sma(self, prices: pd.Series, period: int) -> pd.Series:
        """Calculate Simple Moving Average."""
        return prices.rolling(window=period).mean()

    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index."""
        delta = prices.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = (-delta).where(delta < 0, 0.0)
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    def _calculate_macd(
        self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
    ) -> tuple[pd.Series, pd.Series]:
        """Calculate MACD line and signal line."""
        fast_ema = prices.ewm(span=fast, adjust=False).mean()
        slow_ema = prices.ewm(span=slow, adjust=False).mean()
        macd_line = fast_ema - slow_ema
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        return macd_line, signal_line


    # -------------------------------------------------------------------------
    # Strategy Implementations
    # -------------------------------------------------------------------------

    def _run_sma_crossover_strategy(
        self, df: pd.DataFrame, initial_capital: float
    ) -> tuple[pd.Series, list[dict]]:
        """Run SMA crossover strategy (50/200 day).

        Buy when 50-day SMA crosses above 200-day SMA.
        Sell when 50-day SMA crosses below 200-day SMA.

        Args:
            df: Historical price data.
            initial_capital: Starting capital.

        Returns:
            Tuple of (portfolio values series, list of trades).
        """
        close = df["Close"]
        sma_50 = self._calculate_sma(close, 50)
        sma_200 = self._calculate_sma(close, 200)

        # Generate signals
        signals = pd.Series(0, index=df.index)
        signals[sma_50 > sma_200] = 1  # Long signal
        signals[sma_50 <= sma_200] = 0  # No position

        # Calculate positions (shift to avoid look-ahead bias)
        positions = signals.shift(1).fillna(0)

        # Calculate returns
        daily_returns = close.pct_change().fillna(0)
        strategy_returns = positions * daily_returns

        # Calculate portfolio value
        portfolio_values = initial_capital * (1 + strategy_returns).cumprod()

        # Track trades
        trades = []
        position_changes = positions.diff().fillna(0)
        for date, change in position_changes.items():
            if change == 1:
                trades.append({
                    "date": str(date.date()),
                    "action": "buy",
                    "price": float(close.loc[date]),
                })
            elif change == -1:
                trades.append({
                    "date": str(date.date()),
                    "action": "sell",
                    "price": float(close.loc[date]),
                })

        return portfolio_values, trades

    def _run_rsi_strategy(
        self, df: pd.DataFrame, initial_capital: float
    ) -> tuple[pd.Series, list[dict]]:
        """Run RSI strategy.

        Buy when RSI < 30 (oversold).
        Sell when RSI > 70 (overbought).

        Args:
            df: Historical price data.
            initial_capital: Starting capital.

        Returns:
            Tuple of (portfolio values series, list of trades).
        """
        close = df["Close"]
        rsi = self._calculate_rsi(close, 14)

        # Generate signals
        signals = pd.Series(0, index=df.index)
        position = 0

        for i in range(1, len(df)):
            if rsi.iloc[i] < 30 and position == 0:
                position = 1
                signals.iloc[i] = 1
            elif rsi.iloc[i] > 70 and position == 1:
                position = 0
                signals.iloc[i] = 0
            else:
                signals.iloc[i] = position

        # Calculate positions (shift to avoid look-ahead bias)
        positions = signals.shift(1).fillna(0)

        # Calculate returns
        daily_returns = close.pct_change().fillna(0)
        strategy_returns = positions * daily_returns

        # Calculate portfolio value
        portfolio_values = initial_capital * (1 + strategy_returns).cumprod()

        # Track trades
        trades = []
        position_changes = positions.diff().fillna(0)
        for date, change in position_changes.items():
            if change == 1:
                trades.append({
                    "date": str(date.date()),
                    "action": "buy",
                    "price": float(close.loc[date]),
                })
            elif change == -1:
                trades.append({
                    "date": str(date.date()),
                    "action": "sell",
                    "price": float(close.loc[date]),
                })

        return portfolio_values, trades

    def _run_buy_and_hold_strategy(
        self, df: pd.DataFrame, initial_capital: float
    ) -> tuple[pd.Series, list[dict]]:
        """Run buy and hold strategy.

        Buy at the start and hold until the end.

        Args:
            df: Historical price data.
            initial_capital: Starting capital.

        Returns:
            Tuple of (portfolio values series, list of trades).
        """
        close = df["Close"]

        # Always in position
        positions = pd.Series(1, index=df.index)

        # Calculate returns
        daily_returns = close.pct_change().fillna(0)
        strategy_returns = positions * daily_returns

        # Calculate portfolio value
        portfolio_values = initial_capital * (1 + strategy_returns).cumprod()

        # Single buy trade at start
        trades = [{
            "date": str(df.index[0].date()),
            "action": "buy",
            "price": float(close.iloc[0]),
        }]

        return portfolio_values, trades

    def _run_macd_strategy(
        self, df: pd.DataFrame, initial_capital: float
    ) -> tuple[pd.Series, list[dict]]:
        """Run MACD crossover strategy.

        Buy when MACD crosses above signal line.
        Sell when MACD crosses below signal line.

        Args:
            df: Historical price data.
            initial_capital: Starting capital.

        Returns:
            Tuple of (portfolio values series, list of trades).
        """
        close = df["Close"]
        macd_line, signal_line = self._calculate_macd(close)

        # Generate signals
        signals = pd.Series(0, index=df.index)
        signals[macd_line > signal_line] = 1  # Long signal
        signals[macd_line <= signal_line] = 0  # No position

        # Calculate positions (shift to avoid look-ahead bias)
        positions = signals.shift(1).fillna(0)

        # Calculate returns
        daily_returns = close.pct_change().fillna(0)
        strategy_returns = positions * daily_returns

        # Calculate portfolio value
        portfolio_values = initial_capital * (1 + strategy_returns).cumprod()

        # Track trades
        trades = []
        position_changes = positions.diff().fillna(0)
        for date, change in position_changes.items():
            if change == 1:
                trades.append({
                    "date": str(date.date()),
                    "action": "buy",
                    "price": float(close.loc[date]),
                })
            elif change == -1:
                trades.append({
                    "date": str(date.date()),
                    "action": "sell",
                    "price": float(close.loc[date]),
                })

        return portfolio_values, trades


    # -------------------------------------------------------------------------
    # Tool Implementations
    # -------------------------------------------------------------------------

    async def _run_backtest(
        self,
        ticker: str,
        strategy: str,
        period: str = "1y",
        initial_capital: float = 10000,
    ) -> dict[str, Any]:
        """Execute a strategy backtest.

        Args:
            ticker: Stock ticker symbol.
            strategy: Strategy name ('sma_crossover', 'rsi', 'buy_and_hold', 'macd').
            period: Backtest period ('1y', '2y', '5y').
            initial_capital: Starting capital.

        Returns:
            Backtest result dictionary.

        Raises:
            MCPInvalidParamsError: If parameters are invalid.
            MCPError: If backtest fails.
        """
        ticker = ticker.upper().strip()
        if not ticker:
            raise MCPInvalidParamsError("Ticker symbol is required", param="ticker")

        valid_strategies = ["sma_crossover", "rsi", "buy_and_hold", "macd"]
        if strategy not in valid_strategies:
            raise MCPInvalidParamsError(
                f"Invalid strategy. Must be one of: {valid_strategies}",
                param="strategy",
            )

        valid_periods = ["1y", "2y", "5y"]
        if period not in valid_periods:
            raise MCPInvalidParamsError(
                f"Invalid period. Must be one of: {valid_periods}",
                param="period",
            )

        if initial_capital <= 0:
            raise MCPInvalidParamsError(
                "Initial capital must be positive",
                param="initial_capital",
            )

        try:
            df = self._fetch_historical_data(ticker, period)

            if df is None or df.empty:
                raise MCPInvalidParamsError(
                    f"Ticker '{ticker}' not found or no data available",
                    param="ticker",
                )

            # Run the selected strategy
            strategy_map = {
                "sma_crossover": self._run_sma_crossover_strategy,
                "rsi": self._run_rsi_strategy,
                "buy_and_hold": self._run_buy_and_hold_strategy,
                "macd": self._run_macd_strategy,
            }

            portfolio_values, trades = strategy_map[strategy](df, initial_capital)

            # Calculate results
            final_value = float(portfolio_values.iloc[-1])
            total_return = (final_value - initial_capital) / initial_capital

            # Count winning/losing trades
            winning_trades = 0
            losing_trades = 0
            for i in range(0, len(trades) - 1, 2):
                if i + 1 < len(trades):
                    buy_price = trades[i]["price"]
                    sell_price = trades[i + 1]["price"]
                    if sell_price > buy_price:
                        winning_trades += 1
                    else:
                        losing_trades += 1

            # Generate backtest ID
            backtest_id = str(uuid.uuid4())[:8]

            result = BacktestResult(
                backtest_id=backtest_id,
                ticker=ticker,
                strategy=strategy,
                start_date=str(df.index[0].date()),
                end_date=str(df.index[-1].date()),
                initial_capital=initial_capital,
                final_value=round(final_value, 2),
                total_return=round(total_return, 4),
                total_trades=len(trades),
                winning_trades=winning_trades,
                losing_trades=losing_trades,
            ).model_dump()

            result["created_at"] = result["created_at"].isoformat()

            # Store for later retrieval
            self._backtest_results[backtest_id] = {
                "result": result,
                "portfolio_values": portfolio_values,
                "trades": trades,
                "df": df,
            }

            logger.info(
                f"Backtest completed for {ticker} with {strategy}: "
                f"return={total_return:.2%}"
            )
            return result

        except MCPInvalidParamsError:
            raise
        except Exception as e:
            logger.error(f"Backtest error for {ticker}: {e}")
            raise MCPError(
                message=f"Failed to run backtest for {ticker}: {str(e)}",
                code=MCPErrorCode.SERVER_ERROR,
                details={"ticker": ticker, "strategy": strategy},
            )

    async def _get_equity_curve(self, backtest_id: str) -> dict[str, Any]:
        """Get equity curve for a completed backtest.

        Args:
            backtest_id: Backtest ID from run_backtest.

        Returns:
            Equity curve with daily portfolio values.

        Raises:
            MCPInvalidParamsError: If backtest_id is invalid.
        """
        if not backtest_id:
            raise MCPInvalidParamsError("Backtest ID is required", param="backtest_id")

        if backtest_id not in self._backtest_results:
            raise MCPInvalidParamsError(
                f"Backtest '{backtest_id}' not found. Run a backtest first.",
                param="backtest_id",
            )

        try:
            stored = self._backtest_results[backtest_id]
            portfolio_values = stored["portfolio_values"]
            result_data = stored["result"]

            # Build equity curve points
            points = []
            initial_value = portfolio_values.iloc[0]
            prev_value = initial_value

            for date, value in portfolio_values.items():
                daily_return = (value - prev_value) / prev_value if prev_value != 0 else 0
                cumulative_return = (value - initial_value) / initial_value

                points.append(EquityPoint(
                    date=str(date.date()),
                    portfolio_value=round(float(value), 2),
                    daily_return=round(float(daily_return), 6),
                    cumulative_return=round(float(cumulative_return), 4),
                ).model_dump())

                prev_value = value

            result = EquityCurve(
                backtest_id=backtest_id,
                ticker=result_data["ticker"],
                strategy=result_data["strategy"],
                points=points,
                start_date=result_data["start_date"],
                end_date=result_data["end_date"],
            ).model_dump()

            logger.info(f"Retrieved equity curve for backtest {backtest_id}")
            return result

        except MCPInvalidParamsError:
            raise
        except Exception as e:
            logger.error(f"Error getting equity curve for {backtest_id}: {e}")
            raise MCPError(
                message=f"Failed to get equity curve: {str(e)}",
                code=MCPErrorCode.SERVER_ERROR,
                details={"backtest_id": backtest_id},
            )


    async def _calculate_metrics(self, backtest_id: str) -> dict[str, Any]:
        """Calculate performance metrics for a backtest.

        Args:
            backtest_id: Backtest ID from run_backtest.

        Returns:
            Performance metrics including Sharpe ratio, max drawdown, etc.

        Raises:
            MCPInvalidParamsError: If backtest_id is invalid.
        """
        if not backtest_id:
            raise MCPInvalidParamsError("Backtest ID is required", param="backtest_id")

        if backtest_id not in self._backtest_results:
            raise MCPInvalidParamsError(
                f"Backtest '{backtest_id}' not found. Run a backtest first.",
                param="backtest_id",
            )

        try:
            stored = self._backtest_results[backtest_id]
            portfolio_values = stored["portfolio_values"]
            result_data = stored["result"]
            trades = stored["trades"]

            # Calculate daily returns
            daily_returns = portfolio_values.pct_change().dropna()

            # Total return
            total_return = result_data["total_return"]

            # Annualized return (assuming 252 trading days)
            num_days = len(portfolio_values)
            annualized_return = None
            if num_days > 1:
                annualized_return = ((1 + total_return) ** (252 / num_days)) - 1

            # Volatility (annualized)
            volatility = None
            if len(daily_returns) > 1:
                volatility = float(daily_returns.std() * np.sqrt(252))

            # Sharpe ratio (assuming risk-free rate of 0.02)
            sharpe_ratio = None
            risk_free_rate = 0.02
            if volatility and volatility > 0 and annualized_return is not None:
                sharpe_ratio = (annualized_return - risk_free_rate) / volatility

            # Max drawdown
            cumulative_max = portfolio_values.cummax()
            drawdown = (portfolio_values - cumulative_max) / cumulative_max
            max_drawdown = float(drawdown.min())

            # Max drawdown duration
            max_drawdown_duration = None
            if max_drawdown < 0:
                in_drawdown = drawdown < 0
                drawdown_periods = []
                current_period = 0
                for is_dd in in_drawdown:
                    if is_dd:
                        current_period += 1
                    else:
                        if current_period > 0:
                            drawdown_periods.append(current_period)
                        current_period = 0
                if current_period > 0:
                    drawdown_periods.append(current_period)
                if drawdown_periods:
                    max_drawdown_duration = max(drawdown_periods)

            # Win rate
            win_rate = None
            total_trades = result_data["winning_trades"] + result_data["losing_trades"]
            if total_trades > 0:
                win_rate = result_data["winning_trades"] / total_trades

            # Profit factor
            profit_factor = None
            if trades and len(trades) >= 2:
                gross_profit = 0
                gross_loss = 0
                for i in range(0, len(trades) - 1, 2):
                    if i + 1 < len(trades):
                        buy_price = trades[i]["price"]
                        sell_price = trades[i + 1]["price"]
                        pnl = sell_price - buy_price
                        if pnl > 0:
                            gross_profit += pnl
                        else:
                            gross_loss += abs(pnl)
                if gross_loss > 0:
                    profit_factor = gross_profit / gross_loss

            result = PerformanceMetrics(
                backtest_id=backtest_id,
                ticker=result_data["ticker"],
                total_return=round(total_return, 4),
                annualized_return=round(annualized_return, 4) if annualized_return else None,
                sharpe_ratio=round(sharpe_ratio, 4) if sharpe_ratio else None,
                max_drawdown=round(max_drawdown, 4),
                max_drawdown_duration_days=max_drawdown_duration,
                volatility=round(volatility, 4) if volatility else None,
                win_rate=round(win_rate, 4) if win_rate else None,
                profit_factor=round(profit_factor, 4) if profit_factor else None,
            ).model_dump()

            result["calculated_at"] = result["calculated_at"].isoformat()

            logger.info(
                f"Calculated metrics for backtest {backtest_id}: "
                f"sharpe={sharpe_ratio}, max_dd={max_drawdown}"
            )
            return result

        except MCPInvalidParamsError:
            raise
        except Exception as e:
            logger.error(f"Error calculating metrics for {backtest_id}: {e}")
            raise MCPError(
                message=f"Failed to calculate metrics: {str(e)}",
                code=MCPErrorCode.SERVER_ERROR,
                details={"backtest_id": backtest_id},
            )

    async def _get_historical_recommendations(
        self, ticker: str, limit: int = 10
    ) -> dict[str, Any]:
        """Get historical recommendations and outcomes for a ticker.

        This generates simulated historical recommendations based on
        technical indicators for demonstration purposes.

        Args:
            ticker: Stock ticker symbol.
            limit: Maximum number of recommendations to return.

        Returns:
            List of historical recommendations with outcomes.

        Raises:
            MCPInvalidParamsError: If ticker is invalid.
        """
        ticker = ticker.upper().strip()
        if not ticker:
            raise MCPInvalidParamsError("Ticker symbol is required", param="ticker")

        if limit < 1 or limit > 100:
            raise MCPInvalidParamsError(
                "Limit must be between 1 and 100",
                param="limit",
            )

        try:
            df = self._fetch_historical_data(ticker, "1y")

            if df is None or df.empty:
                raise MCPInvalidParamsError(
                    f"Ticker '{ticker}' not found or no data available",
                    param="ticker",
                )

            close = df["Close"]
            rsi = self._calculate_rsi(close, 14)

            recommendations = []
            rec_dates = []

            # Generate recommendations based on RSI signals
            for i in range(30, len(df) - 5, 10):  # Every 10 days, starting from day 30
                if len(recommendations) >= limit:
                    break

                rsi_val = rsi.iloc[i]
                date = df.index[i]
                price = float(close.iloc[i])

                # Determine recommendation
                if rsi_val < 30:
                    rec = "buy"
                elif rsi_val > 70:
                    rec = "sell"
                else:
                    rec = "hold"

                # Calculate outcome (5 days later)
                future_price = float(close.iloc[i + 5])
                return_pct = (future_price - price) / price

                if rec == "buy":
                    outcome = "profit" if return_pct > 0 else "loss"
                elif rec == "sell":
                    outcome = "profit" if return_pct < 0 else "loss"
                else:
                    outcome = "neutral"

                recommendations.append(HistoricalRecommendation(
                    recommendation_id=str(uuid.uuid4())[:8],
                    ticker=ticker,
                    recommendation=rec,
                    date=str(date.date()),
                    price_at_recommendation=round(price, 2),
                    outcome=outcome,
                    return_pct=round(return_pct, 4),
                ).model_dump())

            result = {
                "ticker": ticker,
                "recommendations": recommendations,
                "total_count": len(recommendations),
            }

            logger.info(f"Retrieved {len(recommendations)} historical recommendations for {ticker}")
            return result

        except MCPInvalidParamsError:
            raise
        except Exception as e:
            logger.error(f"Error getting recommendations for {ticker}: {e}")
            raise MCPError(
                message=f"Failed to get historical recommendations: {str(e)}",
                code=MCPErrorCode.SERVER_ERROR,
                details={"ticker": ticker},
            )

    # -------------------------------------------------------------------------
    # Resource Implementation
    # -------------------------------------------------------------------------

    async def read_resource(self, uri: str) -> ContentBlock:
        """Read a resource by URI.

        Args:
            uri: Resource URI (e.g., 'backtest://results/abc123').

        Returns:
            Content block with resource data.

        Raises:
            MCPInvalidParamsError: If URI is invalid.
            MCPError: If resource fetch fails.
        """
        if not uri.startswith("backtest://"):
            raise MCPInvalidParamsError(f"Invalid URI scheme: {uri}")

        path = uri.replace("backtest://", "")
        parts = path.split("/")

        if len(parts) != 2:
            raise MCPInvalidParamsError(f"Invalid URI format: {uri}")

        resource_type, identifier = parts

        if resource_type == "results":
            if identifier not in self._backtest_results:
                raise MCPInvalidParamsError(f"Backtest '{identifier}' not found")
            data = self._backtest_results[identifier]["result"]
        elif resource_type == "history":
            data = await self._get_historical_recommendations(identifier)
        else:
            raise MCPInvalidParamsError(f"Unknown resource type: {resource_type}")

        return ContentBlock(type=ContentType.JSON, data=data)

    # -------------------------------------------------------------------------
    # Health Check
    # -------------------------------------------------------------------------

    async def health_check(self) -> HealthStatus:
        """Check server health by testing data fetch.

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
