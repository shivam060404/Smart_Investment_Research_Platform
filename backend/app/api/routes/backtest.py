import asyncio
import logging
import math
from datetime import datetime, timedelta, date
from typing import Any, Optional, Literal

import yfinance as yf
import pandas as pd
import numpy as np
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from app.models.responses import (
    BacktestResponse,
    BacktestTrade,
    EquityPoint,
    ErrorCode,
    ErrorResponse,
    AIBacktestResponse,
    AISignalRecord,
    AgentSignalSummary,
    BaselineComparison,
    StrategyMetrics,
)
from app.models.requests import AnalysisInterval
from app.services.ai_backtest_service import (
    AIBacktestService,
    get_ai_backtest_service,
    AIBacktestError,
    MistralUnavailableError,
    AnalysisInterval as ServiceAnalysisInterval,
    MistralRetryConfig,
    BacktestWarning,
)
from app.utils.validators import validate_ticker_http

logger = logging.getLogger(__name__)

router = APIRouter()


# =============================================================================
# Request Models
# =============================================================================

class BacktestRequest(BaseModel):
    """Request model for backtesting endpoint."""
    
    ticker: str = Field(..., min_length=1, max_length=10, description="Stock ticker symbol")
    start_date: str = Field(..., description="Backtest start date (YYYY-MM-DD)")
    end_date: str = Field(..., description="Backtest end date (YYYY-MM-DD)")
    initial_capital: float = Field(default=10000.0, gt=0, description="Initial capital")
    strategy: Optional[str] = Field(
        default="auto",
        description="Strategy: 'auto', 'ai', 'ai_live', 'sma_crossover', 'rsi', 'macd', 'buy_hold'"
    )
    analysis_interval: Optional[str] = Field(
        default="weekly",
        description="Analysis interval for AI strategies: 'daily', 'weekly', 'biweekly', 'monthly'"
    )
    include_baseline_comparison: bool = Field(
        default=True,
        description="Whether to include baseline strategy comparison for AI backtests"
    )


# =============================================================================
# Technical Indicator Functions
# =============================================================================

def calculate_sma(prices: pd.Series, period: int) -> pd.Series:
    """Calculate Simple Moving Average."""
    return prices.rolling(window=period).mean()


def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """Calculate Relative Strength Index."""
    delta = prices.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def calculate_macd(
    prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Calculate MACD line, signal line, and histogram."""
    fast_ema = prices.ewm(span=fast, adjust=False).mean()
    slow_ema = prices.ewm(span=slow, adjust=False).mean()
    macd_line = fast_ema - slow_ema
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def calculate_bollinger_bands(prices: pd.Series, period: int = 20, std_dev: float = 2.0):
    """Calculate Bollinger Bands."""
    sma = prices.rolling(window=period).mean()
    std = prices.rolling(window=period).std()
    upper = sma + (std * std_dev)
    lower = sma - (std * std_dev)
    return upper, sma, lower


# =============================================================================
# Strategy Implementations
# =============================================================================

def run_sma_crossover_strategy(
    df: pd.DataFrame, initial_capital: float
) -> tuple[pd.Series, list[dict], str]:
    """Run SMA crossover strategy (50/200 day)."""
    close = df["Close"]
    sma_50 = calculate_sma(close, 50)
    sma_200 = calculate_sma(close, 200)

    signals = pd.Series(0, index=df.index)
    signals[sma_50 > sma_200] = 1
    signals[sma_50 <= sma_200] = 0

    positions = signals.shift(1).fillna(0)
    daily_returns = close.pct_change().fillna(0)
    strategy_returns = positions * daily_returns
    portfolio_values = initial_capital * (1 + strategy_returns).cumprod()

    trades = []
    position_changes = positions.diff().fillna(0)
    for date, change in position_changes.items():
        if change == 1:
            trades.append({"date": date, "action": "BUY", "price": float(close.loc[date])})
        elif change == -1:
            trades.append({"date": date, "action": "SELL", "price": float(close.loc[date])})

    return portfolio_values, trades, "SMA Crossover (50/200)"


def run_rsi_strategy(
    df: pd.DataFrame, initial_capital: float
) -> tuple[pd.Series, list[dict], str]:
    """Run RSI strategy (buy oversold, sell overbought)."""
    close = df["Close"]
    rsi = calculate_rsi(close, 14)

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

    positions = signals.shift(1).fillna(0)
    daily_returns = close.pct_change().fillna(0)
    strategy_returns = positions * daily_returns
    portfolio_values = initial_capital * (1 + strategy_returns).cumprod()

    trades = []
    position_changes = positions.diff().fillna(0)
    for date, change in position_changes.items():
        if change == 1:
            trades.append({"date": date, "action": "BUY", "price": float(close.loc[date])})
        elif change == -1:
            trades.append({"date": date, "action": "SELL", "price": float(close.loc[date])})

    return portfolio_values, trades, "RSI (14)"


def run_macd_strategy(
    df: pd.DataFrame, initial_capital: float
) -> tuple[pd.Series, list[dict], str]:
    """Run MACD crossover strategy."""
    close = df["Close"]
    macd_line, signal_line, _ = calculate_macd(close)

    signals = pd.Series(0, index=df.index)
    signals[macd_line > signal_line] = 1
    signals[macd_line <= signal_line] = 0

    positions = signals.shift(1).fillna(0)
    daily_returns = close.pct_change().fillna(0)
    strategy_returns = positions * daily_returns
    portfolio_values = initial_capital * (1 + strategy_returns).cumprod()

    trades = []
    position_changes = positions.diff().fillna(0)
    for date, change in position_changes.items():
        if change == 1:
            trades.append({"date": date, "action": "BUY", "price": float(close.loc[date])})
        elif change == -1:
            trades.append({"date": date, "action": "SELL", "price": float(close.loc[date])})

    return portfolio_values, trades, "MACD Crossover"


def run_buy_and_hold_strategy(
    df: pd.DataFrame, initial_capital: float
) -> tuple[pd.Series, list[dict], str]:
    """Run buy and hold strategy."""
    close = df["Close"]
    positions = pd.Series(1, index=df.index)
    daily_returns = close.pct_change().fillna(0)
    strategy_returns = positions * daily_returns
    portfolio_values = initial_capital * (1 + strategy_returns).cumprod()

    trades = [{"date": df.index[0], "action": "BUY", "price": float(close.iloc[0])}]

    return portfolio_values, trades, "Buy & Hold"


async def run_ai_strategy(
    df: pd.DataFrame, 
    ticker: str,
    initial_capital: float,
    analysis_interval: int = 5  # Analyze every N days
) -> tuple[pd.Series, list[dict], str, list[dict]]:
    """Run AI multi-agent strategy.
    
    Simulates the multi-agent system by generating signals based on:
    - Fundamental metrics (P/E, growth indicators)
    - Technical indicators (SMA, RSI, MACD)
    - Sentiment proxy (price momentum as sentiment indicator)
    - Risk metrics (volatility, drawdown)
    
    Returns portfolio values, trades, strategy name, and AI signals log.
    """
    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    volume = df["Volume"]
    
    # Pre-calculate all indicators
    sma_20 = calculate_sma(close, 20)
    sma_50 = calculate_sma(close, 50)
    sma_200 = calculate_sma(close, 200)
    rsi = calculate_rsi(close, 14)
    macd_line, macd_signal, macd_hist = calculate_macd(close)
    bb_upper, bb_middle, bb_lower = calculate_bollinger_bands(close, 20)
    
    # Volatility (20-day rolling std)
    volatility = close.pct_change().rolling(window=20).std() * np.sqrt(252) * 100
    
    # Momentum (10-day price change)
    momentum = close.pct_change(periods=10) * 100
    
    # Volume trend
    volume_sma = volume.rolling(window=20).mean()
    volume_ratio = volume / volume_sma
    
    signals = pd.Series(0, index=df.index)
    ai_signals_log = []
    position = 0
    
    for i in range(max(200, analysis_interval), len(df), analysis_interval):
        date = df.index[i]
        current_price = close.iloc[i]
        
        # =================================================================
        # FUNDAMENTAL AGENT (simulated with price-based metrics)
        # =================================================================
        # Use price relative to moving averages as value indicator
        price_to_sma200 = (current_price / sma_200.iloc[i] - 1) * 100 if not pd.isna(sma_200.iloc[i]) else 0
        
        fundamental_score = 50  # Base score
        if price_to_sma200 < -10:  # Undervalued (price below SMA200)
            fundamental_score = 70
        elif price_to_sma200 > 20:  # Overvalued
            fundamental_score = 30
        else:
            fundamental_score = 50 - price_to_sma200 * 0.5
        
        fundamental_score = max(0, min(100, fundamental_score))
        fundamental_signal = "bullish" if fundamental_score > 55 else "bearish" if fundamental_score < 45 else "neutral"
        
        # =================================================================
        # TECHNICAL AGENT
        # =================================================================
        tech_score = 50
        tech_factors = []
        
        # SMA signals
        if not pd.isna(sma_50.iloc[i]) and not pd.isna(sma_200.iloc[i]):
            if sma_50.iloc[i] > sma_200.iloc[i]:
                tech_score += 15
                tech_factors.append("Golden cross pattern")
            else:
                tech_score -= 15
                tech_factors.append("Death cross pattern")
        
        # RSI signals
        rsi_val = rsi.iloc[i] if not pd.isna(rsi.iloc[i]) else 50
        if rsi_val < 30:
            tech_score += 20
            tech_factors.append("RSI oversold")
        elif rsi_val > 70:
            tech_score -= 20
            tech_factors.append("RSI overbought")
        
        # MACD signals
        if not pd.isna(macd_line.iloc[i]) and not pd.isna(macd_signal.iloc[i]):
            if macd_line.iloc[i] > macd_signal.iloc[i]:
                tech_score += 10
                tech_factors.append("MACD bullish")
            else:
                tech_score -= 10
                tech_factors.append("MACD bearish")
        
        # Bollinger Band signals
        if not pd.isna(bb_lower.iloc[i]) and current_price < bb_lower.iloc[i]:
            tech_score += 10
            tech_factors.append("Price below lower BB")
        elif not pd.isna(bb_upper.iloc[i]) and current_price > bb_upper.iloc[i]:
            tech_score -= 10
            tech_factors.append("Price above upper BB")
        
        tech_score = max(0, min(100, tech_score))
        tech_signal = "bullish" if tech_score > 55 else "bearish" if tech_score < 45 else "neutral"
        
        # =================================================================
        # SENTIMENT AGENT (using momentum and volume as proxy)
        # =================================================================
        sentiment_score = 50
        
        mom_val = momentum.iloc[i] if not pd.isna(momentum.iloc[i]) else 0
        if mom_val > 5:
            sentiment_score += 25
        elif mom_val < -5:
            sentiment_score -= 25
        else:
            sentiment_score += mom_val * 3
        
        vol_ratio = volume_ratio.iloc[i] if not pd.isna(volume_ratio.iloc[i]) else 1
        if vol_ratio > 1.5 and mom_val > 0:
            sentiment_score += 10  # High volume on up move
        elif vol_ratio > 1.5 and mom_val < 0:
            sentiment_score -= 10  # High volume on down move
        
        sentiment_score = max(0, min(100, sentiment_score))
        sentiment_signal = "bullish" if sentiment_score > 55 else "bearish" if sentiment_score < 45 else "neutral"
        
        # =================================================================
        # RISK AGENT
        # =================================================================
        risk_score = 50  # Higher = more risky
        risk_factors = []
        
        vol_val = volatility.iloc[i] if not pd.isna(volatility.iloc[i]) else 20
        if vol_val > 40:
            risk_score = 80
            risk_factors.append("High volatility")
        elif vol_val > 25:
            risk_score = 60
            risk_factors.append("Elevated volatility")
        else:
            risk_score = 30
            risk_factors.append("Low volatility")
        
        # Drawdown risk
        rolling_max = close.iloc[max(0, i-60):i+1].max()
        drawdown = (current_price / rolling_max - 1) * 100
        if drawdown < -15:
            risk_score += 20
            risk_factors.append(f"In drawdown ({drawdown:.1f}%)")
        
        risk_score = max(0, min(100, risk_score))
        risk_signal = "high" if risk_score > 60 else "low" if risk_score < 40 else "moderate"
        
        # =================================================================
        # ORCHESTRATOR: Weighted Synthesis
        # =================================================================
        # Weights: Fundamental 30%, Technical 25%, Sentiment 25%, Risk 20%
        # For risk, invert the score (high risk = bearish)
        inverted_risk = 100 - risk_score
        
        weighted_score = (
            fundamental_score * 0.30 +
            tech_score * 0.25 +
            sentiment_score * 0.25 +
            inverted_risk * 0.20
        )
        
        # Confidence based on signal agreement and score variance
        signals_list = [fundamental_signal, tech_signal, sentiment_signal]
        scores_list = [fundamental_score, tech_score, sentiment_score, inverted_risk]
        bullish_count = sum(1 for s in signals_list if s == "bullish")
        bearish_count = sum(1 for s in signals_list if s == "bearish")
        
        # Calculate confidence based on multiple factors
        # 1. Signal agreement (base confidence)
        if bullish_count == 3 or bearish_count == 3:
            base_confidence = 85  # All signals agree
        elif bullish_count >= 2 or bearish_count >= 2:
            base_confidence = 70  # Majority agree
        else:
            base_confidence = 45  # Mixed signals
        
        # 2. Score variance adjustment (lower variance = higher confidence)
        score_variance = np.var(scores_list)
        variance_penalty = min(20, score_variance / 50)  # Max 20% penalty
        
        # 3. Weighted score strength adjustment
        score_strength = abs(weighted_score - 50) / 50  # 0 to 1
        strength_bonus = score_strength * 10  # Up to 10% bonus for strong signals
        
        # Calculate final confidence
        confidence = base_confidence - variance_penalty + strength_bonus
        
        # Reduce confidence if high risk
        if risk_score > 70:
            confidence -= 15
        elif risk_score > 50:
            confidence -= 5
        
        # Clamp confidence to valid range
        confidence = max(25, min(95, confidence))
        
        # Final recommendation
        if weighted_score >= 60 and confidence >= 50:
            recommendation = "BUY"
            new_position = 1
        elif weighted_score <= 40 or (risk_score > 75 and position == 1):
            recommendation = "SELL"
            new_position = 0
        else:
            recommendation = "HOLD"
            new_position = position
        
        # Log AI signal
        ai_signals_log.append({
            "date": str(date.date()) if hasattr(date, 'date') else str(date)[:10],
            "weighted_score": round(weighted_score, 1),
            "recommendation": recommendation,
            "confidence": round(confidence, 1),
            "fundamental": {"score": round(fundamental_score, 1), "signal": fundamental_signal},
            "technical": {"score": round(tech_score, 1), "signal": tech_signal},
            "sentiment": {"score": round(sentiment_score, 1), "signal": sentiment_signal},
            "risk": {"score": round(risk_score, 1), "signal": risk_signal},
        })
        
        # Update position for the next interval
        for j in range(i, min(i + analysis_interval, len(df))):
            signals.iloc[j] = new_position
        
        position = new_position
    
    # Fill initial period with no position
    signals.iloc[:max(200, analysis_interval)] = 0
    
    # Calculate portfolio values
    positions = signals.shift(1).fillna(0)
    daily_returns = close.pct_change().fillna(0)
    strategy_returns = positions * daily_returns
    portfolio_values = initial_capital * (1 + strategy_returns).cumprod()
    
    # Extract trades
    trades = []
    position_changes = positions.diff().fillna(0)
    for date, change in position_changes.items():
        if change == 1:
            trades.append({"date": date, "action": "BUY", "price": float(close.loc[date])})
        elif change == -1:
            trades.append({"date": date, "action": "SELL", "price": float(close.loc[date])})
    
    return portfolio_values, trades, "Multi-Agent AI Strategy", ai_signals_log


# =============================================================================
# Metrics Calculation
# =============================================================================

def calculate_sharpe_ratio(
    returns: list[float],
    risk_free_rate: float = 0.02,
    periods_per_year: int = 252,
) -> float:
    """Calculate the Sharpe ratio."""
    if not returns or len(returns) < 2:
        return 0.0
    
    mean_return = sum(returns) / len(returns)
    variance = sum((r - mean_return) ** 2 for r in returns) / (len(returns) - 1)
    std_dev = math.sqrt(variance) if variance > 0 else 0.0
    
    if std_dev == 0:
        return 0.0
    
    annualized_return = mean_return * periods_per_year
    annualized_std = std_dev * math.sqrt(periods_per_year)
    
    sharpe = (annualized_return - risk_free_rate) / annualized_std
    return round(sharpe, 4)


def calculate_max_drawdown(equity_curve: list[float]) -> float:
    """Calculate maximum drawdown."""
    if not equity_curve or len(equity_curve) < 2:
        return 0.0
    
    peak = equity_curve[0]
    max_drawdown = 0.0
    
    for equity in equity_curve:
        if equity > peak:
            peak = equity
        drawdown = (peak - equity) / peak * 100 if peak > 0 else 0
        max_drawdown = max(max_drawdown, drawdown)
    
    return round(max_drawdown, 2)


# =============================================================================
# Extended Response Model
# =============================================================================

class ExtendedBacktestResponse(BacktestResponse):
    """Extended backtest response with strategy info and AI signals."""
    
    strategy_name: str = Field(default="", description="Name of the strategy used")
    ai_signals: Optional[list[dict]] = Field(default=None, description="AI agent signals log")
    benchmark_comparison: Optional[dict] = Field(default=None, description="Comparison with benchmark")
    baseline_comparison: Optional[BaselineComparison] = Field(default=None, description="Detailed baseline comparison")
    win_rate: Optional[float] = Field(default=None, description="Win rate percentage")
    total_trades: Optional[int] = Field(default=None, description="Total number of trades")
    warnings: list[str] = Field(default_factory=list, description="Warnings encountered during backtest")


# =============================================================================
# Baseline Comparison Helper Functions
# =============================================================================

def calculate_strategy_metrics(
    portfolio_values: pd.Series,
    initial_capital: float,
    trades: Optional[list[dict]] = None,
) -> StrategyMetrics:
    """Calculate performance metrics for a strategy.
    
    Args:
        portfolio_values: Series of portfolio values over time.
        initial_capital: Starting capital.
        trades: Optional list of trade dictionaries with action and price.
        
    Returns:
        StrategyMetrics with total_return, sharpe_ratio, max_drawdown, win_rate, total_trades.
    """
    equity_values = [float(v) for v in portfolio_values.values]
    
    # Total return
    final_equity = equity_values[-1] if equity_values else initial_capital
    total_return = ((final_equity - initial_capital) / initial_capital) * 100
    
    # Daily returns for Sharpe ratio
    daily_returns = []
    for i in range(1, len(equity_values)):
        if equity_values[i-1] > 0:
            ret = (equity_values[i] - equity_values[i-1]) / equity_values[i-1]
            daily_returns.append(ret)
    
    sharpe_ratio = calculate_sharpe_ratio(daily_returns)
    max_drawdown = calculate_max_drawdown(equity_values)
    
    # Calculate win rate and total trades if trades provided
    win_rate = None
    total_trades = None
    if trades is not None:
        win_rate = calculate_win_rate(trades)
        total_trades = len(trades)
    
    return StrategyMetrics(
        total_return=round(total_return, 2),
        sharpe_ratio=round(sharpe_ratio, 4),
        max_drawdown=round(max_drawdown, 2),
        win_rate=round(win_rate, 2) if win_rate is not None else None,
        total_trades=total_trades,
    )


def run_baseline_comparison(
    df: pd.DataFrame,
    initial_capital: float,
    ai_return: float,
    include_extended: bool = False,
) -> BaselineComparison:
    """Run baseline strategies and compare against AI performance.
    
    Implements Requirements 5.1, 5.2, 5.3, 5.4.
    
    Args:
        df: DataFrame with price history.
        initial_capital: Starting capital.
        ai_return: AI strategy total return.
        include_extended: Whether to include SMA and MACD comparisons.
        
    Returns:
        BaselineComparison with all strategy metrics.
    """
    # Always run Buy & Hold (Requirement 5.1)
    bh_values, bh_trades, _ = run_buy_and_hold_strategy(df, initial_capital)
    bh_metrics = calculate_strategy_metrics(bh_values, initial_capital, bh_trades)
    
    # Calculate outperformance (Requirement 5.2)
    ai_outperformance = round(ai_return - bh_metrics.total_return, 2)
    
    # Optional extended comparison (Requirement 5.4)
    sma_metrics = None
    macd_metrics = None
    
    if include_extended:
        if len(df) >= 200:
            sma_values, sma_trades, _ = run_sma_crossover_strategy(df, initial_capital)
            sma_metrics = calculate_strategy_metrics(sma_values, initial_capital, sma_trades)
        
        if len(df) >= 26:
            macd_values, macd_trades, _ = run_macd_strategy(df, initial_capital)
            macd_metrics = calculate_strategy_metrics(macd_values, initial_capital, macd_trades)
    
    return BaselineComparison(
        buy_hold=bh_metrics,
        sma_crossover=sma_metrics,
        macd=macd_metrics,
        ai_outperformance=ai_outperformance,
    )


def calculate_win_rate(trades: list[dict]) -> float:
    """Calculate win rate from trades.
    
    Args:
        trades: List of trade dictionaries with action and price.
        
    Returns:
        Win rate as percentage (0-100).
    """
    if len(trades) < 2:
        return 0.0
    
    profitable_trades = 0
    total_round_trips = 0
    buy_price = None
    
    for trade in trades:
        if trade["action"] == "BUY":
            buy_price = trade["price"]
        elif trade["action"] == "SELL" and buy_price is not None:
            total_round_trips += 1
            if trade["price"] > buy_price:
                profitable_trades += 1
            buy_price = None
    
    if total_round_trips == 0:
        return 0.0
    
    return round((profitable_trades / total_round_trips) * 100, 2)


def parse_analysis_interval(interval_str: str) -> ServiceAnalysisInterval:
    """Parse analysis interval string to enum.
    
    Args:
        interval_str: Interval string (daily, weekly, biweekly, monthly).
        
    Returns:
        ServiceAnalysisInterval enum value.
    """
    mapping = {
        "daily": ServiceAnalysisInterval.DAILY,
        "weekly": ServiceAnalysisInterval.WEEKLY,
        "biweekly": ServiceAnalysisInterval.BIWEEKLY,
        "monthly": ServiceAnalysisInterval.MONTHLY,
    }
    return mapping.get(interval_str.lower(), ServiceAnalysisInterval.WEEKLY)


# =============================================================================
# AI Live Strategy Runner
# =============================================================================

async def run_ai_live_strategy(
    ticker: str,
    start_date: date,
    end_date: date,
    initial_capital: float,
    analysis_interval: str = "weekly",
) -> tuple[ExtendedBacktestResponse, list[str]]:
    """Run AI backtest using the real multi-agent system.
    
    Implements Requirements 1.1, 1.3, 1.4.
    
    Args:
        ticker: Stock ticker symbol.
        start_date: Backtest start date.
        end_date: Backtest end date.
        initial_capital: Starting capital.
        analysis_interval: Interval for AI signal generation.
        
    Returns:
        Tuple of (response data dict, warnings list).
        
    Raises:
        MistralUnavailableError: If Mistral API is unavailable after retries.
    """
    warnings: list[str] = []
    
    # Get AI backtest service
    ai_service = get_ai_backtest_service()
    
    # Parse interval
    interval = parse_analysis_interval(analysis_interval)
    
    # Run AI backtest
    result = await ai_service.run_ai_backtest(
        ticker=ticker,
        start_date=start_date,
        end_date=end_date,
        initial_capital=initial_capital,
        analysis_interval=interval,
    )
    
    # Collect warnings from service
    warnings.extend(result.warnings)
    
    return result, warnings


# =============================================================================
# Main Endpoint
# =============================================================================

@router.post(
    "/backtest",
    response_model=ExtendedBacktestResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid request parameters"},
        404: {"model": ErrorResponse, "description": "No historical data found"},
    },
    summary="Run historical backtest",
    description="Evaluates historical performance of trading strategies including AI multi-agent strategy.",
)
async def run_backtest(request: BacktestRequest) -> ExtendedBacktestResponse:
    """Run historical backtesting for a stock.
    
    Supports multiple strategies:
    - 'ai_live': Real multi-agent AI strategy using Mistral API
    - 'ai': Simulated multi-agent AI strategy (fallback)
    - 'sma_crossover': SMA 50/200 crossover
    - 'rsi': RSI oversold/overbought
    - 'macd': MACD crossover
    - 'buy_hold': Simple buy and hold
    - 'auto': Automatically select best baseline strategy
    """
    # Validate ticker
    ticker = validate_ticker_http(request.ticker)
    
    # Parse dates
    try:
        start_date = datetime.strptime(request.start_date, "%Y-%m-%d").date()
        end_date = datetime.strptime(request.end_date, "%Y-%m-%d").date()
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error_code": ErrorCode.VALIDATION_ERROR.value,
                "message": "Invalid date format. Use YYYY-MM-DD.",
            }
        )
    
    if start_date >= end_date:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error_code": ErrorCode.VALIDATION_ERROR.value,
                "message": "Start date must be before end date.",
            }
        )
    
    logger.info(f"Running backtest for {ticker} from {start_date} to {end_date}, strategy: {request.strategy}")
    
    initial_capital = request.initial_capital
    strategy = request.strategy or "auto"
    warnings: list[str] = []
    ai_signals_log = None
    benchmark_comparison = None
    baseline_comparison = None
    win_rate = None
    total_trades_count = None
    
    # ==========================================================================
    # AI Live Strategy (Real Multi-Agent System)
    # ==========================================================================
    if strategy == "ai_live":
        try:
            result, ai_warnings = await run_ai_live_strategy(
                ticker=ticker,
                start_date=start_date,
                end_date=end_date,
                initial_capital=initial_capital,
                analysis_interval=request.analysis_interval or "weekly",
            )
            warnings.extend(ai_warnings)
            
            # Convert AI signals to dict format for response
            ai_signals_log = [s.to_dict() for s in result.ai_signals]
            
            # Build equity curve
            equity_curve: list[EquityPoint] = []
            for ep in result.equity_curve:
                equity_curve.append(EquityPoint(
                    date=datetime.combine(ep.date, datetime.min.time()) if isinstance(ep.date, date) else ep.date,
                    equity=round(ep.equity, 2),
                    returns=round(ep.returns, 2),
                ))
            
            # Build trades list
            backtest_trades: list[BacktestTrade] = []
            for t in result.trades:
                backtest_trades.append(BacktestTrade(
                    date=datetime.combine(t.date, datetime.min.time()) if isinstance(t.date, date) else t.date,
                    action=t.action,
                    price=round(t.price, 2),
                    shares=round(t.shares, 4),
                    value=round(t.value, 2),
                ))
            
            # Run baseline comparison if requested (Requirement 5.1)
            if request.include_baseline_comparison:
                # Fetch data for baseline strategies
                stock = yf.Ticker(ticker)
                df = stock.history(
                    start=start_date.strftime("%Y-%m-%d"),
                    end=(end_date + timedelta(days=1)).strftime("%Y-%m-%d")
                )
                if not df.empty:
                    baseline_comparison = run_baseline_comparison(
                        df=df,
                        initial_capital=initial_capital,
                        ai_return=result.total_return,
                        include_extended=True,  # Include SMA and MACD
                    )
            
            logger.info(
                f"AI Live backtest complete for {ticker}: return={result.total_return:.2f}%, "
                f"trades={result.total_trades}, win_rate={result.win_rate:.1f}%"
            )
            
            return ExtendedBacktestResponse(
                ticker=ticker,
                total_return=round(result.total_return, 2),
                sharpe_ratio=round(result.sharpe_ratio, 4),
                max_drawdown=round(result.max_drawdown, 2),
                trades=backtest_trades,
                equity_curve=equity_curve,
                strategy_name=result.strategy_name,
                ai_signals=ai_signals_log,
                benchmark_comparison=None,
                baseline_comparison=baseline_comparison,
                win_rate=round(result.win_rate, 2),
                total_trades=result.total_trades,
                warnings=warnings,
            )
            
        except MistralUnavailableError as e:
            # Fallback to simulated AI strategy (Requirement 1.4)
            logger.warning(f"Mistral unavailable, falling back to simulated AI: {e}")
            warnings.append(f"{BacktestWarning.MISTRAL_FALLBACK.value}: Falling back to simulated AI strategy")
            strategy = "ai"  # Fall through to simulated AI
            
        except AIBacktestError as e:
            logger.error(f"AI backtest error: {e}")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail={
                    "error_code": ErrorCode.EXTERNAL_API_ERROR.value,
                    "message": f"AI backtest failed: {str(e)}",
                }
            )
    
    # ==========================================================================
    # Fetch Historical Data for Non-AI-Live Strategies
    # ==========================================================================
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(
            start=start_date.strftime("%Y-%m-%d"),
            end=(end_date + timedelta(days=1)).strftime("%Y-%m-%d")
        )
        
        if df.empty:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={
                    "error_code": ErrorCode.VALIDATION_ERROR.value,
                    "message": f"No historical data found for {ticker} in the specified date range",
                }
            )
        
        logger.info(f"Fetched {len(df)} days of historical data for {ticker}")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to fetch historical data for {ticker}: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "error_code": ErrorCode.EXTERNAL_API_ERROR.value,
                "message": f"Failed to fetch historical data: {str(e)}",
            }
        )
    
    # ==========================================================================
    # Run Selected Strategy
    # ==========================================================================
    if strategy == "ai":
        # Run simulated AI multi-agent strategy
        portfolio_values, trades, strategy_name, ai_signals_log = await run_ai_strategy(
            df, ticker, initial_capital
        )
        win_rate = calculate_win_rate(trades)
        total_trades_count = len(trades)
        
        # Run baseline comparison (Requirement 5.1)
        if request.include_baseline_comparison:
            final_equity = float(portfolio_values.iloc[-1])
            ai_return = ((final_equity - initial_capital) / initial_capital) * 100
            baseline_comparison = run_baseline_comparison(
                df=df,
                initial_capital=initial_capital,
                ai_return=ai_return,
                include_extended=True,
            )
        
        # Legacy benchmark comparison for backwards compatibility
        bh_values, _, _ = run_buy_and_hold_strategy(df, initial_capital)
        bh_return = ((bh_values.iloc[-1] - initial_capital) / initial_capital) * 100
        benchmark_comparison = {
            "buy_hold_return": round(bh_return, 2),
            "ai_outperformance": None
        }
        
    elif strategy == "sma_crossover":
        if len(df) < 200:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "error_code": ErrorCode.VALIDATION_ERROR.value,
                    "message": "SMA Crossover requires at least 200 days of data",
                }
            )
        portfolio_values, trades, strategy_name = run_sma_crossover_strategy(df, initial_capital)
        
    elif strategy == "rsi":
        if len(df) < 14:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "error_code": ErrorCode.VALIDATION_ERROR.value,
                    "message": "RSI strategy requires at least 14 days of data",
                }
            )
        portfolio_values, trades, strategy_name = run_rsi_strategy(df, initial_capital)
        
    elif strategy == "macd":
        if len(df) < 26:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "error_code": ErrorCode.VALIDATION_ERROR.value,
                    "message": "MACD strategy requires at least 26 days of data",
                }
            )
        portfolio_values, trades, strategy_name = run_macd_strategy(df, initial_capital)
        
    elif strategy == "buy_hold":
        portfolio_values, trades, strategy_name = run_buy_and_hold_strategy(df, initial_capital)
        
    else:  # auto
        # Select best strategy based on data availability
        if len(df) >= 200:
            portfolio_values, trades, strategy_name = run_sma_crossover_strategy(df, initial_capital)
        elif len(df) >= 26:
            portfolio_values, trades, strategy_name = run_macd_strategy(df, initial_capital)
        else:
            portfolio_values, trades, strategy_name = run_buy_and_hold_strategy(df, initial_capital)
    
    # ==========================================================================
    # Build Response
    # ==========================================================================
    
    # Build equity curve
    equity_curve: list[EquityPoint] = []
    equity_values: list[float] = []
    daily_returns: list[float] = []
    prev_equity = initial_capital
    
    for idx, equity in portfolio_values.items():
        equity_value = float(equity)
        equity_values.append(equity_value)
        
        if prev_equity > 0:
            daily_return = (equity_value - prev_equity) / prev_equity
            daily_returns.append(daily_return)
        
        prev_equity = equity_value
        
        equity_curve.append(EquityPoint(
            date=idx.to_pydatetime() if hasattr(idx, 'to_pydatetime') else idx,
            equity=round(equity_value, 2),
            returns=round(((equity_value - initial_capital) / initial_capital) * 100, 2),
        ))
    
    # Convert trades to BacktestTrade objects
    backtest_trades: list[BacktestTrade] = []
    shares_held = 0.0
    
    for trade in trades:
        trade_date = trade["date"]
        if hasattr(trade_date, 'to_pydatetime'):
            trade_date = trade_date.to_pydatetime()
        
        price = trade["price"]
        action = trade["action"]
        
        if action == "BUY":
            shares = initial_capital / price if shares_held == 0 else 0
            shares_held = shares
            value = shares * price
        else:
            shares = shares_held
            value = shares * price
            shares_held = 0
        
        backtest_trades.append(BacktestTrade(
            date=trade_date,
            action=action,
            price=round(price, 2),
            shares=round(shares, 4),
            value=round(value, 2),
        ))
    
    # Calculate final metrics
    final_equity = equity_values[-1] if equity_values else initial_capital
    total_return = ((final_equity - initial_capital) / initial_capital) * 100
    sharpe_ratio = calculate_sharpe_ratio(daily_returns)
    max_drawdown = calculate_max_drawdown(equity_values)
    
    # Calculate AI outperformance if applicable
    if benchmark_comparison:
        benchmark_comparison["ai_outperformance"] = round(
            total_return - benchmark_comparison["buy_hold_return"], 2
        )
    
    # Calculate win rate and total trades if not already set
    if win_rate is None:
        win_rate = calculate_win_rate(trades)
    if total_trades_count is None:
        total_trades_count = len(backtest_trades)
    
    logger.info(
        f"Backtest complete for {ticker} ({strategy_name}): return={total_return:.2f}%, "
        f"sharpe={sharpe_ratio:.2f}, max_drawdown={max_drawdown:.2f}%"
    )
    
    return ExtendedBacktestResponse(
        ticker=ticker,
        total_return=round(total_return, 2),
        sharpe_ratio=sharpe_ratio,
        max_drawdown=max_drawdown,
        trades=backtest_trades,
        equity_curve=equity_curve,
        strategy_name=strategy_name,
        ai_signals=ai_signals_log,
        benchmark_comparison=benchmark_comparison,
        baseline_comparison=baseline_comparison,
        win_rate=win_rate,
        total_trades=total_trades_count,
        warnings=warnings,
    )
