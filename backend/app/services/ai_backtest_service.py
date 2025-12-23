import asyncio
import logging
from datetime import date, datetime, timedelta
from enum import Enum
from typing import Optional, Any

import pandas as pd
import numpy as np
import yfinance as yf

from app.agents.orchestrator import OrchestratorAgent, get_orchestrator
from app.models.signals import AgentSignal, SynthesisResult, Recommendation
from app.services.data_fetcher import DataFetcher, get_data_fetcher

logger = logging.getLogger(__name__)


class AnalysisInterval(str, Enum):
    """Analysis interval for AI backtest signal generation."""
    
    DAILY = "daily"        # Every trading day
    WEEKLY = "weekly"      # Every 5 trading days (default)
    BIWEEKLY = "biweekly"  # Every 10 trading days
    MONTHLY = "monthly"    # Every 21 trading days
    
    def to_trading_days(self) -> int:
        """Convert interval to number of trading days."""
        mapping = {
            AnalysisInterval.DAILY: 1,
            AnalysisInterval.WEEKLY: 5,
            AnalysisInterval.BIWEEKLY: 10,
            AnalysisInterval.MONTHLY: 21,
        }
        return mapping[self]


class AIBacktestError(Exception):
    """Base exception for AI backtest errors."""
    pass


class MistralUnavailableError(AIBacktestError):
    """Raised when Mistral API is unavailable."""
    pass


# =============================================================================
# Retry Configuration
# =============================================================================

class MistralRetryConfig:
    """Configuration for Mistral API retry logic."""
    MAX_RETRIES = 3
    BASE_DELAY_SECONDS = 2
    MAX_DELAY_SECONDS = 30


class BacktestWarning(str, Enum):
    """Warning types for backtest issues."""
    MISTRAL_FALLBACK = "mistral_fallback"
    PARTIAL_AGENT_FAILURE = "partial_agent_failure"
    INSUFFICIENT_NEWS_DATA = "insufficient_news_data"
    RATE_LIMIT_DELAYS = "rate_limit_delays"
    SKIPPED_SIGNALS = "skipped_signals"


# =============================================================================
# Data Models for AI Backtest
# =============================================================================

class AgentSignalSummary:
    """Summary of an individual agent's signal."""
    
    def __init__(
        self,
        score: float,
        signal_type: str,
        confidence: float,
        key_factors: list[str],
    ):
        self.score = score
        self.signal_type = signal_type
        self.confidence = confidence
        self.key_factors = key_factors
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "score": round(self.score, 2),
            "signal_type": self.signal_type,
            "confidence": round(self.confidence, 2),
            "key_factors": self.key_factors,
        }


class AISignalRecord:
    """Record of an AI-generated signal at a specific date."""
    
    def __init__(
        self,
        date: date,
        recommendation: str,
        confidence: float,
        weighted_score: float,
        agent_signals: dict[str, AgentSignalSummary],
        reasoning: str,
        position_action: str,
    ):
        self.date = date
        self.recommendation = recommendation
        self.confidence = confidence
        self.weighted_score = weighted_score
        self.agent_signals = agent_signals
        self.reasoning = reasoning
        self.position_action = position_action
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for API response."""
        return {
            "date": self.date.isoformat() if isinstance(self.date, date) else str(self.date),
            "recommendation": self.recommendation,
            "confidence": round(self.confidence, 2),
            "weighted_score": round(self.weighted_score, 2),
            "agent_signals": {
                name: summary.to_dict() 
                for name, summary in self.agent_signals.items()
            },
            "reasoning": self.reasoning,
            "position_action": self.position_action,
        }


class BacktestTrade:
    """Individual trade executed during backtest."""
    
    def __init__(
        self,
        date: date,
        action: str,
        price: float,
        shares: float,
        value: float,
        signal_confidence: Optional[float] = None,
        signal_reasoning: Optional[str] = None,
    ):
        self.date = date
        self.action = action
        self.price = price
        self.shares = shares
        self.value = value
        self.signal_confidence = signal_confidence
        self.signal_reasoning = signal_reasoning
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "date": self.date.isoformat() if isinstance(self.date, date) else str(self.date),
            "action": self.action,
            "price": round(self.price, 2),
            "shares": round(self.shares, 4),
            "value": round(self.value, 2),
            "signal_confidence": round(self.signal_confidence, 2) if self.signal_confidence else None,
            "signal_reasoning": self.signal_reasoning,
        }


class EquityPoint:
    """Point on the equity curve."""
    
    def __init__(self, date: date, equity: float, returns: float):
        self.date = date
        self.equity = equity
        self.returns = returns
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "date": self.date.isoformat() if isinstance(self.date, date) else str(self.date),
            "equity": round(self.equity, 2),
            "returns": round(self.returns, 2),
        }


class AIBacktestResult:
    """Result of an AI-driven backtest."""
    
    def __init__(
        self,
        ticker: str,
        strategy_name: str,
        total_return: float,
        sharpe_ratio: float,
        max_drawdown: float,
        win_rate: float,
        total_trades: int,
        trades: list[BacktestTrade],
        equity_curve: list[EquityPoint],
        ai_signals: list[AISignalRecord],
        warnings: list[str],
    ):
        self.ticker = ticker
        self.strategy_name = strategy_name
        self.total_return = total_return
        self.sharpe_ratio = sharpe_ratio
        self.max_drawdown = max_drawdown
        self.win_rate = win_rate
        self.total_trades = total_trades
        self.trades = trades
        self.equity_curve = equity_curve
        self.ai_signals = ai_signals
        self.warnings = warnings
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for API response."""
        return {
            "ticker": self.ticker,
            "strategy_name": self.strategy_name,
            "total_return": round(self.total_return, 2),
            "sharpe_ratio": round(self.sharpe_ratio, 4),
            "max_drawdown": round(self.max_drawdown, 2),
            "win_rate": round(self.win_rate, 2),
            "total_trades": self.total_trades,
            "trades": [t.to_dict() for t in self.trades],
            "equity_curve": [e.to_dict() for e in self.equity_curve],
            "ai_signals": [s.to_dict() for s in self.ai_signals],
            "warnings": self.warnings,
        }


# =============================================================================
# Portfolio Simulator
# =============================================================================

class PortfolioSimulator:
    """Simulates portfolio based on AI signals.
    
    Handles trade execution logic, position tracking, and equity curve calculation.
    Implements Requirements 3.1, 3.2, 3.3, 3.4, 3.5.
    """
    
    def __init__(self, initial_capital: float):
        """Initialize the portfolio simulator.
        
        Args:
            initial_capital: Starting capital for the portfolio.
        """
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.shares = 0.0
        self.position = False
        self.trades: list[BacktestTrade] = []
        self.equity_history: list[EquityPoint] = []
        self._entry_price: Optional[float] = None
    
    def process_signal(
        self,
        signal: AISignalRecord,
        price_data: pd.DataFrame,
        signal_date: date,
    ) -> Optional[BacktestTrade]:
        """Process AI signal and execute trade if needed.
        
        Implements:
        - Req 3.1: BUY signal with no position -> enter full position
        - Req 3.2: SELL signal with position -> exit full position
        - Req 3.3: HOLD signal -> maintain current position
        
        Args:
            signal: The AI signal record.
            price_data: DataFrame with price history.
            signal_date: Date of the signal.
            
        Returns:
            BacktestTrade if a trade was executed, None otherwise.
        """
        execution_price = self._get_execution_price(price_data, signal_date)
        
        if execution_price is None:
            logger.warning(f"Could not get execution price for {signal_date}")
            return None
        
        # BUY: Enter position if not already in one
        if signal.recommendation == "BUY" and not self.position:
            shares = self.cash / execution_price
            trade = BacktestTrade(
                date=signal_date,
                action="BUY",
                price=execution_price,
                shares=shares,
                value=shares * execution_price,
                signal_confidence=signal.confidence,
                signal_reasoning=signal.reasoning[:200] if signal.reasoning else None,
            )
            
            self.shares = shares
            self.cash = 0.0
            self.position = True
            self._entry_price = execution_price
            self.trades.append(trade)
            
            logger.debug(f"BUY executed: {shares:.4f} shares at ${execution_price:.2f}")
            return trade
        
        # SELL: Exit position if currently holding
        elif signal.recommendation == "SELL" and self.position:
            value = self.shares * execution_price
            trade = BacktestTrade(
                date=signal_date,
                action="SELL",
                price=execution_price,
                shares=self.shares,
                value=value,
                signal_confidence=signal.confidence,
                signal_reasoning=signal.reasoning[:200] if signal.reasoning else None,
            )
            
            self.cash = value
            self.shares = 0.0
            self.position = False
            self._entry_price = None
            self.trades.append(trade)
            
            logger.debug(f"SELL executed: {trade.shares:.4f} shares at ${execution_price:.2f}")
            return trade
        
        # HOLD: No action needed
        return None
    
    def _get_execution_price(
        self,
        price_data: pd.DataFrame,
        signal_date: date,
    ) -> Optional[float]:
        """Get execution price for the next trading day after signal.
        
        Uses the next trading day's open price for realistic execution.
        
        Args:
            price_data: DataFrame with OHLCV data.
            signal_date: Date of the signal.
            
        Returns:
            Execution price or None if not available.
        """
        try:
            # Convert signal_date to datetime for comparison
            if isinstance(signal_date, date) and not isinstance(signal_date, datetime):
                signal_datetime = datetime.combine(signal_date, datetime.min.time())
            else:
                signal_datetime = signal_date
            
            # Find the next trading day after signal_date
            future_dates = price_data.index[price_data.index > signal_datetime]
            
            if len(future_dates) > 0:
                next_date = future_dates[0]
                # Use Open price of next day for execution
                if "Open" in price_data.columns:
                    return float(price_data.loc[next_date, "Open"])
                else:
                    return float(price_data.loc[next_date, "Close"])
            else:
                # If no future date, use current day's close
                if signal_datetime in price_data.index:
                    return float(price_data.loc[signal_datetime, "Close"])
                # Find closest date
                closest_idx = price_data.index.get_indexer([signal_datetime], method="nearest")[0]
                if closest_idx >= 0:
                    return float(price_data.iloc[closest_idx]["Close"])
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting execution price: {e}")
            return None
    
    def update_equity(self, current_date: date, current_price: float) -> None:
        """Update equity curve with current portfolio value.
        
        Implements Requirement 3.5: Track equity curve.
        
        Args:
            current_date: Current date.
            current_price: Current stock price.
        """
        equity = self.cash + (self.shares * current_price)
        returns = ((equity - self.initial_capital) / self.initial_capital) * 100
        
        self.equity_history.append(EquityPoint(
            date=current_date,
            equity=equity,
            returns=returns,
        ))
    
    def get_current_equity(self, current_price: float) -> float:
        """Get current portfolio equity value.
        
        Args:
            current_price: Current stock price.
            
        Returns:
            Current equity value.
        """
        return self.cash + (self.shares * current_price)
    
    def calculate_win_rate(self) -> float:
        """Calculate win rate from completed trades.
        
        Returns:
            Win rate as percentage (0-100).
        """
        if len(self.trades) < 2:
            return 0.0
        
        # Pair up BUY and SELL trades
        profitable_trades = 0
        total_round_trips = 0
        
        buy_price = None
        for trade in self.trades:
            if trade.action == "BUY":
                buy_price = trade.price
            elif trade.action == "SELL" and buy_price is not None:
                total_round_trips += 1
                if trade.price > buy_price:
                    profitable_trades += 1
                buy_price = None
        
        if total_round_trips == 0:
            return 0.0
        
        return (profitable_trades / total_round_trips) * 100


# =============================================================================
# AI Backtest Service
# =============================================================================

class AIBacktestService:
    """Service for running AI-driven backtests using the multi-agent system.
    
    Integrates the Orchestrator with all specialist agents (Fundamental, Technical,
    Sentiment, Risk) to generate trading signals for historical backtesting.
    
    Implements Requirements 1.1, 1.2, 1.3, 1.4, 1.5.
    """
    
    def __init__(
        self,
        orchestrator: Optional[OrchestratorAgent] = None,
        data_fetcher: Optional[DataFetcher] = None,
    ):
        """Initialize the AI Backtest Service.
        
        Args:
            orchestrator: OrchestratorAgent instance. Uses singleton if not provided.
            data_fetcher: DataFetcher instance. Uses singleton if not provided.
        """
        self.orchestrator = orchestrator or get_orchestrator()
        self.data_fetcher = data_fetcher or get_data_fetcher()
        self._warnings: list[str] = []
    
    async def run_ai_backtest(
        self,
        ticker: str,
        start_date: date,
        end_date: date,
        initial_capital: float = 10000.0,
        analysis_interval: AnalysisInterval = AnalysisInterval.WEEKLY,
    ) -> AIBacktestResult:
        """Execute AI-driven backtest over historical period.
        
        Implements Requirements 1.1, 1.2, 1.3, 1.5.
        
        Args:
            ticker: Stock ticker symbol.
            start_date: Backtest start date.
            end_date: Backtest end date.
            initial_capital: Starting capital.
            analysis_interval: Frequency of AI signal generation.
            
        Returns:
            AIBacktestResult with signals, trades, and metrics.
            
        Raises:
            AIBacktestError: If backtest fails completely.
        """
        self._warnings = []
        logger.info(
            f"Starting AI backtest for {ticker} from {start_date} to {end_date}, "
            f"interval: {analysis_interval.value}"
        )
        
        # Fetch historical price data
        price_data = await self._fetch_historical_data(ticker, start_date, end_date)
        
        if price_data.empty:
            raise AIBacktestError(f"No historical data available for {ticker}")
        
        # Initialize portfolio simulator
        simulator = PortfolioSimulator(initial_capital)
        
        # Generate signals at each interval
        ai_signals: list[AISignalRecord] = []
        interval_days = analysis_interval.to_trading_days()
        
        # Get analysis dates
        analysis_dates = self._get_analysis_dates(price_data, interval_days)
        
        logger.info(f"Generating {len(analysis_dates)} AI signals for {ticker}")
        
        for analysis_date in analysis_dates:
            try:
                # Generate signal at this date
                signal = await self.generate_signal_at_date(
                    ticker=ticker,
                    analysis_date=analysis_date,
                    historical_data=price_data,
                    current_position=simulator.position,
                )
                
                if signal:
                    ai_signals.append(signal)
                    
                    # Process signal through portfolio simulator
                    simulator.process_signal(signal, price_data, analysis_date)
                    
            except Exception as e:
                logger.warning(f"Failed to generate signal for {analysis_date}: {e}")
                self._warnings.append(f"Skipped signal at {analysis_date}: {str(e)}")
        
        # Update equity curve for all trading days
        for idx, row in price_data.iterrows():
            trade_date = idx.date() if hasattr(idx, 'date') else idx
            simulator.update_equity(trade_date, float(row["Close"]))
        
        # Calculate final metrics
        metrics = self._calculate_metrics(simulator, price_data)
        
        logger.info(
            f"AI backtest complete for {ticker}: return={metrics['total_return']:.2f}%, "
            f"trades={metrics['total_trades']}, win_rate={metrics['win_rate']:.1f}%"
        )
        
        return AIBacktestResult(
            ticker=ticker,
            strategy_name="Multi-Agent AI Strategy (Live)",
            total_return=metrics["total_return"],
            sharpe_ratio=metrics["sharpe_ratio"],
            max_drawdown=metrics["max_drawdown"],
            win_rate=metrics["win_rate"],
            total_trades=metrics["total_trades"],
            trades=simulator.trades,
            equity_curve=simulator.equity_history,
            ai_signals=ai_signals,
            warnings=self._warnings,
        )
    
    async def generate_signal_at_date(
        self,
        ticker: str,
        analysis_date: date,
        historical_data: pd.DataFrame,
        current_position: bool = False,
    ) -> Optional[AISignalRecord]:
        """Generate a single AI signal for a specific date.
        
        Uses the Orchestrator to coordinate all specialist agents and
        generate a trading signal.
        
        Implements Requirement 1.3.
        
        Args:
            ticker: Stock ticker symbol.
            analysis_date: Date to generate signal for.
            historical_data: DataFrame with price history.
            current_position: Whether currently holding a position.
            
        Returns:
            AISignalRecord or None if generation fails.
        """
        try:
            # Call orchestrator with retry logic
            synthesis, agent_signals, _ = await self._call_orchestrator_with_retry(ticker)
            
            # Convert agent signals to summaries
            agent_summaries = self._convert_to_summaries(agent_signals)
            
            # Determine position action
            position_action = self._determine_position_action(
                synthesis.recommendation,
                current_position,
            )
            
            return AISignalRecord(
                date=analysis_date,
                recommendation=synthesis.recommendation.value,
                confidence=synthesis.confidence,
                weighted_score=synthesis.weighted_score,
                agent_signals=agent_summaries,
                reasoning=synthesis.reasoning,
                position_action=position_action,
            )
            
        except MistralUnavailableError:
            self._warnings.append(BacktestWarning.MISTRAL_FALLBACK.value)
            logger.warning(f"Mistral unavailable at {analysis_date}, using fallback")
            return None
            
        except Exception as e:
            logger.error(f"Error generating signal at {analysis_date}: {e}")
            self._warnings.append(f"Signal generation failed at {analysis_date}")
            return None
    
    async def _call_orchestrator_with_retry(
        self,
        ticker: str,
    ) -> tuple[SynthesisResult, list[AgentSignal], Any]:
        """Call orchestrator with exponential backoff retry.
        
        Implements Requirement 7.2.
        
        Args:
            ticker: Stock ticker symbol.
            
        Returns:
            Tuple of (SynthesisResult, agent_signals, stock_data).
            
        Raises:
            MistralUnavailableError: If all retries fail.
        """
        last_error = None
        
        for attempt in range(MistralRetryConfig.MAX_RETRIES):
            try:
                return await self.orchestrator.orchestrate(ticker)
                
            except Exception as e:
                last_error = e
                error_str = str(e).lower()
                
                # Check if it's a rate limit error
                if "rate" in error_str or "limit" in error_str or "429" in error_str:
                    if attempt < MistralRetryConfig.MAX_RETRIES - 1:
                        delay = min(
                            MistralRetryConfig.BASE_DELAY_SECONDS * (2 ** attempt),
                            MistralRetryConfig.MAX_DELAY_SECONDS
                        )
                        logger.warning(
                            f"Rate limit hit, retrying in {delay}s (attempt {attempt + 1})"
                        )
                        self._warnings.append(BacktestWarning.RATE_LIMIT_DELAYS.value)
                        await asyncio.sleep(delay)
                        continue
                
                # For other errors, don't retry
                raise
        
        raise MistralUnavailableError(f"All retries failed: {last_error}")
    
    def _convert_to_summaries(
        self,
        agent_signals: list[AgentSignal],
    ) -> dict[str, AgentSignalSummary]:
        """Convert AgentSignal list to summary dictionary.
        
        Args:
            agent_signals: List of agent signals.
            
        Returns:
            Dictionary mapping agent name to summary.
        """
        summaries = {}
        
        for signal in agent_signals:
            summaries[signal.agent_name.lower()] = AgentSignalSummary(
                score=signal.score,
                signal_type=signal.signal_type.value,
                confidence=signal.confidence,
                key_factors=signal.key_factors[:5],  # Limit to 5 factors
            )
        
        return summaries
    
    def _determine_position_action(
        self,
        recommendation: Recommendation,
        current_position: bool,
    ) -> str:
        """Determine the position action based on recommendation and current state.
        
        Args:
            recommendation: The AI recommendation.
            current_position: Whether currently holding a position.
            
        Returns:
            Position action string: "ENTER", "EXIT", or "HOLD".
        """
        if recommendation == Recommendation.BUY and not current_position:
            return "ENTER"
        elif recommendation == Recommendation.SELL and current_position:
            return "EXIT"
        else:
            return "HOLD"
    
    async def _fetch_historical_data(
        self,
        ticker: str,
        start_date: date,
        end_date: date,
    ) -> pd.DataFrame:
        """Fetch historical price data for the ticker.
        
        Args:
            ticker: Stock ticker symbol.
            start_date: Start date.
            end_date: End date.
            
        Returns:
            DataFrame with OHLCV data.
        """
        try:
            stock = yf.Ticker(ticker)
            df = stock.history(
                start=start_date.strftime("%Y-%m-%d"),
                end=(end_date + timedelta(days=1)).strftime("%Y-%m-%d")
            )
            return df
            
        except Exception as e:
            logger.error(f"Failed to fetch historical data for {ticker}: {e}")
            raise AIBacktestError(f"Failed to fetch historical data: {e}")
    
    def _get_analysis_dates(
        self,
        price_data: pd.DataFrame,
        interval_days: int,
    ) -> list[date]:
        """Get dates for AI signal generation based on interval.
        
        Args:
            price_data: DataFrame with price history.
            interval_days: Number of trading days between analyses.
            
        Returns:
            List of analysis dates.
        """
        dates = []
        trading_days = list(price_data.index)
        
        # Start after initial warmup period (need some history for agents)
        start_idx = min(20, len(trading_days) - 1)
        
        for i in range(start_idx, len(trading_days), interval_days):
            idx = trading_days[i]
            analysis_date = idx.date() if hasattr(idx, 'date') else idx
            dates.append(analysis_date)
        
        return dates
    
    def _calculate_metrics(
        self,
        simulator: PortfolioSimulator,
        price_data: pd.DataFrame,
    ) -> dict[str, float]:
        """Calculate performance metrics from backtest results.
        
        Implements Requirements 4.1, 4.2, 4.3, 4.4, 4.5.
        
        Args:
            simulator: Portfolio simulator with results.
            price_data: Price data DataFrame.
            
        Returns:
            Dictionary with performance metrics.
        """
        equity_values = [e.equity for e in simulator.equity_history]
        
        if not equity_values:
            return {
                "total_return": 0.0,
                "sharpe_ratio": 0.0,
                "max_drawdown": 0.0,
                "win_rate": 0.0,
                "total_trades": 0,
            }
        
        # Total return
        final_equity = equity_values[-1]
        total_return = ((final_equity - simulator.initial_capital) / simulator.initial_capital) * 100
        
        # Daily returns for Sharpe ratio
        daily_returns = []
        for i in range(1, len(equity_values)):
            if equity_values[i-1] > 0:
                ret = (equity_values[i] - equity_values[i-1]) / equity_values[i-1]
                daily_returns.append(ret)
        
        # Sharpe ratio (annualized, 2% risk-free rate)
        sharpe_ratio = self._calculate_sharpe_ratio(daily_returns)
        
        # Max drawdown
        max_drawdown = self._calculate_max_drawdown(equity_values)
        
        # Win rate
        win_rate = simulator.calculate_win_rate()
        
        # Total trades
        total_trades = len(simulator.trades)
        
        return {
            "total_return": total_return,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
            "win_rate": win_rate,
            "total_trades": total_trades,
        }
    
    def _calculate_sharpe_ratio(
        self,
        returns: list[float],
        risk_free_rate: float = 0.02,
        periods_per_year: int = 252,
    ) -> float:
        """Calculate annualized Sharpe ratio.
        
        Args:
            returns: List of daily returns.
            risk_free_rate: Annual risk-free rate.
            periods_per_year: Trading days per year.
            
        Returns:
            Sharpe ratio.
        """
        if not returns or len(returns) < 2:
            return 0.0
        
        mean_return = sum(returns) / len(returns)
        variance = sum((r - mean_return) ** 2 for r in returns) / (len(returns) - 1)
        std_dev = variance ** 0.5 if variance > 0 else 0.0
        
        if std_dev == 0:
            return 0.0
        
        annualized_return = mean_return * periods_per_year
        annualized_std = std_dev * (periods_per_year ** 0.5)
        
        return (annualized_return - risk_free_rate) / annualized_std
    
    def _calculate_max_drawdown(self, equity_curve: list[float]) -> float:
        """Calculate maximum drawdown percentage.
        
        Args:
            equity_curve: List of equity values.
            
        Returns:
            Maximum drawdown as percentage.
        """
        if not equity_curve or len(equity_curve) < 2:
            return 0.0
        
        peak = equity_curve[0]
        max_drawdown = 0.0
        
        for equity in equity_curve:
            if equity > peak:
                peak = equity
            drawdown = (peak - equity) / peak * 100 if peak > 0 else 0
            max_drawdown = max(max_drawdown, drawdown)
        
        return max_drawdown


# =============================================================================
# Singleton Instance
# =============================================================================

_ai_backtest_service: Optional[AIBacktestService] = None


def get_ai_backtest_service() -> AIBacktestService:
    """Get or create AIBacktestService singleton.
    
    Returns:
        AIBacktestService instance.
    """
    global _ai_backtest_service
    if _ai_backtest_service is None:
        _ai_backtest_service = AIBacktestService()
    return _ai_backtest_service
