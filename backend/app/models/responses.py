from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field

from app.models.signals import AgentSignal, Recommendation, SynthesisResult


class AnalyzeResponse(BaseModel):
    """Response model for stock analysis endpoint."""

    ticker: str = Field(..., description="Analyzed stock ticker")
    recommendation: Recommendation = Field(..., description="Investment recommendation")
    confidence_score: float = Field(
        ...,
        ge=0,
        le=100,
        description="Overall confidence score"
    )
    signals: list[AgentSignal] = Field(
        default_factory=list,
        description="Individual agent signals"
    )
    synthesis: SynthesisResult = Field(..., description="Synthesized analysis result")
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Analysis timestamp"
    )
    cached: bool = Field(default=False, description="Whether result was from cache")


class BacktestTrade(BaseModel):
    """Individual trade in backtest results."""

    date: datetime = Field(..., description="Trade date")
    action: str = Field(..., description="Trade action: BUY or SELL")
    price: float = Field(..., description="Trade price")
    shares: float = Field(..., description="Number of shares")
    value: float = Field(..., description="Trade value")


class EquityPoint(BaseModel):
    """Point on the equity curve."""

    date: datetime = Field(..., description="Date")
    equity: float = Field(..., description="Portfolio equity value")
    returns: float = Field(..., description="Cumulative returns percentage")


class BacktestResponse(BaseModel):
    """Response model for backtesting endpoint."""

    ticker: str = Field(..., description="Backtested stock ticker")
    total_return: float = Field(..., description="Total return percentage")
    sharpe_ratio: float = Field(..., description="Sharpe ratio")
    max_drawdown: float = Field(..., description="Maximum drawdown percentage")
    trades: list[BacktestTrade] = Field(
        default_factory=list,
        description="List of trades executed"
    )
    equity_curve: list[EquityPoint] = Field(
        default_factory=list,
        description="Equity curve data points"
    )


class ServiceStatus(BaseModel):
    """Status of an individual service."""

    name: str = Field(..., description="Service name")
    status: str = Field(..., description="Service status: healthy, unhealthy, unknown")
    latency_ms: Optional[float] = Field(
        default=None,
        description="Response latency in milliseconds"
    )
    message: Optional[str] = Field(default=None, description="Status message")


class HealthResponse(BaseModel):
    """Response model for health check endpoint."""

    status: str = Field(..., description="Overall system status")
    version: str = Field(..., description="Application version")
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Health check timestamp"
    )
    services: dict[str, ServiceStatus] = Field(
        default_factory=dict,
        description="Status of dependent services"
    )


class ErrorCode(str, Enum):
    """Error codes for API responses."""

    INVALID_TICKER = "INVALID_TICKER"
    AGENT_TIMEOUT = "AGENT_TIMEOUT"
    EXTERNAL_API_ERROR = "EXTERNAL_API_ERROR"
    RATE_LIMIT_EXCEEDED = "RATE_LIMIT_EXCEEDED"
    MISTRAL_UNAVAILABLE = "MISTRAL_UNAVAILABLE"
    NEO4J_ERROR = "NEO4J_ERROR"
    VALIDATION_ERROR = "VALIDATION_ERROR"
    INTERNAL_ERROR = "INTERNAL_ERROR"


class ErrorResponse(BaseModel):
    """Response model for error responses."""

    error_code: ErrorCode = Field(..., description="Error code")
    message: str = Field(..., description="Human-readable error message")
    details: Optional[dict] = Field(default=None, description="Additional error details")
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Error timestamp"
    )


# AI Backtest Models

class AgentSignalSummary(BaseModel):
    """Summary of an individual agent's signal for AI backtest logging."""

    score: float = Field(..., description="Agent's score (-100 to 100)")
    signal_type: str = Field(..., description="Signal type: bullish, bearish, neutral")
    confidence: float = Field(..., ge=0, le=100, description="Agent confidence score")
    key_factors: list[str] = Field(
        default_factory=list,
        description="Key factors influencing the signal"
    )


class AISignalRecord(BaseModel):
    """Record of an AI-generated signal during backtest."""

    date: str = Field(..., description="Signal date (YYYY-MM-DD)")
    recommendation: str = Field(..., description="Recommendation: BUY, SELL, HOLD")
    confidence: float = Field(..., ge=0, le=100, description="Overall confidence score")
    weighted_score: float = Field(..., description="Weighted score from all agents")
    agent_signals: dict[str, AgentSignalSummary] = Field(
        default_factory=dict,
        description="Individual agent signal summaries"
    )
    reasoning: str = Field(..., description="Synthesized reasoning from Orchestrator")
    position_action: str = Field(..., description="Position action: ENTER, EXIT, HOLD")


class StrategyMetrics(BaseModel):
    """Performance metrics for a trading strategy."""

    total_return: float = Field(..., description="Total return percentage")
    sharpe_ratio: float = Field(..., description="Sharpe ratio")
    max_drawdown: float = Field(..., description="Maximum drawdown percentage")
    win_rate: Optional[float] = Field(
        default=None,
        ge=0,
        le=100,
        description="Win rate percentage (profitable trades / total trades)"
    )
    total_trades: Optional[int] = Field(
        default=None,
        ge=0,
        description="Total number of trades executed"
    )


class BaselineComparison(BaseModel):
    """Comparison of AI strategy against baseline strategies."""

    buy_hold: StrategyMetrics = Field(..., description="Buy & Hold strategy metrics")
    sma_crossover: Optional[StrategyMetrics] = Field(
        default=None,
        description="SMA Crossover strategy metrics"
    )
    macd: Optional[StrategyMetrics] = Field(
        default=None,
        description="MACD strategy metrics"
    )
    ai_outperformance: float = Field(
        ...,
        description="AI outperformance vs Buy & Hold (percentage points)"
    )


class AIBacktestResponse(BaseModel):
    """Response model for AI-driven backtesting endpoint."""

    ticker: str = Field(..., description="Backtested stock ticker")
    strategy_name: str = Field(..., description="Strategy name used")
    total_return: float = Field(..., description="Total return percentage")
    sharpe_ratio: float = Field(..., description="Sharpe ratio")
    max_drawdown: float = Field(..., description="Maximum drawdown percentage")
    win_rate: float = Field(..., ge=0, le=100, description="Win rate percentage")
    total_trades: int = Field(..., ge=0, description="Total number of trades executed")
    trades: list[BacktestTrade] = Field(
        default_factory=list,
        description="List of trades executed"
    )
    equity_curve: list[EquityPoint] = Field(
        default_factory=list,
        description="Equity curve data points"
    )
    ai_signals: list[AISignalRecord] = Field(
        default_factory=list,
        description="AI signal log throughout backtest"
    )
    baseline_comparison: Optional[BaselineComparison] = Field(
        default=None,
        description="Comparison against baseline strategies"
    )
    warnings: list[str] = Field(
        default_factory=list,
        description="Warnings encountered during backtest"
    )
