from datetime import date
from enum import Enum
from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

from app.utils.validators import (
    validate_ticker,
    TickerValidationError,
    sanitize_string,
)


class AnalysisInterval(str, Enum):
    """Analysis interval for AI backtest signal generation."""

    DAILY = "daily"        # Every trading day
    WEEKLY = "weekly"      # Every 5 trading days (default)
    BIWEEKLY = "biweekly"  # Every 10 trading days
    MONTHLY = "monthly"    # Every 21 trading days


class AgentWeights(BaseModel):
    """Custom weights for agent signal aggregation."""

    fundamental: float = Field(
        default=0.30,
        ge=0,
        le=1,
        description="Weight for fundamental analysis"
    )
    technical: float = Field(
        default=0.25,
        ge=0,
        le=1,
        description="Weight for technical analysis"
    )
    sentiment: float = Field(
        default=0.25,
        ge=0,
        le=1,
        description="Weight for sentiment analysis"
    )
    risk: float = Field(
        default=0.20,
        ge=0,
        le=1,
        description="Weight for risk assessment"
    )

    @model_validator(mode="after")
    def validate_weights_sum(self) -> "AgentWeights":
        """Validate that weights sum to approximately 1.0."""
        total = self.fundamental + self.technical + self.sentiment + self.risk
        if not 0.99 <= total <= 1.01:
            raise ValueError(f"Agent weights must sum to 1.0, got {total:.2f}")
        return self


class AnalyzeRequest(BaseModel):
    """Request model for stock analysis endpoint."""

    ticker: str = Field(..., min_length=1, max_length=10, description="Stock ticker symbol")
    weights: Optional[AgentWeights] = Field(
        default=None,
        description="Custom agent weights (optional)"
    )

    @field_validator("ticker")
    @classmethod
    def validate_ticker_field(cls, v: str) -> str:
        """Validate and normalize ticker symbol."""
        try:
            return validate_ticker(v)
        except TickerValidationError as e:
            raise ValueError(str(e))


class BacktestRequest(BaseModel):
    """Request model for backtesting endpoint."""

    ticker: str = Field(..., min_length=1, max_length=10, description="Stock ticker symbol")
    start_date: date = Field(..., description="Backtest start date")
    end_date: date = Field(..., description="Backtest end date")
    initial_capital: float = Field(
        default=10000.0,
        gt=0,
        description="Initial capital for backtesting"
    )

    @field_validator("ticker")
    @classmethod
    def validate_ticker_field(cls, v: str) -> str:
        """Validate and normalize ticker symbol."""
        try:
            return validate_ticker(v)
        except TickerValidationError as e:
            raise ValueError(str(e))

    @model_validator(mode="after")
    def validate_date_range(self) -> "BacktestRequest":
        """Validate that start_date is before end_date."""
        if self.start_date >= self.end_date:
            raise ValueError("Start date must be before end date")
        return self


class AIBacktestRequest(BaseModel):
    """Request model for AI-driven backtesting endpoint."""

    ticker: str = Field(..., min_length=1, max_length=10, description="Stock ticker symbol")
    start_date: date = Field(..., description="Backtest start date")
    end_date: date = Field(..., description="Backtest end date")
    initial_capital: float = Field(
        default=10000.0,
        gt=0,
        description="Initial capital for backtesting"
    )
    strategy: Literal["ai_live", "ai_simulated", "sma_crossover", "macd", "buy_hold", "auto"] = Field(
        default="ai_live",
        description="Backtest strategy to use"
    )
    analysis_interval: AnalysisInterval = Field(
        default=AnalysisInterval.WEEKLY,
        description="Interval for AI signal generation"
    )
    include_baseline_comparison: bool = Field(
        default=True,
        description="Whether to include baseline strategy comparison"
    )

    @field_validator("ticker")
    @classmethod
    def validate_ticker_field(cls, v: str) -> str:
        """Validate and normalize ticker symbol."""
        try:
            return validate_ticker(v)
        except TickerValidationError as e:
            raise ValueError(str(e))

    @model_validator(mode="after")
    def validate_date_range(self) -> "AIBacktestRequest":
        """Validate that start_date is before end_date."""
        if self.start_date >= self.end_date:
            raise ValueError("Start date must be before end date")
        return self
