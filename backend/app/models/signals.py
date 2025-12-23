from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class SignalType(str, Enum):
    """Signal direction type."""

    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"


class Recommendation(str, Enum):
    """Final investment recommendation."""

    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


class AgentSignal(BaseModel):
    """Individual agent analysis signal."""

    agent_name: str = Field(..., description="Name of the agent producing the signal")
    signal_type: SignalType = Field(..., description="Signal direction")
    score: float = Field(
        ...,
        ge=0,
        le=100,
        description="Signal score from 0-100"
    )
    confidence: float = Field(
        ...,
        ge=0,
        le=100,
        description="Confidence level from 0-100"
    )
    reasoning_trace: str = Field(..., description="Explanation of the analysis")
    key_factors: list[str] = Field(
        default_factory=list,
        description="Key factors influencing the signal"
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Signal generation timestamp"
    )


class SynthesisResult(BaseModel):
    """Synthesized result from all agent signals."""

    recommendation: Recommendation = Field(..., description="Final recommendation")
    confidence: float = Field(
        ...,
        ge=0,
        le=100,
        description="Overall confidence score"
    )
    weighted_score: float = Field(
        ...,
        ge=0,
        le=100,
        description="Weighted aggregate score"
    )
    reasoning: str = Field(..., description="Synthesized reasoning explanation")
    agent_contributions: dict[str, float] = Field(
        default_factory=dict,
        description="Each agent's contribution percentage"
    )
    key_catalysts: list[str] = Field(
        default_factory=list,
        description="Key positive catalysts"
    )
    risk_factors: list[str] = Field(
        default_factory=list,
        description="Key risk factors identified"
    )
