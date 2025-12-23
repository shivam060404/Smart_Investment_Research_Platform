from typing import Optional
from pydantic import BaseModel, Field


class StockData(BaseModel):
    """Basic stock information."""

    ticker: str = Field(..., description="Stock ticker symbol")
    name: str = Field(..., description="Company name")
    sector: str = Field(default="Unknown", description="Business sector")
    industry: str = Field(default="Unknown", description="Industry classification")
    current_price: float = Field(..., description="Current stock price")
    market_cap: float = Field(default=0.0, description="Market capitalization")
    currency: str = Field(default="USD", description="Currency code")


class FundamentalMetrics(BaseModel):
    """Fundamental financial metrics for analysis."""

    pe_ratio: Optional[float] = Field(default=None, description="Price-to-Earnings ratio")
    pb_ratio: Optional[float] = Field(default=None, description="Price-to-Book ratio")
    roe: Optional[float] = Field(default=None, description="Return on Equity")
    roa: Optional[float] = Field(default=None, description="Return on Assets")
    debt_to_equity: Optional[float] = Field(default=None, description="Debt-to-Equity ratio")
    revenue_growth: Optional[float] = Field(default=None, description="Revenue growth rate")
    profit_margin: Optional[float] = Field(default=None, description="Profit margin percentage")
    dividend_yield: Optional[float] = Field(default=None, description="Dividend yield percentage")


class MACDData(BaseModel):
    """MACD indicator data."""

    macd_line: float = Field(..., description="MACD line value")
    signal_line: float = Field(..., description="Signal line value")
    histogram: float = Field(..., description="MACD histogram value")


class BollingerData(BaseModel):
    """Bollinger Bands indicator data."""

    upper_band: float = Field(..., description="Upper Bollinger Band")
    middle_band: float = Field(..., description="Middle Bollinger Band (SMA)")
    lower_band: float = Field(..., description="Lower Bollinger Band")


class TechnicalIndicators(BaseModel):
    """Technical analysis indicators."""

    sma_50: float = Field(..., description="50-day Simple Moving Average")
    sma_200: float = Field(..., description="200-day Simple Moving Average")
    rsi_14: float = Field(..., description="14-day Relative Strength Index")
    macd: MACDData = Field(..., description="MACD indicator data")
    bollinger_bands: BollingerData = Field(..., description="Bollinger Bands data")
    current_trend: str = Field(
        default="neutral",
        description="Current trend direction: bullish, bearish, or neutral"
    )


class SentimentData(BaseModel):
    """Sentiment analysis data."""

    overall_score: float = Field(
        ...,
        ge=-100,
        le=100,
        description="Overall sentiment score from -100 to +100"
    )
    article_count: int = Field(default=0, description="Number of articles analyzed")
    key_catalysts: list[str] = Field(
        default_factory=list,
        description="Key catalysts identified from news"
    )
    source_breakdown: dict[str, float] = Field(
        default_factory=dict,
        description="Sentiment breakdown by source"
    )


class RiskFactor(BaseModel):
    """Individual risk factor."""

    factor_type: str = Field(
        ...,
        description="Risk type: market, sector, company, regulatory, macroeconomic"
    )
    description: str = Field(..., description="Risk factor description")
    severity: str = Field(
        default="medium",
        description="Severity level: low, medium, high"
    )


class RiskMetrics(BaseModel):
    """Risk assessment metrics."""

    volatility: float = Field(..., description="Historical volatility")
    beta: float = Field(..., description="Beta relative to S&P 500")
    var_95: float = Field(default=0.0, description="Value at Risk at 95% confidence")
    risk_score: float = Field(
        ...,
        ge=0,
        le=100,
        description="Overall risk score from 0-100"
    )
    risk_factors: list[RiskFactor] = Field(
        default_factory=list,
        description="Identified risk factors"
    )
