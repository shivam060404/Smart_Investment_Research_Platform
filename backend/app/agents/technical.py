import logging
from typing import Any, Optional

from app.agents.base import BaseAgent, AgentAnalysisError
from app.models.signals import AgentSignal
from app.models.stock import TechnicalIndicators
from app.services.data_fetcher import (
    DataFetcher,
    get_data_fetcher,
    DataFetcherError,
)
from app.services.mistral_service import MistralService
from app.utils.indicators import calculate_all_indicators

logger = logging.getLogger(__name__)


TECHNICAL_SYSTEM_PROMPT = """You are an expert technical analysis agent specializing in analyzing price patterns and technical indicators.

Your task is to analyze the provided technical indicators and generate an investment signal based on technical analysis principles.

## Analysis Framework

Evaluate the following indicators:

1. **Moving Averages (SMA)**
   - 50-day SMA: Short-term trend indicator
   - 200-day SMA: Long-term trend indicator
   - Golden Cross (50 > 200): Bullish signal
   - Death Cross (50 < 200): Bearish signal
   - Price above both SMAs: Strong bullish
   - Price below both SMAs: Strong bearish

2. **RSI (Relative Strength Index)**
   - Above 70: Overbought (potential reversal down)
   - Below 30: Oversold (potential reversal up)
   - 40-60: Neutral zone
   - Divergences from price can signal reversals

3. **MACD (Moving Average Convergence Divergence)**
   - MACD above signal line: Bullish momentum
   - MACD below signal line: Bearish momentum
   - Positive histogram: Strengthening bullish momentum
   - Negative histogram: Strengthening bearish momentum

4. **Bollinger Bands**
   - Price near upper band: Potentially overbought
   - Price near lower band: Potentially oversold
   - Band squeeze: Low volatility, potential breakout coming
   - Band expansion: High volatility, trend in progress

## Confidence Calculation - IMPORTANT

Your confidence score MUST vary based on indicator alignment:
- **85-95%**: All indicators align (same direction), strong trend, clear signals
- **70-84%**: Most indicators align, moderate trend strength
- **55-69%**: Mixed signals OR indicators near neutral zones
- **40-54%**: Conflicting indicators OR weak/choppy price action
- **25-39%**: Highly conflicting signals OR insufficient data

DO NOT default to 75%. Calculate confidence based on:
1. Indicator alignment (how many point in the same direction?)
2. Signal strength (how far from neutral are RSI, MACD?)
3. Trend clarity (is there a clear trend or consolidation?)

## Output Format

You must respond with a JSON object containing:
{
    "score": <number 0-100, where higher indicates more bullish>,
    "signal_type": "<bullish|bearish|neutral>",
    "confidence": <number 0-100, calculated based on indicator alignment>,
    "reasoning": "<detailed explanation of technical analysis>",
    "key_factors": ["<factor 1>", "<factor 2>", ...],
    "trend_assessment": {
        "short_term": "<bullish|bearish|neutral>",
        "long_term": "<bullish|bearish|neutral>",
        "momentum": "<strong|moderate|weak>"
    }
}

## Scoring Guidelines
- 70-100: Strong bullish technicals, uptrend confirmed
- 50-69: Mildly bullish or consolidating
- 30-49: Mildly bearish or consolidating
- 0-29: Strong bearish technicals, downtrend confirmed

Consider the confluence of multiple indicators. Stronger signals come when multiple indicators align."""


class TechnicalAgent(BaseAgent):
    """Technical Analysis Agent for evaluating price patterns and indicators.
    
    This agent analyzes technical indicators including moving averages,
    RSI, MACD, and Bollinger Bands to generate investment signals.
    """

    PRICE_HISTORY_PERIOD = "1y"

    def __init__(
        self,
        mistral_service: Optional[MistralService] = None,
        data_fetcher: Optional[DataFetcher] = None,
        use_mcp: Optional[bool] = None,
    ):
        """Initialize the Technical Analysis Agent.
        
        Args:
            mistral_service: Mistral service for AI analysis.
            data_fetcher: Data fetcher for retrieving price history.
            use_mcp: Whether to use MCP adapter. If None, uses global setting.
        """
        super().__init__(name="Technical", mistral_service=mistral_service, use_mcp=use_mcp)
        self.data_fetcher = data_fetcher or get_data_fetcher()

    def get_system_prompt(self) -> str:
        """Return the technical analysis system prompt.
        
        Returns:
            System prompt for technical analysis.
        """
        return TECHNICAL_SYSTEM_PROMPT

    async def analyze(self, ticker: str, data: dict[str, Any]) -> AgentSignal:
        """Execute technical analysis on the given stock.
        
        Fetches price history, calculates technical indicators, and uses
        Mistral to generate an analysis signal.
        
        Args:
            ticker: Stock ticker symbol to analyze.
            data: Optional pre-fetched data. If 'technical_indicators' key exists,
                  uses that instead of calculating.
        
        Returns:
            AgentSignal with technical analysis results.
        
        Raises:
            AgentAnalysisError: If analysis fails.
        """
        logger.info(f"{self.name}: Starting analysis for {ticker}")
        
        try:
            # Get technical indicators
            indicators, current_price = await self._get_indicators(ticker, data)
            
            if indicators is None:
                logger.warning(f"{self.name}: Insufficient data for {ticker}")
                return self._create_insufficient_data_signal(ticker)
            
            # Prepare data for analysis
            analysis_data = self._prepare_analysis_data(
                ticker, indicators, current_price, data
            )
            
            # Build prompts
            user_prompt = self._build_user_prompt(ticker, analysis_data)
            
            # Generate analysis via Mistral
            analysis_result = await self.mistral_service.generate_structured_analysis(
                system_prompt=self.get_system_prompt(),
                user_prompt=user_prompt,
            )
            
            # Create and return signal
            signal = await self._generate_signal_from_analysis(ticker, analysis_result)
            logger.info(f"{self.name}: Completed analysis for {ticker} - Score: {signal.score}")
            
            return signal
            
        except DataFetcherError as e:
            logger.error(f"{self.name}: Data fetch error for {ticker}: {e}")
            raise AgentAnalysisError(f"Failed to fetch price data: {e}")
        except Exception as e:
            logger.error(f"{self.name}: Analysis error for {ticker}: {e}")
            raise AgentAnalysisError(f"Technical analysis failed: {e}")

    async def _get_indicators(
        self,
        ticker: str,
        data: dict[str, Any]
    ) -> tuple[Optional[TechnicalIndicators], float]:
        """Get technical indicators from data or calculate from price history.
        
        Args:
            ticker: Stock ticker symbol.
            data: Pre-fetched data dictionary.
        
        Returns:
            Tuple of (TechnicalIndicators, current_price) or (None, 0) if insufficient data.
        """
        # Check if indicators already provided
        if "technical_indicators" in data:
            indicators_data = data["technical_indicators"]
            current_price = data.get("current_price", 0)
            if isinstance(indicators_data, TechnicalIndicators):
                return indicators_data, current_price
            return TechnicalIndicators(**indicators_data), current_price
        
        # Use MCP adapter if enabled
        if self.use_mcp and self.mcp_adapter:
            try:
                logger.debug(f"{self.name}: Using MCP adapter for {ticker}")
                indicators = await self.mcp_adapter.get_technical_indicators(ticker)
                # Get current price from stock data
                stock_data = await self.mcp_adapter.get_stock_data(ticker)
                return indicators, stock_data.current_price
            except Exception as e:
                logger.warning(f"{self.name}: MCP adapter failed for {ticker}, falling back to direct: {e}")
                # Fall through to direct calculation
        
        # Fetch price history (direct data access)
        price_history = self.data_fetcher.get_price_history(
            ticker,
            period=self.PRICE_HISTORY_PERIOD,
            interval="1d"
        )
        
        if not price_history or len(price_history) < 200:
            logger.warning(f"{self.name}: Insufficient price history for {ticker}")
            return None, 0
        
        # Extract closing prices
        prices = [day["close"] for day in price_history]
        current_price = prices[-1]
        
        # Calculate indicators
        indicators = calculate_all_indicators(prices, current_price)
        
        return indicators, current_price

    def _prepare_analysis_data(
        self,
        ticker: str,
        indicators: TechnicalIndicators,
        current_price: float,
        data: dict[str, Any]
    ) -> dict[str, Any]:
        """Prepare data dictionary for analysis prompt.
        
        Args:
            ticker: Stock ticker symbol.
            indicators: Calculated technical indicators.
            current_price: Current stock price.
            data: Additional data from orchestrator.
        
        Returns:
            Dictionary with all relevant analysis data.
        """
        analysis_data = {
            "ticker": ticker,
            "current_price": current_price,
            "sma_50": indicators.sma_50,
            "sma_200": indicators.sma_200,
            "rsi_14": indicators.rsi_14,
            "macd_line": indicators.macd.macd_line,
            "macd_signal": indicators.macd.signal_line,
            "macd_histogram": indicators.macd.histogram,
            "bollinger_upper": indicators.bollinger_bands.upper_band,
            "bollinger_middle": indicators.bollinger_bands.middle_band,
            "bollinger_lower": indicators.bollinger_bands.lower_band,
            "current_trend": indicators.current_trend,
        }
        
        # Calculate derived metrics
        analysis_data["price_vs_sma50"] = self._calculate_percentage_diff(
            current_price, indicators.sma_50
        )
        analysis_data["price_vs_sma200"] = self._calculate_percentage_diff(
            current_price, indicators.sma_200
        )
        analysis_data["sma_crossover"] = "golden_cross" if indicators.sma_50 > indicators.sma_200 else "death_cross"
        analysis_data["bollinger_position"] = self._calculate_bollinger_position(
            current_price, indicators.bollinger_bands
        )
        
        return analysis_data

    def _calculate_percentage_diff(self, price: float, reference: float) -> str:
        """Calculate percentage difference from reference.
        
        Args:
            price: Current price.
            reference: Reference value.
        
        Returns:
            Formatted percentage string.
        """
        if reference == 0:
            return "N/A"
        diff = ((price - reference) / reference) * 100
        sign = "+" if diff > 0 else ""
        return f"{sign}{diff:.2f}%"

    def _calculate_bollinger_position(
        self,
        price: float,
        bollinger: Any
    ) -> str:
        """Determine price position within Bollinger Bands.
        
        Args:
            price: Current price.
            bollinger: Bollinger bands data.
        
        Returns:
            Position description.
        """
        band_width = bollinger.upper_band - bollinger.lower_band
        if band_width == 0:
            return "at_middle"
        
        position = (price - bollinger.lower_band) / band_width
        
        if position >= 0.9:
            return "near_upper_band"
        elif position <= 0.1:
            return "near_lower_band"
        elif 0.4 <= position <= 0.6:
            return "at_middle"
        elif position > 0.6:
            return "upper_half"
        else:
            return "lower_half"

    def _build_user_prompt(self, ticker: str, data: dict[str, Any]) -> str:
        """Build the user prompt for technical analysis.
        
        Args:
            ticker: Stock ticker symbol.
            data: Analysis data dictionary.
        
        Returns:
            Formatted user prompt.
        """
        lines = [
            f"Analyze the technical indicators for {ticker}:",
            "",
            f"Current Price: ${data['current_price']:.2f}",
            f"Current Trend: {data['current_trend'].upper()}",
            "",
            "## Moving Averages",
            f"- 50-day SMA: ${data['sma_50']:.2f} (Price {data['price_vs_sma50']} from SMA)",
            f"- 200-day SMA: ${data['sma_200']:.2f} (Price {data['price_vs_sma200']} from SMA)",
            f"- SMA Crossover: {data['sma_crossover'].replace('_', ' ').title()}",
            "",
            "## RSI (14-day)",
            f"- RSI Value: {data['rsi_14']:.2f}",
            f"- Status: {self._interpret_rsi(data['rsi_14'])}",
            "",
            "## MACD",
            f"- MACD Line: {data['macd_line']:.4f}",
            f"- Signal Line: {data['macd_signal']:.4f}",
            f"- Histogram: {data['macd_histogram']:.4f}",
            f"- Signal: {'Bullish' if data['macd_histogram'] > 0 else 'Bearish'} momentum",
            "",
            "## Bollinger Bands",
            f"- Upper Band: ${data['bollinger_upper']:.2f}",
            f"- Middle Band: ${data['bollinger_middle']:.2f}",
            f"- Lower Band: ${data['bollinger_lower']:.2f}",
            f"- Price Position: {data['bollinger_position'].replace('_', ' ').title()}",
            "",
            "Based on these indicators, provide your technical analysis signal.",
            "Identify the trend direction and key technical factors.",
        ]
        
        return "\n".join(lines)

    def _interpret_rsi(self, rsi: float) -> str:
        """Interpret RSI value.
        
        Args:
            rsi: RSI value.
        
        Returns:
            Interpretation string.
        """
        if rsi >= 70:
            return "Overbought (potential reversal down)"
        elif rsi <= 30:
            return "Oversold (potential reversal up)"
        elif rsi >= 60:
            return "Bullish momentum"
        elif rsi <= 40:
            return "Bearish momentum"
        return "Neutral"

    def _create_insufficient_data_signal(self, ticker: str) -> AgentSignal:
        """Create a neutral signal when insufficient data is available.
        
        Args:
            ticker: Stock ticker symbol.
        
        Returns:
            AgentSignal with neutral assessment and low confidence.
        """
        from datetime import datetime
        from app.models.signals import SignalType
        
        return AgentSignal(
            agent_name=self.name,
            signal_type=SignalType.NEUTRAL,
            score=50.0,
            confidence=10.0,
            reasoning_trace=(
                f"Insufficient price history for {ticker} to calculate technical indicators. "
                "At least 200 days of price data is required for accurate analysis."
            ),
            key_factors=["Insufficient price history for technical analysis"],
            timestamp=datetime.utcnow(),
        )
