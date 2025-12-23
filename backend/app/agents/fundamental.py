import logging
from typing import Any, Optional

from app.agents.base import BaseAgent, AgentAnalysisError
from app.models.signals import AgentSignal
from app.models.stock import FundamentalMetrics
from app.services.data_fetcher import (
    DataFetcher,
    get_data_fetcher,
    DataFetcherError,
)
from app.services.mistral_service import MistralService

logger = logging.getLogger(__name__)


FUNDAMENTAL_SYSTEM_PROMPT = """You are an expert fundamental analysis agent specializing in evaluating company financial health and valuation metrics.

Your task is to analyze the provided financial metrics and generate an investment signal based on fundamental analysis principles.

## Analysis Framework

Evaluate the following aspects:

1. **Valuation Metrics**
   - P/E Ratio: Compare to sector average (typically 15-25 is fair value)
   - P/B Ratio: Below 1 may indicate undervaluation, above 3 may indicate overvaluation

2. **Profitability Metrics**
   - ROE (Return on Equity): Above 15% is generally good
   - ROA (Return on Assets): Above 5% is generally good
   - Profit Margin: Higher is better, compare to industry peers

3. **Financial Health**
   - Debt-to-Equity: Below 1 is conservative, above 2 may indicate high leverage risk

4. **Growth Indicators**
   - Revenue Growth: Positive growth is favorable, double-digit growth is strong

## Confidence Calculation - IMPORTANT

Your confidence score MUST vary based on data quality and clarity:
- **85-95%**: All metrics available, clear bullish/bearish signals, no contradictions
- **70-84%**: Most metrics available, generally consistent signals
- **55-69%**: Some metrics missing OR mixed signals
- **40-54%**: Multiple metrics missing OR contradictory signals
- **25-39%**: Limited data OR highly uncertain outlook

DO NOT default to 75%. Calculate confidence based on:
1. Data completeness (how many metrics are available vs N/A)
2. Signal clarity (are metrics pointing in the same direction?)
3. Metric strength (how far from neutral are the values?)

## Output Format

You must respond with a JSON object containing:
{
    "score": <number 0-100, where higher indicates more bullish>,
    "signal_type": "<bullish|bearish|neutral>",
    "confidence": <number 0-100, calculated based on data quality and signal clarity>,
    "reasoning": "<detailed explanation of your analysis>",
    "key_factors": ["<factor 1>", "<factor 2>", ...]
}

## Scoring Guidelines
- 70-100: Strong fundamentals, undervalued, recommend BUY
- 40-69: Mixed signals, fair value, recommend HOLD
- 0-39: Weak fundamentals, overvalued, recommend SELL

Be objective and base your analysis solely on the provided data. If data is missing, note it in your reasoning and REDUCE confidence accordingly."""


class FundamentalAgent(BaseAgent):
    """Fundamental Analysis Agent for evaluating company financial health.
    
    This agent analyzes financial metrics including P/E ratio, P/B ratio,
    ROE, ROA, debt-to-equity, revenue growth, and profit margins to
    generate investment signals.
    """

    def __init__(
        self,
        mistral_service: Optional[MistralService] = None,
        data_fetcher: Optional[DataFetcher] = None,
        use_mcp: Optional[bool] = None,
    ):
        """Initialize the Fundamental Analysis Agent.
        
        Args:
            mistral_service: Mistral service for AI analysis.
            data_fetcher: Data fetcher for retrieving financial metrics.
            use_mcp: Whether to use MCP adapter. If None, uses global setting.
        """
        super().__init__(name="Fundamental", mistral_service=mistral_service, use_mcp=use_mcp)
        self.data_fetcher = data_fetcher or get_data_fetcher()

    def get_system_prompt(self) -> str:
        """Return the fundamental analysis system prompt.
        
        Returns:
            System prompt for fundamental analysis.
        """
        return FUNDAMENTAL_SYSTEM_PROMPT

    async def analyze(self, ticker: str, data: dict[str, Any]) -> AgentSignal:
        """Execute fundamental analysis on the given stock.
        
        Fetches financial metrics from Yahoo Finance (with AlphaVantage fallback)
        and uses Mistral to generate an analysis signal.
        
        Args:
            ticker: Stock ticker symbol to analyze.
            data: Optional pre-fetched data. If 'fundamental_metrics' key exists,
                  uses that instead of fetching.
        
        Returns:
            AgentSignal with fundamental analysis results.
        
        Raises:
            AgentAnalysisError: If analysis fails.
        """
        logger.info(f"{self.name}: Starting analysis for {ticker}")
        
        try:
            # Get fundamental metrics (use provided data or fetch)
            metrics = await self._get_metrics(ticker, data)
            
            # Prepare data for analysis
            analysis_data = self._prepare_analysis_data(ticker, metrics, data)
            
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
            raise AgentAnalysisError(f"Failed to fetch fundamental data: {e}")
        except Exception as e:
            logger.error(f"{self.name}: Analysis error for {ticker}: {e}")
            raise AgentAnalysisError(f"Fundamental analysis failed: {e}")

    async def _get_metrics(
        self,
        ticker: str,
        data: dict[str, Any]
    ) -> FundamentalMetrics:
        """Get fundamental metrics from data or fetch from API.
        
        Args:
            ticker: Stock ticker symbol.
            data: Pre-fetched data dictionary.
        
        Returns:
            FundamentalMetrics for the stock.
        """
        # Check if metrics already provided
        if "fundamental_metrics" in data:
            metrics_data = data["fundamental_metrics"]
            if isinstance(metrics_data, FundamentalMetrics):
                return metrics_data
            return FundamentalMetrics(**metrics_data)
        
        # Use MCP adapter if enabled
        if self.use_mcp and self.mcp_adapter:
            try:
                logger.debug(f"{self.name}: Using MCP adapter for {ticker}")
                return await self.mcp_adapter.get_fundamental_metrics(ticker)
            except Exception as e:
                logger.warning(f"{self.name}: MCP adapter failed for {ticker}, falling back to direct: {e}")
                # Fall through to direct data fetcher
        
        # Fetch with fallback (direct data access)
        return await self.data_fetcher.get_financial_metrics_with_fallback(ticker)

    def _prepare_analysis_data(
        self,
        ticker: str,
        metrics: FundamentalMetrics,
        data: dict[str, Any]
    ) -> dict[str, Any]:
        """Prepare data dictionary for analysis prompt.
        
        Args:
            ticker: Stock ticker symbol.
            metrics: Fundamental metrics.
            data: Additional data from orchestrator.
        
        Returns:
            Dictionary with all relevant analysis data.
        """
        analysis_data = {
            "ticker": ticker,
            "pe_ratio": metrics.pe_ratio,
            "pb_ratio": metrics.pb_ratio,
            "roe": self._format_percentage(metrics.roe),
            "roa": self._format_percentage(metrics.roa),
            "debt_to_equity": metrics.debt_to_equity,
            "revenue_growth": self._format_percentage(metrics.revenue_growth),
            "profit_margin": self._format_percentage(metrics.profit_margin),
            "dividend_yield": self._format_percentage(metrics.dividend_yield),
        }
        
        # Add stock info if available
        if "stock_data" in data:
            stock_data = data["stock_data"]
            if hasattr(stock_data, "sector"):
                analysis_data["sector"] = stock_data.sector
            if hasattr(stock_data, "industry"):
                analysis_data["industry"] = stock_data.industry
            if hasattr(stock_data, "market_cap"):
                analysis_data["market_cap"] = self._format_market_cap(stock_data.market_cap)
            if hasattr(stock_data, "current_price"):
                analysis_data["current_price"] = stock_data.current_price
        
        return analysis_data

    def _format_percentage(self, value: Optional[float]) -> Optional[str]:
        """Format a decimal value as percentage string.
        
        Args:
            value: Decimal value (e.g., 0.15 for 15%).
        
        Returns:
            Formatted percentage string or None.
        """
        if value is None:
            return None
        return f"{value * 100:.2f}%"

    def _format_market_cap(self, value: float) -> str:
        """Format market cap in human-readable form.
        
        Args:
            value: Market cap in dollars.
        
        Returns:
            Formatted string (e.g., "150.5B").
        """
        if value >= 1e12:
            return f"${value / 1e12:.2f}T"
        elif value >= 1e9:
            return f"${value / 1e9:.2f}B"
        elif value >= 1e6:
            return f"${value / 1e6:.2f}M"
        return f"${value:,.0f}"

    def _build_user_prompt(self, ticker: str, data: dict[str, Any]) -> str:
        """Build the user prompt for fundamental analysis.
        
        Args:
            ticker: Stock ticker symbol.
            data: Analysis data dictionary.
        
        Returns:
            Formatted user prompt.
        """
        lines = [f"Analyze the fundamental metrics for {ticker}:", ""]
        
        # Company info
        if data.get("sector"):
            lines.append(f"Sector: {data['sector']}")
        if data.get("industry"):
            lines.append(f"Industry: {data['industry']}")
        if data.get("market_cap"):
            lines.append(f"Market Cap: {data['market_cap']}")
        if data.get("current_price"):
            lines.append(f"Current Price: ${data['current_price']:.2f}")
        
        lines.append("")
        lines.append("## Financial Metrics")
        
        # Valuation
        lines.append("")
        lines.append("### Valuation")
        lines.append(f"- P/E Ratio: {self._format_metric(data.get('pe_ratio'))}")
        lines.append(f"- P/B Ratio: {self._format_metric(data.get('pb_ratio'))}")
        
        # Profitability
        lines.append("")
        lines.append("### Profitability")
        lines.append(f"- ROE: {data.get('roe') or 'N/A'}")
        lines.append(f"- ROA: {data.get('roa') or 'N/A'}")
        lines.append(f"- Profit Margin: {data.get('profit_margin') or 'N/A'}")
        
        # Financial Health
        lines.append("")
        lines.append("### Financial Health")
        lines.append(f"- Debt-to-Equity: {self._format_metric(data.get('debt_to_equity'))}")
        
        # Growth
        lines.append("")
        lines.append("### Growth")
        lines.append(f"- Revenue Growth: {data.get('revenue_growth') or 'N/A'}")
        
        # Dividends
        if data.get("dividend_yield"):
            lines.append("")
            lines.append("### Dividends")
            lines.append(f"- Dividend Yield: {data['dividend_yield']}")
        
        lines.append("")
        lines.append("Based on these metrics, provide your fundamental analysis signal.")
        
        return "\n".join(lines)

    def _format_metric(self, value: Any) -> str:
        """Format a metric value for display.
        
        Args:
            value: Metric value.
        
        Returns:
            Formatted string.
        """
        if value is None:
            return "N/A"
        if isinstance(value, float):
            return f"{value:.2f}"
        return str(value)
