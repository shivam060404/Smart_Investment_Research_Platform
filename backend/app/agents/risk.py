import logging
import math
from typing import Any, Optional

from app.agents.base import BaseAgent, AgentAnalysisError
from app.models.signals import AgentSignal
from app.models.stock import RiskMetrics, RiskFactor
from app.services.data_fetcher import (
    DataFetcher,
    get_data_fetcher,
    DataFetcherError,
)
from app.services.mistral_service import MistralService

logger = logging.getLogger(__name__)


RISK_SYSTEM_PROMPT = """You are an expert risk assessment analyst specializing in evaluating investment risk factors.

Your task is to analyze the provided risk metrics and identify key risk factors that could impact the investment.

## Analysis Framework

Evaluate the following aspects:

1. **Volatility Analysis**
   - Historical volatility: Higher values indicate more price fluctuation
   - Low (<20%): Low volatility, stable stock
   - Medium (20-40%): Moderate volatility
   - High (>40%): High volatility, risky

2. **Beta Analysis**
   - Beta < 0.8: Less volatile than market (defensive)
   - Beta 0.8-1.2: Similar to market
   - Beta > 1.2: More volatile than market (aggressive)
   - Negative beta: Moves opposite to market

3. **Risk Factor Categories**
   Identify and categorize risks into:
   - Market Risk: Overall market conditions, economic cycles
   - Sector Risk: Industry-specific challenges
   - Company Risk: Business model, competition, management
   - Regulatory Risk: Government policies, compliance
   - Macroeconomic Risk: Interest rates, inflation, currency

4. **Value at Risk (VaR)**
   - 95% VaR indicates potential loss in worst 5% of scenarios
   - Higher VaR = higher potential downside

## Confidence Calculation - IMPORTANT

Your confidence score MUST vary based on data quality:
- **85-95%**: Full price history, all metrics calculable, clear risk profile
- **70-84%**: Most metrics available, consistent risk indicators
- **55-69%**: Some metrics missing OR mixed risk signals
- **40-54%**: Limited price history OR inconsistent risk indicators
- **25-39%**: Insufficient data for reliable risk assessment

DO NOT default to 75%. Calculate confidence based on:
1. Data completeness (volatility, beta, VaR all available?)
2. Price history length (longer = more reliable)
3. Risk indicator consistency (do they tell the same story?)

## Output Format

You must respond with a JSON object containing:
{
    "score": <number 0-100, where HIGHER indicates HIGHER RISK>,
    "signal_type": "<bullish|bearish|neutral>",
    "confidence": <number 0-100, calculated based on data quality>,
    "reasoning": "<detailed explanation of risk assessment>",
    "key_factors": ["<risk factor 1>", "<risk factor 2>", ...],
    "risk_breakdown": {
        "market_risk": "<low|medium|high>",
        "sector_risk": "<low|medium|high>",
        "company_risk": "<low|medium|high>",
        "regulatory_risk": "<low|medium|high>",
        "macroeconomic_risk": "<low|medium|high>"
    }
}

## Scoring Guidelines (Risk Score - Higher = More Risky)
- 0-29: Low risk, suitable for conservative investors
- 30-49: Moderate risk, balanced risk-reward
- 50-69: Elevated risk, suitable for growth-oriented investors
- 70-100: High risk, suitable only for aggressive investors

Note: For the signal_type, consider that HIGH risk = BEARISH signal (caution advised)."""


class RiskAgent(BaseAgent):
    """Risk Assessment Agent for evaluating investment risk.
    
    This agent analyzes volatility, beta, and various risk factors
    to generate a comprehensive risk assessment signal.
    """

    PRICE_HISTORY_PERIOD = "1y"
    SP500_TICKER = "^GSPC"

    def __init__(
        self,
        mistral_service: Optional[MistralService] = None,
        data_fetcher: Optional[DataFetcher] = None,
        use_mcp: Optional[bool] = None,
    ):
        """Initialize the Risk Assessment Agent.
        
        Args:
            mistral_service: Mistral service for AI analysis.
            data_fetcher: Data fetcher for retrieving price data.
            use_mcp: Whether to use MCP adapter. If None, uses global setting.
        """
        super().__init__(name="Risk", mistral_service=mistral_service, use_mcp=use_mcp)
        self.data_fetcher = data_fetcher or get_data_fetcher()

    def get_system_prompt(self) -> str:
        """Return the risk assessment system prompt.
        
        Returns:
            System prompt for risk assessment.
        """
        return RISK_SYSTEM_PROMPT

    async def analyze(self, ticker: str, data: dict[str, Any]) -> AgentSignal:
        """Execute risk assessment on the given stock.
        
        Calculates volatility, beta, and uses Mistral to identify
        and categorize risk factors.
        
        Args:
            ticker: Stock ticker symbol to analyze.
            data: Optional pre-fetched data. If 'risk_metrics' key exists,
                  uses that instead of calculating.
        
        Returns:
            AgentSignal with risk assessment results.
        
        Raises:
            AgentAnalysisError: If analysis fails.
        """
        logger.info(f"{self.name}: Starting analysis for {ticker}")
        
        try:
            # Get risk metrics
            risk_metrics = await self._get_risk_metrics(ticker, data)
            
            if risk_metrics is None:
                logger.warning(f"{self.name}: Insufficient data for {ticker}")
                return self._create_insufficient_data_signal(ticker)
            
            # Prepare data for analysis
            analysis_data = self._prepare_analysis_data(ticker, risk_metrics, data)
            
            # Build prompts
            user_prompt = self._build_user_prompt(ticker, analysis_data)
            
            # Generate analysis via Mistral
            analysis_result = await self.mistral_service.generate_structured_analysis(
                system_prompt=self.get_system_prompt(),
                user_prompt=user_prompt,
            )
            
            # Create and return signal
            signal = await self._generate_signal_from_analysis(ticker, analysis_result)
            logger.info(f"{self.name}: Completed analysis for {ticker} - Risk Score: {signal.score}")
            
            return signal
            
        except DataFetcherError as e:
            logger.error(f"{self.name}: Data fetch error for {ticker}: {e}")
            raise AgentAnalysisError(f"Failed to fetch price data: {e}")
        except Exception as e:
            logger.error(f"{self.name}: Analysis error for {ticker}: {e}")
            raise AgentAnalysisError(f"Risk assessment failed: {e}")

    async def _get_risk_metrics(
        self,
        ticker: str,
        data: dict[str, Any]
    ) -> Optional[RiskMetrics]:
        """Get risk metrics from data or calculate from price history.
        
        Args:
            ticker: Stock ticker symbol.
            data: Pre-fetched data dictionary.
        
        Returns:
            RiskMetrics or None if insufficient data.
        """
        # Check if metrics already provided
        if "risk_metrics" in data:
            metrics_data = data["risk_metrics"]
            if isinstance(metrics_data, RiskMetrics):
                return metrics_data
            return RiskMetrics(**metrics_data)
        
        # Fetch price history for stock
        try:
            stock_history = self.data_fetcher.get_price_history(
                ticker,
                period=self.PRICE_HISTORY_PERIOD,
                interval="1d"
            )
        except DataFetcherError:
            return None
        
        if not stock_history or len(stock_history) < 30:
            return None
        
        # Fetch S&P 500 for beta calculation
        try:
            market_history = self.data_fetcher.get_price_history(
                self.SP500_TICKER,
                period=self.PRICE_HISTORY_PERIOD,
                interval="1d"
            )
        except DataFetcherError:
            market_history = None
        
        # Calculate metrics
        stock_returns = self._calculate_returns(stock_history)
        volatility = self._calculate_volatility(stock_returns)
        var_95 = self._calculate_var(stock_returns)
        
        # Calculate beta if market data available
        beta = 1.0  # Default to market beta
        if market_history and len(market_history) >= 30:
            market_returns = self._calculate_returns(market_history)
            beta = self._calculate_beta(stock_returns, market_returns)
        
        # Calculate overall risk score
        risk_score = self._calculate_risk_score(volatility, beta, var_95)
        
        # Get stock info for risk factors
        stock_data = data.get("stock_data")
        risk_factors = self._identify_basic_risk_factors(
            volatility, beta, stock_data
        )
        
        return RiskMetrics(
            volatility=volatility,
            beta=beta,
            var_95=var_95,
            risk_score=risk_score,
            risk_factors=risk_factors,
        )

    def _calculate_returns(self, price_history: list[dict]) -> list[float]:
        """Calculate daily returns from price history.
        
        Args:
            price_history: List of price data dictionaries.
        
        Returns:
            List of daily returns.
        """
        prices = [day["close"] for day in price_history]
        returns = []
        for i in range(1, len(prices)):
            if prices[i - 1] != 0:
                daily_return = (prices[i] - prices[i - 1]) / prices[i - 1]
                returns.append(daily_return)
        return returns

    def _calculate_volatility(self, returns: list[float]) -> float:
        """Calculate annualized volatility from returns.
        
        Args:
            returns: List of daily returns.
        
        Returns:
            Annualized volatility as percentage.
        """
        if not returns:
            return 0.0
        
        mean_return = sum(returns) / len(returns)
        variance = sum((r - mean_return) ** 2 for r in returns) / len(returns)
        daily_std = math.sqrt(variance)
        
        # Annualize (252 trading days)
        annual_volatility = daily_std * math.sqrt(252) * 100
        return round(annual_volatility, 2)

    def _calculate_beta(
        self,
        stock_returns: list[float],
        market_returns: list[float]
    ) -> float:
        """Calculate beta relative to market.
        
        Args:
            stock_returns: Stock daily returns.
            market_returns: Market daily returns.
        
        Returns:
            Beta coefficient.
        """
        # Align lengths
        min_len = min(len(stock_returns), len(market_returns))
        stock_returns = stock_returns[-min_len:]
        market_returns = market_returns[-min_len:]
        
        if min_len < 30:
            return 1.0
        
        # Calculate means
        stock_mean = sum(stock_returns) / len(stock_returns)
        market_mean = sum(market_returns) / len(market_returns)
        
        # Calculate covariance and market variance
        covariance = sum(
            (s - stock_mean) * (m - market_mean)
            for s, m in zip(stock_returns, market_returns)
        ) / len(stock_returns)
        
        market_variance = sum(
            (m - market_mean) ** 2 for m in market_returns
        ) / len(market_returns)
        
        if market_variance == 0:
            return 1.0
        
        beta = covariance / market_variance
        return round(beta, 2)

    def _calculate_var(self, returns: list[float], confidence: float = 0.95) -> float:
        """Calculate Value at Risk at given confidence level.
        
        Args:
            returns: List of daily returns.
            confidence: Confidence level (default 95%).
        
        Returns:
            VaR as percentage (positive number representing potential loss).
        """
        if not returns:
            return 0.0
        
        sorted_returns = sorted(returns)
        index = int((1 - confidence) * len(sorted_returns))
        var = -sorted_returns[index] * 100  # Convert to positive percentage
        return round(var, 2)

    def _calculate_risk_score(
        self,
        volatility: float,
        beta: float,
        var_95: float
    ) -> float:
        """Calculate overall risk score from metrics.
        
        Args:
            volatility: Annualized volatility percentage.
            beta: Beta coefficient.
            var_95: Value at Risk at 95%.
        
        Returns:
            Risk score from 0-100.
        """
        # Volatility component (0-40 points)
        vol_score = min(40, volatility)
        
        # Beta component (0-30 points)
        beta_score = min(30, abs(beta - 1) * 20 + (10 if beta > 1.2 else 0))
        
        # VaR component (0-30 points)
        var_score = min(30, var_95 * 5)
        
        total_score = vol_score + beta_score + var_score
        return round(min(100, max(0, total_score)), 2)

    def _identify_basic_risk_factors(
        self,
        volatility: float,
        beta: float,
        stock_data: Any
    ) -> list[RiskFactor]:
        """Identify basic risk factors from metrics.
        
        Args:
            volatility: Annualized volatility.
            beta: Beta coefficient.
            stock_data: Stock data object.
        
        Returns:
            List of identified risk factors.
        """
        factors = []
        
        # Volatility risk
        if volatility > 40:
            factors.append(RiskFactor(
                factor_type="market",
                description=f"High volatility ({volatility:.1f}% annualized)",
                severity="high"
            ))
        elif volatility > 25:
            factors.append(RiskFactor(
                factor_type="market",
                description=f"Moderate volatility ({volatility:.1f}% annualized)",
                severity="medium"
            ))
        
        # Beta risk
        if beta > 1.5:
            factors.append(RiskFactor(
                factor_type="market",
                description=f"High beta ({beta:.2f}) - significantly more volatile than market",
                severity="high"
            ))
        elif beta > 1.2:
            factors.append(RiskFactor(
                factor_type="market",
                description=f"Elevated beta ({beta:.2f}) - more volatile than market",
                severity="medium"
            ))
        
        return factors

    def _prepare_analysis_data(
        self,
        ticker: str,
        risk_metrics: RiskMetrics,
        data: dict[str, Any]
    ) -> dict[str, Any]:
        """Prepare data dictionary for analysis prompt.
        
        Args:
            ticker: Stock ticker symbol.
            risk_metrics: Calculated risk metrics.
            data: Additional data from orchestrator.
        
        Returns:
            Dictionary with all relevant analysis data.
        """
        analysis_data = {
            "ticker": ticker,
            "volatility": risk_metrics.volatility,
            "beta": risk_metrics.beta,
            "var_95": risk_metrics.var_95,
            "calculated_risk_score": risk_metrics.risk_score,
            "identified_risk_factors": [
                {"type": rf.factor_type, "description": rf.description, "severity": rf.severity}
                for rf in risk_metrics.risk_factors
            ],
        }
        
        # Add stock info if available
        if "stock_data" in data:
            stock_data = data["stock_data"]
            if hasattr(stock_data, "sector"):
                analysis_data["sector"] = stock_data.sector
            if hasattr(stock_data, "industry"):
                analysis_data["industry"] = stock_data.industry
            if hasattr(stock_data, "market_cap"):
                analysis_data["market_cap"] = stock_data.market_cap
        
        return analysis_data

    def _build_user_prompt(self, ticker: str, data: dict[str, Any]) -> str:
        """Build the user prompt for risk assessment.
        
        Args:
            ticker: Stock ticker symbol.
            data: Analysis data dictionary.
        
        Returns:
            Formatted user prompt.
        """
        lines = [
            f"Assess the risk profile for {ticker}:",
            "",
        ]
        
        # Company info
        if data.get("sector"):
            lines.append(f"Sector: {data['sector']}")
        if data.get("industry"):
            lines.append(f"Industry: {data['industry']}")
        if data.get("market_cap"):
            lines.append(f"Market Cap: ${data['market_cap']:,.0f}")
        
        lines.extend([
            "",
            "## Quantitative Risk Metrics",
            "",
            f"- Annualized Volatility: {data['volatility']:.2f}%",
            f"  - Interpretation: {self._interpret_volatility(data['volatility'])}",
            "",
            f"- Beta (vs S&P 500): {data['beta']:.2f}",
            f"  - Interpretation: {self._interpret_beta(data['beta'])}",
            "",
            f"- Value at Risk (95%): {data['var_95']:.2f}%",
            f"  - Meaning: In the worst 5% of scenarios, daily loss could exceed {data['var_95']:.2f}%",
            "",
            f"- Preliminary Risk Score: {data['calculated_risk_score']:.0f}/100",
            "",
        ])
        
        # Pre-identified risk factors
        if data["identified_risk_factors"]:
            lines.append("## Pre-identified Risk Factors")
            lines.append("")
            for rf in data["identified_risk_factors"]:
                lines.append(f"- [{rf['severity'].upper()}] {rf['type'].title()}: {rf['description']}")
            lines.append("")
        
        lines.extend([
            "Based on these metrics and the company's sector/industry,",
            "provide a comprehensive risk assessment.",
            "Identify up to 5 key risk factors categorized by type",
            "(market, sector, company, regulatory, macroeconomic).",
        ])
        
        return "\n".join(lines)

    def _interpret_volatility(self, volatility: float) -> str:
        """Interpret volatility level.
        
        Args:
            volatility: Annualized volatility percentage.
        
        Returns:
            Interpretation string.
        """
        if volatility < 15:
            return "Low volatility - relatively stable stock"
        elif volatility < 25:
            return "Moderate volatility - typical for established companies"
        elif volatility < 40:
            return "Elevated volatility - higher price swings expected"
        return "High volatility - significant price fluctuations"

    def _interpret_beta(self, beta: float) -> str:
        """Interpret beta value.
        
        Args:
            beta: Beta coefficient.
        
        Returns:
            Interpretation string.
        """
        if beta < 0:
            return "Negative beta - tends to move opposite to market"
        elif beta < 0.8:
            return "Low beta - less volatile than market (defensive)"
        elif beta <= 1.2:
            return "Market beta - moves similarly to overall market"
        elif beta <= 1.5:
            return "Elevated beta - more volatile than market"
        return "High beta - significantly more volatile than market (aggressive)"

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
                f"Insufficient price history for {ticker} to calculate risk metrics. "
                "At least 30 days of price data is required for risk assessment."
            ),
            key_factors=["Insufficient data for risk assessment"],
            timestamp=datetime.utcnow(),
        )
