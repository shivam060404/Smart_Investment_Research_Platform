import asyncio
import logging
import time
from datetime import datetime
from typing import Any, Optional

from app.agents.base import AgentError
from app.agents.fundamental import FundamentalAgent
from app.agents.sentiment import SentimentAgent
from app.agents.technical import TechnicalAgent
from app.agents.risk import RiskAgent
from app.logging_config import AgentLogger
from app.models.signals import (
    AgentSignal,
    Recommendation,
    SignalType,
    SynthesisResult,
)
from app.models.stock import StockData
from app.services.data_fetcher import (
    DataFetcher,
    get_data_fetcher,
    DataFetcherError,
)
from app.services.mistral_service import (
    MistralService,
    MistralServiceError,
    get_mistral_service,
)
from app.services.neo4j_service import Neo4jService, get_neo4j_service

logger = logging.getLogger(__name__)
agent_logger = AgentLogger("Orchestrator")


class OrchestratorError(Exception):
    """Base exception for orchestrator errors."""
    pass


class InsufficientSignalsError(OrchestratorError):
    """Raised when not enough agent signals are available."""
    pass


class AgentWeights:
    """Configuration for agent signal weighting."""

    DEFAULT_WEIGHTS = {
        "Fundamental": 0.30,
        "Technical": 0.25,
        "Sentiment": 0.25,
        "Risk": 0.20,
    }

    def __init__(
        self,
        fundamental: float = 0.30,
        technical: float = 0.25,
        sentiment: float = 0.25,
        risk: float = 0.20,
    ):
        """Initialize agent weights.
        
        Args:
            fundamental: Weight for Fundamental agent (default 30%).
            technical: Weight for Technical agent (default 25%).
            sentiment: Weight for Sentiment agent (default 25%).
            risk: Weight for Risk agent (default 20%).
        """
        self.weights = {
            "Fundamental": fundamental,
            "Technical": technical,
            "Sentiment": sentiment,
            "Risk": risk,
        }
        self._normalize()

    def _normalize(self) -> None:
        """Normalize weights to sum to 1.0."""
        total = sum(self.weights.values())
        if total > 0:
            self.weights = {k: v / total for k, v in self.weights.items()}

    def get(self, agent_name: str) -> float:
        """Get weight for an agent.
        
        Args:
            agent_name: Name of the agent.
            
        Returns:
            Weight value (0-1).
        """
        return self.weights.get(agent_name, 0.0)

    def adjust_for_missing(self, available_agents: list[str]) -> "AgentWeights":
        """Create adjusted weights when some agents are missing.
        
        Args:
            available_agents: List of agent names that provided signals.
            
        Returns:
            New AgentWeights with redistributed weights.
        """
        adjusted = AgentWeights()
        adjusted.weights = {
            name: weight
            for name, weight in self.weights.items()
            if name in available_agents
        }
        adjusted._normalize()
        return adjusted


SYNTHESIS_SYSTEM_PROMPT = """You are a Portfolio Manager AI responsible for synthesizing investment signals from multiple specialist analysts.

Your task is to combine the analyses from Fundamental, Technical, Sentiment, and Risk agents into a coherent investment recommendation.

## Input Signals

You will receive signals from up to 4 specialist agents:
1. **Fundamental Agent**: Analyzes financial metrics (P/E, ROE, debt levels, growth)
2. **Technical Agent**: Analyzes price patterns and indicators (SMA, RSI, MACD)
3. **Sentiment Agent**: Analyzes news and market sentiment
4. **Risk Agent**: Assesses volatility, beta, and risk factors

Each signal includes:
- Score (0-100): Higher = more bullish (except Risk where higher = more risky)
- Confidence (0-100): How certain the agent is
- Signal Type: bullish, bearish, or neutral
- Key Factors: Important points from the analysis

## Your Task

1. Weigh each agent's signal according to the provided weights
2. Consider the confidence levels when synthesizing
3. For Risk signals, remember that HIGH risk score = BEARISH implication
4. Generate a final BUY/SELL/HOLD recommendation
5. Calculate a UNIQUE confidence score based on the analysis
6. Provide a synthesized reasoning that explains how you combined the signals

## Confidence Calculation Guidelines

Your confidence score MUST be calculated based on these factors - DO NOT default to 75%:

1. **Signal Agreement** (40% weight):
   - All 4 agents agree: +40 points
   - 3 agents agree: +30 points
   - 2 agents agree: +20 points
   - No agreement: +10 points

2. **Average Agent Confidence** (30% weight):
   - Take the weighted average of all agent confidence scores
   - Multiply by 0.3

3. **Score Strength** (20% weight):
   - If weighted score is far from 50 (strong signal): +20 points
   - If weighted score is near 50 (weak signal): +5 points

4. **Risk Adjustment** (10% weight):
   - Low risk (score < 40): +10 points
   - Moderate risk (40-60): +5 points
   - High risk (> 60): +0 points

The final confidence should vary significantly based on the actual data:
- Strong unanimous bullish/bearish with low risk: 85-95%
- Majority agreement with moderate risk: 65-80%
- Mixed signals or high uncertainty: 40-60%
- Conflicting signals with high risk: 25-40%

## Output Format

Respond with a JSON object:
{
    "recommendation": "<BUY|SELL|HOLD>",
    "confidence": <number 0-100, calculated using the guidelines above>,
    "reasoning": "<synthesized explanation combining all agent insights>",
    "key_catalysts": ["<positive factor 1>", "<positive factor 2>", ...],
    "risk_factors": ["<risk 1>", "<risk 2>", ...]
}

## Decision Guidelines

- **BUY**: Weighted score >= 60, majority bullish signals, acceptable risk
- **SELL**: Weighted score <= 40, majority bearish signals, or high risk concerns
- **HOLD**: Mixed signals, weighted score 40-60, or high uncertainty

Be decisive but acknowledge uncertainty. Your reasoning should clearly show how each agent's analysis contributed to the final decision. The confidence score MUST reflect the actual certainty of your recommendation based on the input data."""



class OrchestratorAgent:
    """Portfolio Manager orchestrator for coordinating specialist agents.
    
    This agent coordinates the execution of all specialist agents in parallel,
    aggregates their signals using configurable weights, and generates a
    final investment recommendation with synthesized reasoning.
    """

    AGENT_TIMEOUT_SECONDS = 30

    def __init__(
        self,
        mistral_service: Optional[MistralService] = None,
        data_fetcher: Optional[DataFetcher] = None,
        neo4j_service: Optional[Neo4jService] = None,
        weights: Optional[AgentWeights] = None,
    ):
        """Initialize the Orchestrator Agent.
        
        Args:
            mistral_service: Mistral service for AI synthesis.
            data_fetcher: Data fetcher for stock data.
            neo4j_service: Neo4j service for storing results.
            weights: Custom agent weights. Uses defaults if not provided.
        """
        self.mistral_service = mistral_service or get_mistral_service()
        self.data_fetcher = data_fetcher or get_data_fetcher()
        self.neo4j_service = neo4j_service or get_neo4j_service()
        self.weights = weights or AgentWeights()
        
        # Initialize specialist agents
        self.fundamental_agent = FundamentalAgent(
            mistral_service=self.mistral_service,
            data_fetcher=self.data_fetcher,
        )
        self.sentiment_agent = SentimentAgent(
            mistral_service=self.mistral_service,
            data_fetcher=self.data_fetcher,
        )
        self.technical_agent = TechnicalAgent(
            mistral_service=self.mistral_service,
            data_fetcher=self.data_fetcher,
        )
        self.risk_agent = RiskAgent(
            mistral_service=self.mistral_service,
            data_fetcher=self.data_fetcher,
        )
        
        logger.info("Initialized OrchestratorAgent with weights: %s", self.weights.weights)

    async def orchestrate(
        self,
        ticker: str,
        weights: Optional[AgentWeights] = None,
    ) -> tuple[SynthesisResult, list[AgentSignal], Optional[StockData]]:
        """Orchestrate full analysis for a stock ticker.
        
        Coordinates all specialist agents in parallel, handles partial failures,
        synthesizes signals, and stores results in Neo4j.
        
        Args:
            ticker: Stock ticker symbol to analyze.
            weights: Optional custom weights for this analysis.
            
        Returns:
            Tuple of (SynthesisResult, list of AgentSignals, StockData).
            
        Raises:
            OrchestratorError: If orchestration fails completely.
            InsufficientSignalsError: If no valid signals are received.
        """
        start_time = time.perf_counter()
        effective_weights = weights or self.weights
        agent_logger.log_analysis_start(ticker, weights=effective_weights.weights if weights else "default")
        
        try:
            # Step 1: Fetch base stock data
            stock_data = await self._fetch_stock_data(ticker)
            
            # Step 2: Run all agents in parallel
            signals = await self._run_agents_parallel(ticker, stock_data)
            
            # Step 3: Filter valid signals and handle partial failures
            valid_signals, failed_agents = self._filter_valid_signals(signals)
            
            if not valid_signals:
                raise InsufficientSignalsError(
                    f"No valid signals received for {ticker}. All agents failed."
                )
            
            if failed_agents:
                logger.warning(
                    "Partial failure for %s. Failed agents: %s",
                    ticker,
                    failed_agents,
                    extra={
                        "event": "partial_agent_failure",
                        "ticker": ticker,
                        "failed_agents": failed_agents,
                        "successful_agents": [s.agent_name for s in valid_signals],
                    }
                )
            
            # Step 4: Adjust weights for available signals
            available_agents = [s.agent_name for s in valid_signals]
            adjusted_weights = effective_weights.adjust_for_missing(available_agents)
            
            # Step 5: Synthesize signals into final recommendation
            synthesis = await self._synthesize_signals(
                ticker,
                valid_signals,
                adjusted_weights,
                failed_agents,
            )
            
            # Step 6: Store results in Neo4j
            await self._store_results(ticker, synthesis, valid_signals, stock_data)
            
            duration_ms = (time.perf_counter() - start_time) * 1000
            agent_logger.log_analysis_complete(
                ticker=ticker,
                score=synthesis.weighted_score,
                signal_type=synthesis.recommendation.value.lower(),
                confidence=synthesis.confidence,
                duration_ms=duration_ms,
                failed_agents=failed_agents,
                successful_agents=available_agents,
            )
            
            return synthesis, valid_signals, stock_data
            
        except Exception as e:
            duration_ms = (time.perf_counter() - start_time) * 1000
            agent_logger.log_analysis_error(ticker, e, duration_ms)
            raise

    async def _fetch_stock_data(self, ticker: str) -> Optional[StockData]:
        """Fetch base stock data for the ticker.
        
        Args:
            ticker: Stock ticker symbol.
            
        Returns:
            StockData or None if fetch fails.
        """
        try:
            return await self.data_fetcher.get_stock_data_with_fallback(ticker)
        except DataFetcherError as e:
            logger.warning("Failed to fetch stock data for %s: %s", ticker, e)
            return None

    async def _run_agents_parallel(
        self,
        ticker: str,
        stock_data: Optional[StockData],
    ) -> list[AgentSignal | Exception]:
        """Run all specialist agents in parallel.
        
        Uses asyncio.gather with return_exceptions=True to handle
        individual agent failures gracefully.
        
        Args:
            ticker: Stock ticker symbol.
            stock_data: Pre-fetched stock data.
            
        Returns:
            List of AgentSignals or Exceptions for failed agents.
        """
        # Prepare shared data for agents
        shared_data: dict[str, Any] = {}
        if stock_data:
            shared_data["stock_data"] = stock_data
        
        # Create agent tasks with timeout
        async def run_agent_with_timeout(agent, name: str) -> AgentSignal:
            """Run agent with timeout wrapper."""
            try:
                return await asyncio.wait_for(
                    agent.safe_analyze(ticker, shared_data),
                    timeout=self.AGENT_TIMEOUT_SECONDS
                )
            except asyncio.TimeoutError:
                logger.error("%s agent timed out for %s", name, ticker)
                raise AgentError(f"{name} agent timed out after {self.AGENT_TIMEOUT_SECONDS}s")
        
        # Run all agents in parallel
        results = await asyncio.gather(
            run_agent_with_timeout(self.fundamental_agent, "Fundamental"),
            run_agent_with_timeout(self.sentiment_agent, "Sentiment"),
            run_agent_with_timeout(self.technical_agent, "Technical"),
            run_agent_with_timeout(self.risk_agent, "Risk"),
            return_exceptions=True
        )
        
        return results

    def _filter_valid_signals(
        self,
        results: list[AgentSignal | Exception],
    ) -> tuple[list[AgentSignal], list[str]]:
        """Filter valid signals from agent results.
        
        Args:
            results: List of AgentSignals or Exceptions.
            
        Returns:
            Tuple of (valid_signals, failed_agent_names).
        """
        valid_signals = []
        failed_agents = []
        agent_names = ["Fundamental", "Sentiment", "Technical", "Risk"]
        
        for i, result in enumerate(results):
            agent_name = agent_names[i]
            
            if isinstance(result, Exception):
                logger.error("%s agent failed: %s", agent_name, result)
                failed_agents.append(agent_name)
            elif isinstance(result, AgentSignal):
                # Check if signal indicates an error (confidence = 0)
                if result.confidence == 0 and "failed" in result.reasoning_trace.lower():
                    logger.warning("%s agent returned error signal", agent_name)
                    failed_agents.append(agent_name)
                else:
                    valid_signals.append(result)
            else:
                logger.error("Unexpected result type from %s: %s", agent_name, type(result))
                failed_agents.append(agent_name)
        
        return valid_signals, failed_agents

    async def _synthesize_signals(
        self,
        ticker: str,
        signals: list[AgentSignal],
        weights: AgentWeights,
        failed_agents: list[str],
    ) -> SynthesisResult:
        """Synthesize agent signals into final recommendation.
        
        Args:
            ticker: Stock ticker symbol.
            signals: List of valid agent signals.
            weights: Adjusted weights for available agents.
            failed_agents: List of agents that failed.
            
        Returns:
            SynthesisResult with final recommendation.
        """
        # Calculate weighted score
        weighted_score = self._calculate_weighted_score(signals, weights)
        
        # Calculate agent contributions
        agent_contributions = {
            signal.agent_name: weights.get(signal.agent_name) * 100
            for signal in signals
        }
        
        # Extract key catalysts and risk factors
        key_catalysts = self._extract_catalysts(signals)
        risk_factors = self._extract_risk_factors(signals)
        
        # Generate synthesized reasoning via Mistral
        try:
            synthesis_result = await self._generate_synthesis_via_mistral(
                ticker,
                signals,
                weights,
                weighted_score,
                failed_agents,
            )
            
            return SynthesisResult(
                recommendation=Recommendation(synthesis_result.get("recommendation", "HOLD")),
                confidence=float(synthesis_result.get("confidence", 50)),
                weighted_score=weighted_score,
                reasoning=synthesis_result.get("reasoning", ""),
                agent_contributions=agent_contributions,
                key_catalysts=synthesis_result.get("key_catalysts", key_catalysts),
                risk_factors=synthesis_result.get("risk_factors", risk_factors),
            )
        except MistralServiceError as e:
            logger.warning("Mistral synthesis failed, using rule-based fallback: %s", e)
            return self._rule_based_synthesis(
                signals,
                weights,
                weighted_score,
                agent_contributions,
                key_catalysts,
                risk_factors,
                failed_agents,
            )

    def _calculate_weighted_score(
        self,
        signals: list[AgentSignal],
        weights: AgentWeights,
    ) -> float:
        """Calculate weighted aggregate score from signals.
        
        For the Risk agent, the score is inverted since high risk = bearish.
        
        Args:
            signals: List of agent signals.
            weights: Agent weights.
            
        Returns:
            Weighted score (0-100).
        """
        total_score = 0.0
        
        for signal in signals:
            weight = weights.get(signal.agent_name)
            score = signal.score
            
            # Invert risk score (high risk = low investment score)
            if signal.agent_name == "Risk":
                score = 100 - score
            
            # Weight by confidence as well
            confidence_factor = signal.confidence / 100
            adjusted_score = score * confidence_factor
            
            total_score += adjusted_score * weight
        
        return round(total_score, 2)

    def _extract_catalysts(self, signals: list[AgentSignal]) -> list[str]:
        """Extract positive catalysts from signals.
        
        Args:
            signals: List of agent signals.
            
        Returns:
            List of key catalysts.
        """
        catalysts = []
        
        for signal in signals:
            if signal.agent_name in ["Fundamental", "Sentiment", "Technical"]:
                if signal.signal_type == SignalType.BULLISH:
                    catalysts.extend(signal.key_factors[:2])
        
        return catalysts[:5]  # Limit to 5 catalysts

    def _extract_risk_factors(self, signals: list[AgentSignal]) -> list[str]:
        """Extract risk factors from signals.
        
        Args:
            signals: List of agent signals.
            
        Returns:
            List of risk factors.
        """
        risk_factors = []
        
        for signal in signals:
            if signal.agent_name == "Risk":
                risk_factors.extend(signal.key_factors)
            elif signal.signal_type == SignalType.BEARISH:
                risk_factors.extend(signal.key_factors[:1])
        
        return risk_factors[:5]  # Limit to 5 risk factors

    async def _generate_synthesis_via_mistral(
        self,
        ticker: str,
        signals: list[AgentSignal],
        weights: AgentWeights,
        weighted_score: float,
        failed_agents: list[str],
    ) -> dict[str, Any]:
        """Generate synthesized reasoning using Mistral.
        
        Args:
            ticker: Stock ticker symbol.
            signals: List of agent signals.
            weights: Agent weights.
            weighted_score: Pre-calculated weighted score.
            failed_agents: List of failed agents.
            
        Returns:
            Parsed synthesis result from Mistral.
        """
        user_prompt = self._build_synthesis_prompt(
            ticker,
            signals,
            weights,
            weighted_score,
            failed_agents,
        )
        
        return await self.mistral_service.generate_structured_analysis(
            system_prompt=SYNTHESIS_SYSTEM_PROMPT,
            user_prompt=user_prompt,
        )

    def _build_synthesis_prompt(
        self,
        ticker: str,
        signals: list[AgentSignal],
        weights: AgentWeights,
        weighted_score: float,
        failed_agents: list[str],
    ) -> str:
        """Build the user prompt for synthesis.
        
        Args:
            ticker: Stock ticker symbol.
            signals: List of agent signals.
            weights: Agent weights.
            weighted_score: Pre-calculated weighted score.
            failed_agents: List of failed agents.
            
        Returns:
            Formatted user prompt.
        """
        lines = [
            f"Synthesize the following agent signals for {ticker}:",
            "",
            f"Pre-calculated Weighted Score: {weighted_score:.2f}/100",
            "",
        ]
        
        if failed_agents:
            lines.append(f"⚠️ Missing Signals: {', '.join(failed_agents)} agents failed to respond")
            lines.append("")
        
        lines.append("## Agent Signals")
        lines.append("")
        
        for signal in signals:
            weight_pct = weights.get(signal.agent_name) * 100
            lines.extend([
                f"### {signal.agent_name} Agent (Weight: {weight_pct:.0f}%)",
                f"- Signal: {signal.signal_type.value.upper()}",
                f"- Score: {signal.score:.1f}/100",
                f"- Confidence: {signal.confidence:.1f}%",
                f"- Key Factors:",
            ])
            for factor in signal.key_factors[:3]:
                lines.append(f"  - {factor}")
            lines.append(f"- Reasoning: {signal.reasoning_trace[:300]}...")
            lines.append("")
        
        lines.extend([
            "Based on these signals and their weights, provide your synthesized recommendation.",
            "Remember: For Risk signals, HIGH score means HIGH risk (bearish implication).",
        ])
        
        return "\n".join(lines)

    def _rule_based_synthesis(
        self,
        signals: list[AgentSignal],
        weights: AgentWeights,
        weighted_score: float,
        agent_contributions: dict[str, float],
        key_catalysts: list[str],
        risk_factors: list[str],
        failed_agents: list[str],
    ) -> SynthesisResult:
        """Generate synthesis using rule-based logic as fallback.
        
        Used when Mistral is unavailable.
        
        Args:
            signals: List of agent signals.
            weights: Agent weights.
            weighted_score: Pre-calculated weighted score.
            agent_contributions: Agent contribution percentages.
            key_catalysts: Extracted catalysts.
            risk_factors: Extracted risk factors.
            failed_agents: List of failed agents.
            
        Returns:
            SynthesisResult with rule-based recommendation.
        """
        # Determine recommendation based on weighted score
        if weighted_score >= 60:
            recommendation = Recommendation.BUY
        elif weighted_score <= 40:
            recommendation = Recommendation.SELL
        else:
            recommendation = Recommendation.HOLD
        
        # Calculate confidence based on signal agreement and confidence levels
        avg_confidence = sum(s.confidence for s in signals) / len(signals)
        
        # Reduce confidence if agents failed
        confidence_penalty = len(failed_agents) * 10
        confidence = max(20, avg_confidence - confidence_penalty)
        
        # Check signal agreement
        bullish_count = sum(1 for s in signals if s.signal_type == SignalType.BULLISH)
        bearish_count = sum(1 for s in signals if s.signal_type == SignalType.BEARISH)
        
        if bullish_count == len(signals) or bearish_count == len(signals):
            confidence = min(95, confidence + 10)  # Boost for unanimous signals
        
        # Build reasoning
        reasoning_parts = []
        
        for signal in signals:
            direction = signal.signal_type.value
            reasoning_parts.append(
                f"{signal.agent_name} analysis is {direction} "
                f"(score: {signal.score:.0f}, confidence: {signal.confidence:.0f}%)"
            )
        
        if failed_agents:
            reasoning_parts.append(
                f"Note: {', '.join(failed_agents)} agent(s) failed to provide signals"
            )
        
        reasoning = (
            f"Based on weighted analysis (score: {weighted_score:.1f}/100), "
            f"the recommendation is {recommendation.value}. "
            + ". ".join(reasoning_parts) + "."
        )
        
        return SynthesisResult(
            recommendation=recommendation,
            confidence=round(confidence, 2),
            weighted_score=weighted_score,
            reasoning=reasoning,
            agent_contributions=agent_contributions,
            key_catalysts=key_catalysts,
            risk_factors=risk_factors,
        )

    async def _store_results(
        self,
        ticker: str,
        synthesis: SynthesisResult,
        signals: list[AgentSignal],
        stock_data: Optional[StockData],
    ) -> None:
        """Store analysis results in Neo4j.
        
        Args:
            ticker: Stock ticker symbol.
            synthesis: Synthesis result.
            signals: List of agent signals.
            stock_data: Stock data.
        """
        try:
            await self.neo4j_service.store_analysis(
                ticker=ticker,
                synthesis=synthesis,
                signals=signals,
                stock_data=stock_data,
            )
        except Exception as e:
            # Log but don't fail the orchestration
            logger.error("Failed to store results in Neo4j for %s: %s", ticker, e)


# Singleton instance
_orchestrator: Optional[OrchestratorAgent] = None


def get_orchestrator() -> OrchestratorAgent:
    """Get or create OrchestratorAgent singleton.
    
    Returns:
        OrchestratorAgent instance.
    """
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = OrchestratorAgent()
    return _orchestrator
