import logging
import time
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Optional

from app.logging_config import AgentLogger
from app.models.signals import AgentSignal, SignalType
from app.services.mistral_service import (
    MistralService,
    MistralServiceError,
    get_mistral_service,
)

logger = logging.getLogger(__name__)


# Global configuration flag for MCP vs direct data access
_use_mcp_adapter: bool = False


def set_use_mcp_adapter(enabled: bool) -> None:
    """Set whether agents should use MCP adapter for data access.

    Args:
        enabled: True to use MCP adapter, False for direct data access.
    """
    global _use_mcp_adapter
    _use_mcp_adapter = enabled
    logger.info(f"MCP adapter usage set to: {enabled}")


def get_use_mcp_adapter() -> bool:
    """Get whether agents should use MCP adapter for data access.

    Returns:
        True if MCP adapter should be used, False otherwise.
    """
    return _use_mcp_adapter


class AgentError(Exception):
    """Base exception for agent errors."""
    pass


class AgentAnalysisError(AgentError):
    """Raised when agent analysis fails."""
    pass


class AgentTimeoutError(AgentError):
    """Raised when agent analysis times out."""
    pass


class BaseAgent(ABC):
    """Abstract base class for all specialist agents.
    
    All specialist agents (Fundamental, Sentiment, Technical, Risk) must
    inherit from this class and implement the required abstract methods.
    
    Attributes:
        name: The agent's identifier name.
        mistral_service: Service for AI-powered analysis generation.
        mcp_adapter: Optional MCP adapter for data access.
        use_mcp: Whether to use MCP adapter for data access.
    """

    def __init__(
        self,
        name: str,
        mistral_service: Optional[MistralService] = None,
        use_mcp: Optional[bool] = None,
    ):
        """Initialize the base agent.
        
        Args:
            name: Unique identifier for this agent.
            mistral_service: Mistral service instance. Uses global instance if not provided.
            use_mcp: Whether to use MCP adapter. If None, uses global setting.
        """
        self.name = name
        self.mistral_service = mistral_service or get_mistral_service()
        self.agent_logger = AgentLogger(name)
        
        # Determine MCP usage: explicit parameter > global setting
        self.use_mcp = use_mcp if use_mcp is not None else get_use_mcp_adapter()
        self._mcp_adapter = None
        
        logger.info(f"Initialized {self.name} agent (use_mcp={self.use_mcp})")

    @property
    def mcp_adapter(self):
        """Get the MCP adapter instance (lazy initialization).
        
        Returns:
            MCPDataAdapter instance or None if MCP is disabled.
        """
        if not self.use_mcp:
            return None
        
        if self._mcp_adapter is None:
            from app.mcp.adapter import get_mcp_adapter
            self._mcp_adapter = get_mcp_adapter()
        
        return self._mcp_adapter

    @abstractmethod
    async def analyze(self, ticker: str, data: dict[str, Any]) -> AgentSignal:
        """Execute analysis and return a signal.
        
        This method must be implemented by all specialist agents to perform
        their specific type of analysis on the provided stock data.
        
        Args:
            ticker: Stock ticker symbol to analyze.
            data: Dictionary containing relevant data for analysis.
                  The structure varies by agent type.
        
        Returns:
            AgentSignal containing the analysis results.
        
        Raises:
            AgentAnalysisError: If analysis fails.
            AgentTimeoutError: If analysis times out.
        """
        pass

    @abstractmethod
    def get_system_prompt(self) -> str:
        """Return the agent-specific system prompt for Mistral.
        
        Each specialist agent defines its own system prompt that instructs
        the LLM on how to perform its specific type of analysis.
        
        Returns:
            System prompt string for the agent's analysis type.
        """
        pass


    def _build_user_prompt(self, ticker: str, data: dict[str, Any]) -> str:
        """Build the user prompt from ticker and data.
        
        Subclasses can override this method to customize prompt formatting.
        
        Args:
            ticker: Stock ticker symbol.
            data: Analysis data dictionary.
        
        Returns:
            Formatted user prompt string.
        """
        data_str = self._format_data_for_prompt(data)
        return f"Analyze the following data for {ticker}:\n\n{data_str}"

    def _format_data_for_prompt(self, data: dict[str, Any]) -> str:
        """Format data dictionary into a readable string for the prompt.
        
        Args:
            data: Dictionary of analysis data.
        
        Returns:
            Formatted string representation of the data.
        """
        lines = []
        for key, value in data.items():
            if value is not None:
                formatted_key = key.replace("_", " ").title()
                if isinstance(value, float):
                    lines.append(f"- {formatted_key}: {value:.4f}")
                elif isinstance(value, list):
                    if value:
                        lines.append(f"- {formatted_key}:")
                        for item in value[:10]:  # Limit list items
                            lines.append(f"  - {item}")
                elif isinstance(value, dict):
                    lines.append(f"- {formatted_key}:")
                    for k, v in list(value.items())[:10]:  # Limit dict items
                        lines.append(f"  - {k}: {v}")
                else:
                    lines.append(f"- {formatted_key}: {value}")
        return "\n".join(lines) if lines else "No data available"

    async def _generate_signal_from_analysis(
        self,
        ticker: str,
        analysis_result: dict[str, Any],
    ) -> AgentSignal:
        """Create an AgentSignal from parsed analysis results.
        
        Args:
            ticker: Stock ticker symbol.
            analysis_result: Parsed JSON response from Mistral containing
                           score, signal_type, confidence, reasoning, and key_factors.
        
        Returns:
            AgentSignal with the analysis results.
        
        Raises:
            AgentAnalysisError: If required fields are missing.
        """
        try:
            # Extract and validate score
            score = float(analysis_result.get("score", 50))
            score = max(0, min(100, score))  # Clamp to 0-100
            
            # Determine signal type from score or explicit value
            signal_type_str = analysis_result.get("signal_type", "").lower()
            if signal_type_str in ["bullish", "bearish", "neutral"]:
                signal_type = SignalType(signal_type_str)
            else:
                signal_type = self._score_to_signal_type(score)
            
            # Extract confidence
            confidence = float(analysis_result.get("confidence", 70))
            confidence = max(0, min(100, confidence))
            
            # Extract reasoning and key factors
            reasoning = analysis_result.get("reasoning", "Analysis completed.")
            key_factors = analysis_result.get("key_factors", [])
            if isinstance(key_factors, str):
                key_factors = [key_factors]
            
            return AgentSignal(
                agent_name=self.name,
                signal_type=signal_type,
                score=score,
                confidence=confidence,
                reasoning_trace=reasoning,
                key_factors=key_factors,
                timestamp=datetime.utcnow(),
            )
        except Exception as e:
            logger.error(f"{self.name}: Failed to create signal: {e}")
            raise AgentAnalysisError(f"Failed to create signal from analysis: {e}")

    def _score_to_signal_type(self, score: float) -> SignalType:
        """Convert a numeric score to a signal type.
        
        Args:
            score: Numeric score from 0-100.
        
        Returns:
            SignalType based on score thresholds.
        """
        if score >= 60:
            return SignalType.BULLISH
        elif score <= 40:
            return SignalType.BEARISH
        return SignalType.NEUTRAL

    def _create_error_signal(self, error_message: str) -> AgentSignal:
        """Create a neutral signal indicating an error occurred.
        
        Used for graceful degradation when analysis fails.
        
        Args:
            error_message: Description of the error.
        
        Returns:
            AgentSignal with neutral signal and error information.
        """
        return AgentSignal(
            agent_name=self.name,
            signal_type=SignalType.NEUTRAL,
            score=50.0,
            confidence=0.0,
            reasoning_trace=f"Analysis failed: {error_message}",
            key_factors=["Error during analysis"],
            timestamp=datetime.utcnow(),
        )

    async def safe_analyze(self, ticker: str, data: dict[str, Any]) -> AgentSignal:
        """Execute analysis with error handling, returning error signal on failure.
        
        This method wraps analyze() to provide graceful degradation.
        If analysis fails, it returns a neutral signal with error information
        instead of raising an exception.
        
        Args:
            ticker: Stock ticker symbol to analyze.
            data: Dictionary containing relevant data for analysis.
        
        Returns:
            AgentSignal with results or error information.
        """
        start_time = time.perf_counter()
        self.agent_logger.log_analysis_start(ticker)
        
        try:
            signal = await self.analyze(ticker, data)
            duration_ms = (time.perf_counter() - start_time) * 1000
            
            self.agent_logger.log_analysis_complete(
                ticker=ticker,
                score=signal.score,
                signal_type=signal.signal_type.value,
                confidence=signal.confidence,
                duration_ms=duration_ms,
            )
            return signal
            
        except AgentError as e:
            duration_ms = (time.perf_counter() - start_time) * 1000
            self.agent_logger.log_analysis_error(ticker, e, duration_ms)
            return self._create_error_signal(str(e))
        except MistralServiceError as e:
            duration_ms = (time.perf_counter() - start_time) * 1000
            self.agent_logger.log_analysis_error(ticker, e, duration_ms)
            return self._create_error_signal(f"AI service error: {e}")
        except Exception as e:
            duration_ms = (time.perf_counter() - start_time) * 1000
            self.agent_logger.log_analysis_error(ticker, e, duration_ms)
            return self._create_error_signal(f"Unexpected error: {e}")
