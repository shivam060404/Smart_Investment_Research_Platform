import logging
from datetime import datetime
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel

from app.services.neo4j_service import Neo4jService, get_neo4j_service
from app.services.cache_service import CacheService, get_cache_service
from app.services.mistral_service import MistralService, get_mistral_service

logger = logging.getLogger(__name__)


class QuestionType(str, Enum):
    """Types of questions the RAG system can handle."""
    RECOMMENDATION = "recommendation"  # Why is X recommended?
    ANALYSIS = "analysis"  # What does the analysis say about X?
    COMPARISON = "comparison"  # Compare X and Y
    RISK = "risk"  # What are the risks for X?
    TECHNICAL = "technical"  # Technical indicators for X
    SENTIMENT = "sentiment"  # Market sentiment for X
    GENERAL = "general"  # General questions


class Source(BaseModel):
    """A source citation for RAG responses."""
    type: str  # "analysis", "agent", "metric"
    title: str
    content: str
    timestamp: Optional[str] = None
    relevance_score: float = 1.0


class RAGResponse(BaseModel):
    """Response from the RAG chatbot."""
    answer: str
    sources: list[Source]
    question_type: QuestionType
    ticker: Optional[str] = None
    confidence: float


ROUTER_SYSTEM_PROMPT = """You are a question classifier for a stock analysis platform.
Classify the user's question into one of these categories:
- recommendation: Questions about why a stock is recommended (buy/sell/hold)
- analysis: Questions about overall analysis or specific metrics
- comparison: Questions comparing two or more stocks
- risk: Questions about risks or concerns
- technical: Questions about technical indicators (SMA, RSI, MACD)
- sentiment: Questions about news or market sentiment
- general: Other general questions

Also extract any stock ticker symbols mentioned (e.g., AAPL, TSLA, GOOGL).

Respond in JSON format:
{
    "question_type": "<category>",
    "tickers": ["<TICKER1>", "<TICKER2>"],
    "key_terms": ["<term1>", "<term2>"]
}"""


CHATBOT_SYSTEM_PROMPT = """You are an AI investment research assistant. Answer questions about stock analyses based on the provided context.

Rules:
1. Only use information from the provided context
2. If you don't have enough information, say so clearly
3. Always cite your sources using [Source N] format
4. Be concise but comprehensive
5. For recommendations, explain the reasoning from each agent
6. Include relevant metrics and data points
7. Mention any risks or concerns

Context will include:
- Analysis results from AI agents (Fundamental, Technical, Sentiment, Risk)
- Key metrics and indicators
- Recent analysis timestamps

Format your response clearly with:
- Direct answer to the question
- Supporting evidence with citations
- Any caveats or limitations"""


class RAGService:
    """RAG service for answering questions about stock analyses."""

    def __init__(
        self,
        neo4j_service: Optional[Neo4jService] = None,
        cache_service: Optional[CacheService] = None,
        mistral_service: Optional[MistralService] = None,
    ):
        """Initialize RAG service."""
        self.neo4j = neo4j_service or get_neo4j_service()
        self.cache = cache_service or get_cache_service()
        self.mistral = mistral_service or get_mistral_service()

    async def answer_question(self, question: str) -> RAGResponse:
        """Answer a user question using RAG.
        
        Args:
            question: User's question about stock analysis.
            
        Returns:
            RAGResponse with answer and sources.
        """
        # Step 1: Route/classify the question
        classification = await self._classify_question(question)
        question_type = QuestionType(classification.get("question_type", "general"))
        tickers = classification.get("tickers", [])
        
        # Step 2: Retrieve relevant context (will fetch fresh analysis if needed)
        sources = await self._retrieve_context(question_type, tickers, classification.get("key_terms", []))
        
        if not sources:
            # No sources found even after trying fresh analysis
            if tickers:
                ticker_list = ", ".join(tickers)
                return RAGResponse(
                    answer=f"I tried to analyze {ticker_list} but couldn't retrieve the data. "
                           f"This might be due to an invalid ticker symbol or a temporary service issue. "
                           f"Please verify the ticker symbol is correct and try again.",
                    sources=[],
                    question_type=question_type,
                    ticker=tickers[0] if tickers else None,
                    confidence=0.3
                )
            else:
                return RAGResponse(
                    answer="I'd be happy to help! Please mention a specific stock ticker (like AAPL, MSFT, or TSLA) in your question. "
                           "For example: 'Why is AAPL recommended?' or 'What are the risks for TSLA?'",
                    sources=[],
                    question_type=question_type,
                    ticker=None,
                    confidence=0.3
                )
        
        # Step 3: Generate answer with context
        answer, confidence = await self._generate_answer(question, sources, question_type)
        
        return RAGResponse(
            answer=answer,
            sources=sources,
            question_type=question_type,
            ticker=tickers[0] if tickers else None,
            confidence=confidence
        )

    async def _classify_question(self, question: str) -> dict[str, Any]:
        """Classify the question type and extract entities."""
        try:
            result = await self.mistral.generate_structured_analysis(
                system_prompt=ROUTER_SYSTEM_PROMPT,
                user_prompt=f"Classify this question: {question}"
            )
            return result
        except Exception as e:
            logger.error(f"Failed to classify question: {e}")
            # Fallback: simple keyword matching
            question_lower = question.lower()
            
            q_type = "general"
            if any(w in question_lower for w in ["why", "recommend", "buy", "sell", "hold"]):
                q_type = "recommendation"
            elif any(w in question_lower for w in ["risk", "danger", "concern", "worry"]):
                q_type = "risk"
            elif any(w in question_lower for w in ["technical", "sma", "rsi", "macd", "indicator"]):
                q_type = "technical"
            elif any(w in question_lower for w in ["sentiment", "news", "feeling", "market mood"]):
                q_type = "sentiment"
            elif any(w in question_lower for w in ["compare", "versus", "vs", "better"]):
                q_type = "comparison"
            
            # Extract tickers (simple pattern)
            import re
            tickers = re.findall(r'\b[A-Z]{2,5}\b', question)
            
            return {"question_type": q_type, "tickers": tickers, "key_terms": []}

    async def _retrieve_context(
        self,
        question_type: QuestionType,
        tickers: list[str],
        key_terms: list[str]
    ) -> list[Source]:
        """Retrieve relevant context from cache, Neo4j, or fresh analysis."""
        sources: list[Source] = []
        
        for ticker in tickers:
            # Try cache first
            cached = await self._get_cached_analysis(ticker)
            if cached:
                sources.extend(self._extract_sources_from_analysis(cached, ticker))
            else:
                # No cache - run fresh analysis
                fresh_analysis = await self._run_fresh_analysis(ticker)
                if fresh_analysis:
                    sources.extend(self._extract_sources_from_analysis(fresh_analysis, ticker))
            
            # Also try Neo4j for historical data
            neo4j_data = await self._get_neo4j_analysis(ticker)
            if neo4j_data:
                sources.extend(neo4j_data)
        
        # Sort by relevance and limit
        sources.sort(key=lambda s: s.relevance_score, reverse=True)
        return sources[:10]  # Limit to top 10 sources

    async def _run_fresh_analysis(self, ticker: str) -> Optional[dict]:
        """Run fresh analysis for a ticker when cache is empty."""
        try:
            from app.agents.orchestrator import get_orchestrator
            
            orchestrator = get_orchestrator()
            synthesis, signals, stock_data = await orchestrator.orchestrate(ticker=ticker.upper())
            
            # Build the analysis dict
            analysis = {
                "ticker": ticker.upper(),
                "recommendation": synthesis.recommendation.value if hasattr(synthesis.recommendation, 'value') else str(synthesis.recommendation),
                "confidence_score": synthesis.confidence,
                "synthesis": {
                    "reasoning": synthesis.reasoning,
                    "key_catalysts": synthesis.key_catalysts,
                    "risk_factors": synthesis.risk_factors,
                    "weighted_score": synthesis.weighted_score,
                },
                "signals": [
                    {
                        "agent_name": s.agent_name,
                        "signal_type": s.signal_type.value if hasattr(s.signal_type, 'value') else str(s.signal_type),
                        "score": s.score,
                        "confidence": s.confidence,
                        "reasoning_trace": s.reasoning_trace,
                        "key_factors": s.key_factors,
                    }
                    for s in signals
                ],
                "timestamp": datetime.now().isoformat(),
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Failed to run fresh analysis for {ticker}: {e}")
            return None

    async def _get_cached_analysis(self, ticker: str) -> Optional[dict]:
        """Get cached analysis for a ticker."""
        try:
            from app.models.responses import AnalyzeResponse
            
            ticker_upper = ticker.upper()
            cache_key = f"analysis:{ticker_upper}"
            
            # Try to get raw data from cache
            raw_data = await self.cache.get_cached(cache_key)
            if raw_data:
                if isinstance(raw_data, dict):
                    return raw_data
                if hasattr(raw_data, 'model_dump'):
                    return raw_data.model_dump()
                return raw_data
            
            # Try with AnalyzeResponse model validation
            cached_model = await self.cache.get_cached(cache_key, AnalyzeResponse)
            if cached_model:
                if hasattr(cached_model, 'model_dump'):
                    return cached_model.model_dump()
                return cached_model
                
            return None
        except Exception as e:
            logger.warning(f"Cache retrieval failed for {ticker}: {e}")
            return None

    async def _get_neo4j_analysis(self, ticker: str) -> list[Source]:
        """Get analysis data from Neo4j."""
        sources = []
        try:
            # Query for recent analyses
            history = await self.neo4j.get_analysis_history(ticker, limit=3)
            
            for analysis in history:
                sources.append(Source(
                    type="analysis",
                    title=f"{ticker} Analysis ({analysis.get('date', 'Recent')})",
                    content=f"Recommendation: {analysis.get('recommendation', 'N/A')}, "
                           f"Score: {analysis.get('weighted_score', 'N/A')}, "
                           f"Confidence: {analysis.get('confidence', 'N/A')}%",
                    timestamp=analysis.get('timestamp'),
                    relevance_score=0.9
                ))
        except Exception as e:
            logger.warning(f"Neo4j retrieval failed for {ticker}: {e}")
        
        return sources

    def _extract_sources_from_analysis(self, analysis: dict, ticker: str) -> list[Source]:
        """Extract source citations from analysis data."""
        sources = []
        timestamp = analysis.get("timestamp", datetime.now().isoformat())
        if isinstance(timestamp, datetime):
            timestamp = timestamp.isoformat()
        
        # Main recommendation
        if "recommendation" in analysis:
            rec = analysis['recommendation']
            # Handle enum values
            if hasattr(rec, 'value'):
                rec = rec.value
            sources.append(Source(
                type="recommendation",
                title=f"{ticker} Recommendation",
                content=f"Recommendation: {rec}, "
                       f"Confidence: {analysis.get('confidence_score', 'N/A')}%",
                timestamp=timestamp,
                relevance_score=1.0
            ))
        
        # Synthesis reasoning
        synthesis = analysis.get("synthesis")
        if synthesis:
            # Handle both dict and object
            if hasattr(synthesis, 'model_dump'):
                synthesis = synthesis.model_dump()
            elif not isinstance(synthesis, dict):
                synthesis = {}
            
            if synthesis.get("reasoning"):
                sources.append(Source(
                    type="synthesis",
                    title=f"{ticker} AI Synthesis",
                    content=synthesis["reasoning"],
                    timestamp=timestamp,
                    relevance_score=0.95
                ))
            
            # Key catalysts
            catalysts = synthesis.get("key_catalysts", [])
            if catalysts:
                sources.append(Source(
                    type="catalysts",
                    title=f"{ticker} Key Catalysts",
                    content=", ".join(catalysts),
                    timestamp=timestamp,
                    relevance_score=0.85
                ))
            
            # Risk factors
            risks = synthesis.get("risk_factors", [])
            if risks:
                sources.append(Source(
                    type="risks",
                    title=f"{ticker} Risk Factors",
                    content=", ".join(risks),
                    timestamp=timestamp,
                    relevance_score=0.85
                ))
        
        # Individual agent signals
        for signal in analysis.get("signals", []):
            # Handle both dict and object
            if hasattr(signal, 'model_dump'):
                signal = signal.model_dump()
            
            agent_name = signal.get("agent_name", "Unknown")
            signal_type = signal.get("signal_type", "N/A")
            if hasattr(signal_type, 'value'):
                signal_type = signal_type.value
            
            reasoning = signal.get("reasoning_trace", "N/A")
            if len(reasoning) > 300:
                reasoning = reasoning[:300] + "..."
            
            sources.append(Source(
                type="agent",
                title=f"{ticker} {agent_name} Analysis",
                content=f"Signal: {signal_type}, "
                       f"Score: {signal.get('score', 'N/A')}, "
                       f"Reasoning: {reasoning}",
                timestamp=timestamp,
                relevance_score=0.8
            ))
        
        return sources

    async def _generate_answer(
        self,
        question: str,
        sources: list[Source],
        question_type: QuestionType
    ) -> tuple[str, float]:
        """Generate answer using Mistral with retrieved context."""
        # Build context from sources
        context_parts = []
        for i, source in enumerate(sources, 1):
            context_parts.append(f"[Source {i}] {source.title}:\n{source.content}")
        
        context = "\n\n".join(context_parts)
        
        user_prompt = f"""Context:
{context}

Question: {question}

Please answer the question based on the context provided. Use [Source N] citations."""

        try:
            # Use generate_analysis for natural language response (not JSON)
            result = await self.mistral.generate_analysis(
                system_prompt=CHATBOT_SYSTEM_PROMPT,
                user_prompt=user_prompt
            )
            
            # Result is a string response
            answer = str(result)
            
            # Calculate confidence based on source quality
            confidence = self._calculate_answer_confidence(sources, question_type)
            
            return answer, confidence
            
        except Exception as e:
            logger.error(f"Failed to generate answer: {e}")
            # Fallback: simple summary
            if sources:
                answer = f"Based on the available analysis:\n\n"
                for i, source in enumerate(sources[:3], 1):
                    answer += f"• {source.content} [Source {i}]\n"
                return answer, 0.5
            return "I couldn't generate an answer. Please try rephrasing your question.", 0.2

    def _calculate_answer_confidence(
        self,
        sources: list[Source],
        question_type: QuestionType
    ) -> float:
        """Calculate confidence score based on source quality and relevance.
        
        Args:
            sources: List of sources used for the answer.
            question_type: Type of question being answered.
            
        Returns:
            Confidence score between 0.0 and 1.0.
        """
        if not sources:
            return 0.2
        
        # Base confidence from number of sources
        source_count_score = min(len(sources) / 5, 1.0) * 0.3  # Max 30% from source count
        
        # Average relevance score from sources
        avg_relevance = sum(s.relevance_score for s in sources) / len(sources)
        relevance_score = avg_relevance * 0.4  # Max 40% from relevance
        
        # Source type diversity bonus
        source_types = set(s.type for s in sources)
        diversity_score = min(len(source_types) / 4, 1.0) * 0.2  # Max 20% from diversity
        
        # Recency bonus (if timestamps available)
        recency_score = 0.1  # Base 10%
        for source in sources:
            if source.timestamp:
                try:
                    from datetime import datetime
                    ts = datetime.fromisoformat(source.timestamp.replace('Z', '+00:00'))
                    age_hours = (datetime.now(ts.tzinfo) - ts).total_seconds() / 3600
                    if age_hours < 24:
                        recency_score = 0.1
                        break
                except:
                    pass
        
        # Calculate total confidence
        confidence = source_count_score + relevance_score + diversity_score + recency_score
        
        # Clamp to valid range
        return max(0.25, min(0.95, confidence))


# Singleton instance
_rag_service: Optional[RAGService] = None


def get_rag_service() -> RAGService:
    """Get or create RAG service singleton."""
    global _rag_service
    if _rag_service is None:
        _rag_service = RAGService()
    return _rag_service
