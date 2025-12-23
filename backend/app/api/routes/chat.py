import logging
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.services.rag_service import RAGService, get_rag_service, RAGResponse, Source

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/chat", tags=["Chat"])


class ChatMessage(BaseModel):
    """A chat message."""
    role: str = Field(..., description="Message role: 'user' or 'assistant'")
    content: str = Field(..., description="Message content")
    sources: Optional[list[dict]] = Field(default=None, description="Source citations")
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())


class ChatRequest(BaseModel):
    """Request body for chat endpoint."""
    message: str = Field(..., min_length=1, max_length=1000, description="User's question")
    conversation_id: Optional[str] = Field(default=None, description="Optional conversation ID for context")


class ChatResponse(BaseModel):
    """Response from chat endpoint."""
    message: ChatMessage
    question_type: str
    ticker: Optional[str]
    confidence: float
    sources: list[dict]


class SourceResponse(BaseModel):
    """Source citation in response."""
    type: str
    title: str
    content: str
    timestamp: Optional[str]


@router.post("", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    """Send a message to the RAG chatbot.
    
    The chatbot can answer questions about stock analyses, recommendations,
    risks, technical indicators, and market sentiment.
    
    Example questions:
    - "Why is AAPL recommended?"
    - "What are the risks for TSLA?"
    - "Explain the technical analysis for GOOGL"
    - "What does the sentiment say about MSFT?"
    
    Args:
        request: Chat request with user message.
        
    Returns:
        ChatResponse with AI-generated answer and source citations.
    """
    logger.info(f"Chat request: {request.message[:100]}...")
    
    try:
        rag_service = get_rag_service()
        response: RAGResponse = await rag_service.answer_question(request.message)
        
        # Convert sources to dict format
        sources_dict = [
            {
                "type": s.type,
                "title": s.title,
                "content": s.content,
                "timestamp": s.timestamp
            }
            for s in response.sources
        ]
        
        return ChatResponse(
            message=ChatMessage(
                role="assistant",
                content=response.answer,
                sources=sources_dict
            ),
            question_type=response.question_type.value,
            ticker=response.ticker,
            confidence=response.confidence,
            sources=sources_dict
        )
        
    except Exception as e:
        logger.error(f"Chat error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={
                "error_code": "CHAT_ERROR",
                "message": "Failed to process your question. Please try again."
            }
        )


@router.get("/suggestions")
async def get_suggestions(ticker: Optional[str] = None) -> dict:
    """Get suggested questions for the chatbot.
    
    Args:
        ticker: Optional ticker to customize suggestions.
        
    Returns:
        List of suggested questions.
    """
    base_suggestions = [
        "What stocks have you analyzed recently?",
        "How does the AI analysis work?",
        "What factors do you consider for recommendations?",
    ]
    
    if ticker:
        ticker_suggestions = [
            f"Why is {ticker} recommended?",
            f"What are the risks for {ticker}?",
            f"Explain the technical analysis for {ticker}",
            f"What does the sentiment say about {ticker}?",
            f"What are the key catalysts for {ticker}?",
        ]
        return {"suggestions": ticker_suggestions + base_suggestions}
    
    return {"suggestions": base_suggestions}
