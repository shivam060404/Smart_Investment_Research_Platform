import logging
from datetime import datetime
from typing import Optional

import yfinance as yf
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel

from app.agents.orchestrator import (
    OrchestratorAgent,
    OrchestratorError,
    InsufficientSignalsError,
    AgentWeights,
    get_orchestrator,
)
from app.models.requests import AgentWeights as RequestAgentWeights
from app.models.responses import AnalyzeResponse, ErrorCode, ErrorResponse
from app.services.cache_service import CacheService, get_cache_service
from app.utils.validators import validate_ticker_http

logger = logging.getLogger(__name__)

router = APIRouter()

# List of major exchanges supported
SUPPORTED_EXCHANGES = ["NYSE", "NASDAQ", "AMEX", "LSE", "TSE", "HKEX"]


class StockPriceResponse(BaseModel):
    """Response model for stock price endpoint."""
    ticker: str
    price: float
    currency: str = "USD"
    timestamp: datetime


@router.get(
    "/stock/{ticker}/price",
    response_model=StockPriceResponse,
    summary="Get current stock price",
    description="Fetches the current market price for a stock ticker.",
)
async def get_stock_price(ticker: str) -> StockPriceResponse:
    """Get current stock price from Yahoo Finance.
    
    Args:
        ticker: Stock ticker symbol.
        
    Returns:
        StockPriceResponse with current price.
    """
    normalized_ticker = validate_ticker_http(ticker)
    
    try:
        stock = yf.Ticker(normalized_ticker)
        info = stock.info
        
        # Try different price fields
        price = (
            info.get("regularMarketPrice") or
            info.get("currentPrice") or
            info.get("previousClose")
        )
        
        if price is None:
            # Fallback to history
            hist = stock.history(period="1d")
            if not hist.empty:
                price = float(hist["Close"].iloc[-1])
        
        if price is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={"error_code": "PRICE_NOT_FOUND", "message": f"Could not fetch price for {normalized_ticker}"}
            )
        
        return StockPriceResponse(
            ticker=normalized_ticker,
            price=float(price),
            currency=info.get("currency", "USD"),
            timestamp=datetime.utcnow(),
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to fetch price for {normalized_ticker}: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"error_code": "EXTERNAL_API_ERROR", "message": f"Failed to fetch price: {str(e)}"}
        )


@router.post(
    "/analyze/{ticker}",
    response_model=AnalyzeResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid ticker symbol"},
        503: {"model": ErrorResponse, "description": "Service unavailable"},
    },
    summary="Analyze a stock",
    description="Initiates parallel analysis across all specialist agents and returns a comprehensive investment recommendation.",
)
async def analyze_stock(
    ticker: str,
    weights: Optional[RequestAgentWeights] = None,
) -> AnalyzeResponse:
    """Analyze a stock and generate investment recommendation.
    
    This endpoint coordinates four specialist agents (Fundamental, Technical,
    Sentiment, and Risk) to analyze the given stock ticker and produce a
    synthesized investment recommendation with full explainability.
    
    Args:
        ticker: Stock ticker symbol to analyze (e.g., AAPL, GOOGL).
        weights: Optional custom weights for agent signal aggregation.
        
    Returns:
        AnalyzeResponse: Analysis results including recommendation,
            confidence score, individual agent signals, and synthesis.
            
    Raises:
        HTTPException: If ticker is invalid or analysis fails.
    """
    # Validate ticker using centralized validator
    normalized_ticker = validate_ticker_http(ticker)
    logger.info(f"Received analysis request for ticker: {normalized_ticker}")
    
    # Get services
    cache_service = get_cache_service()
    
    # Generate cache key
    weights_dict = weights.model_dump() if weights else None
    cache_key = cache_service.generate_analysis_key(normalized_ticker, weights_dict)
    
    # Check cache first
    cached_result = await cache_service.get_cached(cache_key, AnalyzeResponse)
    if cached_result:
        logger.info(f"Cache hit for {normalized_ticker}")
        cached_result.cached = True
        return cached_result
    
    logger.info(f"Cache miss for {normalized_ticker}, running analysis")
    
    # Convert request weights to orchestrator weights
    orchestrator_weights = None
    if weights:
        orchestrator_weights = AgentWeights(
            fundamental=weights.fundamental,
            technical=weights.technical,
            sentiment=weights.sentiment,
            risk=weights.risk,
        )
    
    # Run orchestration
    try:
        orchestrator = get_orchestrator()
        synthesis, signals, stock_data = await orchestrator.orchestrate(
            ticker=normalized_ticker,
            weights=orchestrator_weights,
        )
        
        # Build response
        response = AnalyzeResponse(
            ticker=normalized_ticker,
            recommendation=synthesis.recommendation,
            confidence_score=synthesis.confidence,
            signals=signals,
            synthesis=synthesis,
            timestamp=datetime.utcnow(),
            cached=False,
        )
        
        # Cache the result
        await cache_service.set_cached(cache_key, response)
        logger.info(f"Analysis complete for {normalized_ticker}: {synthesis.recommendation.value}")
        
        return response
        
    except InsufficientSignalsError as e:
        logger.error(f"Insufficient signals for {normalized_ticker}: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "error_code": ErrorCode.AGENT_TIMEOUT.value,
                "message": f"Unable to analyze {normalized_ticker}: all agents failed to respond",
                "details": str(e),
            }
        )
    except OrchestratorError as e:
        logger.error(f"Orchestration error for {normalized_ticker}: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "error_code": ErrorCode.EXTERNAL_API_ERROR.value,
                "message": f"Analysis failed for {normalized_ticker}",
                "details": str(e),
            }
        )
    except Exception as e:
        logger.exception(f"Unexpected error analyzing {normalized_ticker}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error_code": ErrorCode.INTERNAL_ERROR.value,
                "message": "An unexpected error occurred during analysis",
            }
        )
