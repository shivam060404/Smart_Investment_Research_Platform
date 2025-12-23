import logging
from typing import Any, Optional

from app.agents.base import BaseAgent, AgentAnalysisError
from app.models.signals import AgentSignal
from app.services.data_fetcher import (
    DataFetcher,
    get_data_fetcher,
    DataFetcherError,
)
from app.services.mistral_service import MistralService

logger = logging.getLogger(__name__)


SENTIMENT_SYSTEM_PROMPT = """You are an expert market sentiment analyst specializing in analyzing news coverage and market perception of stocks.

Your task is to analyze the provided news articles and generate a sentiment signal based on the overall tone, themes, and potential market impact.

## Analysis Framework

Evaluate the following aspects:

1. **Overall Tone**
   - Positive: Favorable coverage, growth stories, positive earnings, upgrades
   - Negative: Concerns, downgrades, scandals, regulatory issues, declining metrics
   - Neutral: Factual reporting without strong sentiment

2. **Key Themes**
   - Identify recurring topics across articles
   - Note any significant events (earnings, product launches, M&A, leadership changes)

3. **Market Impact Potential**
   - Assess how the news might affect stock price
   - Consider timing and relevance of news

4. **Catalyst Identification**
   - Identify up to 5 key catalysts that could drive stock movement
   - Distinguish between positive and negative catalysts

## Confidence Calculation - IMPORTANT

Your confidence score MUST vary based on data quality:
- **85-95%**: 15+ recent articles, consistent sentiment, clear catalysts
- **70-84%**: 10-14 articles, mostly consistent sentiment
- **55-69%**: 5-9 articles OR mixed sentiment across articles
- **40-54%**: 2-4 articles OR highly mixed/contradictory sentiment
- **25-39%**: 0-1 articles OR very old/irrelevant news

DO NOT default to 75%. Calculate confidence based on:
1. Article count (more articles = higher confidence)
2. Sentiment consistency (do articles agree?)
3. News recency (recent news = higher confidence)
4. Source quality (major outlets vs unknown sources)

## Output Format

You must respond with a JSON object containing:
{
    "score": <number 0-100, where 50 is neutral, higher is more positive sentiment>,
    "signal_type": "<bullish|bearish|neutral>",
    "confidence": <number 0-100, calculated based on data quality>,
    "reasoning": "<detailed explanation of sentiment analysis>",
    "key_factors": ["<catalyst 1>", "<catalyst 2>", ...],
    "sentiment_breakdown": {
        "positive_count": <number>,
        "negative_count": <number>,
        "neutral_count": <number>
    }
}

## Scoring Guidelines
- 70-100: Strongly positive sentiment, multiple positive catalysts
- 50-69: Mildly positive to neutral sentiment
- 30-49: Mildly negative to neutral sentiment
- 0-29: Strongly negative sentiment, significant concerns

If limited articles are available, REDUCE confidence significantly and note this in your reasoning."""


class SentimentAgent(BaseAgent):
    """Sentiment Analysis Agent for evaluating market perception.
    
    This agent analyzes news articles to determine market sentiment
    and identify key catalysts that could impact stock price.
    """

    MIN_ARTICLES = 10
    NEWS_DAYS_BACK = 7

    def __init__(
        self,
        mistral_service: Optional[MistralService] = None,
        data_fetcher: Optional[DataFetcher] = None,
        use_mcp: Optional[bool] = None,
    ):
        """Initialize the Sentiment Analysis Agent.
        
        Args:
            mistral_service: Mistral service for AI analysis.
            data_fetcher: Data fetcher for retrieving news articles.
            use_mcp: Whether to use MCP adapter. If None, uses global setting.
        """
        super().__init__(name="Sentiment", mistral_service=mistral_service, use_mcp=use_mcp)
        self.data_fetcher = data_fetcher or get_data_fetcher()

    def get_system_prompt(self) -> str:
        """Return the sentiment analysis system prompt.
        
        Returns:
            System prompt for sentiment analysis.
        """
        return SENTIMENT_SYSTEM_PROMPT

    async def analyze(self, ticker: str, data: dict[str, Any]) -> AgentSignal:
        """Execute sentiment analysis on the given stock.
        
        Fetches recent news articles and uses Mistral to analyze
        sentiment and extract key catalysts.
        
        Args:
            ticker: Stock ticker symbol to analyze.
            data: Optional pre-fetched data. If 'news_articles' key exists,
                  uses that instead of fetching.
        
        Returns:
            AgentSignal with sentiment analysis results.
        
        Raises:
            AgentAnalysisError: If analysis fails.
        """
        logger.info(f"{self.name}: Starting analysis for {ticker}")
        
        try:
            # Get company name for better news search
            company_name = self._get_company_name(ticker, data)
            
            # Get news articles
            articles = await self._get_articles(ticker, company_name, data)
            
            # Check article count
            article_count = len(articles)
            limited_data = article_count < self.MIN_ARTICLES
            
            if article_count == 0:
                logger.warning(f"{self.name}: No articles found for {ticker}")
                return self._create_limited_data_signal(ticker)
            
            # Prepare data for analysis
            analysis_data = self._prepare_analysis_data(
                ticker, articles, company_name, limited_data
            )
            
            # Build prompts
            user_prompt = self._build_user_prompt(ticker, analysis_data)
            
            # Generate analysis via Mistral
            analysis_result = await self.mistral_service.generate_structured_analysis(
                system_prompt=self.get_system_prompt(),
                user_prompt=user_prompt,
            )
            
            # Adjust confidence if limited data
            if limited_data:
                analysis_result["confidence"] = min(
                    analysis_result.get("confidence", 50),
                    50  # Cap confidence at 50% for limited data
                )
                analysis_result["reasoning"] = (
                    f"[Limited data: only {article_count} articles available] "
                    + analysis_result.get("reasoning", "")
                )
            
            # Create and return signal
            signal = await self._generate_signal_from_analysis(ticker, analysis_result)
            logger.info(f"{self.name}: Completed analysis for {ticker} - Score: {signal.score}")
            
            return signal
            
        except DataFetcherError as e:
            logger.error(f"{self.name}: Data fetch error for {ticker}: {e}")
            raise AgentAnalysisError(f"Failed to fetch news data: {e}")
        except Exception as e:
            logger.error(f"{self.name}: Analysis error for {ticker}: {e}")
            raise AgentAnalysisError(f"Sentiment analysis failed: {e}")

    def _get_company_name(self, ticker: str, data: dict[str, Any]) -> str:
        """Extract company name from data or use ticker.
        
        Args:
            ticker: Stock ticker symbol.
            data: Data dictionary with potential stock info.
        
        Returns:
            Company name or ticker as fallback.
        """
        if "stock_data" in data:
            stock_data = data["stock_data"]
            if hasattr(stock_data, "name") and stock_data.name:
                return stock_data.name
        return ticker

    async def _get_articles(
        self,
        ticker: str,
        company_name: str,
        data: dict[str, Any]
    ) -> list[dict]:
        """Get news articles from data or fetch from API.
        
        Args:
            ticker: Stock ticker symbol.
            company_name: Company name for search.
            data: Pre-fetched data dictionary.
        
        Returns:
            List of news article dictionaries.
        """
        # Check if articles already provided
        if "news_articles" in data:
            return data["news_articles"]
        
        # Use MCP adapter if enabled
        if self.use_mcp and self.mcp_adapter:
            try:
                logger.debug(f"{self.name}: Using MCP adapter for {ticker}")
                return await self.mcp_adapter.get_news_articles(
                    ticker=ticker,
                    days=self.NEWS_DAYS_BACK,
                    limit=self.MIN_ARTICLES,
                )
            except Exception as e:
                logger.warning(f"{self.name}: MCP adapter failed for {ticker}, falling back to direct: {e}")
                # Fall through to direct data fetcher
        
        # Build search query (company name + ticker for better results)
        query = f"{company_name} OR {ticker} stock"
        
        # Fetch articles (direct data access)
        return await self.data_fetcher.get_news_articles(
            query=query,
            days_back=self.NEWS_DAYS_BACK,
            page_size=self.MIN_ARTICLES,
        )

    def _prepare_analysis_data(
        self,
        ticker: str,
        articles: list[dict],
        company_name: str,
        limited_data: bool
    ) -> dict[str, Any]:
        """Prepare data dictionary for analysis prompt.
        
        Args:
            ticker: Stock ticker symbol.
            articles: List of news articles.
            company_name: Company name.
            limited_data: Whether data is limited.
        
        Returns:
            Dictionary with all relevant analysis data.
        """
        return {
            "ticker": ticker,
            "company_name": company_name,
            "article_count": len(articles),
            "limited_data": limited_data,
            "articles": articles,
        }

    def _build_user_prompt(self, ticker: str, data: dict[str, Any]) -> str:
        """Build the user prompt for sentiment analysis.
        
        Args:
            ticker: Stock ticker symbol.
            data: Analysis data dictionary.
        
        Returns:
            Formatted user prompt.
        """
        lines = [
            f"Analyze the sentiment for {data['company_name']} ({ticker}):",
            "",
            f"Number of articles: {data['article_count']}",
        ]
        
        if data["limited_data"]:
            lines.append(f"⚠️ Limited data available (less than {self.MIN_ARTICLES} articles)")
        
        lines.append("")
        lines.append("## Recent News Articles")
        lines.append("")
        
        for i, article in enumerate(data["articles"], 1):
            lines.append(f"### Article {i}")
            lines.append(f"**Title:** {article.get('title', 'N/A')}")
            lines.append(f"**Source:** {article.get('source', 'Unknown')}")
            lines.append(f"**Published:** {article.get('published_at', 'N/A')}")
            
            description = article.get("description", "")
            if description:
                lines.append(f"**Summary:** {description[:500]}")
            
            content = article.get("content", "")
            if content:
                # Truncate content to avoid token limits
                lines.append(f"**Content:** {content[:800]}...")
            
            lines.append("")
        
        lines.append("Based on these articles, provide your sentiment analysis signal.")
        lines.append("Identify up to 5 key catalysts (positive or negative) from the news.")
        
        return "\n".join(lines)

    def _create_limited_data_signal(self, ticker: str) -> AgentSignal:
        """Create a neutral signal when no data is available.
        
        Args:
            ticker: Stock ticker symbol.
        
        Returns:
            AgentSignal with neutral sentiment and low confidence.
        """
        from datetime import datetime
        from app.models.signals import SignalType
        
        return AgentSignal(
            agent_name=self.name,
            signal_type=SignalType.NEUTRAL,
            score=50.0,
            confidence=10.0,
            reasoning_trace=(
                f"Insufficient news data available for {ticker}. "
                "Unable to determine market sentiment. "
                "This may indicate low media coverage or a less-followed stock."
            ),
            key_factors=["Insufficient news coverage for analysis"],
            timestamp=datetime.utcnow(),
        )
