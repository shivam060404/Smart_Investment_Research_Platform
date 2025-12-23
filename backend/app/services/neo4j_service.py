import logging
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, Optional
from uuid import uuid4

from neo4j import AsyncGraphDatabase, AsyncDriver, AsyncSession
from neo4j.exceptions import ServiceUnavailable, AuthError, Neo4jError

from app.config import get_settings
from app.models.signals import AgentSignal, SynthesisResult, Recommendation
from app.models.stock import StockData

logger = logging.getLogger(__name__)


class Neo4jService:
    """Service for Neo4j knowledge graph operations."""

    def __init__(self):
        """Initialize Neo4j service with configuration."""
        self._settings = get_settings()
        self._driver: Optional[AsyncDriver] = None

    async def connect(self) -> None:
        """Establish connection to Neo4j database."""
        if self._driver is not None:
            return

        try:
            self._driver = AsyncGraphDatabase.driver(
                self._settings.neo4j_uri,
                auth=(self._settings.neo4j_user, self._settings.neo4j_password),
            )
            # Verify connectivity
            await self._driver.verify_connectivity()
            logger.info("Connected to Neo4j at %s", self._settings.neo4j_uri)
        except AuthError as e:
            logger.error("Neo4j authentication failed: %s", e)
            raise
        except ServiceUnavailable as e:
            logger.error("Neo4j service unavailable: %s", e)
            raise
        except Exception as e:
            logger.error("Failed to connect to Neo4j: %s", e)
            raise

    async def disconnect(self) -> None:
        """Close Neo4j connection."""
        if self._driver is not None:
            await self._driver.close()
            self._driver = None
            logger.info("Disconnected from Neo4j")

    @asynccontextmanager
    async def get_session(self):
        """Get a Neo4j session context manager.
        
        Yields:
            AsyncSession: Neo4j async session.
        """
        if self._driver is None:
            await self.connect()
        
        session = self._driver.session()
        try:
            yield session
        finally:
            await session.close()

    async def health_check(self) -> dict[str, Any]:
        """Check Neo4j connection health.
        
        Returns:
            dict: Health status with connectivity info.
        """
        start_time = datetime.utcnow()
        
        try:
            if self._driver is None:
                await self.connect()
            
            await self._driver.verify_connectivity()
            
            latency_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            return {
                "status": "healthy",
                "latency_ms": round(latency_ms, 2),
                "message": "Neo4j connection successful"
            }
        except ServiceUnavailable as e:
            return {
                "status": "unhealthy",
                "latency_ms": None,
                "message": f"Neo4j service unavailable: {str(e)}"
            }
        except AuthError as e:
            return {
                "status": "unhealthy",
                "latency_ms": None,
                "message": f"Neo4j authentication failed: {str(e)}"
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "latency_ms": None,
                "message": f"Neo4j error: {str(e)}"
            }

    async def initialize_schema(self) -> None:
        """Initialize Neo4j schema with constraints and indexes."""
        async with self.get_session() as session:
            # Create constraints for unique identifiers
            constraints = [
                "CREATE CONSTRAINT stock_ticker IF NOT EXISTS FOR (s:Stock) REQUIRE s.ticker IS UNIQUE",
                "CREATE CONSTRAINT analysis_id IF NOT EXISTS FOR (a:Analysis) REQUIRE a.id IS UNIQUE",
                "CREATE CONSTRAINT sector_name IF NOT EXISTS FOR (sec:Sector) REQUIRE sec.name IS UNIQUE",
            ]
            
            # Create indexes for common queries
            indexes = [
                "CREATE INDEX analysis_timestamp IF NOT EXISTS FOR (a:Analysis) ON (a.timestamp)",
                "CREATE INDEX stock_sector IF NOT EXISTS FOR (s:Stock) ON (s.sector)",
            ]
            
            for constraint in constraints:
                try:
                    await session.run(constraint)
                except Neo4jError as e:
                    logger.warning("Constraint creation warning: %s", e)
            
            for index in indexes:
                try:
                    await session.run(index)
                except Neo4jError as e:
                    logger.warning("Index creation warning: %s", e)
            
            logger.info("Neo4j schema initialized")


    async def store_analysis(
        self,
        ticker: str,
        synthesis: SynthesisResult,
        signals: list[AgentSignal],
        stock_data: Optional[StockData] = None,
    ) -> str:
        """Store analysis results in the knowledge graph.
        
        Args:
            ticker: Stock ticker symbol.
            synthesis: Synthesized analysis result.
            signals: List of individual agent signals.
            stock_data: Optional stock metadata.
            
        Returns:
            str: Analysis ID.
        """
        analysis_id = str(uuid4())
        timestamp = datetime.utcnow().isoformat()
        
        async with self.get_session() as session:
            # Create or update Stock node
            stock_query = """
            MERGE (s:Stock {ticker: $ticker})
            SET s.name = COALESCE($name, s.name, $ticker),
                s.sector = COALESCE($sector, s.sector, 'Unknown'),
                s.industry = COALESCE($industry, s.industry, 'Unknown'),
                s.last_updated = $timestamp
            RETURN s
            """
            
            stock_params = {
                "ticker": ticker.upper(),
                "name": stock_data.name if stock_data else None,
                "sector": stock_data.sector if stock_data else None,
                "industry": stock_data.industry if stock_data else None,
                "timestamp": timestamp,
            }
            
            await session.run(stock_query, stock_params)
            
            # Create Sector node and relationship
            if stock_data and stock_data.sector != "Unknown":
                sector_query = """
                MERGE (sec:Sector {name: $sector})
                WITH sec
                MATCH (s:Stock {ticker: $ticker})
                MERGE (s)-[:BELONGS_TO]->(sec)
                """
                await session.run(sector_query, {
                    "sector": stock_data.sector,
                    "ticker": ticker.upper(),
                })
            
            # Create Analysis node
            analysis_query = """
            MATCH (s:Stock {ticker: $ticker})
            CREATE (a:Analysis {
                id: $analysis_id,
                timestamp: datetime($timestamp),
                recommendation: $recommendation,
                confidence: $confidence,
                weighted_score: $weighted_score,
                reasoning: $reasoning
            })
            CREATE (s)-[:HAS_ANALYSIS]->(a)
            RETURN a
            """
            
            analysis_params = {
                "ticker": ticker.upper(),
                "analysis_id": analysis_id,
                "timestamp": timestamp,
                "recommendation": synthesis.recommendation.value,
                "confidence": synthesis.confidence,
                "weighted_score": synthesis.weighted_score,
                "reasoning": synthesis.reasoning,
            }
            
            await session.run(analysis_query, analysis_params)
            
            # Create Signal nodes for each agent
            for signal in signals:
                signal_query = """
                MATCH (a:Analysis {id: $analysis_id})
                CREATE (sig:Signal {
                    agent_name: $agent_name,
                    signal_type: $signal_type,
                    score: $score,
                    confidence: $confidence,
                    reasoning_trace: $reasoning_trace,
                    key_factors: $key_factors,
                    timestamp: datetime($timestamp)
                })
                CREATE (a)-[:CONTAINS_SIGNAL]->(sig)
                """
                
                signal_params = {
                    "analysis_id": analysis_id,
                    "agent_name": signal.agent_name,
                    "signal_type": signal.signal_type.value,
                    "score": signal.score,
                    "confidence": signal.confidence,
                    "reasoning_trace": signal.reasoning_trace,
                    "key_factors": signal.key_factors,
                    "timestamp": signal.timestamp.isoformat(),
                }
                
                await session.run(signal_query, signal_params)
            
            # Store risk factors if present
            for risk_factor in synthesis.risk_factors:
                risk_query = """
                MATCH (a:Analysis {id: $analysis_id})
                MERGE (rf:RiskFactor {description: $description})
                ON CREATE SET rf.type = $type
                CREATE (a)-[:IDENTIFIED_RISK]->(rf)
                """
                
                await session.run(risk_query, {
                    "analysis_id": analysis_id,
                    "description": risk_factor,
                    "type": "general",
                })
            
            logger.info("Stored analysis %s for ticker %s", analysis_id, ticker)
            return analysis_id

    async def store_stock_relationships(
        self,
        ticker: str,
        correlations: dict[str, float],
    ) -> None:
        """Store stock correlation relationships.
        
        Args:
            ticker: Source stock ticker.
            correlations: Dict mapping related tickers to correlation values.
        """
        async with self.get_session() as session:
            for related_ticker, correlation in correlations.items():
                query = """
                MERGE (s1:Stock {ticker: $ticker1})
                MERGE (s2:Stock {ticker: $ticker2})
                MERGE (s1)-[r:CORRELATES_WITH]->(s2)
                SET r.correlation = $correlation,
                    r.updated_at = datetime()
                """
                
                await session.run(query, {
                    "ticker1": ticker.upper(),
                    "ticker2": related_ticker.upper(),
                    "correlation": correlation,
                })
            
            logger.info(
                "Stored %d correlations for ticker %s",
                len(correlations),
                ticker
            )

    async def get_analysis_history(
        self,
        ticker: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Retrieve historical analyses for backtesting.
        
        Args:
            ticker: Stock ticker symbol.
            start_date: Optional start date filter.
            end_date: Optional end date filter.
            limit: Maximum number of results.
            
        Returns:
            list: Historical analysis records.
        """
        async with self.get_session() as session:
            # Build query with optional date filters
            query = """
            MATCH (s:Stock {ticker: $ticker})-[:HAS_ANALYSIS]->(a:Analysis)
            """
            
            params: dict[str, Any] = {
                "ticker": ticker.upper(),
                "limit": limit,
            }
            
            where_clauses = []
            if start_date:
                where_clauses.append("a.timestamp >= datetime($start_date)")
                params["start_date"] = start_date.isoformat()
            if end_date:
                where_clauses.append("a.timestamp <= datetime($end_date)")
                params["end_date"] = end_date.isoformat()
            
            if where_clauses:
                query += "WHERE " + " AND ".join(where_clauses) + "\n"
            
            query += """
            OPTIONAL MATCH (a)-[:CONTAINS_SIGNAL]->(sig:Signal)
            WITH a, collect({
                agent_name: sig.agent_name,
                signal_type: sig.signal_type,
                score: sig.score,
                confidence: sig.confidence,
                reasoning_trace: sig.reasoning_trace,
                key_factors: sig.key_factors
            }) as signals
            RETURN a.id as id,
                   a.timestamp as timestamp,
                   a.recommendation as recommendation,
                   a.confidence as confidence,
                   a.weighted_score as weighted_score,
                   a.reasoning as reasoning,
                   signals
            ORDER BY a.timestamp DESC
            LIMIT $limit
            """
            
            result = await session.run(query, params)
            records = await result.data()
            
            # Convert Neo4j datetime to Python datetime
            analyses = []
            for record in records:
                analysis = dict(record)
                if analysis.get("timestamp"):
                    analysis["timestamp"] = analysis["timestamp"].to_native()
                # Filter out empty signal entries
                analysis["signals"] = [
                    s for s in analysis.get("signals", [])
                    if s.get("agent_name")
                ]
                analyses.append(analysis)
            
            logger.info(
                "Retrieved %d analyses for ticker %s",
                len(analyses),
                ticker
            )
            return analyses

    async def get_stock_with_sector(self, ticker: str) -> Optional[dict[str, Any]]:
        """Get stock information with sector relationship.
        
        Args:
            ticker: Stock ticker symbol.
            
        Returns:
            dict: Stock data with sector info, or None if not found.
        """
        async with self.get_session() as session:
            query = """
            MATCH (s:Stock {ticker: $ticker})
            OPTIONAL MATCH (s)-[:BELONGS_TO]->(sec:Sector)
            RETURN s.ticker as ticker,
                   s.name as name,
                   s.sector as sector,
                   s.industry as industry,
                   sec.name as sector_name
            """
            
            result = await session.run(query, {"ticker": ticker.upper()})
            record = await result.single()
            
            if record:
                return dict(record)
            return None

    async def get_correlated_stocks(
        self,
        ticker: str,
        min_correlation: float = 0.5,
    ) -> list[dict[str, Any]]:
        """Get stocks correlated with the given ticker.
        
        Args:
            ticker: Stock ticker symbol.
            min_correlation: Minimum correlation threshold.
            
        Returns:
            list: Correlated stocks with correlation values.
        """
        async with self.get_session() as session:
            query = """
            MATCH (s1:Stock {ticker: $ticker})-[r:CORRELATES_WITH]->(s2:Stock)
            WHERE r.correlation >= $min_correlation
            RETURN s2.ticker as ticker,
                   s2.name as name,
                   r.correlation as correlation
            ORDER BY r.correlation DESC
            """
            
            result = await session.run(query, {
                "ticker": ticker.upper(),
                "min_correlation": min_correlation,
            })
            
            return await result.data()

    async def get_latest_analysis(self, ticker: str) -> Optional[dict[str, Any]]:
        """Get the most recent analysis for a stock.
        
        Args:
            ticker: Stock ticker symbol.
            
        Returns:
            dict: Latest analysis data, or None if not found.
        """
        analyses = await self.get_analysis_history(ticker, limit=1)
        return analyses[0] if analyses else None

    async def delete_old_analyses(self, days_to_keep: int = 730) -> int:
        """Delete analyses older than specified days.
        
        Args:
            days_to_keep: Number of days of data to retain (default 2 years).
            
        Returns:
            int: Number of analyses deleted.
        """
        async with self.get_session() as session:
            query = """
            MATCH (a:Analysis)
            WHERE a.timestamp < datetime() - duration({days: $days})
            OPTIONAL MATCH (a)-[:CONTAINS_SIGNAL]->(sig:Signal)
            OPTIONAL MATCH (a)-[:IDENTIFIED_RISK]->(rf:RiskFactor)
            WITH a, collect(sig) as signals, collect(rf) as risks
            DETACH DELETE a
            FOREACH (s IN signals | DELETE s)
            WITH count(a) as deleted_count
            RETURN deleted_count
            """
            
            result = await session.run(query, {"days": days_to_keep})
            record = await result.single()
            
            deleted = record["deleted_count"] if record else 0
            logger.info("Deleted %d old analyses", deleted)
            return deleted


# Singleton instance
_neo4j_service: Optional[Neo4jService] = None


def get_neo4j_service() -> Neo4jService:
    """Get or create Neo4j service singleton.
    
    Returns:
        Neo4jService: Neo4j service instance.
    """
    global _neo4j_service
    if _neo4j_service is None:
        _neo4j_service = Neo4jService()
    return _neo4j_service
