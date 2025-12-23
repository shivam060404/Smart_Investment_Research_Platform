import hashlib
import json
import logging
from datetime import datetime
from typing import Any, Optional

import redis.asyncio as redis
from pydantic import BaseModel

from app.config import get_settings

logger = logging.getLogger(__name__)


class CacheService:
    """Redis-based caching service for analysis results."""

    def __init__(self, redis_url: Optional[str] = None, ttl_seconds: Optional[int] = None):
        """Initialize the cache service.
        
        Args:
            redis_url: Redis connection URL. Defaults to settings value.
            ttl_seconds: Default TTL for cached items. Defaults to settings value.
        """
        settings = get_settings()
        self._redis_url = redis_url or settings.redis_url
        self._ttl_seconds = ttl_seconds or settings.cache_ttl_seconds
        self._client: Optional[redis.Redis] = None

    async def connect(self) -> None:
        """Establish connection to Redis."""
        if self._client is None:
            self._client = redis.from_url(
                self._redis_url,
                encoding="utf-8",
                decode_responses=True
            )
            logger.info(f"Connected to Redis at {self._redis_url}")

    async def disconnect(self) -> None:
        """Close Redis connection."""
        if self._client:
            await self._client.close()
            self._client = None
            logger.info("Disconnected from Redis")

    async def _ensure_connected(self) -> redis.Redis:
        """Ensure Redis connection is established.
        
        Returns:
            Redis client instance.
        """
        if self._client is None:
            await self.connect()
        return self._client

    @staticmethod
    def generate_cache_key(prefix: str, *args: Any) -> str:
        """Generate a cache key from prefix and arguments.
        
        Args:
            prefix: Key prefix (e.g., 'analysis', 'stock_data').
            *args: Additional arguments to include in the key.
            
        Returns:
            Generated cache key string.
        """
        key_parts = [prefix] + [str(arg) for arg in args]
        key_string = ":".join(key_parts)
        return key_string

    @staticmethod
    def generate_analysis_key(ticker: str, weights: Optional[dict] = None) -> str:
        """Generate a cache key for analysis results.
        
        Args:
            ticker: Stock ticker symbol.
            weights: Optional custom agent weights.
            
        Returns:
            Cache key for the analysis.
        """
        key_parts = ["analysis", ticker.upper()]
        
        if weights:
            # Create a hash of weights to include in key
            weights_str = json.dumps(weights, sort_keys=True)
            weights_hash = hashlib.md5(weights_str.encode()).hexdigest()[:8]
            key_parts.append(weights_hash)
        
        return ":".join(key_parts)

    async def get_cached(
        self,
        key: str,
        model_class: Optional[type[BaseModel]] = None
    ) -> Optional[Any]:
        """Retrieve a cached value.
        
        Args:
            key: Cache key to retrieve.
            model_class: Optional Pydantic model class for deserialization.
            
        Returns:
            Cached value or None if not found.
        """
        try:
            client = await self._ensure_connected()
            value = await client.get(key)
            
            if value is None:
                logger.debug(f"Cache miss for key: {key}")
                return None
            
            logger.debug(f"Cache hit for key: {key}")
            data = json.loads(value)
            
            if model_class:
                return model_class.model_validate(data)
            
            return data
            
        except redis.RedisError as e:
            logger.error(f"Redis error getting key {key}: {e}")
            return None
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error for key {key}: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error getting cached value: {e}")
            return None

    async def set_cached(
        self,
        key: str,
        value: Any,
        ttl_seconds: Optional[int] = None
    ) -> bool:
        """Store a value in cache.
        
        Args:
            key: Cache key.
            value: Value to cache (dict, Pydantic model, or JSON-serializable).
            ttl_seconds: Optional TTL override. Defaults to configured TTL.
            
        Returns:
            True if successfully cached, False otherwise.
        """
        try:
            client = await self._ensure_connected()
            ttl = ttl_seconds or self._ttl_seconds
            
            # Handle Pydantic models
            if isinstance(value, BaseModel):
                json_value = value.model_dump_json()
            else:
                json_value = json.dumps(value, default=self._json_serializer)
            
            await client.setex(key, ttl, json_value)
            logger.debug(f"Cached key {key} with TTL {ttl}s")
            return True
            
        except redis.RedisError as e:
            logger.error(f"Redis error setting key {key}: {e}")
            return False
        except (TypeError, ValueError) as e:
            logger.error(f"Serialization error for key {key}: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error setting cached value: {e}")
            return False

    async def delete_cached(self, key: str) -> bool:
        """Delete a cached value.
        
        Args:
            key: Cache key to delete.
            
        Returns:
            True if deleted, False otherwise.
        """
        try:
            client = await self._ensure_connected()
            result = await client.delete(key)
            logger.debug(f"Deleted cache key {key}: {result > 0}")
            return result > 0
        except redis.RedisError as e:
            logger.error(f"Redis error deleting key {key}: {e}")
            return False

    async def exists(self, key: str) -> bool:
        """Check if a key exists in cache.
        
        Args:
            key: Cache key to check.
            
        Returns:
            True if key exists, False otherwise.
        """
        try:
            client = await self._ensure_connected()
            return await client.exists(key) > 0
        except redis.RedisError as e:
            logger.error(f"Redis error checking key {key}: {e}")
            return False

    async def get_ttl(self, key: str) -> int:
        """Get remaining TTL for a key.
        
        Args:
            key: Cache key.
            
        Returns:
            TTL in seconds, -1 if no TTL, -2 if key doesn't exist.
        """
        try:
            client = await self._ensure_connected()
            return await client.ttl(key)
        except redis.RedisError as e:
            logger.error(f"Redis error getting TTL for key {key}: {e}")
            return -2

    async def health_check(self) -> tuple[bool, Optional[float]]:
        """Check Redis connection health.
        
        Returns:
            Tuple of (is_healthy, latency_ms).
        """
        try:
            client = await self._ensure_connected()
            start = datetime.utcnow()
            await client.ping()
            latency = (datetime.utcnow() - start).total_seconds() * 1000
            return True, latency
        except Exception as e:
            logger.error(f"Redis health check failed: {e}")
            return False, None

    @staticmethod
    def _json_serializer(obj: Any) -> Any:
        """Custom JSON serializer for non-standard types.
        
        Args:
            obj: Object to serialize.
            
        Returns:
            JSON-serializable representation.
        """
        if isinstance(obj, datetime):
            return obj.isoformat()
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


# Global cache service instance
_cache_service: Optional[CacheService] = None


def get_cache_service() -> CacheService:
    """Get the global cache service instance.
    
    Returns:
        CacheService instance.
    """
    global _cache_service
    if _cache_service is None:
        _cache_service = CacheService()
    return _cache_service
