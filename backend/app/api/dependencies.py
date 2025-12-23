from functools import lru_cache
from typing import Annotated

from fastapi import Depends

from app.config import Settings, get_settings
from app.services.cache_service import CacheService, get_cache_service


def get_config() -> Settings:
    """Get application configuration.
    
    Returns:
        Settings: Application settings instance.
    """
    return get_settings()


def get_cache() -> CacheService:
    """Get cache service instance.
    
    Returns:
        CacheService: Cache service instance.
    """
    return get_cache_service()


# Type aliases for dependency injection
ConfigDep = Annotated[Settings, Depends(get_config)]
CacheDep = Annotated[CacheService, Depends(get_cache)]
