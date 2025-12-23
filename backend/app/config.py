from functools import lru_cache
from pathlib import Path
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Application
    app_name: str = Field(default="AI Investment Research Platform")
    app_version: str = Field(default="0.1.0")
    debug: bool = Field(default=False)
    log_level: str = Field(default="INFO")

    # API Keys
    mistral_api_key: str = Field(default="")
    news_api_key: str = Field(default="")
    alpha_vantage_key: str = Field(default="")

    # Neo4j Configuration
    neo4j_uri: str = Field(default="bolt://localhost:7687")
    neo4j_user: str = Field(default="neo4j")
    neo4j_password: str = Field(default="password")

    # Redis Configuration
    redis_url: str = Field(default="redis://localhost:6379")
    cache_ttl_seconds: int = Field(default=900)  # 15 minutes

    # Rate Limiting
    api_rate_limit: int = Field(default=100)  # requests per minute
    rate_limit_window_seconds: int = Field(default=60)

    # Agent Configuration
    agent_timeout_seconds: int = Field(default=30)
    fundamental_weight: float = Field(default=0.30)
    technical_weight: float = Field(default=0.25)
    sentiment_weight: float = Field(default=0.25)
    risk_weight: float = Field(default=0.20)

    # MCP Configuration
    mcp_enabled: bool = Field(default=False, description="Enable MCP adapter for agent data access")
    mcp_server_timeout: int = Field(default=30, description="MCP server timeout in seconds")
    mcp_config_path: str = Field(default="config/mcp_servers.yaml", description="Path to MCP servers YAML config")

    # Individual MCP Server Configuration (environment variable overrides)
    mcp_yfinance_enabled: bool = Field(default=True, description="Enable YFinance MCP server")
    mcp_yfinance_transport: str = Field(default="stdio", description="YFinance MCP transport type")
    mcp_yfinance_command: Optional[str] = Field(default="python -m app.mcp_servers.yfinance_server")
    mcp_yfinance_timeout: int = Field(default=30)

    mcp_news_enabled: bool = Field(default=True, description="Enable News MCP server")
    mcp_news_transport: str = Field(default="stdio", description="News MCP transport type")
    mcp_news_command: Optional[str] = Field(default="python -m app.mcp_servers.news_server")
    mcp_news_timeout: int = Field(default=15)

    mcp_technical_enabled: bool = Field(default=True, description="Enable Technical MCP server")
    mcp_technical_transport: str = Field(default="stdio", description="Technical MCP transport type")
    mcp_technical_command: Optional[str] = Field(default="python -m app.mcp_servers.technical_server")
    mcp_technical_timeout: int = Field(default=20)

    mcp_neo4j_enabled: bool = Field(default=True, description="Enable Neo4j MCP server")
    mcp_neo4j_transport: str = Field(default="stdio", description="Neo4j MCP transport type")
    mcp_neo4j_command: Optional[str] = Field(default="python -m app.mcp_servers.neo4j_server")
    mcp_neo4j_timeout: int = Field(default=30)

    mcp_backtesting_enabled: bool = Field(default=True, description="Enable Backtesting MCP server")
    mcp_backtesting_transport: str = Field(default="stdio", description="Backtesting MCP transport type")
    mcp_backtesting_command: Optional[str] = Field(default="python -m app.mcp_servers.backtest_server")
    mcp_backtesting_timeout: int = Field(default=60)

    # CORS Configuration
    cors_origins: list[str] = Field(default=["http://localhost:3000"])
    cors_allow_credentials: bool = Field(default=True)
    cors_allow_methods: list[str] = Field(default=["*"])
    cors_allow_headers: list[str] = Field(default=["*"])

    # API Key Authentication
    api_key_auth_enabled: bool = Field(default=True)
    api_keys: str = Field(default="", description="Comma-separated list of valid API keys")

    def get_mcp_config_path(self) -> Path:
        """Get the absolute path to the MCP configuration file.

        Returns:
            Path to the MCP servers YAML configuration file.
        """
        config_path = Path(self.mcp_config_path)
        if not config_path.is_absolute():
            # Resolve relative to the backend directory
            backend_dir = Path(__file__).parent.parent
            config_path = backend_dir / config_path
        return config_path


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance.
    
    Returns:
        Settings: Application settings loaded from environment.
    """
    return Settings()
