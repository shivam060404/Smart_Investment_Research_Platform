# AI Investment Research Platform

A multi-agent AI investment research platform that autonomously analyzes stocks for investment recommendations. The platform coordinates specialized financial agents (Fundamental, Sentiment, Technical, and Risk) through an orchestrator agent, leveraging real-time data sources, a Neo4j knowledge graph, and an explainability layer.

## 🚀 Quick Deploy to Railway

[![Deploy on Railway](https://railway.app/button.svg)](https://railway.app/new/template)

See [Railway Deployment](#railway-deployment) section for detailed instructions.

## Architecture Overview

- **Backend**: FastAPI (Python 3.11+) with Mistral AI-powered agents
- **Frontend**: Next.js 14 with TypeScript and Tailwind CSS
- **Database**: Neo4j for knowledge graph storage
- **Cache**: Redis for API response caching
- **MCP Layer**: Model Context Protocol servers for standardized data access
- **Containerization**: Docker and Docker Compose

### MCP Architecture

The platform uses Model Context Protocol (MCP) to provide a standardized interface between AI agents and external data sources. This enables distributed, scalable architecture with real-time market data from multiple providers.

```
┌─────────────────────────────────────────────────────────────────┐
│                     AGENT LAYER                                  │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐           │
│  │Fundamental│ │Sentiment │ │Technical │ │  Risk    │           │
│  │  Agent   │ │  Agent   │ │  Agent   │ │  Agent   │           │
│  └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘           │
│       └────────────┴────────────┴────────────┘                   │
│                          │                                       │
│                    MCPDataAdapter                                │
│                          │                                       │
│                     MCPClient                                    │
└──────────────────────────┼──────────────────────────────────────┘
                           │ JSON-RPC
┌──────────────────────────┼──────────────────────────────────────┐
│                     MCP SERVER LAYER                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐              │
│  │  YFinance   │  │    News     │  │  Technical  │              │
│  │ MCP Server  │  │ MCP Server  │  │ MCP Server  │              │
│  └─────────────┘  └─────────────┘  └─────────────┘              │
│  ┌─────────────┐  ┌─────────────┐                               │
│  │   Neo4j     │  │ Backtesting │                               │
│  │ MCP Server  │  │ MCP Server  │                               │
│  └─────────────┘  └─────────────┘                               │
└─────────────────────────────────────────────────────────────────┘
```

## Prerequisites

- Docker and Docker Compose (recommended)
- OR for local development:
  - Python 3.11+
  - Node.js 20+
  - Neo4j 5.x
  - Redis 7.x

## Quick Start with Docker Compose

The easiest way to run the platform is using Docker Compose:

```bash
# 1. Clone the repository
git clone <repository-url>
cd ai-investment-platform

# 2. Edit backend/.env with your API keys (see Environment Variables section)

# 3. Start all services
docker-compose up -d

# 4. Access the application
# Frontend: http://localhost:3000
# Backend API: http://localhost:8000
# API Docs: http://localhost:8000/docs
# Neo4j Browser: http://localhost:7474
```

### Docker Compose Commands

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# View logs for specific service
docker-compose logs -f backend

# Stop all services
docker-compose down

# Rebuild and start (after code changes)
docker-compose up -d --build

# Remove all data (volumes)
docker-compose down -v
```

## Local Development Setup

### Backend Setup

```bash
# Navigate to backend directory
cd backend

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Edit .env with your API keys

# Start the backend server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Frontend Setup

```bash
# Navigate to frontend directory
cd frontend

# Install dependencies
npm install

# Start the development server
npm run dev
```

### Neo4j Setup (Local)

1. Download and install Neo4j Desktop from https://neo4j.com/download/
2. Create a new database with password `password` (or update `.env`)
3. Start the database
4. Access Neo4j Browser at http://localhost:7474

### Redis Setup (Local)

```bash
# Using Docker (recommended)
docker run -d --name redis -p 6379:6379 redis:7-alpine

# Or install locally
# macOS: brew install redis && brew services start redis
# Ubuntu: sudo apt install redis-server && sudo systemctl start redis
```

## Environment Variables

### Backend (`backend/.env`)

| Variable | Description | Required | Default |
|----------|-------------|----------|---------|
| `MISTRAL_API_KEY` | Mistral AI API key for agent LLM | Yes | - |
| `NEWS_API_KEY` | NewsAPI key for sentiment analysis | Yes | - |
| `ALPHA_VANTAGE_KEY` | AlphaVantage API key (fallback data source) | No | - |
| `NEO4J_URI` | Neo4j connection URI | Yes | `bolt://localhost:7687` |
| `NEO4J_USER` | Neo4j username | Yes | `neo4j` |
| `NEO4J_PASSWORD` | Neo4j password | Yes | `password` |
| `REDIS_URL` | Redis connection URL | Yes | `redis://localhost:6379` |
| `API_RATE_LIMIT` | Requests per minute per client | No | `100` |
| `CACHE_TTL_SECONDS` | Cache time-to-live in seconds | No | `900` |
| `LOG_LEVEL` | Logging level (DEBUG, INFO, WARNING, ERROR) | No | `INFO` |
| `API_KEY_AUTH_ENABLED` | Enable API key authentication | No | `true` |
| `API_KEYS` | Comma-separated list of valid API keys | Yes* | - |
| `HOST` | Server host | No | `0.0.0.0` |
| `PORT` | Server port | No | `8000` |
| `MCP_ENABLED` | Enable MCP adapter for agent data access | No | `false` |
| `MCP_SERVER_TIMEOUT` | Global MCP server timeout in seconds | No | `30` |
| `MCP_CONFIG_PATH` | Path to MCP servers YAML config | No | `config/mcp_servers.yaml` |

*Required when `API_KEY_AUTH_ENABLED=true`

### Frontend (`frontend/.env`)

| Variable | Description | Required | Default |
|----------|-------------|----------|---------|
| `NEXT_PUBLIC_API_URL` | Backend API URL | Yes | `http://localhost:8000` |

## Obtaining API Keys

### Mistral AI API Key
1. Visit https://console.mistral.ai/
2. Create an account or sign in
3. Navigate to API Keys section
4. Generate a new API key

### NewsAPI Key
1. Visit https://newsapi.org/
2. Create a free account
3. Copy your API key from the dashboard

### AlphaVantage Key (Optional)
1. Visit https://www.alphavantage.co/support/#api-key
2. Request a free API key
3. Used as fallback when Yahoo Finance is unavailable

## API Documentation

Once the backend is running, access the interactive API documentation:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **OpenAPI JSON**: http://localhost:8000/openapi.json

### Key Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/analyze/{ticker}` | POST | Analyze a stock ticker |
| `/api/backtest` | POST | Run historical backtesting |
| `/api/chat` | POST | Send message to RAG chatbot |
| `/api/chat/suggestions` | GET | Get suggested questions for chatbot |
| `/health` | GET | Check system health status |

### AI-Driven Backtesting

The platform supports AI-driven backtesting that uses the multi-agent system to generate trading signals:

```bash
# Example: Run AI backtest via API
curl -X POST http://localhost:8000/api/backtest \
  -H "Content-Type: application/json" \
  -d '{
    "ticker": "AAPL",
    "start_date": "2023-01-01",
    "end_date": "2024-01-01",
    "initial_capital": 10000,
    "strategy": "ai_live",
    "analysis_interval": "weekly",
    "include_baseline_comparison": true
  }'
```

**Available Strategies:**
- `ai_live` - Real multi-agent AI strategy using Mistral API
- `ai` - Simulated AI strategy (faster, no API calls)
- `sma_crossover` - SMA 50/200 crossover strategy
- `macd` - MACD crossover strategy
- `buy_hold` - Simple buy and hold
- `auto` - Automatically select best strategy

**Analysis Intervals (for AI strategies):**
- `daily` - Generate signals every trading day
- `weekly` - Generate signals every 5 trading days (recommended)
- `biweekly` - Generate signals every 10 trading days
- `monthly` - Generate signals every 21 trading days

### RAG Chatbot

The platform includes a RAG (Retrieval-Augmented Generation) chatbot that answers questions about stock analyses using data from Neo4j and cached analysis results.

```bash
# Example: Ask a question via API
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Why is AAPL recommended?"}'

# Get suggested questions
curl http://localhost:8000/api/chat/suggestions?ticker=AAPL
```

**Supported Question Types:**
- `recommendation` - Why is a stock recommended (buy/sell/hold)?
- `analysis` - What does the analysis say about a stock?
- `comparison` - Compare two or more stocks
- `risk` - What are the risks for a stock?
- `technical` - Technical indicators (SMA, RSI, MACD)
- `sentiment` - Market sentiment and news analysis
- `general` - Other general questions

**Example Questions:**
- "Why is AAPL recommended?"
- "What are the risks for TSLA?"
- "Explain the technical analysis for GOOGL"
- "What does the sentiment say about MSFT?"
- "Compare AAPL and GOOGL"

The chatbot automatically:
1. Classifies the question type and extracts ticker symbols
2. Retrieves relevant context from cache and Neo4j
3. Runs fresh analysis if no cached data exists
4. Generates answers with source citations

## Project Structure

```
.
├── backend/
│   ├── app/
│   │   ├── agents/          # AI agents (fundamental, sentiment, technical, risk)
│   │   ├── api/routes/      # API endpoints
│   │   ├── mcp/             # MCP infrastructure (client, adapter, registry)
│   │   ├── mcp_servers/     # MCP server implementations
│   │   ├── models/          # Pydantic data models
│   │   ├── services/        # External service integrations
│   │   │   ├── ai_backtest_service.py  # AI-driven backtesting
│   │   │   ├── mistral_service.py      # Mistral AI integration
│   │   │   ├── rag_service.py          # RAG chatbot service
│   │   │   └── ...
│   │   └── utils/           # Utility functions
│   ├── config/              # Configuration files (mcp_servers.yaml)
│   ├── tests/               # Backend tests
│   │   └── test_ai_backtest_integration.py  # AI backtest integration tests
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── pytest.ini           # Pytest configuration
│   └── .env.example
├── frontend/
│   ├── src/
│   │   ├── app/             # Next.js pages
│   │   ├── components/      # React components
│   │   ├── lib/             # API client and utilities
│   │   └── types/           # TypeScript types
│   ├── Dockerfile
│   └── .env.example
├── docker-compose.yml
└── README.md
```

## MCP Servers

The platform includes five MCP servers that provide standardized access to financial data and analysis tools.

### YFinance MCP Server

Provides real-time and historical stock market data from Yahoo Finance.

| Tool | Description | Parameters |
|------|-------------|------------|
| `get_stock_price` | Fetch OHLCV price data | `ticker`, `period` |
| `get_fundamentals` | Get P/E, ROE, ROA, profit margins | `ticker` |
| `get_company_profile` | Get sector, industry, market cap | `ticker` |
| `calculate_ratios` | Calculate financial ratios | `ticker` |

Resources: `yfinance://price/{ticker}`, `yfinance://profile/{ticker}`, `yfinance://fundamentals/{ticker}`

### News Sentiment MCP Server

Aggregates news articles and calculates sentiment scores.

| Tool | Description | Parameters |
|------|-------------|------------|
| `get_news` | Fetch recent news articles | `ticker`, `days` |
| `get_headlines` | Get top headlines | `ticker`, `limit` |
| `calculate_sentiment` | Calculate aggregate sentiment | `ticker` |

Resources: `news://articles/{ticker}`, `news://sentiment/{ticker}`

### Technical Indicators MCP Server

Calculates technical analysis indicators and detects patterns.

| Tool | Description | Parameters |
|------|-------------|------------|
| `calculate_sma` | Simple Moving Average | `ticker`, `period` |
| `calculate_rsi` | Relative Strength Index | `ticker`, `period` |
| `calculate_macd` | MACD indicator | `ticker` |
| `detect_patterns` | Detect Golden/Death Cross | `ticker` |
| `get_signals` | Aggregate all signals | `ticker` |

Resources: `technical://indicators/{ticker}`, `technical://signals/{ticker}`

### Neo4j MCP Server

Queries the knowledge graph for correlations and historical analysis.

| Tool | Description | Parameters |
|------|-------------|------------|
| `get_correlations` | Stock correlations within sector | `ticker`, `sector` |
| `find_similar_stocks` | Find similar stocks | `ticker`, `limit` |
| `get_analysis_history` | Historical analysis signals | `ticker` |
| `get_sector_performance` | Sector performance metrics | `sector` |

Resources: `neo4j://correlations/{ticker}`, `neo4j://history/{ticker}`

### Backtesting MCP Server

Runs strategy backtests and calculates performance metrics.

| Tool | Description | Parameters |
|------|-------------|------------|
| `run_backtest` | Execute strategy backtest | `ticker`, `strategy`, `period` |
| `calculate_metrics` | Get performance metrics | `backtest_id` |
| `get_equity_curve` | Daily portfolio values | `backtest_id` |
| `compare_strategies` | Compare multiple strategies | `ticker`, `strategies` |

Resources: `backtest://results/{id}`, `backtest://history/{ticker}`

### MCP Configuration

MCP servers are configured via `backend/config/mcp_servers.yaml` or environment variables:

```yaml
# backend/config/mcp_servers.yaml
servers:
  yfinance:
    enabled: true
    transport: "stdio"
    command: "python -m app.mcp_servers.yfinance_server"
    timeout_seconds: 30
```

Environment variable overrides follow the pattern `MCP_{SERVER}_{SETTING}`:
- `MCP_YFINANCE_ENABLED=true`
- `MCP_YFINANCE_TIMEOUT=30`

## Services and Ports

| Service | Port | Description |
|---------|------|-------------|
| Frontend | 3000 | Next.js web application |
| Backend | 8000 | FastAPI REST API |
| Neo4j HTTP | 7474 | Neo4j Browser interface |
| Neo4j Bolt | 7687 | Neo4j database connection |
| Redis | 6379 | Redis cache |

## Troubleshooting

### Common Issues

**Backend fails to start**
- Ensure all required environment variables are set
- Check Neo4j and Redis are running and accessible
- Verify API keys are valid

**Neo4j connection refused**
- Ensure Neo4j is running: `docker-compose logs neo4j`
- Check the URI matches your configuration
- Verify credentials are correct

**Redis connection error**
- Ensure Redis is running: `docker-compose logs redis`
- Check REDIS_URL is correct

**API rate limit exceeded**
- Wait for the rate limit window to reset (1 minute)
- Increase `API_RATE_LIMIT` in configuration

**Mistral API errors**
- Verify your API key is valid
- Check your Mistral account has available credits
- Review rate limits on your Mistral plan

### MCP Troubleshooting

**MCP server fails to start**
- Check `MCP_ENABLED=true` in your `.env` file
- Verify the MCP config file exists at `backend/config/mcp_servers.yaml`
- Check server logs for initialization errors: `docker-compose logs -f backend`

**MCP server timeout errors**
- Increase timeout in config: `MCP_SERVER_TIMEOUT=60`
- Check if external APIs (Yahoo Finance, NewsAPI) are accessible
- Verify network connectivity to data sources

**MCP tool returns empty results**
- Verify the ticker symbol is valid
- Check if the data source has data for the requested period
- Review server logs for API errors

**MCP health check fails**
- Check `/health` endpoint for MCP server status
- Verify all required environment variables are set
- Ensure Neo4j is running for the Neo4j MCP server

**Switching between MCP and direct data access**
- Set `MCP_ENABLED=false` to use direct API calls
- Set `MCP_ENABLED=true` to route through MCP servers
- Both modes use the same agent interfaces

### Viewing Logs

```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f backend
docker-compose logs -f frontend
docker-compose logs -f neo4j
docker-compose logs -f redis
```

## Development

### Running Tests

```bash
# Backend tests
cd backend
pytest

# Run specific test file
pytest tests/test_ai_backtest_integration.py -v

# Run with coverage
pytest --cov=app tests/

# Frontend tests
cd frontend
npm test
```

### Backend Test Suite

The backend includes comprehensive integration tests for the AI-driven backtesting functionality:

| Test Category | Description |
|---------------|-------------|
| Portfolio Simulator | Tests for position tracking, equity calculations, and win rate |
| AI Backtest Service | Tests for signal generation, orchestrator integration, and fallback behavior |
| API Endpoints | Integration tests for `/api/backtest` with various strategies |
| Metrics Calculation | Tests for Sharpe ratio, max drawdown, and performance metrics |
| Warning System | Tests for error handling and warning generation |

### Code Formatting

```bash
# Backend (Python)
cd backend
pip install black isort
black .
isort .

# Frontend (TypeScript)
cd frontend
npm run lint
```

## Railway Deployment

Railway is a modern cloud platform that makes deploying applications simple. This project is configured for easy deployment on Railway.

### Prerequisites

1. Create a [Railway account](https://railway.app/)
2. Install the Railway CLI (optional but recommended):
   ```bash
   npm install -g @railway/cli
   railway login
   ```

### One-Click Deploy

[![Deploy on Railway](https://railway.app/button.svg)](https://railway.app/template)

### Manual Deployment Steps

#### Step 1: Create a New Project

1. Go to [Railway Dashboard](https://railway.app/dashboard)
2. Click "New Project"
3. Select "Deploy from GitHub repo"
4. Connect your GitHub account and select this repository

#### Step 2: Add Required Services

Add the following services to your Railway project:

**Neo4j Database:**
1. Click "New" → "Database" → "Add Neo4j"
2. Railway will automatically provision a Neo4j instance
3. Note the connection variables: `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD`

**Redis Cache:**
1. Click "New" → "Database" → "Add Redis"
2. Railway will automatically provision a Redis instance
3. Note the `REDIS_URL` variable

#### Step 3: Deploy Backend Service

1. Click "New" → "GitHub Repo" → Select your repo
2. Set the root directory to `backend`
3. Railway will auto-detect the Dockerfile
4. Add environment variables (Settings → Variables):

```env
# Required API Keys
MISTRAL_API_KEY=your_mistral_api_key
NEWS_API_KEY=your_newsapi_key

# Neo4j (use Railway's provided variables)
NEO4J_URI=${{Neo4j.NEO4J_URI}}
NEO4J_USER=${{Neo4j.NEO4J_USER}}
NEO4J_PASSWORD=${{Neo4j.NEO4J_PASSWORD}}

# Redis (use Railway's provided variable)
REDIS_URL=${{Redis.REDIS_URL}}

# App Config
API_RATE_LIMIT=100
CACHE_TTL_SECONDS=900
LOG_LEVEL=INFO
API_KEY_AUTH_ENABLED=false
MCP_ENABLED=true
HOST=0.0.0.0
PORT=8000
```

5. Set the port to `8000` in Settings → Networking → Public Networking

#### Step 4: Deploy Frontend Service

1. Click "New" → "GitHub Repo" → Select your repo
2. Set the root directory to `frontend`
3. Railway will auto-detect the Dockerfile
4. Add environment variables:

```env
NEXT_PUBLIC_API_URL=https://your-backend-service.railway.app
```

5. Set the port to `3000` in Settings → Networking → Public Networking

#### Step 5: Configure Service Dependencies

Ensure the backend service starts before the frontend by setting up service dependencies in Railway's settings.

### Railway Configuration Files

The project includes Railway-specific configuration files:

**`railway.toml`** (Backend):
```toml
[build]
builder = "dockerfile"
dockerfilePath = "Dockerfile"

[deploy]
healthcheckPath = "/health"
healthcheckTimeout = 300
restartPolicyType = "on_failure"
restartPolicyMaxRetries = 3
```

**`railway.toml`** (Frontend):
```toml
[build]
builder = "dockerfile"
dockerfilePath = "Dockerfile"

[deploy]
healthcheckPath = "/"
healthcheckTimeout = 300
restartPolicyType = "on_failure"
restartPolicyMaxRetries = 3
```

### Environment Variables Reference (Railway)

| Variable | Service | Description |
|----------|---------|-------------|
| `MISTRAL_API_KEY` | Backend | Mistral AI API key |
| `NEWS_API_KEY` | Backend | NewsAPI key |
| `NEO4J_URI` | Backend | Neo4j connection URI (from Railway) |
| `NEO4J_USER` | Backend | Neo4j username (from Railway) |
| `NEO4J_PASSWORD` | Backend | Neo4j password (from Railway) |
| `REDIS_URL` | Backend | Redis URL (from Railway) |
| `NEXT_PUBLIC_API_URL` | Frontend | Backend API URL |

### Railway CLI Deployment

```bash
# Login to Railway
railway login

# Initialize project (run from project root)
railway init

# Link to existing project
railway link

# Deploy backend
cd backend
railway up

# Deploy frontend
cd ../frontend
railway up

# View logs
railway logs

# Open deployed app
railway open
```

### Monitoring and Logs

- View real-time logs in Railway Dashboard → Service → Logs
- Monitor resource usage in Railway Dashboard → Service → Metrics
- Set up alerts for service health in Railway Dashboard → Settings → Alerts

### Cost Estimation

Railway offers a free tier with:
- $5 free credit per month
- 500 hours of compute
- 1GB RAM per service

For production workloads, expect approximately:
- Backend: ~$5-10/month
- Frontend: ~$5-10/month
- Neo4j: ~$10-20/month
- Redis: ~$5/month

### Troubleshooting Railway Deployment

**Build fails:**
- Check build logs in Railway Dashboard
- Ensure Dockerfile is valid
- Verify all dependencies are in requirements.txt/package.json

**Service won't start:**
- Check runtime logs for errors
- Verify all environment variables are set
- Ensure health check endpoint is accessible

**Database connection issues:**
- Verify Railway variable references are correct (e.g., `${{Neo4j.NEO4J_URI}}`)
- Check if database service is running
- Ensure network connectivity between services

**Frontend can't reach backend:**
- Verify `NEXT_PUBLIC_API_URL` points to the correct backend URL
- Ensure backend has public networking enabled
- Check CORS settings in backend

## Alternative Deployment Options

### Render

Similar to Railway, Render supports Docker deployments:
1. Create a new Web Service
2. Connect your GitHub repo
3. Set root directory and environment variables
4. Deploy

### Fly.io

```bash
# Install flyctl
curl -L https://fly.io/install.sh | sh

# Deploy backend
cd backend
fly launch
fly deploy

# Deploy frontend
cd ../frontend
fly launch
fly deploy
```

### AWS App Runner

1. Push Docker images to ECR
2. Create App Runner services for backend and frontend
3. Configure environment variables and networking

## License

MIT License - See LICENSE file for details
