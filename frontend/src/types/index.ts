/**
 * TypeScript interfaces for the AI Investment Research Platform
 * Matching backend Pydantic models (snake_case from API)
 */

// ============================================
// Stock Data Models
// ============================================

export interface StockData {
  ticker: string
  name: string
  sector: string
  industry: string
  current_price: number
  market_cap: number
  currency: string
}

export interface FundamentalMetrics {
  pe_ratio: number | null
  pb_ratio: number | null
  roe: number | null
  roa: number | null
  debt_to_equity: number | null
  revenue_growth: number | null
  profit_margin: number | null
  dividend_yield: number | null
}

export interface MACDData {
  macd_line: number
  signal_line: number
  histogram: number
}

export interface BollingerData {
  upper: number
  middle: number
  lower: number
}

export interface TechnicalIndicators {
  sma_50: number
  sma_200: number
  rsi_14: number
  macd: MACDData
  bollinger_bands: BollingerData
  current_trend: SignalType
}

export interface SentimentData {
  overall_score: number // -100 to +100
  article_count: number
  key_catalysts: string[]
  source_breakdown: Record<string, number>
}

export interface RiskFactor {
  type: 'market' | 'sector' | 'company' | 'regulatory' | 'macroeconomic'
  description: string
  severity: 'low' | 'medium' | 'high'
}

export interface RiskMetrics {
  volatility: number
  beta: number
  var_95: number // Value at Risk
  risk_score: number // 0-100
  risk_factors: RiskFactor[]
}

// ============================================
// Signal Models
// ============================================

export type SignalType = 'bullish' | 'bearish' | 'neutral'

export interface AgentSignal {
  agent_name: string
  signal_type: SignalType
  score: number // 0-100
  confidence: number // 0-100
  reasoning_trace: string
  key_factors: string[]
  timestamp: string
}

export type Recommendation = 'BUY' | 'SELL' | 'HOLD'

export interface SynthesisResult {
  recommendation: Recommendation
  confidence: number
  weighted_score: number
  reasoning: string
  agent_contributions: Record<string, number>
  key_catalysts: string[]
  risk_factors: string[]
}

// ============================================
// Request Models
// ============================================

export interface AgentWeights {
  fundamental: number
  technical: number
  sentiment: number
  risk: number
}

export interface AnalyzeRequest {
  ticker: string
  weights?: AgentWeights
}

export interface BacktestRequest {
  ticker: string
  start_date: string // ISO date string
  end_date: string // ISO date string
  initial_capital: number
}

// ============================================
// AI Backtest Models
// ============================================

export type AnalysisInterval = 'daily' | 'weekly' | 'biweekly' | 'monthly'

export type BacktestStrategy = 'auto' | 'ai_live' | 'ai_simulated' | 'sma_crossover' | 'macd' | 'buy_hold'

export interface AIBacktestRequest {
  ticker: string
  start_date: string
  end_date: string
  initial_capital: number
  strategy: BacktestStrategy
  analysis_interval: AnalysisInterval
  include_baseline_comparison: boolean
}

export interface AgentSignalSummary {
  score: number
  signal_type: SignalType
  confidence: number
  key_factors: string[]
}

export interface AISignalRecord {
  date: string
  recommendation: Recommendation
  confidence: number
  weighted_score: number
  agent_signals: Record<string, AgentSignalSummary>
  reasoning: string
  position_action: 'ENTER' | 'EXIT' | 'HOLD'
}

export interface StrategyMetrics {
  total_return: number
  sharpe_ratio: number
  max_drawdown: number
}

export interface BaselineComparison {
  buy_hold: StrategyMetrics
  sma_crossover?: StrategyMetrics
  macd?: StrategyMetrics
  ai_outperformance: number
}

export interface AIBacktestResponse {
  ticker: string
  strategy_name: string
  total_return: number
  sharpe_ratio: number
  max_drawdown: number
  win_rate: number
  total_trades: number
  trades: BacktestTrade[]
  equity_curve: EquityPoint[]
  ai_signals: AISignalRecord[]
  baseline_comparison: BaselineComparison
  warnings: string[]
}

// ============================================
// Response Models
// ============================================

export interface AnalyzeResponse {
  ticker: string
  recommendation: Recommendation
  confidence_score: number
  signals: AgentSignal[]
  synthesis: SynthesisResult
  timestamp: string
  cached: boolean
}

export interface BacktestTrade {
  date: string
  action: 'BUY' | 'SELL'
  price: number
  shares: number
  value: number
}

export interface EquityPoint {
  date: string
  equity: number
  returns: number
}

export interface BacktestResponse {
  ticker: string
  total_return: number
  sharpe_ratio: number
  max_drawdown: number
  trades: BacktestTrade[]
  equity_curve: EquityPoint[]
}

export interface ServiceStatus {
  name: string
  status: 'healthy' | 'unhealthy' | 'unknown'
  latency_ms?: number
  message?: string
}

export interface HealthResponse {
  status: string
  version: string
  timestamp: string
  services: Record<string, ServiceStatus>
}

// ============================================
// Error Models
// ============================================

export type ErrorCode =
  | 'INVALID_TICKER'
  | 'AGENT_TIMEOUT'
  | 'EXTERNAL_API_ERROR'
  | 'RATE_LIMIT_EXCEEDED'
  | 'MISTRAL_UNAVAILABLE'
  | 'NEO4J_ERROR'
  | 'VALIDATION_ERROR'
  | 'INTERNAL_ERROR'
  | 'NETWORK_ERROR'

export interface ErrorResponse {
  error_code: ErrorCode
  message: string
  details?: Record<string, unknown>
  timestamp: string
}


// ============================================
// Chat Models
// ============================================

export interface ChatSource {
  type: string
  title: string
  content: string
  timestamp?: string
}

export interface ChatMessage {
  role: 'user' | 'assistant'
  content: string
  sources?: ChatSource[]
  timestamp: string
}

export interface ChatResponse {
  message: ChatMessage
  question_type: string
  ticker?: string
  confidence: number
  sources: ChatSource[]
}
