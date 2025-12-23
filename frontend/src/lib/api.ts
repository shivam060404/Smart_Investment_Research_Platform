import axios, { AxiosInstance, AxiosError } from 'axios'
import {
  AnalyzeResponse,
  BacktestResponse,
  AIBacktestResponse,
  HealthResponse,
  ErrorResponse,
  AgentWeights,
  ChatResponse,
  BacktestStrategy,
  AnalysisInterval,
} from '@/types'

/**
 * API client for the AI Investment Research Platform backend
 */
class ApiClient {
  private client: AxiosInstance

  constructor() {
    this.client = axios.create({
      baseURL: process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000',
      timeout: 120000, // 120 seconds for analysis requests (agents can take time)
      headers: {
        'Content-Type': 'application/json',
      },
    })

    // Add response interceptor for error handling
    this.client.interceptors.response.use(
      (response) => response,
      (error: AxiosError<ErrorResponse>) => {
        if (error.response?.data) {
          return Promise.reject(error.response.data)
        }
        return Promise.reject({
          error_code: 'NETWORK_ERROR',
          message: error.message || 'Network error occurred',
          timestamp: new Date().toISOString(),
        } as ErrorResponse)
      }
    )
  }

  /**
   * Analyze a stock ticker
   * POST /api/analyze/{ticker}
   */
  async analyzeStock(
    ticker: string,
    weights?: AgentWeights
  ): Promise<AnalyzeResponse> {
    const response = await this.client.post<AnalyzeResponse>(
      `/api/analyze/${ticker.toUpperCase()}`,
      weights ? { weights } : {}
    )
    return response.data
  }

  /**
   * Run backtesting on historical data
   * POST /api/backtest
   */
  async runBacktest(
    ticker: string,
    startDate: string,
    endDate: string,
    initialCapital: number = 10000,
    strategy: BacktestStrategy = 'auto',
    analysisInterval: AnalysisInterval = 'weekly'
  ): Promise<BacktestResponse | AIBacktestResponse> {
    const response = await this.client.post<BacktestResponse | AIBacktestResponse>(
      '/api/backtest',
      {
        ticker: ticker.toUpperCase(),
        start_date: startDate,
        end_date: endDate,
        initial_capital: initialCapital,
        strategy,
        analysis_interval: analysisInterval,
        include_baseline_comparison: true,
      }
    )
    return response.data
  }

  /**
   * Check system health status
   * GET /health
   */
  async getHealth(): Promise<HealthResponse> {
    const response = await this.client.get<HealthResponse>('/health')
    return response.data
  }

  /**
   * Send a message to the RAG chatbot
   * POST /api/chat
   */
  async sendChatMessage(message: string): Promise<ChatResponse> {
    const response = await this.client.post<ChatResponse>('/api/chat', {
      message,
    })
    return response.data
  }

  /**
   * Get chat suggestions
   * GET /api/chat/suggestions
   */
  async getChatSuggestions(ticker?: string): Promise<{ suggestions: string[] }> {
    const params = ticker ? { ticker } : {}
    const response = await this.client.get<{ suggestions: string[] }>(
      '/api/chat/suggestions',
      { params }
    )
    return response.data
  }

  /**
   * Set API key for authenticated requests
   */
  setApiKey(apiKey: string): void {
    this.client.defaults.headers.common['X-API-Key'] = apiKey
  }

  /**
   * Remove API key from requests
   */
  clearApiKey(): void {
    delete this.client.defaults.headers.common['X-API-Key']
  }
}

// Export singleton instance
const api = new ApiClient()
export default api

// Export class for testing purposes
export { ApiClient }
