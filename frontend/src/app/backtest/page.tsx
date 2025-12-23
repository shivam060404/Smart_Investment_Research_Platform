'use client'

import { useState, useEffect } from 'react'
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
  Legend,
  Scatter,
  ComposedChart,
} from 'recharts'
import api from '@/lib/api'
import {
  BacktestResponse,
  AIBacktestResponse,
  ErrorResponse,
  BacktestStrategy,
  AnalysisInterval,
} from '@/types'
import AISignalTimeline from '@/components/AISignalTimeline'
import PerformanceComparisonTable from '@/components/PerformanceComparisonTable'
import PerformanceSummaryCard from '@/components/PerformanceSummaryCard'

function isAIBacktestResponse(
  response: BacktestResponse | AIBacktestResponse
): response is AIBacktestResponse {
  return 'ai_signals' in response && 'baseline_comparison' in response
}

export default function BacktestPage() {
  const [ticker, setTicker] = useState('')
  const [startDate, setStartDate] = useState('')
  const [endDate, setEndDate] = useState('')
  const [initialCapital, setInitialCapital] = useState('10000')
  const [strategy, setStrategy] = useState<BacktestStrategy>('auto')
  const [analysisInterval, setAnalysisInterval] = useState<AnalysisInterval>('weekly')
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [result, setResult] = useState<BacktestResponse | AIBacktestResponse | null>(null)

  useEffect(() => {
    const end = new Date()
    const start = new Date()
    start.setFullYear(start.getFullYear() - 1)
    setStartDate(start.toISOString().split('T')[0])
    setEndDate(end.toISOString().split('T')[0])
  }, [])

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!ticker.trim()) {
      setError('Please enter a ticker symbol')
      return
    }
    setIsLoading(true)
    setError(null)
    try {
      const data = await api.runBacktest(
        ticker.toUpperCase(),
        startDate,
        endDate,
        parseFloat(initialCapital) || 10000,
        strategy,
        analysisInterval
      )
      setResult(data)
    } catch (err) {
      const errorResponse = err as ErrorResponse
      setError(errorResponse.message || 'Failed to run backtest')
    } finally {
      setIsLoading(false)
    }
  }

  const formatCurrency = (value: number) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 0,
      maximumFractionDigits: 0,
    }).format(value)
  }

  const formatPercent = (value: number) => {
    const sign = value >= 0 ? '+' : ''
    return `${sign}${value.toFixed(2)}%`
  }

  const prepareChartData = () => {
    if (!result) return []
    if (isAIBacktestResponse(result) && result.baseline_comparison) {
      const aiEquity = result.equity_curve
      const initialValue = parseFloat(initialCapital) || 10000
      const buyHoldReturn = result.baseline_comparison.buy_hold.total_return / 100
      return aiEquity.map((point, index) => {
        const progress = index / (aiEquity.length - 1 || 1)
        const buyHoldEquity = initialValue * (1 + buyHoldReturn * progress)
        const signal = result.ai_signals?.find((s) => s.date === point.date)
        return {
          date: point.date,
          aiEquity: point.equity,
          buyHoldEquity: buyHoldEquity,
          signal: signal?.recommendation,
          signalAction: signal?.position_action,
        }
      })
    }
    return result.equity_curve.map((point) => ({
      date: point.date,
      equity: point.equity,
    }))
  }

  const chartData = prepareChartData()
  const isAIResult = result && isAIBacktestResponse(result)

  return (
    <div className="space-y-8">
      <div>
        <h1 className="text-4xl font-bold text-white">Backtesting</h1>
        <p className="text-lg text-gray-400 mt-2">
          Test trading strategies against historical market data
        </p>
      </div>

      <div className="glass-card p-8">
        <form onSubmit={handleSubmit} className="space-y-6">
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            <div>
              <label htmlFor="ticker" className="block text-base font-medium text-gray-300 mb-2">
                Stock Ticker
              </label>
              <input
                id="ticker"
                type="text"
                value={ticker}
                onChange={(e) => setTicker(e.target.value.toUpperCase())}
                placeholder="e.g., AAPL"
                className="input-glass text-lg"
              />
            </div>
            <div>
              <label htmlFor="startDate" className="block text-base font-medium text-gray-300 mb-2">
                Start Date
              </label>
              <input
                id="startDate"
                type="date"
                value={startDate}
                onChange={(e) => setStartDate(e.target.value)}
                className="input-glass text-lg"
              />
            </div>
            <div>
              <label htmlFor="endDate" className="block text-base font-medium text-gray-300 mb-2">
                End Date
              </label>
              <input
                id="endDate"
                type="date"
                value={endDate}
                onChange={(e) => setEndDate(e.target.value)}
                className="input-glass text-lg"
              />
            </div>
            <div>
              <label htmlFor="capital" className="block text-base font-medium text-gray-300 mb-2">
                Initial Capital ($)
              </label>
              <input
                id="capital"
                type="number"
                value={initialCapital}
                onChange={(e) => setInitialCapital(e.target.value)}
                min="1000"
                step="1000"
                className="input-glass text-lg"
              />
            </div>
            <div>
              <label htmlFor="strategy" className="block text-base font-medium text-gray-300 mb-2">
                Strategy
              </label>
              <select
                id="strategy"
                value={strategy}
                onChange={(e) => setStrategy(e.target.value as BacktestStrategy)}
                className="input-glass text-lg"
              >
                <option value="auto">Auto Select</option>
                <option value="ai_live">AI Multi-Agent (Live)</option>
                <option value="ai_simulated">AI Simulated</option>
                <option value="sma_crossover">SMA Crossover</option>
                <option value="macd">MACD</option>
                <option value="buy_hold">Buy & Hold</option>
              </select>
            </div>
            {(strategy === 'ai_live' || strategy === 'ai_simulated') && (
              <div>
                <label htmlFor="interval" className="block text-base font-medium text-gray-300 mb-2">
                  Analysis Interval
                </label>
                <select
                  id="interval"
                  value={analysisInterval}
                  onChange={(e) => setAnalysisInterval(e.target.value as AnalysisInterval)}
                  className="input-glass text-lg"
                >
                  <option value="daily">Daily</option>
                  <option value="weekly">Weekly (Recommended)</option>
                  <option value="biweekly">Bi-weekly</option>
                  <option value="monthly">Monthly</option>
                </select>
              </div>
            )}
          </div>

          <div className="p-4 rounded-xl bg-white/5 border border-white/10">
            {strategy === 'ai_live' ? (
              <p className="text-base text-gray-400">
                <span className="text-purple-400 font-medium">AI Multi-Agent (Live):</span> Uses the
                real multi-agent system (Fundamental, Technical, Sentiment, Risk agents) to generate
                trading signals at each analysis interval.
              </p>
            ) : strategy === 'ai_simulated' ? (
              <p className="text-base text-gray-400">
                <span className="text-purple-400 font-medium">AI Simulated:</span> Uses a simulated
                AI strategy for faster backtesting without calling the actual agents.
              </p>
            ) : (
              <p className="text-base text-gray-400">
                <span className="text-purple-400 font-medium">Strategy:</span> The system
                automatically selects the best strategy based on data availability.
              </p>
            )}
          </div>
          <div className="flex justify-end">
            <button type="submit" disabled={isLoading} className="btn-primary text-lg px-8 py-4 disabled:opacity-50">
              {isLoading ? (
                <span className="flex items-center gap-3">
                  <svg className="animate-spin w-5 h-5" viewBox="0 0 24 24">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
                  </svg>
                  {strategy === 'ai_live' ? 'Running AI Backtest...' : 'Running Backtest...'}
                </span>
              ) : (
                'Run Backtest'
              )}
            </button>
          </div>
        </form>
      </div>

      {error && (
        <div className="glass-card border-red-500/30 p-6">
          <div className="flex items-start gap-4">
            <div className="w-12 h-12 rounded-xl bg-red-500/20 flex items-center justify-center flex-shrink-0">
              <svg className="w-6 h-6 text-red-400" fill="currentColor" viewBox="0 0 20 20">
                <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7 4a1 1 0 11-2 0 1 1 0 012 0zm-1-9a1 1 0 00-1 1v4a1 1 0 102 0V6a1 1 0 00-1-1z" clipRule="evenodd" />
              </svg>
            </div>
            <div>
              <h3 className="text-lg font-semibold text-white mb-2">Backtest Failed</h3>
              <p className="text-gray-400">{error}</p>
            </div>
          </div>
        </div>
      )}

      {isAIResult && result.warnings && result.warnings.length > 0 && (
        <div className="glass-card border-yellow-500/30 p-6">
          <div className="flex items-start gap-4">
            <div className="w-12 h-12 rounded-xl bg-yellow-500/20 flex items-center justify-center flex-shrink-0">
              <svg className="w-6 h-6 text-yellow-400" fill="currentColor" viewBox="0 0 20 20">
                <path fillRule="evenodd" d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z" clipRule="evenodd" />
              </svg>
            </div>
            <div>
              <h3 className="text-lg font-semibold text-white mb-2">Warnings</h3>
              <ul className="text-gray-400 space-y-1">
                {result.warnings.map((warning, index) => (
                  <li key={index}>• {warning}</li>
                ))}
              </ul>
            </div>
          </div>
        </div>
      )}

      {result && (
        <div className="space-y-6">
          {isAIResult && (
            <PerformanceSummaryCard
              totalReturn={result.total_return}
              sharpeRatio={result.sharpe_ratio}
              maxDrawdown={result.max_drawdown}
              winRate={result.win_rate}
              totalTrades={result.total_trades}
            />
          )}

          {!isAIResult && (
            <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
              <MetricCard label="Total Return" value={formatPercent(result.total_return)} isPositive={result.total_return >= 0} />
              <MetricCard label="Sharpe Ratio" value={result.sharpe_ratio.toFixed(2)} isPositive={result.sharpe_ratio >= 1} />
              <MetricCard label="Max Drawdown" value={formatPercent(-Math.abs(result.max_drawdown))} isPositive={false} />
              <MetricCard label="Total Trades" value={result.trades.length.toString()} isNeutral />
            </div>
          )}

          <div className="glass-card p-6">
            <h3 className="text-xl font-semibold text-white mb-6">
              {isAIResult ? 'Equity Curve Comparison' : 'Equity Curve'}
            </h3>
            <div className="h-96">
              <ResponsiveContainer width="100%" height="100%">
                {isAIResult ? (
                  <ComposedChart data={chartData} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
                    <XAxis
                      dataKey="date"
                      tick={{ fontSize: 12, fill: '#9ca3af' }}
                      tickFormatter={(value) => new Date(value).toLocaleDateString('en-US', { month: 'short', year: '2-digit' })}
                      stroke="rgba(255,255,255,0.2)"
                    />
                    <YAxis
                      tick={{ fontSize: 12, fill: '#9ca3af' }}
                      tickFormatter={(value) => `${(value / 1000).toFixed(0)}k`}
                      stroke="rgba(255,255,255,0.2)"
                    />
                    <Tooltip
                      formatter={(value: number, name: string) => [formatCurrency(value), name === 'aiEquity' ? 'AI Strategy' : 'Buy & Hold']}
                      labelFormatter={(label) => new Date(label).toLocaleDateString()}
                      contentStyle={{ borderRadius: '12px', border: '1px solid rgba(255,255,255,0.1)', backgroundColor: 'rgba(0,0,0,0.8)', color: '#fff' }}
                    />
                    <Legend formatter={(value) => (value === 'aiEquity' ? 'AI Strategy' : 'Buy & Hold')} />
                    <ReferenceLine y={parseFloat(initialCapital)} stroke="#9ca3af" strokeDasharray="5 5" label={{ value: 'Initial', position: 'right', fill: '#9ca3af', fontSize: 12 }} />
                    <Line type="monotone" dataKey="aiEquity" stroke="#a855f7" strokeWidth={2} dot={false} name="aiEquity" />
                    <Line type="monotone" dataKey="buyHoldEquity" stroke="#6b7280" strokeWidth={2} dot={false} strokeDasharray="5 5" name="buyHoldEquity" />
                    <Scatter dataKey="aiEquity" data={chartData.filter((d) => d.signalAction === 'ENTER')} fill="#22c55e" shape="triangle" />
                    <Scatter dataKey="aiEquity" data={chartData.filter((d) => d.signalAction === 'EXIT')} fill="#ef4444" shape="triangle" />
                  </ComposedChart>
                ) : (
                  <LineChart data={result.equity_curve} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
                    <XAxis
                      dataKey="date"
                      tick={{ fontSize: 12, fill: '#9ca3af' }}
                      tickFormatter={(value) => new Date(value).toLocaleDateString('en-US', { month: 'short', year: '2-digit' })}
                      stroke="rgba(255,255,255,0.2)"
                    />
                    <YAxis
                      tick={{ fontSize: 12, fill: '#9ca3af' }}
                      tickFormatter={(value) => `${(value / 1000).toFixed(0)}k`}
                      stroke="rgba(255,255,255,0.2)"
                    />
                    <Tooltip
                      formatter={(value: number) => [formatCurrency(value), 'Portfolio Value']}
                      labelFormatter={(label) => new Date(label).toLocaleDateString()}
                      contentStyle={{ borderRadius: '12px', border: '1px solid rgba(255,255,255,0.1)', backgroundColor: 'rgba(0,0,0,0.8)', color: '#fff' }}
                    />
                    <ReferenceLine y={parseFloat(initialCapital)} stroke="#9ca3af" strokeDasharray="5 5" label={{ value: 'Initial', position: 'right', fill: '#9ca3af', fontSize: 12 }} />
                    <Line type="monotone" dataKey="equity" stroke={result.total_return >= 0 ? '#22c55e' : '#ef4444'} strokeWidth={2} dot={false} />
                  </LineChart>
                )}
              </ResponsiveContainer>
            </div>
          </div>

          {isAIResult && result.ai_signals && result.ai_signals.length > 0 && (
            <AISignalTimeline signals={result.ai_signals} />
          )}

          {isAIResult && result.baseline_comparison && (
            <PerformanceComparisonTable
              aiMetrics={{
                total_return: result.total_return,
                sharpe_ratio: result.sharpe_ratio,
                max_drawdown: result.max_drawdown,
                win_rate: result.win_rate,
                total_trades: result.total_trades,
              }}
              baselineComparison={result.baseline_comparison}
            />
          )}

          <div className="glass-card overflow-hidden">
            <div className="px-6 py-5 border-b border-white/10">
              <h3 className="text-xl font-semibold text-white">Trade History</h3>
            </div>
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead className="bg-white/5">
                  <tr>
                    <th className="px-6 py-4 text-left text-sm font-medium text-gray-400 uppercase tracking-wider">Date</th>
                    <th className="px-6 py-4 text-left text-sm font-medium text-gray-400 uppercase tracking-wider">Action</th>
                    <th className="px-6 py-4 text-right text-sm font-medium text-gray-400 uppercase tracking-wider">Price</th>
                    <th className="px-6 py-4 text-right text-sm font-medium text-gray-400 uppercase tracking-wider">Shares</th>
                    <th className="px-6 py-4 text-right text-sm font-medium text-gray-400 uppercase tracking-wider">Value</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-white/5">
                  {result.trades.length === 0 ? (
                    <tr>
                      <td colSpan={5} className="px-6 py-10 text-center text-gray-500 text-base">
                        No trades executed during this period (Buy & Hold strategy)
                      </td>
                    </tr>
                  ) : (
                    result.trades.map((trade, index) => (
                      <tr key={index} className="hover:bg-white/5">
                        <td className="px-6 py-4 whitespace-nowrap text-base text-white">
                          {new Date(trade.date).toLocaleDateString()}
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap">
                          <span className={`inline-flex px-3 py-1.5 text-sm font-semibold rounded-full ${trade.action === 'BUY' ? 'bg-emerald-500/20 text-emerald-400' : 'bg-red-500/20 text-red-400'}`}>
                            {trade.action}
                          </span>
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap text-base text-white text-right">${trade.price.toFixed(2)}</td>
                        <td className="px-6 py-4 whitespace-nowrap text-base text-white text-right">{trade.shares.toFixed(2)}</td>
                        <td className="px-6 py-4 whitespace-nowrap text-base text-white text-right">{formatCurrency(trade.value)}</td>
                      </tr>
                    ))
                  )}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}

      {!result && !isLoading && !error && (
        <div className="glass-card p-16 text-center">
          <svg className="w-20 h-20 text-gray-600 mx-auto mb-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
          </svg>
          <h3 className="text-2xl font-semibold text-white mb-3">Run Your First Backtest</h3>
          <p className="text-lg text-gray-400 max-w-lg mx-auto">
            Enter a stock ticker and date range above to see how trading strategies would have performed historically.
            Try the AI Multi-Agent strategy for intelligent signal generation.
          </p>
        </div>
      )}
    </div>
  )
}

function MetricCard({ label, value, isPositive, isNeutral }: { label: string; value: string; isPositive?: boolean; isNeutral?: boolean }) {
  const colorClass = isNeutral ? 'text-white' : isPositive ? 'text-emerald-400' : 'text-red-400'
  return (
    <div className="glass-card p-6">
      <p className="text-base text-gray-400 mb-2">{label}</p>
      <p className={`text-3xl font-bold ${colorClass}`}>{value}</p>
    </div>
  )
}
