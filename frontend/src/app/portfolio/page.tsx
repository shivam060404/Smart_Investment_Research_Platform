'use client'

import { useState, useEffect, useCallback } from 'react'
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell,
} from 'recharts'

// Portfolio position interface
interface Position {
  id: string
  ticker: string
  name: string
  shares: number
  avgCost: number
  currentPrice: number
  recommendation?: 'BUY' | 'SELL' | 'HOLD'
  lastUpdated?: string
}

// Portfolio return data point
interface ReturnPoint {
  date: string
  value: number
  return: number
}

// Local storage key
const PORTFOLIO_KEY = 'ai-invest-portfolio'

// Colors for pie chart
const COLORS = ['#8b5cf6', '#06b6d4', '#f59e0b', '#ec4899', '#22c55e', '#3b82f6', '#ef4444', '#14b8a6']

// Fetch real stock price from Yahoo Finance via backend proxy or direct API
async function fetchStockPrice(ticker: string): Promise<number | null> {
  try {
    // Try to fetch from our backend API first
    const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'}/api/stock/${ticker}/price`, {
      method: 'GET',
      headers: { 'Content-Type': 'application/json' },
    })
    
    if (response.ok) {
      const data = await response.json()
      return data.price
    }
    
    // Fallback: Use Yahoo Finance chart API (CORS-friendly)
    const yahooResponse = await fetch(
      `https://query1.finance.yahoo.com/v8/finance/chart/${ticker}?interval=1d&range=1d`
    )
    
    if (yahooResponse.ok) {
      const data = await yahooResponse.json()
      const price = data?.chart?.result?.[0]?.meta?.regularMarketPrice
      if (price) return price
    }
    
    return null
  } catch (error) {
    console.error(`Failed to fetch price for ${ticker}:`, error)
    return null
  }
}

export default function PortfolioPage() {
  const [positions, setPositions] = useState<Position[]>([])
  const [showAddModal, setShowAddModal] = useState(false)
  const [isRefreshing, setIsRefreshing] = useState(false)
  const [lastRefresh, setLastRefresh] = useState<Date | null>(null)
  const [newPosition, setNewPosition] = useState({
    ticker: '',
    shares: '',
    avgCost: '',
  })

  // Load positions from localStorage
  useEffect(() => {
    const stored = localStorage.getItem(PORTFOLIO_KEY)
    if (stored) {
      try {
        setPositions(JSON.parse(stored))
      } catch {
        localStorage.removeItem(PORTFOLIO_KEY)
      }
    }
  }, [])

  // Save positions to localStorage
  useEffect(() => {
    if (positions.length > 0) {
      localStorage.setItem(PORTFOLIO_KEY, JSON.stringify(positions))
    }
  }, [positions])

  // Refresh all stock prices
  const refreshPrices = useCallback(async () => {
    if (positions.length === 0) return
    
    setIsRefreshing(true)
    
    const updatedPositions = await Promise.all(
      positions.map(async (position) => {
        const price = await fetchStockPrice(position.ticker)
        return {
          ...position,
          currentPrice: price ?? position.currentPrice,
          lastUpdated: new Date().toISOString(),
        }
      })
    )
    
    setPositions(updatedPositions)
    setLastRefresh(new Date())
    setIsRefreshing(false)
  }, [positions])

  // Auto-refresh prices on mount and every 5 minutes
  useEffect(() => {
    if (positions.length > 0) {
      refreshPrices()
      
      const interval = setInterval(refreshPrices, 5 * 60 * 1000) // 5 minutes
      return () => clearInterval(interval)
    }
  }, [positions.length]) // Only re-run when position count changes

  // Calculate portfolio metrics
  const totalValue = positions.reduce((sum, p) => sum + p.shares * p.currentPrice, 0)
  const totalCost = positions.reduce((sum, p) => sum + p.shares * p.avgCost, 0)
  const totalReturn = totalCost > 0 ? ((totalValue - totalCost) / totalCost) * 100 : 0
  const totalGain = totalValue - totalCost

  // Generate mock cumulative returns data
  const generateReturnsData = (): ReturnPoint[] => {
    const data: ReturnPoint[] = []
    const startValue = totalCost || 10000
    let currentValue = startValue
    
    for (let i = 30; i >= 0; i--) {
      const date = new Date()
      date.setDate(date.getDate() - i)
      
      // Simulate daily returns
      const dailyReturn = (Math.random() - 0.48) * 2 // Slight positive bias
      currentValue = currentValue * (1 + dailyReturn / 100)
      
      data.push({
        date: date.toISOString().split('T')[0],
        value: currentValue,
        return: ((currentValue - startValue) / startValue) * 100,
      })
    }
    
    // Adjust final value to match actual portfolio
    if (data.length > 0 && totalValue > 0) {
      data[data.length - 1].value = totalValue
      data[data.length - 1].return = totalReturn
    }
    
    return data
  }

  const returnsData = generateReturnsData()

  // Pie chart data
  const pieData = positions.map((p) => ({
    name: p.ticker,
    value: p.shares * p.currentPrice,
  }))

  // Add new position with real price fetch
  const handleAddPosition = async (e: React.FormEvent) => {
    e.preventDefault()
    
    if (!newPosition.ticker || !newPosition.shares || !newPosition.avgCost) return

    // Fetch real current price
    const realPrice = await fetchStockPrice(newPosition.ticker.toUpperCase())
    
    const position: Position = {
      id: Date.now().toString(),
      ticker: newPosition.ticker.toUpperCase(),
      name: newPosition.ticker.toUpperCase(),
      shares: parseFloat(newPosition.shares),
      avgCost: parseFloat(newPosition.avgCost),
      currentPrice: realPrice ?? parseFloat(newPosition.avgCost), // Fallback to avg cost if fetch fails
      lastUpdated: new Date().toISOString(),
    }

    setPositions([...positions, position])
    setNewPosition({ ticker: '', shares: '', avgCost: '' })
    setShowAddModal(false)
  }

  // Remove position
  const handleRemovePosition = (id: string) => {
    setPositions(positions.filter((p) => p.id !== id))
  }

  // Format currency
  const formatCurrency = (value: number) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 2,
    }).format(value)
  }

  // Format percentage
  const formatPercent = (value: number) => {
    const sign = value >= 0 ? '+' : ''
    return `${sign}${value.toFixed(2)}%`
  }

  return (
    <div className="space-y-8">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-white">Portfolio Tracker</h1>
          <p className="text-slate-400 mt-1">Track your positions and performance</p>
          {lastRefresh && (
            <p className="text-xs text-slate-500 mt-1">
              Last updated: {lastRefresh.toLocaleTimeString()}
            </p>
          )}
        </div>
        <div className="flex gap-3">
          {positions.length > 0 && (
            <button
              onClick={refreshPrices}
              disabled={isRefreshing}
              className="px-4 py-2 bg-slate-700 text-white rounded-lg hover:bg-slate-600 transition-colors flex items-center gap-2 disabled:opacity-50"
            >
              <svg 
                className={`w-5 h-5 ${isRefreshing ? 'animate-spin' : ''}`} 
                fill="none" 
                stroke="currentColor" 
                viewBox="0 0 24 24"
              >
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
              </svg>
              {isRefreshing ? 'Refreshing...' : 'Refresh Prices'}
            </button>
          )}
          <button
            onClick={() => setShowAddModal(true)}
            className="px-4 py-2 bg-primary-600 text-white rounded-lg hover:bg-primary-700 transition-colors flex items-center gap-2"
          >
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
            </svg>
            Add Position
          </button>
        </div>
      </div>

      {/* Portfolio Summary */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <SummaryCard
          label="Total Value"
          value={formatCurrency(totalValue)}
          isNeutral
        />
        <SummaryCard
          label="Total Cost"
          value={formatCurrency(totalCost)}
          isNeutral
        />
        <SummaryCard
          label="Total Return"
          value={formatPercent(totalReturn)}
          isPositive={totalReturn >= 0}
        />
        <SummaryCard
          label="Total Gain/Loss"
          value={formatCurrency(totalGain)}
          isPositive={totalGain >= 0}
        />
      </div>

      {positions.length > 0 ? (
        <>
          {/* Charts Row */}
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            {/* Cumulative Returns Chart */}
            <div className="lg:col-span-2 bg-white rounded-xl border border-gray-200 p-6">
              <h3 className="text-lg font-semibold text-gray-900 mb-4">Cumulative Returns</h3>
              <div className="h-64">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={returnsData} margin={{ top: 5, right: 20, left: 10, bottom: 5 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                    <XAxis
                      dataKey="date"
                      tick={{ fontSize: 11 }}
                      tickFormatter={(value) => new Date(value).toLocaleDateString('en-US', { month: 'short', day: 'numeric' })}
                    />
                    <YAxis
                      tick={{ fontSize: 11 }}
                      tickFormatter={(value) => `${value.toFixed(0)}%`}
                    />
                    <Tooltip
                      formatter={(value: number) => [`${value.toFixed(2)}%`, 'Return']}
                      labelFormatter={(label) => new Date(label).toLocaleDateString()}
                      contentStyle={{ borderRadius: '8px', border: '1px solid #e5e7eb' }}
                    />
                    <Line
                      type="monotone"
                      dataKey="return"
                      stroke={totalReturn >= 0 ? '#22c55e' : '#ef4444'}
                      strokeWidth={2}
                      dot={false}
                    />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </div>

            {/* Allocation Pie Chart */}
            <div className="bg-white rounded-xl border border-gray-200 p-6">
              <h3 className="text-lg font-semibold text-gray-900 mb-4">Allocation</h3>
              <div className="h-64">
                <ResponsiveContainer width="100%" height="100%">
                  <PieChart>
                    <Pie
                      data={pieData}
                      cx="50%"
                      cy="50%"
                      innerRadius={50}
                      outerRadius={80}
                      paddingAngle={2}
                      dataKey="value"
                      label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
                      labelLine={false}
                    >
                      {pieData.map((_, index) => (
                        <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                      ))}
                    </Pie>
                    <Tooltip
                      formatter={(value: number) => [formatCurrency(value), 'Value']}
                      contentStyle={{ borderRadius: '8px', border: '1px solid #e5e7eb' }}
                    />
                  </PieChart>
                </ResponsiveContainer>
              </div>
            </div>
          </div>

          {/* Positions Table */}
          <div className="bg-white rounded-xl border border-gray-200 overflow-hidden">
            <div className="px-6 py-4 border-b border-gray-100">
              <h3 className="text-lg font-semibold text-gray-900">Positions</h3>
            </div>
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead className="bg-gray-50">
                  <tr>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Ticker
                    </th>
                    <th className="px-6 py-3 text-right text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Shares
                    </th>
                    <th className="px-6 py-3 text-right text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Avg Cost
                    </th>
                    <th className="px-6 py-3 text-right text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Current Price
                    </th>
                    <th className="px-6 py-3 text-right text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Market Value
                    </th>
                    <th className="px-6 py-3 text-right text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Gain/Loss
                    </th>
                    <th className="px-6 py-3 text-right text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Return
                    </th>
                    <th className="px-6 py-3 text-center text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Actions
                    </th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-gray-100">
                  {positions.map((position) => {
                    const marketValue = position.shares * position.currentPrice
                    const costBasis = position.shares * position.avgCost
                    const gain = marketValue - costBasis
                    const returnPct = ((marketValue - costBasis) / costBasis) * 100

                    return (
                      <tr key={position.id} className="hover:bg-gray-50">
                        <td className="px-6 py-4 whitespace-nowrap">
                          <span className="font-semibold text-gray-900">{position.ticker}</span>
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900 text-right">
                          {position.shares}
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900 text-right">
                          {formatCurrency(position.avgCost)}
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900 text-right">
                          {formatCurrency(position.currentPrice)}
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900 text-right font-medium">
                          {formatCurrency(marketValue)}
                        </td>
                        <td className={`px-6 py-4 whitespace-nowrap text-sm text-right font-medium ${gain >= 0 ? 'text-bullish' : 'text-bearish'}`}>
                          {formatCurrency(gain)}
                        </td>
                        <td className={`px-6 py-4 whitespace-nowrap text-sm text-right font-medium ${returnPct >= 0 ? 'text-bullish' : 'text-bearish'}`}>
                          {formatPercent(returnPct)}
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap text-center">
                          <button
                            onClick={() => handleRemovePosition(position.id)}
                            className="text-slate-400 hover:text-bearish transition-colors"
                            title="Remove position"
                          >
                            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                            </svg>
                          </button>
                        </td>
                      </tr>
                    )
                  })}
                </tbody>
              </table>
            </div>
          </div>
        </>
      ) : (
        /* Empty State */
        <div className="bg-slate-800/50 border border-slate-700 rounded-xl p-12 text-center">
          <svg className="w-16 h-16 text-slate-600 mx-auto mb-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10" />
          </svg>
          <h3 className="text-xl font-semibold text-white mb-2">No Positions Yet</h3>
          <p className="text-slate-400 max-w-md mx-auto mb-6">
            Start tracking your portfolio by adding your first position. 
            You can track multiple stocks and see your overall performance.
          </p>
          <button
            onClick={() => setShowAddModal(true)}
            className="px-6 py-2 bg-primary-600 text-white rounded-lg hover:bg-primary-700 transition-colors"
          >
            Add Your First Position
          </button>
        </div>
      )}

      {/* Add Position Modal */}
      {showAddModal && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
          <div className="bg-white rounded-xl p-6 w-full max-w-md mx-4">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-lg font-semibold text-gray-900">Add Position</h3>
              <button
                onClick={() => setShowAddModal(false)}
                className="text-gray-400 hover:text-gray-600"
              >
                <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                </svg>
              </button>
            </div>
            <form onSubmit={handleAddPosition} className="space-y-4">
              <div>
                <label htmlFor="modal-ticker" className="block text-sm font-medium text-gray-700 mb-1">
                  Ticker Symbol
                </label>
                <input
                  id="modal-ticker"
                  type="text"
                  value={newPosition.ticker}
                  onChange={(e) => setNewPosition({ ...newPosition, ticker: e.target.value.toUpperCase() })}
                  placeholder="e.g., AAPL"
                  className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:outline-none focus:border-primary-500"
                  required
                />
              </div>
              <div>
                <label htmlFor="modal-shares" className="block text-sm font-medium text-gray-700 mb-1">
                  Number of Shares
                </label>
                <input
                  id="modal-shares"
                  type="number"
                  value={newPosition.shares}
                  onChange={(e) => setNewPosition({ ...newPosition, shares: e.target.value })}
                  placeholder="e.g., 10"
                  min="0.01"
                  step="0.01"
                  className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:outline-none focus:border-primary-500"
                  required
                />
              </div>
              <div>
                <label htmlFor="modal-cost" className="block text-sm font-medium text-gray-700 mb-1">
                  Average Cost per Share ($)
                </label>
                <input
                  id="modal-cost"
                  type="number"
                  value={newPosition.avgCost}
                  onChange={(e) => setNewPosition({ ...newPosition, avgCost: e.target.value })}
                  placeholder="e.g., 150.00"
                  min="0.01"
                  step="0.01"
                  className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:outline-none focus:border-primary-500"
                  required
                />
              </div>
              <div className="flex gap-3 pt-2">
                <button
                  type="button"
                  onClick={() => setShowAddModal(false)}
                  className="flex-1 px-4 py-2 border border-gray-300 text-gray-700 rounded-lg hover:bg-gray-50 transition-colors"
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  className="flex-1 px-4 py-2 bg-primary-600 text-white rounded-lg hover:bg-primary-700 transition-colors"
                >
                  Add Position
                </button>
              </div>
            </form>
          </div>
        </div>
      )}
    </div>
  )
}

// Summary Card Component
function SummaryCard({
  label,
  value,
  isPositive,
  isNeutral,
}: {
  label: string
  value: string
  isPositive?: boolean
  isNeutral?: boolean
}) {
  const colorClass = isNeutral
    ? 'text-white'
    : isPositive
    ? 'text-bullish'
    : 'text-bearish'

  return (
    <div className="bg-slate-800 rounded-xl p-4">
      <p className="text-sm text-slate-400 mb-1">{label}</p>
      <p className={`text-2xl font-bold ${colorClass}`}>{value}</p>
    </div>
  )
}
