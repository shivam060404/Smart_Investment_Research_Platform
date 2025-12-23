'use client'

interface PerformanceSummaryCardProps {
  totalReturn: number
  sharpeRatio: number
  maxDrawdown: number
  winRate: number
  totalTrades: number
}

export default function PerformanceSummaryCard({
  totalReturn,
  sharpeRatio,
  maxDrawdown,
  winRate,
  totalTrades,
}: PerformanceSummaryCardProps) {
  const formatPercent = (value: number) => {
    const sign = value >= 0 ? '+' : ''
    return `${sign}${value.toFixed(2)}%`
  }

  const getReturnQuality = (value: number) => {
    if (value >= 20) return { label: 'Excellent', color: 'text-emerald-400', bg: 'bg-emerald-500/20' }
    if (value >= 10) return { label: 'Good', color: 'text-green-400', bg: 'bg-green-500/20' }
    if (value >= 0) return { label: 'Positive', color: 'text-yellow-400', bg: 'bg-yellow-500/20' }
    return { label: 'Negative', color: 'text-red-400', bg: 'bg-red-500/20' }
  }

  const getSharpeQuality = (value: number) => {
    if (value >= 2) return { label: 'Excellent', color: 'text-emerald-400' }
    if (value >= 1) return { label: 'Good', color: 'text-green-400' }
    if (value >= 0.5) return { label: 'Moderate', color: 'text-yellow-400' }
    return { label: 'Poor', color: 'text-red-400' }
  }

  const getDrawdownQuality = (value: number) => {
    const absValue = Math.abs(value)
    if (absValue <= 10) return { label: 'Low Risk', color: 'text-emerald-400' }
    if (absValue <= 20) return { label: 'Moderate', color: 'text-yellow-400' }
    return { label: 'High Risk', color: 'text-red-400' }
  }

  const getWinRateQuality = (value: number) => {
    if (value >= 60) return { label: 'Strong', color: 'text-emerald-400' }
    if (value >= 50) return { label: 'Balanced', color: 'text-yellow-400' }
    return { label: 'Weak', color: 'text-red-400' }
  }

  const returnQuality = getReturnQuality(totalReturn)
  const sharpeQuality = getSharpeQuality(sharpeRatio)
  const drawdownQuality = getDrawdownQuality(maxDrawdown)
  const winRateQuality = getWinRateQuality(winRate)

  return (
    <div className="glass-card p-6">
      <div className="flex items-center justify-between mb-6">
        <h3 className="text-xl font-semibold text-white">AI Strategy Performance</h3>
        <span className={`px-3 py-1 rounded-full text-sm font-medium ${returnQuality.bg} ${returnQuality.color}`}>
          {returnQuality.label}
        </span>
      </div>

      <div className="grid grid-cols-2 lg:grid-cols-5 gap-4">
        {/* Total Return */}
        <div className="p-4 rounded-xl bg-white/5 border border-white/10">
          <p className="text-sm text-gray-400 mb-1">Total Return</p>
          <p className={`text-2xl font-bold ${totalReturn >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
            {formatPercent(totalReturn)}
          </p>
        </div>

        {/* Sharpe Ratio */}
        <div className="p-4 rounded-xl bg-white/5 border border-white/10">
          <p className="text-sm text-gray-400 mb-1">Sharpe Ratio</p>
          <p className={`text-2xl font-bold ${sharpeQuality.color}`}>
            {sharpeRatio.toFixed(2)}
          </p>
          <p className={`text-xs ${sharpeQuality.color}`}>{sharpeQuality.label}</p>
        </div>

        {/* Max Drawdown */}
        <div className="p-4 rounded-xl bg-white/5 border border-white/10">
          <p className="text-sm text-gray-400 mb-1">Max Drawdown</p>
          <p className="text-2xl font-bold text-red-400">
            {formatPercent(-Math.abs(maxDrawdown))}
          </p>
          <p className={`text-xs ${drawdownQuality.color}`}>{drawdownQuality.label}</p>
        </div>

        {/* Win Rate */}
        <div className="p-4 rounded-xl bg-white/5 border border-white/10">
          <p className="text-sm text-gray-400 mb-1">Win Rate</p>
          <p className={`text-2xl font-bold ${winRateQuality.color}`}>
            {winRate.toFixed(1)}%
          </p>
          <p className={`text-xs ${winRateQuality.color}`}>{winRateQuality.label}</p>
        </div>

        {/* Total Trades */}
        <div className="p-4 rounded-xl bg-white/5 border border-white/10">
          <p className="text-sm text-gray-400 mb-1">Total Trades</p>
          <p className="text-2xl font-bold text-white">{totalTrades}</p>
          <p className="text-xs text-gray-500">Executed</p>
        </div>
      </div>
    </div>
  )
}
