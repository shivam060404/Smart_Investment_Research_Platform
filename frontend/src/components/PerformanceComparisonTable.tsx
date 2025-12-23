'use client'

import { BaselineComparison, StrategyMetrics } from '@/types'

interface PerformanceComparisonTableProps {
  aiMetrics: StrategyMetrics & { win_rate?: number; total_trades?: number }
  baselineComparison: BaselineComparison
}

export default function PerformanceComparisonTable({
  aiMetrics,
  baselineComparison,
}: PerformanceComparisonTableProps) {
  const formatPercent = (value: number) => {
    const sign = value >= 0 ? '+' : ''
    return `${sign}${value.toFixed(2)}%`
  }

  const getComparisonClass = (aiValue: number, baselineValue: number) => {
    if (aiValue > baselineValue) return 'text-emerald-400'
    if (aiValue < baselineValue) return 'text-red-400'
    return 'text-gray-400'
  }

  const strategies = [
    { name: 'AI Multi-Agent', metrics: aiMetrics, isAI: true },
    { name: 'Buy & Hold', metrics: baselineComparison.buy_hold, isAI: false },
    ...(baselineComparison.sma_crossover
      ? [{ name: 'SMA Crossover', metrics: baselineComparison.sma_crossover, isAI: false }]
      : []),
    ...(baselineComparison.macd
      ? [{ name: 'MACD', metrics: baselineComparison.macd, isAI: false }]
      : []),
  ]

  return (
    <div className="glass-card overflow-hidden">
      <div className="px-6 py-5 border-b border-white/10">
        <h3 className="text-xl font-semibold text-white">Strategy Comparison</h3>
        <p className="text-sm text-gray-400 mt-1">
          AI outperformance vs Buy & Hold:{' '}
          <span
            className={
              baselineComparison.ai_outperformance >= 0 ? 'text-emerald-400' : 'text-red-400'
            }
          >
            {formatPercent(baselineComparison.ai_outperformance)}
          </span>
        </p>
      </div>
      <div className="overflow-x-auto">
        <table className="w-full">
          <thead className="bg-white/5">
            <tr>
              <th className="px-6 py-4 text-left text-sm font-medium text-gray-400 uppercase tracking-wider">
                Strategy
              </th>
              <th className="px-6 py-4 text-right text-sm font-medium text-gray-400 uppercase tracking-wider">
                Return
              </th>
              <th className="px-6 py-4 text-right text-sm font-medium text-gray-400 uppercase tracking-wider">
                Sharpe
              </th>
              <th className="px-6 py-4 text-right text-sm font-medium text-gray-400 uppercase tracking-wider">
                Max DD
              </th>
            </tr>
          </thead>
          <tbody className="divide-y divide-white/5">
            {strategies.map((strategy, index) => (
              <tr
                key={strategy.name}
                className={`hover:bg-white/5 ${strategy.isAI ? 'bg-purple-500/10' : ''}`}
              >
                <td className="px-6 py-4 whitespace-nowrap">
                  <div className="flex items-center gap-2">
                    {strategy.isAI && (
                      <span className="w-2 h-2 rounded-full bg-purple-500"></span>
                    )}
                    <span className={`text-base ${strategy.isAI ? 'text-white font-medium' : 'text-gray-300'}`}>
                      {strategy.name}
                    </span>
                  </div>
                </td>
                <td className="px-6 py-4 whitespace-nowrap text-right">
                  <span
                    className={`text-base font-medium ${
                      strategy.isAI
                        ? getComparisonClass(
                            strategy.metrics.total_return,
                            baselineComparison.buy_hold.total_return
                          )
                        : strategy.metrics.total_return >= 0
                        ? 'text-emerald-400'
                        : 'text-red-400'
                    }`}
                  >
                    {formatPercent(strategy.metrics.total_return)}
                  </span>
                </td>
                <td className="px-6 py-4 whitespace-nowrap text-right">
                  <span
                    className={`text-base ${
                      strategy.metrics.sharpe_ratio >= 1 ? 'text-emerald-400' : 'text-gray-300'
                    }`}
                  >
                    {strategy.metrics.sharpe_ratio.toFixed(2)}
                  </span>
                </td>
                <td className="px-6 py-4 whitespace-nowrap text-right">
                  <span className="text-base text-red-400">
                    {formatPercent(-Math.abs(strategy.metrics.max_drawdown))}
                  </span>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}
