'use client'

import { useState } from 'react'
import { AISignalRecord, AgentSignalSummary } from '@/types'

interface AISignalTimelineProps {
  signals: AISignalRecord[]
}

export default function AISignalTimeline({ signals }: AISignalTimelineProps) {
  const [expandedIndex, setExpandedIndex] = useState<number | null>(null)

  const getRecommendationStyle = (recommendation: string) => {
    switch (recommendation) {
      case 'BUY':
        return 'bg-emerald-500/20 text-emerald-400 border-emerald-500/30'
      case 'SELL':
        return 'bg-red-500/20 text-red-400 border-red-500/30'
      default:
        return 'bg-yellow-500/20 text-yellow-400 border-yellow-500/30'
    }
  }

  const getActionStyle = (action: string) => {
    switch (action) {
      case 'ENTER':
        return 'text-emerald-400'
      case 'EXIT':
        return 'text-red-400'
      default:
        return 'text-gray-400'
    }
  }

  const getSignalTypeColor = (signalType: string) => {
    switch (signalType) {
      case 'bullish':
        return 'text-emerald-400'
      case 'bearish':
        return 'text-red-400'
      default:
        return 'text-yellow-400'
    }
  }

  const renderAgentContribution = (name: string, signal: AgentSignalSummary) => (
    <div key={name} className="p-3 rounded-lg bg-white/5 border border-white/10">
      <div className="flex items-center justify-between mb-2">
        <span className="text-sm font-medium text-white capitalize">{name}</span>
        <span className={`text-sm font-medium ${getSignalTypeColor(signal.signal_type)}`}>
          {signal.signal_type}
        </span>
      </div>
      <div className="flex items-center gap-4 text-xs text-gray-400 mb-2">
        <span>Score: {signal.score.toFixed(1)}</span>
        <span>Confidence: {signal.confidence.toFixed(0)}%</span>
      </div>
      {signal.key_factors.length > 0 && (
        <ul className="text-xs text-gray-500 space-y-1">
          {signal.key_factors.slice(0, 3).map((factor, i) => (
            <li key={i} className="flex items-start gap-1">
              <span className="text-purple-400">•</span>
              <span>{factor}</span>
            </li>
          ))}
        </ul>
      )}
    </div>
  )

  if (signals.length === 0) {
    return (
      <div className="glass-card p-8 text-center">
        <p className="text-gray-400">No AI signals generated during this period.</p>
      </div>
    )
  }

  return (
    <div className="glass-card overflow-hidden">
      <div className="px-6 py-5 border-b border-white/10">
        <h3 className="text-xl font-semibold text-white">AI Signal Timeline</h3>
        <p className="text-sm text-gray-400 mt-1">{signals.length} signals generated</p>
      </div>
      <div className="divide-y divide-white/5 max-h-[500px] overflow-y-auto">
        {signals.map((signal, index) => (
          <div key={index} className="hover:bg-white/5 transition-colors">
            <button
              onClick={() => setExpandedIndex(expandedIndex === index ? null : index)}
              className="w-full px-6 py-4 flex items-center justify-between text-left"
            >
              <div className="flex items-center gap-4">
                <span className="text-sm text-gray-400 w-24">
                  {new Date(signal.date).toLocaleDateString()}
                </span>
                <span
                  className={`px-3 py-1 text-sm font-semibold rounded-full border ${getRecommendationStyle(
                    signal.recommendation
                  )}`}
                >
                  {signal.recommendation}
                </span>
                <span className={`text-sm ${getActionStyle(signal.position_action)}`}>
                  {signal.position_action !== 'HOLD' && `→ ${signal.position_action}`}
                </span>
              </div>
              <div className="flex items-center gap-4">
                <div className="text-right">
                  <div className="text-sm text-white">
                    {signal.confidence.toFixed(0)}% confidence
                  </div>
                  <div className="text-xs text-gray-500">
                    Score: {signal.weighted_score.toFixed(1)}
                  </div>
                </div>
                <svg
                  className={`w-5 h-5 text-gray-400 transition-transform ${
                    expandedIndex === index ? 'rotate-180' : ''
                  }`}
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                </svg>
              </div>
            </button>
            {expandedIndex === index && (
              <div className="px-6 pb-4 space-y-4">
                <div className="p-4 rounded-lg bg-white/5 border border-white/10">
                  <h4 className="text-sm font-medium text-gray-300 mb-2">Reasoning</h4>
                  <p className="text-sm text-gray-400">{signal.reasoning}</p>
                </div>
                <div>
                  <h4 className="text-sm font-medium text-gray-300 mb-3">Agent Contributions</h4>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                    {Object.entries(signal.agent_signals).map(([name, agentSignal]) =>
                      renderAgentContribution(name, agentSignal)
                    )}
                  </div>
                </div>
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  )
}
