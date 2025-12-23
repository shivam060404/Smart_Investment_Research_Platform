'use client'

import { useState } from 'react'
import { AgentSignal, SignalType } from '@/types'

interface ReasoningTraceProps {
  signals: AgentSignal[]
  keyCatalysts: string[]
  riskFactors: string[]
}

const AGENT_CONFIG: Record<string, { icon: string; gradient: string; bgColor: string }> = {
  Fundamental: {
    icon: 'M9 7h6m0 10v-3m-3 3h.01M9 17h.01M9 14h.01M12 14h.01M15 11h.01M12 11h.01M9 11h.01M7 21h10a2 2 0 002-2V5a2 2 0 00-2-2H7a2 2 0 00-2 2v14a2 2 0 002 2z',
    gradient: 'from-indigo-500 to-purple-500',
    bgColor: 'bg-indigo-500/10',
  },
  Technical: {
    icon: 'M7 12l3-3 3 3 4-4M8 21l4-4 4 4M3 4h18M4 4h16v12a1 1 0 01-1 1H5a1 1 0 01-1-1V4z',
    gradient: 'from-emerald-500 to-teal-500',
    bgColor: 'bg-emerald-500/10',
  },
  Sentiment: {
    icon: 'M19 20H5a2 2 0 01-2-2V6a2 2 0 012-2h10a2 2 0 012 2v1m2 13a2 2 0 01-2-2V7m2 13a2 2 0 002-2V9a2 2 0 00-2-2h-2m-4-3H9M7 16h6M7 8h6v4H7V8z',
    gradient: 'from-amber-500 to-orange-500',
    bgColor: 'bg-amber-500/10',
  },
  Risk: {
    icon: 'M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z',
    gradient: 'from-pink-500 to-rose-500',
    bgColor: 'bg-pink-500/10',
  },
}

export default function ReasoningTrace({ signals, keyCatalysts, riskFactors }: ReasoningTraceProps) {
  const [expandedAgents, setExpandedAgents] = useState<Set<string>>(new Set())

  const toggleAgent = (agentName: string) => {
    const newExpanded = new Set(expandedAgents)
    if (newExpanded.has(agentName)) {
      newExpanded.delete(agentName)
    } else {
      newExpanded.add(agentName)
    }
    setExpandedAgents(newExpanded)
  }

  const expandAll = () => {
    setExpandedAgents(new Set(signals.map((s) => s.agent_name)))
  }

  const collapseAll = () => {
    setExpandedAgents(new Set())
  }

  const getSignalBadge = (signalType: SignalType) => {
    const config = {
      bullish: { bg: 'bg-emerald-500/20', text: 'text-emerald-400', border: 'border-emerald-500/30' },
      bearish: { bg: 'bg-red-500/20', text: 'text-red-400', border: 'border-red-500/30' },
      neutral: { bg: 'bg-amber-500/20', text: 'text-amber-400', border: 'border-amber-500/30' },
    }
    return config[signalType] || config.neutral
  }

  return (
    <div className="glass-card p-8">
      <div className="flex items-center justify-between mb-6">
        <div>
          <h3 className="text-xl font-semibold text-white mb-1">Detailed Analysis</h3>
          <p className="text-gray-400 text-sm">Reasoning from each AI agent</p>
        </div>
        <div className="flex items-center gap-3">
          <button
            onClick={expandAll}
            className="text-sm text-purple-400 hover:text-purple-300 transition-colors"
          >
            Expand All
          </button>
          <span className="text-gray-600">|</span>
          <button
            onClick={collapseAll}
            className="text-sm text-gray-400 hover:text-gray-300 transition-colors"
          >
            Collapse All
          </button>
        </div>
      </div>

      {/* Agent Signals */}
      <div className="space-y-4 mb-8">
        {signals.map((signal) => {
          const config = AGENT_CONFIG[signal.agent_name] || AGENT_CONFIG.Fundamental
          const isExpanded = expandedAgents.has(signal.agent_name)
          const badgeConfig = getSignalBadge(signal.signal_type)

          return (
            <div
              key={signal.agent_name}
              className={`rounded-xl border border-white/10 overflow-hidden transition-all ${config.bgColor}`}
            >
              {/* Header */}
              <button
                onClick={() => toggleAgent(signal.agent_name)}
                className="w-full p-5 flex items-center justify-between hover:bg-white/5 transition-colors"
              >
                <div className="flex items-center gap-4">
                  <div className={`w-10 h-10 rounded-xl bg-gradient-to-br ${config.gradient} flex items-center justify-center`}>
                    <svg className="w-5 h-5 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d={config.icon} />
                    </svg>
                  </div>
                  <div className="text-left">
                    <h4 className="font-semibold text-white">{signal.agent_name} Analysis</h4>
                    <span className={`inline-flex items-center px-2 py-0.5 rounded text-xs font-medium ${badgeConfig.bg} ${badgeConfig.text} border ${badgeConfig.border}`}>
                      {signal.signal_type.toUpperCase()}
                    </span>
                  </div>
                </div>
                <div className="flex items-center gap-6">
                  <div className="text-right">
                    <p className="text-sm text-gray-400">Score</p>
                    <p className="text-lg font-bold text-white">{signal.score.toFixed(0)}</p>
                  </div>
                  <svg
                    className={`w-5 h-5 text-gray-400 transition-transform ${isExpanded ? 'rotate-180' : ''}`}
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                  >
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                  </svg>
                </div>
              </button>

              {/* Expanded Content */}
              {isExpanded && (
                <div className="px-5 pb-5 border-t border-white/5">
                  <div className="pt-5">
                    {/* Key Factors */}
                    {signal.key_factors.length > 0 && (
                      <div className="mb-4">
                        <h5 className="text-sm font-medium text-gray-400 mb-2">Key Factors</h5>
                        <ul className="space-y-2">
                          {signal.key_factors.map((factor, idx) => (
                            <li key={idx} className="flex items-start gap-2 text-gray-300 text-sm">
                              <span className="text-purple-400 mt-1">•</span>
                              {factor}
                            </li>
                          ))}
                        </ul>
                      </div>
                    )}

                    {/* Reasoning */}
                    <div>
                      <h5 className="text-sm font-medium text-gray-400 mb-2">Analysis</h5>
                      <p className="text-gray-300 text-sm leading-relaxed">
                        {signal.reasoning_trace}
                      </p>
                    </div>

                    {/* Confidence */}
                    <div className="mt-4 pt-4 border-t border-white/5">
                      <div className="flex items-center justify-between text-sm">
                        <span className="text-gray-400">Confidence Level</span>
                        <span className="text-white font-medium">{signal.confidence.toFixed(0)}%</span>
                      </div>
                      <div className="progress-bar mt-2 h-1.5">
                        <div
                          className={`progress-fill ${
                            signal.confidence >= 70 ? 'success' : 
                            signal.confidence >= 40 ? 'warning' : 'danger'
                          }`}
                          style={{ width: `${signal.confidence}%` }}
                        />
                      </div>
                    </div>
                  </div>
                </div>
              )}
            </div>
          )
        })}
      </div>

      {/* Key Catalysts & Risk Factors */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Key Catalysts */}
        {keyCatalysts.length > 0 && (
          <div className="p-5 rounded-xl bg-emerald-500/10 border border-emerald-500/20">
            <div className="flex items-center gap-2 mb-4">
              <svg className="w-5 h-5 text-emerald-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7h8m0 0v8m0-8l-8 8-4-4-6 6" />
              </svg>
              <h4 className="font-semibold text-emerald-400">Key Catalysts</h4>
            </div>
            <ul className="space-y-2">
              {keyCatalysts.map((catalyst, idx) => (
                <li key={idx} className="flex items-start gap-2 text-gray-300 text-sm">
                  <span className="text-emerald-400 mt-0.5">✓</span>
                  {catalyst}
                </li>
              ))}
            </ul>
          </div>
        )}

        {/* Risk Factors */}
        {riskFactors.length > 0 && (
          <div className="p-5 rounded-xl bg-red-500/10 border border-red-500/20">
            <div className="flex items-center gap-2 mb-4">
              <svg className="w-5 h-5 text-red-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
              </svg>
              <h4 className="font-semibold text-red-400">Risk Factors</h4>
            </div>
            <ul className="space-y-2">
              {riskFactors.map((risk, idx) => (
                <li key={idx} className="flex items-start gap-2 text-gray-300 text-sm">
                  <span className="text-red-400 mt-0.5">⚠</span>
                  {risk}
                </li>
              ))}
            </ul>
          </div>
        )}
      </div>
    </div>
  )
}
