'use client'

import { Recommendation } from '@/types'

interface RecommendationCardProps {
  recommendation: Recommendation
  confidenceScore: number
  ticker: string
  reasoning: string
}

export default function RecommendationCard({
  recommendation,
  confidenceScore,
  ticker,
  reasoning,
}: RecommendationCardProps) {
  const config = {
    BUY: {
      gradient: 'from-emerald-500 to-teal-500',
      bgGlow: 'bg-emerald-500/20',
      textColor: 'text-emerald-400',
      icon: 'M13 7h8m0 0v8m0-8l-8 8-4-4-6 6',
      label: 'Strong Buy Signal',
    },
    SELL: {
      gradient: 'from-red-500 to-rose-500',
      bgGlow: 'bg-red-500/20',
      textColor: 'text-red-400',
      icon: 'M13 17h8m0 0v-8m0 8l-8-8-4 4-6-6',
      label: 'Sell Signal',
    },
    HOLD: {
      gradient: 'from-amber-500 to-orange-500',
      bgGlow: 'bg-amber-500/20',
      textColor: 'text-amber-400',
      icon: 'M20 12H4',
      label: 'Hold Position',
    },
  }

  const recConfig = config[recommendation]

  return (
    <div className="glass-card overflow-hidden">
      {/* Header with gradient */}
      <div className={`relative p-8 ${recConfig.bgGlow}`}>
        <div className="absolute inset-0 bg-gradient-to-br opacity-10" style={{
          backgroundImage: `linear-gradient(135deg, var(--tw-gradient-from), var(--tw-gradient-to))`
        }} />
        
        <div className="relative flex items-start justify-between">
          <div>
            <div className="flex items-center gap-3 mb-4">
              <div className={`w-14 h-14 rounded-2xl bg-gradient-to-br ${recConfig.gradient} flex items-center justify-center shadow-lg`}>
                <svg className="w-7 h-7 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d={recConfig.icon} />
                </svg>
              </div>
              <div>
                <p className="text-sm text-gray-400 uppercase tracking-wider">Recommendation</p>
                <h2 className={`text-4xl font-bold ${recConfig.textColor}`}>
                  {recommendation}
                </h2>
              </div>
            </div>
            <p className="text-gray-400">{recConfig.label}</p>
          </div>
          
          <div className="text-right">
            <p className="text-sm text-gray-400 mb-1">Confidence</p>
            <div className={`text-3xl font-bold ${recConfig.textColor}`}>
              {confidenceScore.toFixed(1)}%
            </div>
          </div>
        </div>
      </div>

      {/* Reasoning */}
      <div className="p-8 border-t border-white/5">
        <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
          <svg className="w-5 h-5 text-purple-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
          </svg>
          AI Analysis Summary
        </h3>
        <p className="text-gray-300 leading-relaxed text-base">
          {reasoning}
        </p>
      </div>

      {/* Confidence Bar */}
      <div className="px-8 pb-8">
        <div className="progress-bar h-3">
          <div 
            className={`progress-fill ${
              confidenceScore >= 70 ? 'success' : 
              confidenceScore >= 40 ? 'warning' : 'danger'
            }`}
            style={{ width: `${confidenceScore}%` }}
          />
        </div>
        <div className="flex justify-between mt-2 text-xs text-gray-500">
          <span>Low Confidence</span>
          <span>High Confidence</span>
        </div>
      </div>
    </div>
  )
}
