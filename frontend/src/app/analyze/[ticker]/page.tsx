'use client'

import { useState, useEffect, useCallback } from 'react'
import { useRouter } from 'next/navigation'
import api from '@/lib/api'
import { AnalyzeResponse, ErrorResponse } from '@/types'
import RecommendationCard from '@/components/RecommendationCard'
import AgentSignalChart from '@/components/AgentSignalChart'
import ConfidenceGauge from '@/components/ConfidenceGauge'
import ReasoningTrace from '@/components/ReasoningTrace'

interface AgentProgress {
  fundamental: 'pending' | 'running' | 'complete' | 'error'
  technical: 'pending' | 'running' | 'complete' | 'error'
  sentiment: 'pending' | 'running' | 'complete' | 'error'
  risk: 'pending' | 'running' | 'complete' | 'error'
}

const AGENT_NAMES = ['fundamental', 'technical', 'sentiment', 'risk'] as const

export default function AnalyzePage({ params }: { params: { ticker: string } }) {
  const router = useRouter()
  const ticker = params.ticker.toUpperCase()
  
  const [isLoading, setIsLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [result, setResult] = useState<AnalyzeResponse | null>(null)
  const [searchValue, setSearchValue] = useState('')
  const [agentProgress, setAgentProgress] = useState<AgentProgress>({
    fundamental: 'pending',
    technical: 'pending',
    sentiment: 'pending',
    risk: 'pending',
  })

  useEffect(() => {
    if (!isLoading) return

    const timers: NodeJS.Timeout[] = []
    
    AGENT_NAMES.forEach((agent, index) => {
      timers.push(
        setTimeout(() => {
          setAgentProgress((prev) => ({ ...prev, [agent]: 'running' }))
        }, index * 400)
      )
      
      timers.push(
        setTimeout(() => {
          setAgentProgress((prev) => ({ ...prev, [agent]: 'complete' }))
        }, 1500 + index * 1200)
      )
    })

    return () => timers.forEach(clearTimeout)
  }, [isLoading])

  const fetchAnalysis = useCallback(async () => {
    setIsLoading(true)
    setError(null)
    setAgentProgress({
      fundamental: 'pending',
      technical: 'pending',
      sentiment: 'pending',
      risk: 'pending',
    })

    try {
      const data = await api.analyzeStock(ticker)
      setResult(data)
      setAgentProgress({
        fundamental: 'complete',
        technical: 'complete',
        sentiment: 'complete',
        risk: 'complete',
      })
    } catch (err) {
      const errorResponse = err as ErrorResponse
      setError(errorResponse.message || 'Failed to analyze stock')
      setAgentProgress({
        fundamental: 'error',
        technical: 'error',
        sentiment: 'error',
        risk: 'error',
      })
    } finally {
      setIsLoading(false)
    }
  }, [ticker])

  useEffect(() => {
    fetchAnalysis()
  }, [fetchAnalysis])

  const handleSearch = (e: React.FormEvent) => {
    e.preventDefault()
    if (searchValue.trim()) {
      router.push(`/analyze/${searchValue.toUpperCase().trim()}`)
    }
  }

  return (
    <div className="space-y-8">
      {/* Header */}
      <div className="flex flex-col lg:flex-row items-start lg:items-center justify-between gap-6">
        <div className="flex items-center gap-4">
          <div className="w-16 h-16 rounded-2xl bg-gradient-to-br from-indigo-500 to-purple-500 flex items-center justify-center shadow-lg shadow-purple-500/30">
            <span className="text-2xl font-bold text-white">{ticker.charAt(0)}</span>
          </div>
          <div>
            <h1 className="text-4xl font-bold text-white">{ticker}</h1>
            <p className="text-gray-400 mt-1">AI-Powered Stock Analysis</p>
          </div>
        </div>
        
        {/* Search Bar */}
        <form onSubmit={handleSearch} className="w-full lg:w-auto">
          <div className="relative flex items-center">
            <input
              type="text"
              value={searchValue}
              onChange={(e) => setSearchValue(e.target.value.toUpperCase())}
              placeholder="Analyze another stock..."
              className="input-glass w-full lg:w-80 pr-24"
            />
            <button
              type="submit"
              disabled={isLoading}
              className="absolute right-2 px-4 py-2 bg-gradient-to-r from-indigo-500 to-purple-500 text-white text-sm font-semibold rounded-lg hover:from-indigo-600 hover:to-purple-600 transition-all disabled:opacity-50"
            >
              Analyze
            </button>
          </div>
        </form>
      </div>

      {/* Loading State */}
      {isLoading && (
        <div className="glass-card p-10">
          <div className="text-center mb-8">
            <div className="inline-flex items-center gap-3 px-6 py-3 rounded-full bg-white/5 border border-white/10 mb-6">
              <div className="w-3 h-3 rounded-full bg-purple-500 animate-pulse" />
              <span className="text-lg text-white font-medium">Analyzing {ticker}</span>
            </div>
            <p className="text-gray-400">Our AI agents are working on your analysis...</p>
          </div>
          
          <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
            {AGENT_NAMES.map((agent) => (
              <AgentProgressCard key={agent} name={agent} status={agentProgress[agent]} />
            ))}
          </div>
          
          {/* Progress bar */}
          <div className="mt-8">
            <div className="progress-bar">
              <div 
                className="progress-fill success"
                style={{ 
                  width: `${Object.values(agentProgress).filter(s => s === 'complete').length * 25}%`,
                  transition: 'width 0.5s ease-out'
                }}
              />
            </div>
          </div>
        </div>
      )}

      {/* Error State */}
      {error && !isLoading && (
        <div className="glass-card p-8 border-red-500/30">
          <div className="flex items-start gap-4">
            <div className="w-12 h-12 rounded-xl bg-red-500/20 flex items-center justify-center flex-shrink-0">
              <svg className="w-6 h-6 text-red-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
              </svg>
            </div>
            <div className="flex-1">
              <h3 className="text-xl font-semibold text-white mb-2">Analysis Failed</h3>
              <p className="text-gray-400 mb-4">{error}</p>
              <button onClick={fetchAnalysis} className="btn-primary">
                Try Again
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Results */}
      {result && !isLoading && (
        <div className="space-y-8">
          {/* Main Results Grid */}
          <div className="grid grid-cols-1 xl:grid-cols-3 gap-6">
            {/* Recommendation Card - Takes 2 columns */}
            <div className="xl:col-span-2">
              <RecommendationCard
                recommendation={result.recommendation}
                confidenceScore={result.confidence_score}
                ticker={result.ticker}
                reasoning={result.synthesis.reasoning}
              />
            </div>
            
            {/* Confidence Gauge */}
            <div className="glass-card p-8 flex flex-col items-center justify-center">
              <h3 className="text-lg font-semibold text-white mb-6">Overall Confidence</h3>
              <ConfidenceGauge score={result.confidence_score} size="lg" />
              <div className="mt-6 text-center">
                <span className={`text-4xl font-bold ${
                  result.confidence_score >= 70 ? 'score-high' : 
                  result.confidence_score >= 40 ? 'score-medium' : 'score-low'
                }`}>
                  {result.confidence_score.toFixed(0)}%
                </span>
                <p className="text-gray-400 text-sm mt-2">
                  {result.confidence_score >= 70 ? 'High Confidence' : 
                   result.confidence_score >= 40 ? 'Moderate Confidence' : 'Low Confidence'}
                </p>
              </div>
            </div>
          </div>

          {/* Agent Signals */}
          <AgentSignalChart
            signals={result.signals}
            agentContributions={result.synthesis.agent_contributions}
          />

          {/* Reasoning Trace */}
          <ReasoningTrace
            signals={result.signals}
            keyCatalysts={result.synthesis.key_catalysts}
            riskFactors={result.synthesis.risk_factors}
          />

          {/* Footer Metadata */}
          <div className="glass-card p-5 flex flex-wrap items-center justify-between gap-4">
            <div className="flex items-center gap-6 text-sm">
              <div className="flex items-center gap-2">
                <svg className="w-4 h-4 text-gray-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
                <span className="text-gray-400">
                  {new Date(result.timestamp).toLocaleString()}
                </span>
              </div>
              {result.cached && (
                <span className="px-3 py-1 rounded-full bg-white/5 border border-white/10 text-gray-400 text-xs">
                  Cached Result
                </span>
              )}
            </div>
            <button
              onClick={fetchAnalysis}
              className="flex items-center gap-2 text-purple-400 hover:text-purple-300 transition-colors font-medium"
            >
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
              </svg>
              Refresh Analysis
            </button>
          </div>
        </div>
      )}
    </div>
  )
}

function AgentProgressCard({ name, status }: { name: string; status: 'pending' | 'running' | 'complete' | 'error' }) {
  const config: Record<string, { icon: string; gradient: string; label: string }> = {
    fundamental: {
      icon: 'M9 7h6m0 10v-3m-3 3h.01M9 17h.01M9 14h.01M12 14h.01M15 11h.01M12 11h.01M9 11h.01M7 21h10a2 2 0 002-2V5a2 2 0 00-2-2H7a2 2 0 00-2 2v14a2 2 0 002 2z',
      gradient: 'from-indigo-500 to-purple-500',
      label: 'Fundamental',
    },
    technical: {
      icon: 'M7 12l3-3 3 3 4-4M8 21l4-4 4 4M3 4h18M4 4h16v12a1 1 0 01-1 1H5a1 1 0 01-1-1V4z',
      gradient: 'from-emerald-500 to-teal-500',
      label: 'Technical',
    },
    sentiment: {
      icon: 'M19 20H5a2 2 0 01-2-2V6a2 2 0 012-2h10a2 2 0 012 2v1m2 13a2 2 0 01-2-2V7m2 13a2 2 0 002-2V9a2 2 0 00-2-2h-2m-4-3H9M7 16h6M7 8h6v4H7V8z',
      gradient: 'from-amber-500 to-orange-500',
      label: 'Sentiment',
    },
    risk: {
      icon: 'M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z',
      gradient: 'from-pink-500 to-rose-500',
      label: 'Risk',
    },
  }

  const agentConfig = config[name]
  
  return (
    <div className={`glass-card p-5 transition-all duration-300 ${status === 'running' ? 'pulse-glow' : ''}`}>
      <div className="flex items-center gap-3 mb-4">
        <div className={`w-10 h-10 rounded-xl bg-gradient-to-br ${agentConfig.gradient} flex items-center justify-center shadow-lg`}>
          <svg className="w-5 h-5 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d={agentConfig.icon} />
          </svg>
        </div>
        <span className="font-semibold text-white">{agentConfig.label}</span>
      </div>
      
      <div className="flex items-center gap-2">
        {status === 'pending' && (
          <>
            <div className="w-2 h-2 rounded-full bg-gray-500" />
            <span className="text-sm text-gray-500">Waiting...</span>
          </>
        )}
        {status === 'running' && (
          <>
            <svg className="animate-spin w-4 h-4 text-purple-400" viewBox="0 0 24 24">
              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
              <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
            </svg>
            <span className="text-sm text-purple-400">Analyzing...</span>
          </>
        )}
        {status === 'complete' && (
          <>
            <svg className="w-4 h-4 text-emerald-400" fill="currentColor" viewBox="0 0 20 20">
              <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
            </svg>
            <span className="text-sm text-emerald-400">Complete</span>
          </>
        )}
        {status === 'error' && (
          <>
            <svg className="w-4 h-4 text-red-400" fill="currentColor" viewBox="0 0 20 20">
              <path fillRule="evenodd" d="M4.293 4.293a1 1 0 011.414 0L10 8.586l4.293-4.293a1 1 0 111.414 1.414L11.414 10l4.293 4.293a1 1 0 01-1.414 1.414L10 11.414l-4.293 4.293a1 1 0 01-1.414-1.414L8.586 10 4.293 5.707a1 1 0 010-1.414z" clipRule="evenodd" />
            </svg>
            <span className="text-sm text-red-400">Failed</span>
          </>
        )}
      </div>
    </div>
  )
}
