'use client'

import { useState } from 'react'
import { AgentSignal } from '@/types'
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell,
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  Radar,
} from 'recharts'

interface AgentSignalChartProps {
  signals: AgentSignal[]
  agentContributions: Record<string, number>
}

const AGENT_COLORS: Record<string, string> = {
  Fundamental: '#8b5cf6',
  Technical: '#10b981',
  Sentiment: '#f59e0b',
  Risk: '#ef4444',
}

const AGENT_GRADIENTS: Record<string, string> = {
  Fundamental: 'url(#fundamentalGradient)',
  Technical: 'url(#technicalGradient)',
  Sentiment: 'url(#sentimentGradient)',
  Risk: 'url(#riskGradient)',
}

export default function AgentSignalChart({ signals, agentContributions }: AgentSignalChartProps) {
  const [chartType, setChartType] = useState<'bar' | 'radar'>('bar')

  const chartData = signals.map((signal) => ({
    name: signal.agent_name,
    score: signal.score,
    confidence: signal.confidence,
    contribution: agentContributions[signal.agent_name] || 0,
    color: AGENT_COLORS[signal.agent_name] || '#6b7280',
  }))

  const radarData = signals.map((signal) => ({
    subject: signal.agent_name,
    score: signal.score,
    fullMark: 100,
  }))

  const CustomTooltip = ({ active, payload }: any) => {
    if (active && payload && payload.length) {
      const data = payload[0].payload
      return (
        <div className="glass-card p-4 border border-white/20">
          <p className="font-semibold text-white mb-2">{data.name}</p>
          <div className="space-y-1 text-sm">
            <p className="text-gray-300">
              Score: <span className="text-white font-medium">{data.score.toFixed(1)}</span>
            </p>
            <p className="text-gray-300">
              Confidence: <span className="text-white font-medium">{data.confidence.toFixed(1)}%</span>
            </p>
            <p className="text-gray-300">
              Weight: <span className="text-white font-medium">{data.contribution.toFixed(0)}%</span>
            </p>
          </div>
        </div>
      )
    }
    return null
  }

  return (
    <div className="glass-card p-8">
      <div className="flex items-center justify-between mb-8">
        <div>
          <h3 className="text-xl font-semibold text-white mb-1">Agent Signals</h3>
          <p className="text-gray-400 text-sm">Individual agent scores and contributions</p>
        </div>
        <div className="flex items-center gap-2 p-1 rounded-xl bg-white/5 border border-white/10">
          <button
            onClick={() => setChartType('bar')}
            className={`px-4 py-2 rounded-lg text-sm font-medium transition-all ${
              chartType === 'bar'
                ? 'bg-gradient-to-r from-indigo-500 to-purple-500 text-white shadow-lg'
                : 'text-gray-400 hover:text-white'
            }`}
          >
            Bar Chart
          </button>
          <button
            onClick={() => setChartType('radar')}
            className={`px-4 py-2 rounded-lg text-sm font-medium transition-all ${
              chartType === 'radar'
                ? 'bg-gradient-to-r from-indigo-500 to-purple-500 text-white shadow-lg'
                : 'text-gray-400 hover:text-white'
            }`}
          >
            Radar Chart
          </button>
        </div>
      </div>

      <div className="h-80">
        <ResponsiveContainer width="100%" height="100%">
          {chartType === 'bar' ? (
            <BarChart data={chartData} layout="vertical" margin={{ left: 20, right: 30 }}>
              <defs>
                <linearGradient id="fundamentalGradient" x1="0" y1="0" x2="1" y2="0">
                  <stop offset="0%" stopColor="#667eea" />
                  <stop offset="100%" stopColor="#764ba2" />
                </linearGradient>
                <linearGradient id="technicalGradient" x1="0" y1="0" x2="1" y2="0">
                  <stop offset="0%" stopColor="#11998e" />
                  <stop offset="100%" stopColor="#38ef7d" />
                </linearGradient>
                <linearGradient id="sentimentGradient" x1="0" y1="0" x2="1" y2="0">
                  <stop offset="0%" stopColor="#f59e0b" />
                  <stop offset="100%" stopColor="#fbbf24" />
                </linearGradient>
                <linearGradient id="riskGradient" x1="0" y1="0" x2="1" y2="0">
                  <stop offset="0%" stopColor="#ef4444" />
                  <stop offset="100%" stopColor="#f87171" />
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" horizontal={false} />
              <XAxis 
                type="number" 
                domain={[0, 100]} 
                tick={{ fill: '#9ca3af', fontSize: 12 }}
                axisLine={{ stroke: 'rgba(255,255,255,0.1)' }}
              />
              <YAxis 
                type="category" 
                dataKey="name" 
                tick={{ fill: '#fff', fontSize: 14, fontWeight: 500 }}
                axisLine={{ stroke: 'rgba(255,255,255,0.1)' }}
                width={100}
              />
              <Tooltip content={<CustomTooltip />} cursor={{ fill: 'rgba(255,255,255,0.05)' }} />
              <Bar dataKey="score" radius={[0, 8, 8, 0]} barSize={40}>
                {chartData.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={AGENT_GRADIENTS[entry.name]} />
                ))}
              </Bar>
            </BarChart>
          ) : (
            <RadarChart data={radarData} margin={{ top: 20, right: 30, bottom: 20, left: 30 }}>
              <PolarGrid stroke="rgba(255,255,255,0.1)" />
              <PolarAngleAxis 
                dataKey="subject" 
                tick={{ fill: '#fff', fontSize: 14, fontWeight: 500 }}
              />
              <PolarRadiusAxis 
                angle={30} 
                domain={[0, 100]} 
                tick={{ fill: '#9ca3af', fontSize: 10 }}
              />
              <Radar
                name="Score"
                dataKey="score"
                stroke="#8b5cf6"
                fill="url(#radarGradient)"
                fillOpacity={0.5}
                strokeWidth={2}
              />
              <defs>
                <linearGradient id="radarGradient" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor="#8b5cf6" stopOpacity={0.8} />
                  <stop offset="100%" stopColor="#6366f1" stopOpacity={0.3} />
                </linearGradient>
              </defs>
            </RadarChart>
          )}
        </ResponsiveContainer>
      </div>

      {/* Agent Legend */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 mt-8 pt-6 border-t border-white/10">
        {signals.map((signal) => (
          <div key={signal.agent_name} className="flex items-center gap-3">
            <div 
              className="w-3 h-3 rounded-full"
              style={{ backgroundColor: AGENT_COLORS[signal.agent_name] }}
            />
            <div>
              <p className="text-white font-medium text-sm">{signal.agent_name}</p>
              <p className="text-gray-500 text-xs">
                {(agentContributions[signal.agent_name] || 0).toFixed(0)}% weight
              </p>
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}
