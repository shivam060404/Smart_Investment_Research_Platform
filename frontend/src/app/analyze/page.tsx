'use client'

import { useState } from 'react'
import { useRouter } from 'next/navigation'

const POPULAR_STOCKS = [
  { ticker: 'AAPL', name: 'Apple Inc.', sector: 'Technology' },
  { ticker: 'GOOGL', name: 'Alphabet Inc.', sector: 'Technology' },
  { ticker: 'MSFT', name: 'Microsoft', sector: 'Technology' },
  { ticker: 'AMZN', name: 'Amazon', sector: 'E-Commerce' },
  { ticker: 'TSLA', name: 'Tesla Inc.', sector: 'Automotive' },
  { ticker: 'NVDA', name: 'NVIDIA', sector: 'Semiconductors' },
  { ticker: 'META', name: 'Meta Platforms', sector: 'Social Media' },
  { ticker: 'JPM', name: 'JPMorgan Chase', sector: 'Banking' },
  { ticker: 'V', name: 'Visa Inc.', sector: 'Financial Services' },
  { ticker: 'JNJ', name: 'Johnson & Johnson', sector: 'Healthcare' },
  { ticker: 'WMT', name: 'Walmart', sector: 'Retail' },
  { ticker: 'DIS', name: 'Walt Disney', sector: 'Entertainment' },
]

const SECTORS = [
  { name: 'Technology', icon: 'M9.75 17L9 20l-1 1h8l-1-1-.75-3M3 13h18M5 17h14a2 2 0 002-2V5a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z', stocks: ['AAPL', 'GOOGL', 'MSFT', 'NVDA'] },
  { name: 'Healthcare', icon: 'M4.318 6.318a4.5 4.5 0 000 6.364L12 20.364l7.682-7.682a4.5 4.5 0 00-6.364-6.364L12 7.636l-1.318-1.318a4.5 4.5 0 00-6.364 0z', stocks: ['JNJ', 'PFE', 'UNH'] },
  { name: 'Finance', icon: 'M12 8c-1.657 0-3 .895-3 2s1.343 2 3 2 3 .895 3 2-1.343 2-3 2m0-8c1.11 0 2.08.402 2.599 1M12 8V7m0 1v8m0 0v1m0-1c-1.11 0-2.08-.402-2.599-1M21 12a9 9 0 11-18 0 9 9 0 0118 0z', stocks: ['JPM', 'V', 'BAC'] },
  { name: 'Consumer', icon: 'M16 11V7a4 4 0 00-8 0v4M5 9h14l1 12H4L5 9z', stocks: ['AMZN', 'WMT', 'NKE'] },
]

export default function AnalyzePage() {
  const router = useRouter()
  const [searchValue, setSearchValue] = useState('')
  const [isSearchFocused, setIsSearchFocused] = useState(false)

  const handleSearch = (ticker: string) => {
    const normalizedTicker = ticker.toUpperCase().trim()
    if (!normalizedTicker) return
    router.push(`/analyze/${normalizedTicker}`)
  }

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    handleSearch(searchValue)
  }

  return (
    <div className="space-y-12">
      {/* Hero Section */}
      <section className="relative py-12">
        <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[800px] h-[600px] bg-purple-500/10 rounded-full blur-[150px] pointer-events-none" />
        
        <div className="relative text-center max-w-4xl mx-auto">
          <h1 className="text-5xl lg:text-6xl font-bold text-white mb-6">
            Analyze Any <span className="gradient-text">Stock</span>
          </h1>
          <p className="text-xl text-gray-400 mb-10 max-w-2xl mx-auto">
            Enter a stock ticker symbol to get comprehensive AI-powered analysis including fundamental metrics, 
            technical indicators, market sentiment, and risk assessment.
          </p>

          {/* Search Bar */}
          <form onSubmit={handleSubmit} className="max-w-2xl mx-auto">
            <div className={`relative transition-all duration-300 ${isSearchFocused ? 'scale-[1.02]' : ''}`}>
              <div className="absolute -inset-1 bg-gradient-to-r from-indigo-500 via-purple-500 to-pink-500 rounded-2xl opacity-50 blur" />
              <div className="relative flex items-center bg-gray-900/95 rounded-xl border border-white/20 overflow-hidden">
                <div className="pl-6">
                  <svg className="w-6 h-6 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
                  </svg>
                </div>
                <input
                  type="text"
                  value={searchValue}
                  onChange={(e) => setSearchValue(e.target.value.toUpperCase())}
                  onFocus={() => setIsSearchFocused(true)}
                  onBlur={() => setIsSearchFocused(false)}
                  placeholder="Enter stock ticker (e.g., AAPL, TSLA, GOOGL)"
                  className="flex-1 px-5 py-6 bg-transparent text-white text-xl placeholder-gray-500 focus:outline-none"
                  autoFocus
                />
                <button
                  type="submit"
                  className="m-3 px-10 py-4 bg-gradient-to-r from-indigo-500 to-purple-500 text-white text-lg font-semibold rounded-xl hover:from-indigo-600 hover:to-purple-600 transition-all shadow-lg shadow-purple-500/30"
                >
                  Analyze
                </button>
              </div>
            </div>
          </form>

          {/* Quick Tips */}
          <div className="mt-8 flex flex-wrap items-center justify-center gap-4">
            <span className="text-base text-gray-500">Try:</span>
            {['AAPL', 'TSLA', 'NVDA', 'GOOGL'].map((ticker) => (
              <button
                key={ticker}
                onClick={() => handleSearch(ticker)}
                className="px-4 py-2 rounded-lg bg-white/5 border border-white/10 text-white text-base font-medium hover:bg-white/10 hover:border-purple-500/50 transition-all"
              >
                {ticker}
              </button>
            ))}
          </div>
        </div>
      </section>

      {/* Popular Stocks */}
      <section>
        <div className="flex items-center justify-between mb-8">
          <div>
            <h2 className="text-3xl font-bold text-white mb-2">Popular Stocks</h2>
            <p className="text-lg text-gray-400">Click any stock to start your analysis</p>
          </div>
        </div>
        <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-6 gap-4">
          {POPULAR_STOCKS.map((stock) => (
            <button
              key={stock.ticker}
              onClick={() => handleSearch(stock.ticker)}
              className="glass-card-hover p-6 text-left group"
            >
              <div className="flex items-center justify-between mb-3">
                <span className="text-2xl font-bold text-white group-hover:text-purple-400 transition-colors">
                  {stock.ticker}
                </span>
                <svg className="w-5 h-5 text-gray-600 group-hover:text-purple-400 transition-colors" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                </svg>
              </div>
              <p className="text-base text-gray-400 truncate">{stock.name}</p>
              <span className="text-sm text-gray-600 mt-1 block">{stock.sector}</span>
            </button>
          ))}
        </div>
      </section>

      {/* Browse by Sector */}
      <section>
        <div className="mb-8">
          <h2 className="text-3xl font-bold text-white mb-2">Browse by Sector</h2>
          <p className="text-lg text-gray-400">Explore stocks organized by industry sector</p>
        </div>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
          {SECTORS.map((sector) => (
            <div key={sector.name} className="glass-card p-6">
              <div className="flex items-center gap-4 mb-5">
                <div className="w-12 h-12 rounded-xl bg-gradient-to-br from-indigo-500 to-purple-500 flex items-center justify-center">
                  <svg className="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d={sector.icon} />
                  </svg>
                </div>
                <h3 className="text-xl font-bold text-white">{sector.name}</h3>
              </div>
              <div className="flex flex-wrap gap-2">
                {sector.stocks.map((ticker) => (
                  <button
                    key={ticker}
                    onClick={() => handleSearch(ticker)}
                    className="px-4 py-2 rounded-lg bg-white/5 border border-white/10 text-white text-base font-medium hover:bg-purple-500/20 hover:border-purple-500/50 transition-all"
                  >
                    {ticker}
                  </button>
                ))}
              </div>
            </div>
          ))}
        </div>
      </section>

      {/* Analysis Features */}
      <section className="glass-card p-10">
        <h2 className="text-3xl font-bold text-white mb-8 text-center">What You'll Get</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-8">
          <div className="text-center">
            <div className="w-16 h-16 rounded-2xl bg-gradient-to-br from-indigo-500 to-purple-500 flex items-center justify-center mx-auto mb-4">
              <svg className="w-8 h-8 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
            </div>
            <h3 className="text-xl font-bold text-white mb-2">Clear Recommendation</h3>
            <p className="text-base text-gray-400">BUY, SELL, or HOLD with confidence score</p>
          </div>
          <div className="text-center">
            <div className="w-16 h-16 rounded-2xl bg-gradient-to-br from-emerald-500 to-teal-500 flex items-center justify-center mx-auto mb-4">
              <svg className="w-8 h-8 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
              </svg>
            </div>
            <h3 className="text-xl font-bold text-white mb-2">Agent Signals</h3>
            <p className="text-base text-gray-400">Insights from 4 specialized AI agents</p>
          </div>
          <div className="text-center">
            <div className="w-16 h-16 rounded-2xl bg-gradient-to-br from-pink-500 to-rose-500 flex items-center justify-center mx-auto mb-4">
              <svg className="w-8 h-8 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
              </svg>
            </div>
            <h3 className="text-xl font-bold text-white mb-2">Detailed Reasoning</h3>
            <p className="text-base text-gray-400">Transparent AI decision explanations</p>
          </div>
          <div className="text-center">
            <div className="w-16 h-16 rounded-2xl bg-gradient-to-br from-orange-500 to-red-500 flex items-center justify-center mx-auto mb-4">
              <svg className="w-8 h-8 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
              </svg>
            </div>
            <h3 className="text-xl font-bold text-white mb-2">Risk Factors</h3>
            <p className="text-base text-gray-400">Key risks and catalysts identified</p>
          </div>
        </div>
      </section>
    </div>
  )
}
