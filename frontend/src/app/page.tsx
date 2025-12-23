'use client'

import { useState, useEffect } from 'react'
import { useRouter } from 'next/navigation'
import Link from 'next/link'

const POPULAR_STOCKS = [
  { ticker: 'AAPL', name: 'Apple Inc.', sector: 'Technology', change: '+2.4%', positive: true },
  { ticker: 'GOOGL', name: 'Alphabet Inc.', sector: 'Technology', change: '+1.8%', positive: true },
  { ticker: 'MSFT', name: 'Microsoft', sector: 'Technology', change: '+0.9%', positive: true },
  { ticker: 'AMZN', name: 'Amazon', sector: 'E-Commerce', change: '-0.5%', positive: false },
  { ticker: 'TSLA', name: 'Tesla Inc.', sector: 'Automotive', change: '+3.2%', positive: true },
  { ticker: 'NVDA', name: 'NVIDIA', sector: 'Semiconductors', change: '+4.1%', positive: true },
]

const FEATURES = [
  {
    icon: 'M9 7h6m0 10v-3m-3 3h.01M9 17h.01M9 14h.01M12 14h.01M15 11h.01M12 11h.01M9 11h.01M7 21h10a2 2 0 002-2V5a2 2 0 00-2-2H7a2 2 0 00-2 2v14a2 2 0 002 2z',
    title: 'Fundamental Analysis',
    description: 'Deep dive into P/E ratios, ROE, debt levels, revenue growth, and profit margins using AI-powered analysis.',
    gradient: 'from-indigo-500 to-purple-500',
  },
  {
    icon: 'M7 12l3-3 3 3 4-4M8 21l4-4 4 4M3 4h18M4 4h16v12a1 1 0 01-1 1H5a1 1 0 01-1-1V4z',
    title: 'Technical Analysis',
    description: 'Advanced indicators including SMA, RSI, MACD, Bollinger Bands, and pattern recognition algorithms.',
    gradient: 'from-emerald-500 to-teal-500',
  },
  {
    icon: 'M19 20H5a2 2 0 01-2-2V6a2 2 0 012-2h10a2 2 0 012 2v1m2 13a2 2 0 01-2-2V7m2 13a2 2 0 002-2V9a2 2 0 00-2-2h-2m-4-3H9M7 16h6M7 8h6v4H7V8z',
    title: 'Sentiment Analysis',
    description: 'Real-time news analysis and market sentiment aggregation from multiple trusted financial sources.',
    gradient: 'from-pink-500 to-rose-500',
  },
  {
    icon: 'M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z',
    title: 'Risk Assessment',
    description: 'Comprehensive volatility analysis, beta calculation, VaR metrics, and risk factor evaluation.',
    gradient: 'from-orange-500 to-red-500',
  },
]

const HOW_IT_WORKS = [
  {
    step: '01',
    title: 'Enter Stock Ticker',
    description: 'Simply type any stock symbol like AAPL, TSLA, or GOOGL to begin your analysis.',
    icon: 'M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z',
  },
  {
    step: '02',
    title: 'AI Agents Analyze',
    description: 'Four specialized AI agents work simultaneously to analyze fundamental, technical, sentiment, and risk factors.',
    icon: 'M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z',
  },
  {
    step: '03',
    title: 'Get Recommendations',
    description: 'Receive a comprehensive BUY, SELL, or HOLD recommendation with detailed reasoning and confidence scores.',
    icon: 'M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z',
  },
]

const SERVICES = [
  {
    title: 'Stock Analysis',
    description: 'Get instant AI-powered analysis on any publicly traded stock with comprehensive insights.',
    icon: 'M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z',
    href: '/analyze',
  },
  {
    title: 'Backtesting',
    description: 'Test your investment strategies against historical data to validate performance.',
    icon: 'M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z',
    href: '/backtest',
  },
  {
    title: 'AI Chat Assistant',
    description: 'Ask questions about your analyzed stocks and get instant answers with source citations.',
    icon: 'M8 10h.01M12 10h.01M16 10h.01M9 16H5a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v8a2 2 0 01-2 2h-5l-5 5v-5z',
    href: '#',
  },
  {
    title: 'Portfolio Tracking',
    description: 'Monitor your investments and track performance across your entire portfolio.',
    icon: 'M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10',
    href: '/portfolio',
  },
]

const STATS = [
  { value: '4', label: 'AI Agents', suffix: '' },
  { value: '100', label: 'Metrics Analyzed', suffix: '+' },
  { value: '24/7', label: 'Availability', suffix: '' },
  { value: '< 30s', label: 'Analysis Time', suffix: '' },
]

export default function Home() {
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
    <div className="space-y-24">
      {/* Hero Section */}
      <section className="relative py-16 lg:py-24">
        <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[1000px] h-[800px] bg-purple-500/10 rounded-full blur-[180px] pointer-events-none" />
        
        <div className="relative text-center max-w-5xl mx-auto">
          <div className="inline-flex items-center gap-3 px-5 py-2.5 rounded-full bg-white/5 border border-white/10 mb-8">
            <div className="w-2.5 h-2.5 rounded-full bg-emerald-400 animate-pulse" />
            <span className="text-base text-gray-300">AI-Powered Investment Analysis Engine</span>
          </div>
          
          <h1 className="text-5xl lg:text-6xl xl:text-7xl font-bold text-white mb-6 leading-tight">
            Make Smarter Investment <br />
            <span className="gradient-text">Decisions with AI</span>
          </h1>
          
          <p className="text-xl lg:text-2xl text-gray-400 max-w-3xl mx-auto mb-10 leading-relaxed">
            Harness the power of multiple AI agents to analyze stocks with unprecedented depth. 
            Get comprehensive insights combining fundamental, technical, sentiment, and risk analysis.
          </p>

          {/* CTA Buttons */}
          <div className="flex flex-col sm:flex-row items-center justify-center gap-4 mb-12">
            <Link href="/analyze" className="btn-primary px-10 py-4 text-lg">
              Start Analyzing Now
              <svg className="w-5 h-5 ml-2 inline" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7l5 5m0 0l-5 5m5-5H6" />
              </svg>
            </Link>
            <button 
              onClick={() => document.getElementById('how-it-works')?.scrollIntoView({ behavior: 'smooth' })}
              className="btn-secondary px-10 py-4 text-lg"
            >
              Learn How It Works
            </button>
          </div>

          {/* Quick Search */}
          <form onSubmit={handleSubmit} className="max-w-2xl mx-auto mb-12">
            <div className={`relative transition-all duration-300 ${isSearchFocused ? 'scale-[1.02]' : ''}`}>
              <div className="absolute -inset-1 bg-gradient-to-r from-indigo-500 via-purple-500 to-pink-500 rounded-2xl opacity-40 blur" />
              <div className="relative flex items-center bg-gray-900/90 rounded-xl border border-white/20 overflow-hidden">
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
                  className="flex-1 px-5 py-5 bg-transparent text-white text-lg placeholder-gray-500 focus:outline-none"
                />
                <button
                  type="submit"
                  className="m-2 px-8 py-3 bg-gradient-to-r from-indigo-500 to-purple-500 text-white text-lg font-semibold rounded-lg hover:from-indigo-600 hover:to-purple-600 transition-all shadow-lg shadow-purple-500/30"
                >
                  Analyze
                </button>
              </div>
            </div>
          </form>

          {/* Stats */}
          <div className="flex flex-wrap items-center justify-center gap-8 lg:gap-16">
            {STATS.map((stat, index) => (
              <div key={index} className="text-center">
                <div className="text-4xl lg:text-5xl font-bold text-white">{stat.value}{stat.suffix}</div>
                <div className="text-base text-gray-500 mt-1">{stat.label}</div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Features Section */}
      <section id="features" className="py-16">
        <div className="text-center mb-16">
          <h2 className="text-4xl lg:text-5xl font-bold text-white mb-4">
            Powered by <span className="gradient-text">4 AI Agents</span>
          </h2>
          <p className="text-xl text-gray-400 max-w-2xl mx-auto">
            Our multi-agent system analyzes stocks from every angle to provide you with comprehensive investment insights.
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {FEATURES.map((feature, index) => (
            <div key={index} className="glass-card p-8 hover:bg-white/[0.08] transition-all group">
              <div className={`w-16 h-16 rounded-2xl bg-gradient-to-br ${feature.gradient} flex items-center justify-center mb-6 group-hover:scale-110 transition-transform`}>
                <svg className="w-8 h-8 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d={feature.icon} />
                </svg>
              </div>
              <h3 className="text-2xl font-bold text-white mb-3">{feature.title}</h3>
              <p className="text-lg text-gray-400 leading-relaxed">{feature.description}</p>
            </div>
          ))}
        </div>
      </section>

      {/* How It Works Section */}
      <section id="how-it-works" className="py-16">
        <div className="text-center mb-16">
          <h2 className="text-4xl lg:text-5xl font-bold text-white mb-4">
            How It <span className="gradient-text">Works</span>
          </h2>
          <p className="text-xl text-gray-400 max-w-2xl mx-auto">
            Get actionable investment insights in three simple steps.
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
          {HOW_IT_WORKS.map((item, index) => (
            <div key={index} className="relative">
              {index < HOW_IT_WORKS.length - 1 && (
                <div className="hidden md:block absolute top-16 left-[60%] w-[80%] h-0.5 bg-gradient-to-r from-purple-500/50 to-transparent" />
              )}
              <div className="glass-card p-8 text-center relative z-10">
                <div className="text-6xl font-bold gradient-text mb-6">{item.step}</div>
                <div className="w-16 h-16 rounded-2xl bg-white/10 flex items-center justify-center mx-auto mb-6">
                  <svg className="w-8 h-8 text-purple-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d={item.icon} />
                  </svg>
                </div>
                <h3 className="text-2xl font-bold text-white mb-3">{item.title}</h3>
                <p className="text-lg text-gray-400">{item.description}</p>
              </div>
            </div>
          ))}
        </div>
      </section>

      {/* Services Section */}
      <section id="services" className="py-16">
        <div className="text-center mb-16">
          <h2 className="text-4xl lg:text-5xl font-bold text-white mb-4">
            Our <span className="gradient-text">Services</span>
          </h2>
          <p className="text-xl text-gray-400 max-w-2xl mx-auto">
            Everything you need to make informed investment decisions.
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
          {SERVICES.map((service, index) => (
            <Link 
              key={index} 
              href={service.href}
              className="glass-card p-8 hover:bg-white/[0.08] transition-all group cursor-pointer"
            >
              <div className="w-14 h-14 rounded-xl bg-gradient-to-br from-indigo-500 to-purple-500 flex items-center justify-center mb-6 group-hover:scale-110 transition-transform">
                <svg className="w-7 h-7 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d={service.icon} />
                </svg>
              </div>
              <h3 className="text-xl font-bold text-white mb-3">{service.title}</h3>
              <p className="text-base text-gray-400 leading-relaxed">{service.description}</p>
              <div className="mt-4 flex items-center text-purple-400 group-hover:text-purple-300 transition-colors">
                <span className="text-base font-medium">Learn more</span>
                <svg className="w-5 h-5 ml-2 group-hover:translate-x-1 transition-transform" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                </svg>
              </div>
            </Link>
          ))}
        </div>
      </section>

      {/* Popular Stocks Section */}
      <section className="py-16">
        <div className="flex items-center justify-between mb-8">
          <div>
            <h2 className="text-3xl font-bold text-white mb-2">Popular Stocks</h2>
            <p className="text-lg text-gray-400">Click any stock to start analyzing</p>
          </div>
          <Link href="/analyze" className="text-purple-400 hover:text-purple-300 text-lg font-medium flex items-center gap-2">
            View all
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
            </svg>
          </Link>
        </div>
        <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-4">
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
                <span className={`text-base font-semibold ${stock.positive ? 'text-emerald-400' : 'text-red-400'}`}>
                  {stock.change}
                </span>
              </div>
              <p className="text-base text-gray-400 truncate">{stock.name}</p>
              <span className="text-sm text-gray-600 mt-1 block">{stock.sector}</span>
            </button>
          ))}
        </div>
      </section>

      {/* About Section */}
      <section id="about" className="py-16">
        <div className="glass-card p-12 lg:p-16">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-12 items-center">
            <div>
              <h2 className="text-4xl lg:text-5xl font-bold text-white mb-6">
                About <span className="gradient-text">AIInvest</span>
              </h2>
              <p className="text-xl text-gray-400 mb-6 leading-relaxed">
                AIInvest is a cutting-edge investment research platform that leverages the power of artificial intelligence 
                to provide comprehensive stock analysis. Our multi-agent system combines fundamental analysis, technical indicators, 
                market sentiment, and risk assessment to deliver actionable investment insights.
              </p>
              <p className="text-xl text-gray-400 mb-8 leading-relaxed">
                Built with advanced AI models and real-time market data, our platform helps investors make informed decisions 
                by analyzing over 100 different metrics and factors for each stock.
              </p>
              <div className="flex flex-wrap gap-4">
                <div className="flex items-center gap-3 px-5 py-3 rounded-xl bg-white/5 border border-white/10">
                  <svg className="w-6 h-6 text-emerald-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                  </svg>
                  <span className="text-lg text-gray-300">Real-time Data</span>
                </div>
                <div className="flex items-center gap-3 px-5 py-3 rounded-xl bg-white/5 border border-white/10">
                  <svg className="w-6 h-6 text-emerald-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                  </svg>
                  <span className="text-lg text-gray-300">AI-Powered</span>
                </div>
                <div className="flex items-center gap-3 px-5 py-3 rounded-xl bg-white/5 border border-white/10">
                  <svg className="w-6 h-6 text-emerald-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                  </svg>
                  <span className="text-lg text-gray-300">Comprehensive Analysis</span>
                </div>
              </div>
            </div>
            <div className="relative">
              <div className="absolute inset-0 bg-gradient-to-r from-indigo-500/20 via-purple-500/20 to-pink-500/20 rounded-3xl blur-3xl" />
              <div className="relative glass-card p-8">
                <div className="grid grid-cols-2 gap-6">
                  <div className="text-center p-6 rounded-2xl bg-white/5">
                    <div className="text-5xl font-bold gradient-text mb-2">4</div>
                    <div className="text-lg text-gray-400">AI Agents</div>
                  </div>
                  <div className="text-center p-6 rounded-2xl bg-white/5">
                    <div className="text-5xl font-bold gradient-text mb-2">100+</div>
                    <div className="text-lg text-gray-400">Metrics</div>
                  </div>
                  <div className="text-center p-6 rounded-2xl bg-white/5">
                    <div className="text-5xl font-bold gradient-text mb-2">24/7</div>
                    <div className="text-lg text-gray-400">Available</div>
                  </div>
                  <div className="text-center p-6 rounded-2xl bg-white/5">
                    <div className="text-5xl font-bold gradient-text mb-2">&lt;30s</div>
                    <div className="text-lg text-gray-400">Analysis</div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Final CTA Section */}
      <section className="py-16">
        <div className="relative">
          <div className="absolute inset-0 bg-gradient-to-r from-indigo-500/10 via-purple-500/10 to-pink-500/10 rounded-3xl" />
          <div className="relative glass-card p-12 lg:p-16 text-center">
            <h2 className="text-4xl lg:text-5xl font-bold text-white mb-6">
              Ready to Make Smarter <span className="gradient-text">Investment Decisions?</span>
            </h2>
            <p className="text-xl text-gray-400 mb-10 max-w-2xl mx-auto">
              Start analyzing any stock with our AI-powered platform. Get instant insights backed by comprehensive data analysis from multiple AI agents.
            </p>
            <Link href="/analyze" className="btn-primary px-12 py-5 text-xl inline-flex items-center">
              Start Analyzing Now
              <svg className="w-6 h-6 ml-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7l5 5m0 0l-5 5m5-5H6" />
              </svg>
            </Link>
          </div>
        </div>
      </section>
    </div>
  )
}
