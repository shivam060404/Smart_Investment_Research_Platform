'use client'

import { useState } from 'react'
import ChatBot from './ChatBot'

interface ChatButtonProps {
  ticker?: string
}

export default function ChatButton({ ticker }: ChatButtonProps) {
  const [isOpen, setIsOpen] = useState(false)

  return (
    <>
      {/* Floating Chat Button - Larger and More Prominent */}
      {!isOpen && (
        <button
          onClick={() => setIsOpen(true)}
          className="fixed bottom-8 right-8 z-40 group"
          aria-label="Open AI Assistant"
        >
          {/* Glow effect */}
          <div className="absolute inset-0 w-16 h-16 rounded-2xl bg-gradient-to-r from-indigo-500 to-purple-500 blur-lg opacity-50 group-hover:opacity-75 transition-opacity" />
          
          {/* Button */}
          <div className="relative w-16 h-16 rounded-2xl bg-gradient-to-r from-indigo-500 to-purple-500 text-white shadow-xl shadow-purple-500/40 hover:shadow-purple-500/60 transition-all hover:scale-105 flex items-center justify-center">
            <svg className="w-7 h-7" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M8 10h.01M12 10h.01M16 10h.01M9 16H5a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v8a2 2 0 01-2 2h-5l-5 5v-5z"
              />
            </svg>
            
            {/* Pulse indicator */}
            <span className="absolute -top-1 -right-1 w-4 h-4 rounded-full bg-emerald-400 border-2 border-gray-900 animate-pulse" />
          </div>
          
          {/* Tooltip */}
          <div className="absolute bottom-full right-0 mb-3 px-3 py-2 rounded-lg bg-gray-900 border border-white/20 text-white text-sm font-medium whitespace-nowrap opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none">
            AI Research Assistant
            <div className="absolute top-full right-4 w-0 h-0 border-l-8 border-r-8 border-t-8 border-transparent border-t-gray-900" />
          </div>
        </button>
      )}

      {/* Chat Window */}
      <ChatBot ticker={ticker} isOpen={isOpen} onClose={() => setIsOpen(false)} />
    </>
  )
}
