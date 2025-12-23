import type { Config } from 'tailwindcss'

const config: Config = {
  content: [
    './src/pages/**/*.{js,ts,jsx,tsx,mdx}',
    './src/components/**/*.{js,ts,jsx,tsx,mdx}',
    './src/app/**/*.{js,ts,jsx,tsx,mdx}',
  ],
  theme: {
    extend: {
      colors: {
        // Primary brand color
        primary: {
          50: '#eff6ff',
          100: '#dbeafe',
          200: '#bfdbfe',
          300: '#93c5fd',
          400: '#60a5fa',
          500: '#3b82f6',
          600: '#2563eb',
          700: '#1d4ed8',
          800: '#1e40af',
          900: '#1e3a8a',
        },
        // Financial signal colors
        bullish: {
          DEFAULT: '#22c55e',
          light: '#86efac',
          dark: '#15803d',
          bg: 'rgba(34, 197, 94, 0.1)',
        },
        bearish: {
          DEFAULT: '#ef4444',
          light: '#fca5a5',
          dark: '#b91c1c',
          bg: 'rgba(239, 68, 68, 0.1)',
        },
        neutral: {
          DEFAULT: '#eab308',
          light: '#fde047',
          dark: '#a16207',
          bg: 'rgba(234, 179, 8, 0.1)',
        },
        // Chart colors
        chart: {
          fundamental: '#8b5cf6', // Purple
          technical: '#06b6d4',   // Cyan
          sentiment: '#f59e0b',   // Amber
          risk: '#ec4899',        // Pink
        },
      },
      animation: {
        'pulse-slow': 'pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite',
        'spin-slow': 'spin 2s linear infinite',
      },
      boxShadow: {
        'glow-green': '0 0 20px rgba(34, 197, 94, 0.3)',
        'glow-red': '0 0 20px rgba(239, 68, 68, 0.3)',
        'glow-yellow': '0 0 20px rgba(234, 179, 8, 0.3)',
      },
    },
  },
  plugins: [],
}

export default config
