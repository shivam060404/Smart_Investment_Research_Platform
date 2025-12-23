'use client'

interface ConfidenceGaugeProps {
  score: number
  size?: 'sm' | 'md' | 'lg'
  label?: string
}

export default function ConfidenceGauge({ score, size = 'md', label }: ConfidenceGaugeProps) {
  const sizeConfig = {
    sm: { width: 120, height: 60, strokeWidth: 8, fontSize: 'text-xl' },
    md: { width: 160, height: 80, strokeWidth: 10, fontSize: 'text-2xl' },
    lg: { width: 200, height: 100, strokeWidth: 12, fontSize: 'text-3xl' },
  }

  const config = sizeConfig[size]
  const radius = config.width / 2 - config.strokeWidth
  const circumference = Math.PI * radius
  const progress = (score / 100) * circumference

  // Determine color based on score
  const getColor = () => {
    if (score >= 70) return { stroke: 'url(#gaugeGradientHigh)', glow: 'rgba(56, 239, 125, 0.5)' }
    if (score >= 40) return { stroke: 'url(#gaugeGradientMedium)', glow: 'rgba(245, 166, 35, 0.5)' }
    return { stroke: 'url(#gaugeGradientLow)', glow: 'rgba(244, 92, 67, 0.5)' }
  }

  const colorConfig = getColor()

  return (
    <div className="flex flex-col items-center">
      <div className="relative" style={{ width: config.width, height: config.height }}>
        <svg
          width={config.width}
          height={config.height + 10}
          viewBox={`0 0 ${config.width} ${config.height + 10}`}
          className="overflow-visible"
        >
          <defs>
            <linearGradient id="gaugeGradientHigh" x1="0%" y1="0%" x2="100%" y2="0%">
              <stop offset="0%" stopColor="#11998e" />
              <stop offset="100%" stopColor="#38ef7d" />
            </linearGradient>
            <linearGradient id="gaugeGradientMedium" x1="0%" y1="0%" x2="100%" y2="0%">
              <stop offset="0%" stopColor="#f5a623" />
              <stop offset="100%" stopColor="#f093fb" />
            </linearGradient>
            <linearGradient id="gaugeGradientLow" x1="0%" y1="0%" x2="100%" y2="0%">
              <stop offset="0%" stopColor="#eb3349" />
              <stop offset="100%" stopColor="#f45c43" />
            </linearGradient>
            <filter id="glow">
              <feGaussianBlur stdDeviation="3" result="coloredBlur" />
              <feMerge>
                <feMergeNode in="coloredBlur" />
                <feMergeNode in="SourceGraphic" />
              </feMerge>
            </filter>
          </defs>

          {/* Background arc */}
          <path
            d={`M ${config.strokeWidth} ${config.height} A ${radius} ${radius} 0 0 1 ${config.width - config.strokeWidth} ${config.height}`}
            fill="none"
            stroke="rgba(255, 255, 255, 0.1)"
            strokeWidth={config.strokeWidth}
            strokeLinecap="round"
          />

          {/* Progress arc */}
          <path
            d={`M ${config.strokeWidth} ${config.height} A ${radius} ${radius} 0 0 1 ${config.width - config.strokeWidth} ${config.height}`}
            fill="none"
            stroke={colorConfig.stroke}
            strokeWidth={config.strokeWidth}
            strokeLinecap="round"
            strokeDasharray={circumference}
            strokeDashoffset={circumference - progress}
            filter="url(#glow)"
            style={{
              transition: 'stroke-dashoffset 1s ease-out',
            }}
          />

          {/* Tick marks */}
          {[0, 25, 50, 75, 100].map((tick) => {
            const angle = Math.PI - (tick / 100) * Math.PI
            const x1 = config.width / 2 + (radius - 15) * Math.cos(angle)
            const y1 = config.height - (radius - 15) * Math.sin(angle)
            const x2 = config.width / 2 + (radius - 5) * Math.cos(angle)
            const y2 = config.height - (radius - 5) * Math.sin(angle)
            
            return (
              <line
                key={tick}
                x1={x1}
                y1={y1}
                x2={x2}
                y2={y2}
                stroke="rgba(255, 255, 255, 0.3)"
                strokeWidth={2}
              />
            )
          })}
        </svg>

        {/* Center indicator */}
        <div 
          className="absolute bottom-0 left-1/2 -translate-x-1/2 w-4 h-4 rounded-full bg-white shadow-lg"
          style={{ boxShadow: `0 0 20px ${colorConfig.glow}` }}
        />
      </div>

      {/* Scale labels */}
      <div className="flex justify-between w-full mt-2 px-2">
        <span className="text-xs text-gray-500">0</span>
        <span className="text-xs text-gray-500">50</span>
        <span className="text-xs text-gray-500">100</span>
      </div>
    </div>
  )
}
