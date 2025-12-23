'use client'

import { Inter } from 'next/font/google'
import Link from 'next/link'
import { usePathname } from 'next/navigation'
import './globals.css'
import ChatButton from '@/components/ChatButton'

const inter = Inter({ subsets: ['latin'], weight: ['300', '400', '500', '600', '700', '800'] })

const navigation = [
  { name: 'Analyze', href: '/', icon: 'M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z' },
  { name: 'Backtest', href: '/backtest', icon: 'M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z' },
  { name: 'Portfolio', href: '/portfolio', icon: 'M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10' },
]

function NavLink({ href, name, icon }: { href: string; name: string; icon: string }) {
  const pathname = usePathname()
  const isActive = pathname === href || (href !== '/' && pathname.startsWith(href))

  return (
    <Link
      href={href}
      className={`nav-link flex items-center gap-2 ${isActive ? 'active' : ''}`}
    >
      <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d={icon} />
      </svg>
      <span>{name}</span>
    </Link>
  )
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en" className="h-full">
      <head>
        <title>AI Investment Research Platform</title>
        <meta name="description" content="Professional AI-powered investment analysis and recommendations" />
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="anonymous" />
      </head>
      <body className={`${inter.className} min-h-screen flex flex-col`}>
        <div className="relative flex flex-col min-h-screen z-10">
          {/* Premium Navigation Header - Full Width */}
          <nav className="sticky top-0 z-50 backdrop-blur-glass border-b border-white/10 w-full">
            <div className="w-full px-6 xl:px-12">
              <div className="flex h-16 items-center justify-center relative">
                {/* Logo - Positioned Left */}
                <Link href="/" className="flex items-center gap-3 group absolute left-0">
                  <div className="relative">
                    <div className="w-10 h-10 rounded-xl flex items-center justify-center bg-gradient-to-br from-indigo-500 via-purple-500 to-pink-500 shadow-lg shadow-purple-500/30 group-hover:shadow-purple-500/50 transition-shadow">
                      <svg
                        className="w-6 h-6 text-white"
                        fill="none"
                        stroke="currentColor"
                        viewBox="0 0 24 24"
                      >
                        <path
                          strokeLinecap="round"
                          strokeLinejoin="round"
                          strokeWidth={2}
                          d="M13 7h8m0 0v8m0-8l-8 8-4-4-6 6"
                        />
                      </svg>
                    </div>
                  </div>
                  <div>
                    <span className="text-xl font-bold text-white tracking-tight">
                      AI<span className="gradient-text">Invest</span>
                    </span>
                    <p className="text-[10px] text-gray-400 -mt-1">Research Platform</p>
                  </div>
                </Link>

                {/* Navigation Links - Centered */}
                <div className="flex items-center gap-1">
                  {navigation.map((item) => (
                    <NavLink key={item.name} href={item.href} name={item.name} icon={item.icon} />
                  ))}
                </div>
              </div>
            </div>
          </nav>

          {/* Main Content - Full Width */}
          <main className="flex-1 w-full px-6 xl:px-12 py-8">
            {children}
          </main>

          {/* Floating Chat Button */}
          <ChatButton />

          {/* Footer - Always at Bottom */}
          <footer className="w-full border-t border-white/10 backdrop-blur-glass mt-auto">
            <div className="w-full px-6 xl:px-12 py-4">
              <div className="flex flex-col md:flex-row items-center justify-between gap-4">
                <div className="flex items-center gap-3">
                  <div className="w-6 h-6 rounded-lg flex items-center justify-center bg-gradient-to-br from-indigo-500 to-purple-500">
                    <svg className="w-4 h-4 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7h8m0 0v8m0-8l-8 8-4-4-6 6" />
                    </svg>
                  </div>
                  <span className="text-gray-400 text-sm">
                    © 2025 AIInvest Research Platform
                  </span>
                </div>
                <p className="text-center text-xs text-gray-500">
                  For educational purposes only. Not financial advice. Past performance does not guarantee future results.
                </p>
                <div className="flex items-center gap-3">
                  <span className="text-xs text-gray-500">Powered by AI Agents</span>
                  <div className="flex -space-x-1.5">
                    <div className="w-5 h-5 rounded-full bg-gradient-to-br from-indigo-500 to-purple-500 border-2 border-gray-900" />
                    <div className="w-5 h-5 rounded-full bg-gradient-to-br from-emerald-500 to-teal-500 border-2 border-gray-900" />
                    <div className="w-5 h-5 rounded-full bg-gradient-to-br from-pink-500 to-rose-500 border-2 border-gray-900" />
                    <div className="w-5 h-5 rounded-full bg-gradient-to-br from-orange-500 to-red-500 border-2 border-gray-900" />
                  </div>
                </div>
              </div>
            </div>
          </footer>
        </div>
      </body>
    </html>
  )
}
