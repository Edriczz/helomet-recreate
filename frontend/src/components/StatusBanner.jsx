import React from 'react'

/**
 * StatusBanner — displays the current security_level with dynamic colour coding.
 * SAFE   → dark green (bg-green-900)
 * UNSAFE → pulsing dark red  (bg-red-900)
 * other  → neutral dark grey
 */
function StatusBanner({ securityLevel, mqttStatus }) {
  const isSafe = securityLevel === 'SAFE'
  const isUnsafe = securityLevel === 'UNSAFE'

  const bannerClasses = [
    'w-full py-5 px-6 flex flex-col sm:flex-row items-center justify-between gap-3 rounded-lg border transition-colors duration-700',
    isSafe
      ? 'bg-green-900 border-green-700'
      : isUnsafe
      ? 'bg-red-900 border-red-700 animate-pulse'
      : 'bg-gray-800 border-gray-700',
  ].join(' ')

  const mqttDot = {
    connected: 'bg-green-400',
    connecting: 'bg-yellow-400 animate-pulse',
    disconnected: 'bg-gray-500',
    error: 'bg-red-500',
  }[mqttStatus] ?? 'bg-gray-500'

  return (
    <div className={bannerClasses} role="status" aria-live="polite">
      {/* Left — security level */}
      <div className="flex items-center gap-4">
        <span className="text-3xl sm:text-4xl font-black tracking-widest uppercase text-white">
          {securityLevel}
        </span>
        {isUnsafe && (
          <span className="text-sm font-semibold bg-red-700 text-red-100 px-3 py-1 rounded-full uppercase tracking-wide">
            ⚠ Alert
          </span>
        )}
        {isSafe && (
          <span className="text-sm font-semibold bg-green-700 text-green-100 px-3 py-1 rounded-full uppercase tracking-wide">
            ✓ All Clear
          </span>
        )}
      </div>

      {/* Right — MQTT connection indicator */}
      <div className="flex items-center gap-2 text-xs text-gray-300 font-mono">
        <span className={`w-2.5 h-2.5 rounded-full ${mqttDot}`} />
        <span className="uppercase tracking-widest">MQTT · {mqttStatus}</span>
      </div>
    </div>
  )
}

export default StatusBanner
