import React from 'react'

/**
 * SystemStatsBar — a horizontal bar displaying edge device system metrics
 * pulled from the telemetry.system_status object.
 */
function SystemStatsBar({ systemStatus }) {
  const stats = systemStatus ?? {}
  const entries = Object.entries(stats)

  if (!entries.length) return null

  return (
    <div className="flex flex-wrap gap-x-6 gap-y-2 bg-gray-800 rounded-lg px-4 py-3 border border-gray-700">
      {entries.map(([key, val]) => (
        <div key={key} className="flex items-center gap-2 font-mono text-xs">
          <span className="text-gray-500 uppercase tracking-wider">{key}</span>
          <span className="text-green-400 font-semibold">{val}</span>
        </div>
      ))}
    </div>
  )
}

export default SystemStatsBar
