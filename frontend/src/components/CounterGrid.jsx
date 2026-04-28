import React from 'react'

/**
 * Metric definition — maps telemetry keys to human-readable labels and icons.
 */
const METRICS = [
  { key: 'person',     label: 'Person',     icon: '🧍', alert: false },
  { key: 'helmet',     label: 'Helmet',     icon: '⛑️',  alert: false },
  { key: 'no_helmet',  label: 'No Helmet',  icon: '🚫',  alert: true  },
  { key: 'vest',       label: 'Vest',       icon: '🦺',  alert: false },
  { key: 'no_vest',    label: 'No Vest',    icon: '🚫',  alert: true  },
]

/**
 * CounterCard — individual metric tile.
 */
function CounterCard({ label, icon, value, isAlert }) {
  const triggered = isAlert && value > 0

  const cardClasses = [
    'relative flex flex-col items-center justify-center gap-2 rounded-xl p-5 border-2 transition-all duration-500',
    triggered
      ? 'bg-red-950 border-red-500 shadow-[0_0_18px_rgba(239,68,68,0.35)]'
      : 'bg-gray-800 border-gray-700 hover:border-gray-500',
  ].join(' ')

  return (
    <div className={cardClasses} role="group" aria-label={`${label}: ${value}`}>
      {triggered && (
        <span className="absolute top-2 right-2 w-2.5 h-2.5 rounded-full bg-red-500 animate-ping" />
      )}
      <span className="text-3xl select-none">{icon}</span>
      <span
        className={[
          'text-5xl font-black tabular-nums',
          triggered ? 'text-red-400' : 'text-white',
        ].join(' ')}
      >
        {value}
      </span>
      <span
        className={[
          'text-xs font-semibold uppercase tracking-widest',
          triggered ? 'text-red-300' : 'text-gray-400',
        ].join(' ')}
      >
        {label}
      </span>
    </div>
  )
}

/**
 * CounterGrid — responsive 5-card grid driven by telemetry data.
 */
function CounterGrid({ telemetry }) {
  return (
    <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-4">
      {METRICS.map(({ key, label, icon, alert }) => (
        <CounterCard
          key={key}
          label={label}
          icon={icon}
          value={telemetry[key] ?? 0}
          isAlert={alert}
        />
      ))}
    </div>
  )
}

export default CounterGrid
