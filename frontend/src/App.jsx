import React from 'react'
import { useMQTT } from './hooks/useMQTT'
import StatusBanner from './components/StatusBanner'
import CounterGrid from './components/CounterGrid'
import SystemStatsBar from './components/SystemStatsBar'

function App() {
  const { telemetry, status } = useMQTT()

  return (
    <div className="min-h-screen bg-gray-900 text-white flex flex-col">
      {/* ── Top Header ────────────────────────────────────────────── */}
      <header className="px-6 pt-6 pb-2 flex items-center justify-between border-b border-gray-800">
        <div>
          <h1 className="text-xl font-bold tracking-tight text-white">
            ⛑ Helomet Dashboard
          </h1>
          <p className="text-xs text-gray-500 mt-0.5 font-mono">
            {import.meta.env.VITE_MQTT_TOPIC}
          </p>
        </div>
        <span className="text-xs text-gray-600 font-mono tabular-nums">
          {new Date().toLocaleString()}
        </span>
      </header>

      {/* ── Main Content ──────────────────────────────────────────── */}
      <main className="flex-1 flex flex-col gap-5 p-6 max-w-7xl w-full mx-auto">

        {/* 1 · Security Status Banner */}
        <section aria-label="Security Status">
          <StatusBanner
            securityLevel={telemetry.security_level}
            mqttStatus={status}
          />
        </section>

        {/* 2 · Object Counter Grid */}
        <section aria-label="Detection Counters">
          <h2 className="text-xs font-semibold uppercase tracking-widest text-gray-500 mb-3">
            Detection Counts
          </h2>
          <CounterGrid telemetry={telemetry} />
        </section>

        {/* 3 · System Stats Bar */}
        <section aria-label="System Status">
          <h2 className="text-xs font-semibold uppercase tracking-widest text-gray-500 mb-3">
            Edge Device Metrics
          </h2>
          <SystemStatsBar systemStatus={telemetry.system_status} />
        </section>

        {/* 4 · Live WebRTC Video Stream (MediaMTX) */}
        <section aria-label="Video Stream">
          <h2 className="text-xs font-semibold uppercase tracking-widest text-gray-500 mb-3">
            Live Video Stream
          </h2>
          {import.meta.env.VITE_WEBRTC_URL ? (
            <iframe
              src={import.meta.env.VITE_WEBRTC_URL}
              title="Live WebRTC Stream — Helomet"
              allow="autoplay"
              className="w-full aspect-video rounded-lg border border-gray-700 bg-gray-950"
            />
          ) : (
            <div className="w-full aspect-video rounded-lg border border-dashed border-gray-700 bg-gray-950 flex items-center justify-center text-gray-600 text-xs font-mono uppercase tracking-widest">
              VITE_WEBRTC_URL not configured
            </div>
          )}
        </section>

      </main>
    </div>
  )
}

export default App
