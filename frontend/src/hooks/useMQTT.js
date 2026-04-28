import { useState, useEffect, useRef } from 'react'
import mqtt from 'mqtt'

const BROKER_URL = import.meta.env.VITE_MQTT_BROKER
const TOPIC = import.meta.env.VITE_MQTT_TOPIC

/**
 * Default telemetry shape — used as initial state before first message arrives.
 */
const DEFAULT_TELEMETRY = {
  person: 0,
  helmet: 0,
  no_helmet: 0,
  vest: 0,
  no_vest: 0,
  security_level: 'UNKNOWN',
  system_status: {
    'Video FPS': '--',
    Inference: '--',
    'App CPU': '--',
    'App RAM': '--',
    'GPU Load': '--',
    'VRAM Use': '--',
  },
}

/**
 * useMQTT — connects to the MQTT broker over WebSocket, subscribes to the
 * telemetry topic, and returns the latest parsed payload along with the
 * current connection status.
 *
 * @returns {{ telemetry: object, status: 'connecting'|'connected'|'disconnected'|'error' }}
 */
export function useMQTT() {
  const [telemetry, setTelemetry] = useState(DEFAULT_TELEMETRY)
  const [status, setStatus] = useState('connecting')
  const clientRef = useRef(null)

  useEffect(() => {
    if (!BROKER_URL || !TOPIC) {
      console.error('[useMQTT] VITE_MQTT_BROKER or VITE_MQTT_TOPIC is not defined.')
      setStatus('error')
      return
    }

    const client = mqtt.connect(BROKER_URL, {
      clientId: `helomet_dashboard_${Math.random().toString(16).slice(2, 10)}`,
      clean: true,
      reconnectPeriod: 3000,   // retry every 3 s on disconnect
      connectTimeout: 10000,
    })

    clientRef.current = client

    client.on('connect', () => {
      console.info('[useMQTT] Connected to broker:', BROKER_URL)
      setStatus('connected')
      client.subscribe(TOPIC, { qos: 0 }, (err) => {
        if (err) {
          console.error('[useMQTT] Subscription error:', err)
          setStatus('error')
        } else {
          console.info('[useMQTT] Subscribed to topic:', TOPIC)
        }
      })
    })

    client.on('message', (_topic, payload) => {
      try {
        const data = JSON.parse(payload.toString())
        setTelemetry((prev) => ({ ...prev, ...data }))
      } catch (err) {
        console.warn('[useMQTT] Failed to parse message payload:', err)
      }
    })

    client.on('reconnect', () => {
      console.info('[useMQTT] Reconnecting…')
      setStatus('connecting')
    })

    client.on('offline', () => {
      console.warn('[useMQTT] Client went offline.')
      setStatus('disconnected')
    })

    client.on('error', (err) => {
      console.error('[useMQTT] Client error:', err)
      setStatus('error')
    })

    return () => {
      client.end(true)
      console.info('[useMQTT] Client disconnected on cleanup.')
    }
  }, []) // run once on mount

  return { telemetry, status }
}
