---
trigger: always_on
---

# FRONTEND ARCHITECTURE RULES & PROTOCOLS

1. **Tech Stack Constraint:** Strictly use React (Vite) and Tailwind CSS. Do NOT install or use any other UI component libraries (e.g., Material UI, Bootstrap, Chakra) unless explicitly requested.

2. **Single Source of Truth:** The application is a real-time Industrial IoT Dashboard. The ONLY data source is a local MQTT Broker (via WebSocket). It is STRICTLY FORBIDDEN to create mock backends, dummy JSON files, or REST API fetches for telemetry data.

3. **Visual Design:** Implement a strict Dark Mode theme suitable for long-hour CCTV monitoring. Use deep dark backgrounds (e.g., `bg-gray-900`) and high-contrast text. Prioritize utilitarian design over unnecessary visual clutter.

4. **Coding Standard:** Write clean, modular code using Functional Components and React Hooks. You must separate the UI rendering logic from the state management/MQTT logic by utilizing custom hooks (e.g., `useMQTT.js`).

5. **Agent Autonomy Limit:** Do NOT execute destructive terminal commands or install heavy NPM packages without asking for confirmation first. Always provide a clear explanation of what component you are about to build or modify.