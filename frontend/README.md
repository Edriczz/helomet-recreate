# Helomet Recreate Frontend

This is the frontend dashboard for the Helomet PPE Detection system. It is built using React, Vite, and Tailwind CSS.

## Overview

The frontend connects to the system via MQTT to receive real-time telemetry data (FPS, CPU/GPU usage, memory) and detection results. It can also embed the RTSP stream (typically via a WebRTC or HLS proxy, or an iframe depending on the setup) to provide a live view of the detections.

## Technologies Used

- **React 19**: UI Library.
- **Vite**: Fast build tool and dev server.
- **Tailwind CSS**: Utility-first CSS framework for styling.
- **MQTT.js**: To subscribe to the telemetry topic broadcasted by the backend.

## Prerequisites

- Node.js (v18+ recommended)
- npm or yarn

## Installation

1. Navigate to the frontend directory.
2. Install the dependencies:
```bash
npm install
```

## Configuration

Create a `.env` file in the `frontend` root to configure the MQTT connection and stream URL.

```env
VITE_MQTT_BROKER=ws://broker.xdevelopment.my.id:8083/mqtt
VITE_MQTT_TOPIC=ai/telemetry
VITE_MQTT_USER=nodered
VITE_MQTT_PASSWORD=nodered
```
*(Note: Ensure you are using the WebSocket port for MQTT in the browser, typically 8083 or 9001).*

## Running Locally

Start the Vite development server:
```bash
npm run dev
```
The application will usually be available at `http://localhost:5173`.

## Building for Production

To build the optimized static files for production deployment:
```bash
npm run build
```
The output will be in the `dist` folder.
