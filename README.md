# Helomet Recreate System

The Helomet Recreate project is a comprehensive Personal Protective Equipment (PPE) detection system designed for real-time inference on edge devices (like the NVIDIA Jetson). It detects helmets and safety vests and broadcasts the live video stream along with telemetry data.

## System Architecture

The system is composed of three main components:

1. **Backend (AI Inference Engine)**
   - Written in Python.
   - Utilizes a YOLOv8 model compiled to TensorRT (`.engine`) for highly optimized, low-latency object detection on NVIDIA hardware.
   - Captures video (webcam, file, or RTSP), runs inference, overlays bounding boxes, and streams the annotated video via FFmpeg to an RTSP server.
   - Broadcasts detection metrics (counts) and system health (CPU, RAM, GPU, FPS) to an MQTT broker.
   - [Backend Documentation](./backend/README.md)

2. **RTSP Server (MediaMTX)**
   - A lightweight, ready-to-use RTSP/WebRTC/HLS server.
   - Receives the FFmpeg stream from the backend and makes it accessible to multiple clients (including the frontend).
   - Located in the `mediamtx` folder.

3. **Frontend (Dashboard)**
   - Built with React, Vite, and Tailwind CSS.
   - Connects to the MQTT broker via WebSockets to display real-time metrics and detection statistics.
   - Can display the live stream originating from the MediaMTX server.
   - [Frontend Documentation](./frontend/README.md)

---

## Global Data Flow

1. **Video Source** -> `Backend`
2. `Backend` -> *(FFmpeg)* -> `MediaMTX RTSP Server` -> `Frontend` (Video feed)
3. `Backend` -> *(MQTT)* -> `MQTT Broker` -> `Frontend` (Telemetry & Stats)

---

## How to Run the Complete System

Follow these steps to spin up the entire architecture locally:

### Step 1: Start the RTSP Server (MediaMTX)
Navigate to the MediaMTX folder and run the executable.
```bash
cd mediamtx/mediamtx_v1.16.3_linux_amd64
./mediamtx
```
This will start an RTSP server on `rtsp://localhost:8554`.

### Step 2: Start the Backend Inference
Open a new terminal.
Make sure you have configured your `backend/.env` properly (especially `STREAM_URL=rtsp://localhost:8554/mystream` and the `MQTT_BROKER` settings).

```bash
cd backend
# pip install -r requirements.txt # (if not already installed)
python run.py
```
*Note: Ensure `ffmpeg` is installed on your system. You should see logs indicating the model is loaded and FFmpeg is streaming.*

### Step 3: Start the Frontend Dashboard
Open a third terminal.
Make sure you have configured your `frontend/.env` to point to the correct WebSocket MQTT broker.

```bash
cd frontend
# npm install # (if not already installed)
npm run dev
```
Open your browser to the URL provided by Vite (usually `http://localhost:5173`) to view the real-time dashboard and stream.

## Requirements Summary
- Python 3.8+ (for backend)
- Node.js & npm (for frontend)
- FFmpeg (for backend streaming)
- TensorRT & CUDA capabilities (for high FPS inference)
- Access to an MQTT broker (local or remote)
