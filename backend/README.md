# Helomet Recreate Backend

This is the backend service for the Helomet (Helmet and Vest) PPE (Personal Protective Equipment) Detection system. 
It uses a YOLOv8 model compiled to a TensorRT engine for high-performance inference, particularly optimized for NVIDIA edge devices (like Jetson).

## Architecture

The backend operates as a continuous pipeline:
1. **Video Capture**: Reads frames from a webcam, video file, or RTSP stream using OpenCV.
2. **Inference**: Preprocesses frames and runs them through a TensorRT engine to detect `helmet`, `no_helmet`, `vest`, and `no_vest`.
3. **Telemetry**: Calculates FPS, memory usage, CPU, and GPU load. Publishes detection metrics and hardware utilization via MQTT.
4. **Streaming**: Draws bounding boxes on the frame and pipes the output via FFmpeg to a MediaMTX RTSP server.

## Requirements

Ensure you have Python 3.8+ installed. The main dependencies are:
- `numpy`
- `opencv-python`
- `paho-mqtt`
- `pycuda`
- `tensorrt`
- `psutil`
- `pynvml` (if running with NVIDIA GPU monitoring)
- `ffmpeg` (must be installed on the system)

You can install the Python dependencies using:
```bash
pip install -r requirements.txt
```

## Configuration

Configuration is handled via a `.env` file and `config/settings.py`. Create a `.env` file in the `backend` directory based on these parameters:

```env
# Model Settings
MODEL_ENGINE_PATH=model/besthelomet_yolov8_MX230.engine
INPUT_WIDTH=640
INPUT_HEIGHT=640
CONF_THRESHOLD=0.25
NMS_THRESHOLD=0.45

# Application Settings
WEBCAM_INDEX=0 # 0 for default webcam, or path to video file / RTSP URL
PROCESS_EVERY_N_FRAMES=5 # Skip frames for performance

# MQTT Settings
MQTT_BROKER=
MQTT_PORT=
MQTT_TOPIC=
MQTT_USER=
MQTT_PASSWORD=

# Stream Settings
STREAM_URL=rtsp://localhost:8554/mystream
```

## Running the Backend

1. Make sure `ffmpeg` is accessible in your system's PATH.
2. Ensure MediaMTX (or your chosen RTSP server) is running and accessible at `STREAM_URL`.
3. Start the application:
```bash
python run.py
```

## Code Structure
- `app/main_app.py`: Core logic combining video reading, inference, drawing, MQTT, and RTSP streaming.
- `config/settings.py`: Configuration loader.
- `core/engine.py`: TensorRT engine initialization and inference handling.
- `helpers/image.py`: Image preprocessing (resize, normalize) and postprocessing (NMS).
- `helpers/visualizer.py`: Drawing bounding boxes and metrics onto frames.
- `services/mqtt_service.py`: MQTT publishing client.
