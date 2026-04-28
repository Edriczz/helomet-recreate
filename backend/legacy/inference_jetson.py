import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import cv2
import time
import json
import sys
import os

# --- INTEGRATION ---
import paho.mqtt.client as mqtt
import jetson.utils

# --- 1. CONFIGURATION ---
ENGINE_PATH = "model/besthelomet_yolov8.onnx.1.1.8201.GPU.FP16.engine"
WEBCAM_INDEX = 0
INPUT_W = 640
INPUT_H = 640

CATEGORIES = ["person", "boots", "helmet", "no_boots", "no_helmet", "no_vest", "vest"]
NUM_CLASSES = len(CATEGORIES)

# Custom color mapping (BGR format for OpenCV)
CLASS_COLORS = {
    "person": (255, 255, 0),        # Cyan
    "boots": (0, 255, 0),           # Green
    "helmet": (50, 100, 200),       # Dark Blue (more saturated)
    "no_boots": (0, 0, 255),        # Red
    "no_helmet": (200, 100, 200),   # Bright Purple (high contrast with dark blue)
    "no_vest": (60, 119, 251),      # Orange (#FB773C in BGR)
    "vest": (120, 54, 235)          # Pink/Magenta (#EB3678 in BGR)
}

CONF_THRESHOLD = 0.25
NMS_THRESHOLD = 0.45

MQTT_BROKER = "broker.xdevelopment.my.id"
MQTT_TOPIC = "ai/portable_helomet/showcase/scase_cam1/telemetry"
MQTT_USER = "nodered"
MQTT_PASSWORD = "nodered"

STREAM_URL = "rtmp://100.109.124.85:1936/portable-helomet"

CLASS_TO_KEY = {0: "person", 2: "helmet", 4: "no_helmet", 6: "vest", 5: "no_vest"}

# CRITICAL OPTIMIZATION: Process every Nth frame
PROCESS_EVERY_N_FRAMES = 5  # Increase to 6-8 for even lower CPU usage

# Pre-allocate padded image buffer
PADDED_BUFFER = np.full((INPUT_H, INPUT_W, 3), 114, dtype=np.uint8)

# --- 2. TENSORRT SETUP ---
class TRTInference:
    def __init__(self, engine_path):
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.engine = self._load_engine(engine_path)
        if self.engine is None:
            raise RuntimeError("Failed to load TensorRT engine")

        self.context = self.engine.create_execution_context()
        self.stream = cuda.Stream()

        # Allocate buffers
        self.inputs, self.outputs, self.bindings = self._allocate_buffers()

    def _load_engine(self, engine_path):
        if not os.path.exists(engine_path):
            print(f"Engine not found: {engine_path}")
            return None
        try:
            with open(engine_path, "rb") as f:
                runtime = trt.Runtime(self.logger)
                return runtime.deserialize_cuda_engine(f.read())
        except Exception as e:
            print(f"Error loading engine: {e}")
            return None

    def _allocate_buffers(self):
        inputs = []
        outputs = []
        bindings = []

        for i in range(self.engine.num_bindings):
            shape = self.engine.get_binding_shape(i)
            dtype = trt.nptype(self.engine.get_binding_dtype(i))
            size = trt.volume(shape) * self.engine.max_batch_size

            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            bindings.append(int(device_mem))

            if self.engine.binding_is_input(i):
                inputs.append({'host': host_mem, 'device': device_mem})
            else:
                outputs.append({'host': host_mem, 'device': device_mem})

        return inputs, outputs, bindings

    def infer(self, image_batch):
        # Copy input to device
        self.inputs[0]['host'][:] = image_batch.ravel()
        cuda.memcpy_htod_async(self.inputs[0]['device'], self.inputs[0]['host'], self.stream)

        # Execute
        self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)

        # Copy output to host
        cuda.memcpy_dtoh_async(self.outputs[0]['host'], self.outputs[0]['device'], self.stream)
        self.stream.synchronize()

        return self.outputs[0]['host']

# --- 3. PREPROCESSING ---
def preprocess_frame_fast(frame):
    """Ultra-fast preprocessing with minimal copies"""
    h, w = frame.shape[:2]
    ratio = min(INPUT_W / w, INPUT_H / h)
    new_w, new_h = int(w * ratio), int(h * ratio)

    # Resize once
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Use pre-allocated buffer
    PADDED_BUFFER[:] = 114
    dw = (INPUT_W - new_w) // 2
    dh = (INPUT_H - new_h) // 2
    PADDED_BUFFER[dh:dh + new_h, dw:dw + new_w] = resized

    # Normalize and transpose in one go
    batch = np.ascontiguousarray(PADDED_BUFFER.transpose(2, 0, 1), dtype=np.float32) / 255.0

    return batch, dw, dh, ratio

# --- 4. POSTPROCESSING ---
def postprocess_fast(raw_output):
    """Optimized postprocessing"""
    prediction = raw_output.reshape((1, NUM_CLASSES + 4, 8400)).transpose(0, 2, 1)

    boxes = prediction[0, :, :4]
    scores = prediction[0, :, 4:]

    class_ids = np.argmax(scores, axis=1)
    max_scores = np.max(scores, axis=1)

    # Filter by confidence
    mask = max_scores > CONF_THRESHOLD
    boxes = boxes[mask]
    scores = max_scores[mask]
    class_ids = class_ids[mask]

    if len(boxes) == 0:
        return []

    # NMS
    indices = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), CONF_THRESHOLD, NMS_THRESHOLD)

    detections = []
    if len(indices) > 0:
        for i in indices.flatten():
            detections.append({
                'box': boxes[i],
                'score': scores[i],
                'class_id': class_ids[i]
            })

    return detections

# --- 5. DRAWING ---
def xywh2xyxy_fast(boxes, dw, dh, ratio):
    """Vectorized coordinate conversion"""
    boxes = boxes.copy()
    boxes[:, 0] -= dw
    boxes[:, 1] -= dh
    boxes[:, :4] /= ratio

    xyxy = np.empty_like(boxes)
    xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
    xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
    xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
    xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2

    return xyxy.astype(int)

def draw_detections_fast(image, detections, dw, dh, ratio):
    """Minimal drawing operations with custom colors"""
    if not detections:
        return

    boxes = np.array([d['box'] for d in detections])
    boxes_xyxy = xywh2xyxy_fast(boxes, dw, dh, ratio)

    for i, det in enumerate(detections):
        x1, y1, x2, y2 = boxes_xyxy[i]
        class_name = CATEGORIES[det['class_id']]

        # Get color from custom color map
        color = CLASS_COLORS.get(class_name, (255, 255, 255))  # Default to white if not found

        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

        label = f"{class_name}:{det['score']:.2f}"
        cv2.putText(image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

# --- 6. MQTT ---
def setup_mqtt():
    """Non-blocking MQTT setup"""
    try:
        client = mqtt.Client()
        client.username_pw_set(MQTT_USER, MQTT_PASSWORD)
        client.connect(MQTT_BROKER, 1883, 60)
        client.loop_start()
        return client
    except Exception as e:
        print(f"MQTT connection failed: {e}")
        return None

def publish_detections(mqtt_client, detections):
    """Publish detection counts"""
    if mqtt_client is None:
        return

    counts = {"person": 0, "helmet": 0, "no_helmet": 0, "vest": 0, "no_vest": 0, "security_level": "UNSAFE"}

    for d in detections:
        cid = d['class_id']
        if cid in CLASS_TO_KEY:
            counts[CLASS_TO_KEY[cid]] += 1

    if counts["no_helmet"] == 0 and counts["no_vest"] == 0 and counts["person"] > 0:
        if counts["helmet"] > 0 and counts["vest"] > 0:
            counts["security_level"] = "SAFE"

    try:
        mqtt_client.publish(MQTT_TOPIC, json.dumps(counts))
    except:
        pass

# --- 7. STREAMING ---
class RTMPStreamer:
    def __init__(self, url):
        try:
            self.output = jetson.utils.videoOutput(url)
        except Exception as e:
            print(f"RTMP streaming disabled: {e}")
            self.output = None

    def send(self, frame):
        if self.output is None:
            return
        try:
            frame_rgba = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
            cuda_img = jetson.utils.cudaFromNumpy(frame_rgba)
            self.output.Render(cuda_img)
        except:
            pass

# --- 8. MAIN LOOP ---
def main():
    print("Initializing PPE Detection System...")

    # Setup webcam
    cap = cv2.VideoCapture(WEBCAM_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        print("Error: Cannot open webcam")
        return

    # Setup TensorRT
    try:
        trt_engine = TRTInference(ENGINE_PATH)
        print("TensorRT engine loaded")
    except Exception as e:
        print(f"Error: {e}")
        return

    # Setup MQTT
    mqtt_client = setup_mqtt()

    # Setup RTMP
    streamer = RTMPStreamer(STREAM_URL)

    # Tracking variables
    frame_count = 0
    last_mqtt_time = time.time()

    # Reuse detection results
    current_detections = []
    current_transform = (0, 0, 1.0)

    print("System ready. Press 'q' to quit.")

    try:
        # Create clean display window
        window_name = "PPE Detection"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1280, 720)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.01)
                continue

            frame_count += 1

            # CRITICAL: Only process every Nth frame
            if frame_count % PROCESS_EVERY_N_FRAMES == 0:
                # Preprocess
                batch, dw, dh, ratio = preprocess_frame_fast(frame)

                # Inference
                output = trt_engine.infer(batch)

                # Postprocess
                current_detections = postprocess_fast(output)
                current_transform = (dw, dh, ratio)

                # MQTT (every 5 seconds)
                if time.time() - last_mqtt_time > 5.0:
                    publish_detections(mqtt_client, current_detections)
                    last_mqtt_time = time.time()

            # Draw on every frame (using cached detections)
            dw, dh, ratio = current_transform
            draw_detections_fast(frame, current_detections, dw, dh, ratio)

            # Stream
            streamer.send(frame)

            # Simple display - just show the frame with detections
            display_frame = frame
            
            # Optional: Resize for better viewing
            if display_frame.shape[1] < 1280 or display_frame.shape[0] < 720:
                display_frame = cv2.resize(display_frame, (1280, 720))

            # Display
            cv2.imshow(window_name, display_frame)

            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

    except KeyboardInterrupt:
        print("\nShutting down...")
    except Exception as e:
        print(f"Error in main loop: {e}")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        if mqtt_client:
            mqtt_client.loop_stop()
        print("Cleanup complete")

if __name__ == "__main__":
    main()