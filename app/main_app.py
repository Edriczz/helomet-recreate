import cv2
import time
import sys
from config.settings import Config
from core.engine import TRTEngine
from helpers.image import ImageProcessor
from helpers.visualizer import Visualizer
from services.mqtt_service import MQTTHandler

class PPEDetectorApp:
    def __init__(self):
        print("🚀 Initializing PPE Detector...")
        
        # 1. Load Config
        self.config = Config()
        
        # 2. Init Modules
        self.mqtt = MQTTHandler(self.config)
        self.visualizer = Visualizer(self.config)
        self.processor = ImageProcessor(self.config)
        
        # 3. Load Engine (Heavy Task)
        print(f"📂 Loading Engine: {self.config.ENGINE_PATH}")
        try:
            self.engine = TRTEngine(self.config.ENGINE_PATH)
            print("✅ Engine Loaded!")
        except Exception as e:
            print(f"❌ Error Loading Engine: {e}")
            sys.exit(1)

    def run(self):
        cap = cv2.VideoCapture(2, cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        if not cap.isOpened():
            print("❌ Cannot open webcam.")
            return

        print("🎥 Camera Started. Press 'Q' to exit.")
        
        frame_count = 0
        last_results = []
        last_transform = (0, 0, 1)
        last_mqtt_time = time.time()
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret: break
                
                frame_count += 1
                
                # --- PROCESS EVERY N FRAMES ---
                if frame_count % self.config.SKIP_FRAMES == 0:
                    batch, dw, dh, ratio = self.processor.preprocess(frame)
                    raw_output = self.engine.infer(batch)
                    last_results = self.processor.postprocess(raw_output)
                    last_transform = (dw, dh, ratio)
                    
                    # Send MQTT
                    if time.time() - last_mqtt_time > 2.0:
                        self.mqtt.publish(last_results)
                        last_mqtt_time = time.time()
                
                # --- VISUALIZE ---
                dw, dh, ratio = last_transform
                frame = self.visualizer.draw(frame, last_results, dw, dh, ratio)
                
                cv2.imshow("PPE Detection System", frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.mqtt.stop()
            print("👋 System Shutdown.")