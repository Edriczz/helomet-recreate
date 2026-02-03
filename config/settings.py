import os
from dotenv import load_dotenv

class Config:
    def __init__(self):
        load_dotenv()
        
        # --- PATHS ---
        self.ENGINE_PATH = os.getenv("MODEL_ENGINE_PATH", "model/besthelomet_yolov8_MX230.engine")
        
        # --- MODEL PARAMS ---
        self.INPUT_W = int(os.getenv("INPUT_WIDTH", 640))
        self.INPUT_H = int(os.getenv("INPUT_HEIGHT", 640))
        self.CONF_THRES = float(os.getenv("CONF_THRESHOLD", 0.25))
        self.NMS_THRES = float(os.getenv("NMS_THRESHOLD", 0.45))
        
        # --- APP CONFIG ---
        self.CAM_INDEX = int(os.getenv("WEBCAM_INDEX", 0))
        self.SKIP_FRAMES = int(os.getenv("PROCESS_EVERY_N_FRAMES", 5))
        
        # --- MQTT CONFIG ---
        self.MQTT_BROKER = os.getenv("MQTT_BROKER", "broker.xdevelopment.my.id")
        self.MQTT_PORT = int(os.getenv("MQTT_PORT", 1883))
        self.MQTT_TOPIC = os.getenv("MQTT_TOPIC", "ai/telemetry")
        self.MQTT_USER = os.getenv("MQTT_USER", "nodered")
        self.MQTT_PASS = os.getenv("MQTT_PASSWORD", "nodered")
        
        # --- CLASSES & COLORS ---
        self.CATEGORIES = ["person", "boots", "helmet", "no_boots", "no_helmet", "no_vest", "vest"]
        self.CLASS_TO_KEY = {0: "person", 2: "helmet", 4: "no_helmet", 6: "vest", 5: "no_vest"}
        
        self.COLORS = {
            "person": (0, 255, 255),    "boots": (0, 255, 0),
            "helmet": (200, 100, 50),   "no_boots": (0, 0, 255),
            "no_helmet": (200, 100, 200),"no_vest": (251, 119, 60),
            "vest": (235, 54, 120)
        }