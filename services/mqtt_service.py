import paho.mqtt.client as mqtt
import json

class MQTTHandler:
    def __init__(self, config):
        self.cfg = config
        self.client = mqtt.Client()
        self.connected = False
        
        if self.cfg.MQTT_USER:
            self.client.username_pw_set(self.cfg.MQTT_USER, self.cfg.MQTT_PASS)
            
        try:
            self.client.connect(self.cfg.MQTT_BROKER, self.cfg.MQTT_PORT, 60)
            self.client.loop_start()
            self.connected = True
            print(f"✅ MQTT Connected to {self.cfg.MQTT_BROKER}")
        except Exception as e:
            print(f"⚠️ MQTT Failed: {e}")

    # Tambahkan parameter metrics=None di sini
    def publish(self, detections, metrics=None):
        if not self.connected: return

        # Gunakan nama variabel payload agar lebih representatif
        payload = {"person": 0, "helmet": 0, "no_helmet": 0, "vest": 0, "no_vest": 0, "security_level": "UNSAFE"}
        
        for det in detections:
            cid = det['class_id']
            if cid in self.cfg.CLASS_TO_KEY:
                key = self.cfg.CLASS_TO_KEY[cid]
                # Pastikan key ada di payload sebelum ditambah
                if key in payload:
                    payload[key] += 1
        
        # Logic Safety Level
        if payload["person"] > 0:
            if payload["no_helmet"] == 0 and payload["no_vest"] == 0:
                 if payload["helmet"] > 0 and payload["vest"] > 0:
                    payload["security_level"] = "SAFE"
                    
        # --- BLOK BARU: Masukkan metrik hardware ke MQTT ---
        if metrics:
            payload["system_status"] = metrics
        
        try:
            self.client.publish(self.cfg.MQTT_TOPIC, json.dumps(payload))
        except Exception as e:
            print(f"MQTT Publish Error: {e}")

    def stop(self):
        if self.connected:
            self.client.loop_stop()
            self.client.disconnect()