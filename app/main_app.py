import cv2
import time
import sys
import subprocess # Tambahkan ini
from config.settings import Config
from core.engine import TRTEngine
from helpers.image import ImageProcessor
from helpers.visualizer import Visualizer
from services.mqtt_service import MQTTHandler

class PPEDetectorApp:
    def __init__(self):
        print("🚀 Initializing PPE Detector...")
        self.config = Config()
        self.mqtt = MQTTHandler(self.config)
        self.visualizer = Visualizer(self.config)
        self.processor = ImageProcessor(self.config)
        
        print(f"📂 Loading Engine: {self.config.ENGINE_PATH}")
        try:
            self.engine = TRTEngine(self.config.ENGINE_PATH)
            print("✅ Engine Loaded!")
        except Exception as e:
            print(f"❌ Error Loading Engine: {e}")
            sys.exit(1)

    def run(self):
        cap = cv2.VideoCapture(2, cv2.CAP_V4L2)
        # Tentukan resolusi. Pastikan sesuai dengan FFmpeg nanti
        width, height = 640, 480 
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        
        if not cap.isOpened():
            print("❌ Cannot open webcam.")
            return

        # --- SETUP FFMPEG UNTUK MEDIAMTX ---
        print(f"📡 Starting FFmpeg stream to: {self.config.STREAM_URL}")
        ffmpeg_cmd = [
            'ffmpeg',
            '-y', '-f', 'rawvideo',
            '-vcodec', 'rawvideo',
            '-pix_fmt', 'bgr24',
            '-s', f"{width}x{height}",
            '-r', '30', # Framerate (sesuaikan dengan FPS kamera)
            '-i', '-',  # Input dari stdin (pipe)
            '-c:v', 'libx264', # Codec h264 agar support WebRTC
            '-preset', 'ultrafast', # Penting agar tidak delay di edge device
            '-tune', 'zerolatency',
            '-b:v', '1M', # Bitrate video
            '-f', 'rtsp',
            self.config.STREAM_URL
        ]
        
        # Jalankan FFmpeg sebagai subprocess
        stream_pipe = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)

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
                    
                    if time.time() - last_mqtt_time > 2.0:
                        self.mqtt.publish(last_results)
                        last_mqtt_time = time.time()
                
                # --- VISUALIZE ---
                dw, dh, ratio = last_transform
                frame = self.visualizer.draw(frame, last_results, dw, dh, ratio)
                
                # Menampilkan lokal di layar PC
                cv2.imshow("PPE Detection System", frame)
                
                # MENGIRIM FRAME KE FFMPEG (MediaMTX)
                try:
                    stream_pipe.stdin.write(frame.tobytes())
                except Exception as e:
                    print(f"⚠️ Failed to write to stream: {e}")
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.mqtt.stop()
            # Tutup stream FFmpeg dengan aman
            if stream_pipe:
                stream_pipe.stdin.close()
                stream_pipe.wait()
            print("👋 System Shutdown.")