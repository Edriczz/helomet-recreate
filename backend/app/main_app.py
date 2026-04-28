import cv2
import time
import os
import sys
import subprocess
import psutil # Tambahkan ini
from config.settings import Config
from core.engine import TRTEngine
from helpers.image import ImageProcessor
from helpers.visualizer import Visualizer
from services.mqtt_service import MQTTHandler

# Inisialisasi library pembaca GPU NVIDIA
try:
    import pynvml
    pynvml.nvmlInit()
    gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    HAS_GPU = True
except Exception as e:
    print(f"⚠️ Peringatan: Tidak dapat menginisialisasi pembacaan GPU ({e})")
    HAS_GPU = False

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
        app_start_time = time.time()
        
        print("🚀 Initializing PPE Detector...")
        cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
        width, height = 640, 480 
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        
        if not cap.isOpened():
            print("❌ Cannot open webcam.")
            return

        print(f"📡 Starting FFmpeg stream to: {self.config.STREAM_URL}")
        ffmpeg_cmd = [
            'ffmpeg', '-y', '-f', 'rawvideo', '-vcodec', 'rawvideo', '-pix_fmt', 'bgr24',
            '-s', f"{width}x{height}", '-r', '30', '-i', '-', '-c:v', 'libx264',
            '-preset', 'ultrafast', '-tune', 'zerolatency', '-b:v', '1M',
            '-f', 'rtsp', self.config.STREAM_URL
        ]
        stream_pipe = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)

        print("🎥 Camera Started. Press 'Q' to exit.")
        
        frame_count = 0
        last_results = []
        last_transform = (0, 0, 1)
        last_metrics = {}
        last_mqtt_time = time.time()
        
        # --- INISIALISASI VARIABEL METRIK (SEBELUM LOOPING) ---
        fps_start_time = time.time()
        fps_frame_count = 0
        current_fps = 0.0
        
        # Buat object process untuk membaca RAM/CPU program ini
        current_process = psutil.Process(os.getpid())
        
        # Pancing pembacaan CPU khusus program ini (Sangat Penting!)
        current_process.cpu_percent(interval=None) 
        
        # Set nilai awal agar tidak 0 murni di detik pertama
        cpu_load, gpu_load = 0.0, 0
        ram_mb = current_process.memory_info().rss / (1024 * 1024)
        vram_mb = 0.0
        t_pre, t_inf, t_post = 0.0, 0.0, 0.0
        
        last_results = []
        last_metrics = {}
        last_mqtt_time = time.time()
        is_first_frame = True

        try:
            while True:
                ret, frame = cap.read()
                if not ret: break
                
                frame_count += 1
                fps_frame_count += 1
                
                # --- UPDATE METRIK SETIAP 1 DETIK ---
                if time.time() - fps_start_time >= 1.0:
                    current_fps = fps_frame_count / (time.time() - fps_start_time)
                    fps_start_time = time.time()
                    fps_frame_count = 0
                    
                    # Update metrik RAM & CPU khusus App ini
                    ram_mb = current_process.memory_info().rss / (1024 * 1024)
                    cpu_load = current_process.cpu_percent(interval=None) / psutil.cpu_count()
                    
                    if HAS_GPU:
                        try:
                            util_info = pynvml.nvmlDeviceGetUtilizationRates(gpu_handle)
                            mem_info = pynvml.nvmlDeviceGetMemoryInfo(gpu_handle)
                            gpu_load = util_info.gpu
                            vram_mb = mem_info.used / (1024 * 1024)
                        except:
                            pass
                            
                # ... (lanjutan logika inferensi Anda) ...
                
                # --- PROCESS INFERENCE ---
                if frame_count % self.config.SKIP_FRAMES == 0:
                    t1 = time.perf_counter()
                    batch, dw, dh, ratio = self.processor.preprocess(frame)
                    t_pre = (time.perf_counter() - t1) * 1000
                    
                    t2 = time.perf_counter()
                    raw_output = self.engine.infer(batch)
                    t_inf = (time.perf_counter() - t2) * 1000
                    
                    t3 = time.perf_counter()
                    last_results = self.processor.postprocess(raw_output)
                    t_post = (time.perf_counter() - t3) * 1000
                    
                    last_transform = (dw, dh, ratio)
                    
                    # --- 3. PEMBUATAN DATA METRIK ---
                last_metrics = {
                    "Video FPS": f"{current_fps:.1f}",
                    "Inference": f"{t_inf:.1f} ms",
                    "App CPU": f"{cpu_load:.1f}%",
                    "App RAM": f"{ram_mb:.1f} MB", # Sekarang pakai MB
                }
                if HAS_GPU:
                    last_metrics["GPU Load"] = f"{gpu_load}%"
                    last_metrics["VRAM Use"] = f"{vram_mb:.1f} MB" # Sekarang pakai MB
                    
                # --- 4. KIRIM MQTT ---
                # Dipindah ke sini agar bisa mengikutsertakan last_metrics
                if time.time() - last_mqtt_time > 2.0:
                    self.mqtt.publish(last_results, last_metrics)
                    last_mqtt_time = time.time()
                
                # --- 5. VISUALIZE & STREAM ---
                dw, dh, ratio = last_transform
                
                # UBAH DISINI: Ganti last_metrics menjadi None
                frame = self.visualizer.draw(frame, last_results, dw, dh, ratio, None)
                
                # cv2.imshow("PPE Detection System", frame)
                if is_first_frame:
                    startup_time = time.time() - app_start_time
                    print(f"\n⏱️ [TENSORRT] Waktu Pemanasan (Time to First Frame): {startup_time:.2f} detik\n")
                    is_first_frame = False
                
                try:
                    stream_pipe.stdin.write(frame.tobytes())
                except Exception:
                    pass
                
                # if cv2.waitKey(1) & 0xFF == ord('q'):
                #     break
                    
        finally:
            cap.release()
            # cv2.destroyAllWindows()
            self.mqtt.stop()
            if stream_pipe:
                stream_pipe.stdin.close()
                stream_pipe.wait()
            if HAS_GPU:
                pynvml.nvmlShutdown()
            print("👋 System Shutdown.")