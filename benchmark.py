import cv2
import os
import time
import sys
import subprocess
import psutil
import torch
import numpy as np
from ultralytics import YOLO
import torchvision
from config.settings import Config
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

# Fungsi manual untuk Pre-process (Sama dengan format TRT)
def letterbox(img, new_shape=(640, 640), color=(114, 114, 114)):
    shape = img.shape[:2]  # [height, width]
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    dw, dh = dw / 2, dh / 2
    if shape[::-1] != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return img

def main():
    # --- 1. MULAI STOPWATCH ---
    app_start_time = time.time() 
    
    print("🚀 Initializing RAW PyTorch Benchmark...")
    config = Config()
    mqtt = MQTTHandler(config)
    # --- LOAD RAW PT MODEL ---
    model_path = "model/best_helomet_v2.pt" 
    print(f"📂 Loading Raw PyTorch Model: {model_path}")
    
    # 1. Gunakan Ultralytics hanya untuk membongkar file .pt
    temp_model = YOLO(model_path)

    
    # 2. Culik inti PyTorch-nya, masuk ke mode Eval, dan Fuse Layer (agar adil dengan TRT)
    raw_model = temp_model.model.eval()
    if hasattr(raw_model, 'fuse'):
        raw_model = raw_model.fuse()
        
    # 3. Pindahkan ke GPU jika tersedia
    if torch.cuda.is_available():
        raw_model = raw_model.cuda()
    
    # --- SETUP KAMERA ---
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    width, height = 640, 480 
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    
    if not cap.isOpened():
        print("❌ Cannot open webcam.")
        return

    # --- SETUP FFMPEG MEDIAMTX ---
    print(f"📡 Starting FFmpeg stream to: {config.STREAM_URL}")
    ffmpeg_cmd = [
        'ffmpeg', '-y', '-f', 'rawvideo', '-vcodec', 'rawvideo', '-pix_fmt', 'bgr24',
        '-s', f"{width}x{height}", '-r', '30', '-i', '-', '-c:v', 'libx264',
        '-preset', 'ultrafast', '-tune', 'zerolatency', '-b:v', '1M',
        '-f', 'rtsp', config.STREAM_URL
    ]
    stream_pipe = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)

    print("🎥 Camera Started. Press 'Q' to exit.")
    
    is_first_frame = True
    frame_count = 0
    # --- INISIALISASI VARIABEL METRIK ---
    fps_start_time = time.time()
    fps_frame_count = 0
    current_fps = 0.0
        
    current_process = psutil.Process(os.getpid())
    current_process.cpu_percent(interval=None) 
        
    cpu_load, gpu_load = 0.0, 0
    ram_mb = current_process.memory_info().rss / (1024 * 1024)
    vram_mb = 0.0
    t_pre, t_inf, t_post = 0.0, 0.0, 0.0
        
    last_results = []
    last_metrics = {}
    last_mqtt_time = time.time()

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
                            
            
            # --- 2. INFERENCE (DENGAN FRAME SKIP) ---
            if frame_count % config.SKIP_FRAMES == 0:
                
                # ==========================================
                # A. PRE-PROCESSING
                # ==========================================
                img_resized = letterbox(frame, new_shape=(640, 640))
                # Konversi BGR ke RGB, lalu HWC ke CHW (channel first)
                img_rgb = img_resized[:, :, ::-1].transpose(2, 0, 1) 
                img_np = np.ascontiguousarray(img_rgb)
                
                # Numpy ke PyTorch Tensor
                input_tensor = torch.from_numpy(img_np).float() / 255.0
                input_tensor = input_tensor.unsqueeze(0) # Tambah batch dimension [1, 3, 640, 640]
                
                if torch.cuda.is_available():
                    input_tensor = input_tensor.cuda()

                # ==========================================
                # B. RAW FORWARD PASS (INFERENSI)
                # ==========================================
                t1 = time.time()
                with torch.no_grad():
                    preds = raw_model(input_tensor)
                t_inf = (time.time() - t1) * 1000
                
                # ==========================================
                # ==========================================
                # C. POST-PROCESSING (RAW PYTORCH NMS)
                # ==========================================
                # Keluaran YOLOv8 berbentuk tuple, ambil tensor pertama
                if isinstance(preds, (list, tuple)):
                    preds = preds[0]
                
                # Bentuk asli preds: [1, 4+classes, 8400]
                # Kita ubah bentuknya menjadi [8400, 4+classes] agar mudah diolah
                preds = preds.squeeze(0).T
                
                # Pisahkan kotak koordinat (4 kolom pertama) dan probabilitas kelas (sisanya)
                boxes_cxcywh = preds[:, :4]
                class_probs = preds[:, 4:]
                
                # Cari kelas dengan skor tertinggi untuk tiap kotak
                scores, class_ids = torch.max(class_probs, dim=1)
                
                # 1. Filter kotak yang skornya di bawah CONF_THRES
                mask = scores > config.CONF_THRES
                boxes_cxcywh = boxes_cxcywh[mask]
                scores = scores[mask]
                class_ids = class_ids[mask]
                
                # 2. Konversi format kotak dari (Center-X, Center-Y, Width, Height) 
                # menjadi format titik sudut (X1, Y1, X2, Y2)
                x1 = boxes_cxcywh[:, 0] - boxes_cxcywh[:, 2] / 2
                y1 = boxes_cxcywh[:, 1] - boxes_cxcywh[:, 3] / 2
                x2 = boxes_cxcywh[:, 0] + boxes_cxcywh[:, 2] / 2
                y2 = boxes_cxcywh[:, 1] + boxes_cxcywh[:, 3] / 2
                boxes_xyxy = torch.stack([x1, y1, x2, y2], dim=1)
                
                # 3. Jalankan PyTorch NMS (Buang kotak yang bertumpuk pada objek yang sama)
                keep_indices = torchvision.ops.nms(boxes_xyxy, scores, iou_threshold=0.45)
                
                final_boxes = boxes_xyxy[keep_indices].cpu().numpy()
                final_scores = scores[keep_indices].cpu().numpy()
                final_class_ids = class_ids[keep_indices].cpu().numpy()
                
                # 4. Skala manual kotak (kembali ke ukuran asli kamera 640x480)
                orig_h, orig_w = frame.shape[:2]
                gain = min(640 / orig_h, 640 / orig_w)
                pad_x = (640 - orig_w * gain) / 2
                pad_y = (640 - orig_h * gain) / 2
                
                last_results = []
                for i in range(len(final_boxes)):
                    bx1, by1, bx2, by2 = final_boxes[i]
                    
                    # Kembalikan koordinat dengan menghapus padding letterbox
                    bx1 = int((bx1 - pad_x) / gain)
                    bx2 = int((bx2 - pad_x) / gain)
                    by1 = int((by1 - pad_y) / gain)
                    by2 = int((by2 - pad_y) / gain)
                    
                    # Clamp agar kotak tidak keluar dari batas layar
                    bx1, bx2 = max(0, bx1), min(orig_w, bx2)
                    by1, by2 = max(0, by1), min(orig_h, by2)
                    
                    cid = int(final_class_ids[i])
                    score = float(final_scores[i])
                    label = config.CLASS_TO_KEY.get(cid, "Unknown")
                    
                    last_results.append({
                        'box': [bx1, by1, bx2, by2],
                        'class_id': cid,
                        'label': label,
                        'score': score
                    })
                
                # Update UI Hardware
                last_metrics = {
                    "Video FPS": f"{current_fps:.1f}",
                    "Inference": f"{t_inf:.1f} ms",
                    "App CPU": f"{cpu_load:.1f}%",
                    "App RAM": f"{ram_mb:.1f} MB", 
                }
                if HAS_GPU:
                    last_metrics["GPU Load"] = f"{gpu_load}%"
                    last_metrics["VRAM Use"] = f"{vram_mb:.1f} MB" 
                
                if time.time() - last_mqtt_time > 2.0:
                    mqtt.publish(last_results, last_metrics)
                    last_mqtt_time = time.time()
            
            # --- 3. VISUALISASI MANUAL ---
            for det in last_results:
                x1, y1, x2, y2 = det['box']
                label = det['label']
                color = config.COLORS.get(label, (255, 255, 255))
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                text_label = f"{label} {det['score']:.2f}"
                (w, h), _ = cv2.getTextSize(text_label, 0, 0.5, 1)
                cv2.rectangle(frame, (x1, y1-20), (x1+w, y1), color, -1)
                cv2.putText(frame, text_label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            y_offset = 30
            for key, val in last_metrics.items():
                text = f"{key}: {val}"
                cv2.putText(frame, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3)
                cv2.putText(frame, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1)
                y_offset += 30
            
            cv2.imshow("PPE Benchmark - PyTorch", frame)
            # --- 3. HENTIKAN STOPWATCH SAAT FRAME PERTAMA MUNCUL ---
            if is_first_frame:
                startup_time = time.time() - app_start_time
                print(f"\n⏱️ [BENCHMARK PT] Waktu Pemanasan (Time to First Frame): {startup_time:.2f} detik\n")
                is_first_frame = False
            
            # Matikan MediaMTX untuk mengukur performa CPU model murni (buka komen untuk tes real)
            # try:
            #     stream_pipe.stdin.write(frame.tobytes())
            # except Exception:
            #     pass
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    finally:
        cap.release()
        cv2.destroyAllWindows()
        mqtt.stop()
        if stream_pipe:
            stream_pipe.stdin.close()
            stream_pipe.wait()
        if HAS_GPU:
            pynvml.nvmlShutdown()
        print("👋 System Shutdown.")

if __name__ == "__main__":
    main()