from ultralytics import YOLO
import shutil
import os

# --- KONFIGURASI ---
INPUT_MODEL = "../model/runs/detect/train/weights/best_helomet_v3.pt"  # <--- Ganti path ini sesuai lokasi file .pt Anda
OUTPUT_NAME = "besthelomet_yolov8_v3.onnx"         # Nama output yang diinginkan
DEST_FOLDER = "../model/"                          # Folder tujuan untuk build engine nanti

def export_to_onnx():
    # 1. Cek apakah file ada
    if not os.path.exists(INPUT_MODEL):
        print(f"❌ Error: File {INPUT_MODEL} tidak ditemukan!")
        print("Pastikan Anda sudah mendownload model dan path-nya benar.")
        return

    print(f"🔄 Memuat model dari: {INPUT_MODEL}")
    model = YOLO(INPUT_MODEL)

    # 2. Lakukan Export ke ONNX
    # imgsz=640 dan opset=12 sangat disarankan untuk kompatibilitas TensorRT
    print("⏳ Sedang meng-export ke ONNX... (Tunggu sebentar)")
    path = model.export(format="onnx", imgsz=640, opset=12)
    
    print(f"✅ Export berhasil! File tersimpan di: {path}")

    # 3. Pindahkan ke folder model/ agar rapi
    if not os.path.exists(DEST_FOLDER):
        os.makedirs(DEST_FOLDER)
        
    final_path = os.path.join(DEST_FOLDER, OUTPUT_NAME)
    shutil.move(path, final_path)
    
    print(f"📂 File dipindahkan ke: {final_path}")
    print("🚀 SIAP UNTUK BUILD ENGINE!")

if __name__ == "__main__":
    export_to_onnx()