from ultralytics import YOLO
import os

def evaluate_pytorch_model():
    print("🚀 Memulai Evaluasi Model PyTorch...")
    model_path="../backend/model/runs/detect/train/weights/last.pt"
    # Load model asli
    model = YOLO(model_path)
    
    # Jalankan validasi
    # Parameter 'plots=True' akan memaksa Ultralytics menggambar Confusion Matrix
    metrics = model.val(
        data="../Data/Helomet v2.1.v1i.yolov8/data.yaml", 
        split="val",
        conf=0.001, # Sesuai dengan threshold di .env Anda
        iou=0.60,  # NMS threshold
        plots=True,
        device="0" # Gunakan GPU
    )
    
    print("\n✅ Evaluasi Selesai!")
    print(f"📊 mAP@50-95 : {metrics.box.map:.4f}")
    print(f"📊 mAP@50    : {metrics.box.map50:.4f}")
    print(f"📊 Fitness   : {metrics.fitness:.4f}")
    
    # Memberi tahu lokasi Confusion Matrix
    save_dir = metrics.save_dir
    print(f"\n📁 Confusion Matrix dan grafik Loss telah disimpan di: {save_dir}")
    print(f"   Silakan cek file 'confusion_matrix.png' untuk presentasi.")

if __name__ == "__main__":
    evaluate_pytorch_model()