import tensorrt as trt
from ultralytics import YOLO
from ultralytics.nn.autobackend import AutoBackend

# ==========================================
# JURUS HACK: MONKEY PATCHING METADATA
# ==========================================
# 1. Simpan fungsi inisialisasi asli milik Ultralytics
original_init = AutoBackend.__init__

# 2. Buat fungsi jebakan untuk menyuntikkan metadata
def patched_init(self, *args, **kwargs):
    original_init(self, *args, **kwargs)
    # Jika engine terdeteksi tidak punya metadata (NoneType), kita buatkan!
    if getattr(self, 'metadata', None) is None:
        self.metadata = {
            "batch": 1,
            "stride": 32,
            "names": {0: 'person', 1: 'helmet', 2: 'no_helmet', 3: 'vest', 4: 'no_vest'} 
        }

# 3. Timpa fungsi asli di dalam library dengan fungsi jebakan kita
AutoBackend.__init__ = patched_init
# ==========================================

def main():
    # File engine Anda
    model_path = "model/besthelomet_yolov8_v2_MX230.engine"
    
    # PERHATIAN: Pastikan path ini mengarah ke dataset VERSI LAMA (yang cuma 5 kelas!)
    dataset_yaml = "Data/Helomet v2.1.v1i.yolov8/data.yaml" 
    
    print(f"🚀 Memulai Evaluasi Akurasi untuk: {model_path}")
    model = YOLO(model_path, task='detect')
    
    # Jalankan validasi
    metrics = model.val(data=dataset_yaml, split='val', imgsz=640, batch=4)
    
    print("\n✅ Evaluasi Selesai!")
    print(f"mAP50    : {metrics.box.map50:.4f}")
    print(f"mAP50-95 : {metrics.box.map:.4f}")

if __name__ == "__main__":
    main()