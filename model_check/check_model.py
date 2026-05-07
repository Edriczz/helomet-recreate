from ultralytics import YOLO
import torch

# 1. Masukkan path ke model baru Anda (bisa .pt atau .engine)
model_path = "../backend/model/best_helomet_v2.pt" # Ganti dengan path model Anda

print(f"Memuat model dari: {model_path}...")
model = YOLO(model_path)

# 2. Mengambil dan mencetak daftar kelas
daftar_kelas = model.names

print("\n" + "="*40)
print("DAFTAR KELAS DALAM MODEL INI:")
print("="*40)
for class_id, class_name in daftar_kelas.items():
    print(f"ID Kelas {class_id} : {class_name}")
print("="*40 + "\n")
print(model.info())
for key, value in model.model.args.items():
    print(f"{key}: {value}")
    
# Muat file .pt
try:
    checkpoint = torch.load('../backend/model/best_helomet_v2.pt', map_location='cpu', weights_only=False)
    
    # Tampilkan kunci apa saja yang ada di dalam file
    print("Keys yang tersedia di dalam file:")
    print(checkpoint.keys())
except Exception as e:
    print(f"Error saat membaca file: {e}")

import torch

# Muat file .pt tanpa error
checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)

# Tampilkan parameter pelatihan
print("=== TRAINING ARGUMENTS ===")
for key, value in checkpoint['train_args'].items():
    print(f"{key}: {value}")

# Tampilkan epoch terakhir dan metrik
print("\n=== INFORMASI LAINNYA ===")
print(f"Epoch Tersimpan: {checkpoint['epoch']}")

# Cek jumlah data yang tersimpan di dalam train_results
if 'train_results' in checkpoint:
    # train_results biasanya berisi list atau dict yang panjangnya sama dengan jumlah epoch yang berjalan
    print(f"Jumlah epoch yang berhasil diproses: {len(checkpoint['train_results'])}")
else:
    print("Data train_results tidak ditemukan dalam file ini.")