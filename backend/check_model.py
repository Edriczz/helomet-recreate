from ultralytics import YOLO

# 1. Masukkan path ke model baru Anda (bisa .pt atau .engine)
model_path = "model/best_helomet_v2.pt" # Ganti dengan path model Anda

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

# 3. (Opsional) Uji coba inferensi pada webcam (index 0 atau 2)
# Hapus atau beri komentar pada baris di bawah ini jika hanya ingin melihat daftar kelas
print("Memulai pengujian kamera (Tekan 'q' untuk keluar)...")
results = model.predict(source=2, show=True) # Ganti source=2 sesuai index kamera Anda