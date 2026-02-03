import tensorrt as trt
import sys

def check_tensorrt():
    print(f"1. Memeriksa Versi TensorRT: {trt.__version__}")
    
    try:
        # Mencoba inisialisasi Logger (Langkah pertama komunikasi ke C++ core)
        logger = trt.Logger(trt.Logger.WARNING)
        print("2. Logger berhasil dibuat.")
        
        # Mencoba inisialisasi Builder (Langkah krusial akses GPU)
        builder = trt.Builder(logger)
        print("3. Builder berhasil dibuat (Akses GPU Oke).")
        
        # Cek apakah support FP16 (Penting untuk MX230)
        if builder.platform_has_fast_fp16:
            print("4. ✅ GPU mendukung FP16 (Mode Cepat).")
        else:
            print("4. ⚠️ GPU tidak mendukung Native FP16 (Akan fallback ke FP32).")
            
        print("\n🎉 KESIMPULAN: TensorRT berfungsi normal!")
        
    except Exception as e:
        print(f"\n❌ ERROR FATAL: {e}")
        print("Kemungkinan LD_LIBRARY_PATH belum di-export.")

if __name__ == "__main__":
    check_tensorrt()