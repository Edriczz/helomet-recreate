import tensorrt as trt
import os

# PASTIKAN NAMA FILE ONNX SESUAI
ONNX_FILE_PATH = "../model/besthelomet_yolov8_v2.onnx"
ENGINE_FILE_PATH = "../model/besthelomet_yolov8_v2_MX230.engine"

def build_engine():
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    
    # Setup network & config
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    config = builder.create_builder_config()
    
    parser = trt.OnnxParser(network, logger)
    
    # Parse ONNX
    print(f"Loading ONNX file from {ONNX_FILE_PATH}...")
    if not os.path.exists(ONNX_FILE_PATH):
        print(f"ERROR: File {ONNX_FILE_PATH} tidak ditemukan!")
        return

    with open(ONNX_FILE_PATH, "rb") as model:
        if not parser.parse(model.read()):
            print("ERROR: Gagal memparsing file ONNX.")
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return

    # Set Memory Pool (PENTING untuk TensorRT 10+)
    # Memberikan 1GB workspace (MX230 punya 2GB VRAM, jadi 1GB aman)
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30) 

    if builder.platform_has_fast_fp16:
        print("Mengahktifkan Mode FP16 (Lebih Cepat)...")
        config.set_flag(trt.BuilderFlag.FP16)
    
    # Build Engine
    print("Sedang membangun Engine... (Bisa memakan waktu 5-10 menit, kipas laptop mungkin ngebut)")
    serialized_engine = builder.build_serialized_network(network, config)
    
    if serialized_engine:
        with open(ENGINE_FILE_PATH, "wb") as f:
            f.write(serialized_engine)
        print(f"SUKSES! Engine tersimpan di: {ENGINE_FILE_PATH}")
    else:
        print("GAGAL membangun engine.")

if __name__ == "__main__":
    build_engine()