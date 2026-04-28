import os
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np

class TRTEngine:
    def __init__(self, engine_path):
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.engine = self._load_engine(engine_path)
        self.context = self.engine.create_execution_context()
        self.stream = cuda.Stream()
        self.inputs, self.outputs, self.bindings = self._allocate_buffers()

    def _load_engine(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Engine file not found: {path}")
        with open(path, "rb") as f:
            runtime = trt.Runtime(self.logger)
            return runtime.deserialize_cuda_engine(f.read())

    def _allocate_buffers(self):
        inputs, outputs, bindings = [], [], []
        for i in range(self.engine.num_bindings):
            shape = self.engine.get_binding_shape(i)
            dtype = trt.nptype(self.engine.get_binding_dtype(i))
            size = trt.volume(shape) * self.engine.max_batch_size
            
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            bindings.append(int(device_mem))
            
            if self.engine.binding_is_input(i):
                inputs.append({'host': host_mem, 'device': device_mem})
            else:
                outputs.append({'host': host_mem, 'device': device_mem})
        return inputs, outputs, bindings

    def infer(self, image_batch):
        np.copyto(self.inputs[0]['host'], image_batch.ravel())
        cuda.memcpy_htod_async(self.inputs[0]['device'], self.inputs[0]['host'], self.stream)
        self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
        cuda.memcpy_dtoh_async(self.outputs[0]['host'], self.outputs[0]['device'], self.stream)
        self.stream.synchronize()
        return self.outputs[0]['host']