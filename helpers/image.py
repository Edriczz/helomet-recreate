import cv2
import numpy as np

class ImageProcessor:
    def __init__(self, config):
        self.cfg = config
        self.padded_buffer = np.full((config.INPUT_H, config.INPUT_W, 3), 114, dtype=np.uint8)

    def preprocess(self, frame):
        h, w = frame.shape[:2]
        r = min(self.cfg.INPUT_W / w, self.cfg.INPUT_H / h)
        nw, nh = int(w * r), int(h * r)
        
        resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
        
        self.padded_buffer[:] = 114
        dw, dh = (self.cfg.INPUT_W - nw) // 2, (self.cfg.INPUT_H - nh) // 2
        self.padded_buffer[dh:dh+nh, dw:dw+nw] = resized
        
        batch = np.ascontiguousarray(self.padded_buffer.transpose(2, 0, 1), dtype=np.float32) / 255.0
        return batch, dw, dh, r

    def postprocess(self, raw_output):
        pred = raw_output.reshape((1, len(self.cfg.CATEGORIES) + 4, 8400)).transpose(0, 2, 1)
        
        boxes = pred[0, :, :4]
        scores = pred[0, :, 4:]
        
        class_ids = np.argmax(scores, axis=1)
        max_scores = np.max(scores, axis=1)
        
        mask = max_scores > self.cfg.CONF_THRES
        boxes = boxes[mask]
        scores = max_scores[mask]
        class_ids = class_ids[mask]
        
        if len(boxes) == 0: return []
        
        indices = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), self.cfg.CONF_THRES, self.cfg.NMS_THRES)
        
        results = []
        if len(indices) > 0:
            for i in indices.flatten():
                results.append({
                    'box': boxes[i], 
                    'score': scores[i], 
                    'class_id': class_ids[i],
                    'label': self.cfg.CATEGORIES[class_ids[i]]
                })
        return results