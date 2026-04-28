import cv2
import numpy as np

class Visualizer:
    def __init__(self, config):
        self.cfg = config

    def draw(self, frame, detections, dw, dh, ratio, metrics=None):
        # 1. Gambar teks metrik performa di pojok kiri atas
        if metrics:
            y_offset = 30
            for key, val in metrics.items():
                text = f"{key}: {val}"
                # Efek teks dengan outline hitam agar terbaca di latar terang
                cv2.putText(frame, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3)
                cv2.putText(frame, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1)
                y_offset += 30

        if not detections: return frame
        
        # 2. Kembalikan koordinat kotak
        boxes = np.array([d['box'] for d in detections])
        boxes[:, 0] -= dw
        boxes[:, 1] -= dh
        boxes[:, :4] /= ratio
        
        x1 = boxes[:, 0].astype(int)
        y1 = boxes[:, 1].astype(int)
        x2 = (boxes[:, 0] + boxes[:, 2]).astype(int)
        y2 = (boxes[:, 1] + boxes[:, 3]).astype(int)
        
        for i, det in enumerate(detections):
            label = det['label']
            color = self.cfg.COLORS.get(label, (255, 255, 255))
            
            cv2.rectangle(frame, (x1[i], y1[i]), (x2[i], y2[i]), color, 2)
            
            text_label = f"{label} {det['score']:.2f}"
            (w, h), _ = cv2.getTextSize(text_label, 0, 0.5, 1)
            cv2.rectangle(frame, (x1[i], y1[i]-20), (x1[i]+w, y1[i]), color, -1)
            cv2.putText(frame, text_label, (x1[i], y1[i] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
        return frame