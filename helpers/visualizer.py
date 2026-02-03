import cv2
import numpy as np

class Visualizer:
    def __init__(self, config):
        self.cfg = config

    def draw(self, frame, detections, dw, dh, ratio):
        if not detections: return frame
        
        boxes = np.array([d['box'] for d in detections])
        boxes[:, 0] -= dw
        boxes[:, 1] -= dh
        boxes[:, :4] /= ratio
        
        x1 = (boxes[:, 0] - boxes[:, 2] / 2).astype(int)
        y1 = (boxes[:, 1] - boxes[:, 3] / 2).astype(int)
        x2 = (boxes[:, 0] + boxes[:, 2] / 2).astype(int)
        y2 = (boxes[:, 1] + boxes[:, 3] / 2).astype(int)
        
        for i, det in enumerate(detections):
            label = det['label']
            color = self.cfg.COLORS.get(label, (255, 255, 255))
            
            cv2.rectangle(frame, (x1[i], y1[i]), (x2[i], y2[i]), color, 2)
            
            text = f"{label} {det['score']:.2f}"
            (w, h), _ = cv2.getTextSize(text, 0, 0.5, 1)
            cv2.rectangle(frame, (x1[i], y1[i]-20), (x1[i]+w, y1[i]), color, -1)
            cv2.putText(frame, text, (x1[i], y1[i] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
        return frame