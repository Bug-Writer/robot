# detector.py
from ultralytics import YOLO
import numpy as np
import cv2

class YoloEDetector:
    def __init__(self, model_path="yoloe-11m-seg.pt", target_names=None):
        self.names = target_names if target_names is not None else ["mouse"]
        self.model = YOLO(model_path)  # 加载 YOLOE 掩膜分割模型

    def detect(self, image):
        # 执行推理（支持掩膜）
        self.model.set_classes(self.names, self.model.get_text_pe(self.names))
        results = self.model.predict(source=image, save=False, imgsz=640, conf=0.25, verbose=False)

        # YOLOE返回的是list，每张图一个结果
        if not results or len(results) == 0:
            return []

        result = results[0]  # 当前仅支持单张图
        boxes = result.boxes.xyxy.cpu().numpy() if result.boxes else []
        masks = result.masks.data.cpu().numpy() if result.masks else []
        cls = result.boxes.cls.cpu().numpy() if result.boxes else []

        detections = []
        for i in range(len(boxes)):
            detections.append({
                "box": boxes[i],
                "mask": masks[i] if len(masks) > i else None,
                "class_id": int(cls[i]),
                "confidence": float(result.boxes.conf[i])
            })

        return detections
