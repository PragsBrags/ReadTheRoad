from ultralytics import YOLO
import cv2

class VehicleDetection :
    def __init__ (self, yolo_model) :
        self.model = YOLO(yolo_model)
    
    def vehicle_class(self, frame):
        results = self.model([frame], stream=True)
        detections = []
        for result in results:
            for box in result.boxes:
                cls = int(box.cls)
                conf = float(box.conf)
                class_name = self.model.names[cls]
                detections.append((class_name, conf))
        return detections

