import os
from ultralytics import YOLO
import cv2
import numpy as np
from paddleocr import PaddleOCR


class LicensePlateDetection :
    def __init__ (self, yolo_path) :
        self.model = YOLO(yolo_path)

    def license_coordinates(self, frame):
        results = self.model([frame], stream=True)
    
        best_box = None
        best_conf = 0.0

        for result in results:
            for box in result.boxes:
                conf = float(box.conf)
                x1, y1, x2, y2 = box.xyxy.numpy()[0]
                print(f"Detection | conf: {conf:.2%} | size: {int(x2-x1)}x{int(y2-y1)}px")

                # No threshold — take the highest confidence box
                if conf > best_conf:
                    best_conf = conf
                    best_box = (x1, y1, x2, y2)

        if best_box is not None:
            print(f"Best detection chosen: conf {best_conf:.2%}")
            return best_box

        print("No detections at all")
        return None

    def crop_into_plate (self, frame, x1, y1, x2, y2) :
        plate_img = frame[int(y1):int(y2), int(x1):int(x2)]
        # Upscaling
        h, w = plate_img.shape[:2]
        min_width = 100
        if w < min_width:
            scale = min_width / w
            new_w = int(w * scale)
            new_h = int(h * scale)
            plate_img = cv2.resize(plate_img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        return plate_img
    
class PaddleInference :

    def __init__(self):
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
        self.pipeline = PaddleOCR(
            use_doc_orientation_classify=False, 
            use_doc_unwarping=False,            
            use_textline_orientation=False,      
        )
        
    def preprocess_for_ocr(self, plate_img):

        # Convert to grayscale
        gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
        
        # Boost contrast
        gray = cv2.equalizeHist(gray)
        
        # Sharpen
        kernel = np.array([[0, -1, 0],
                        [-1, 5, -1],
                        [0, -1, 0]])
        gray = cv2.filter2D(gray, -1, kernel)
        
        # Convert back to BGR
        processed = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        
        # Save debug images
        cv2.imwrite("debug_plate_upscaled.jpg", plate_img)
        cv2.imwrite("debug_plate_processed.jpg", processed)
        
        return processed

    def ocr_inference(self, plate_img):
        processed = self.preprocess_for_ocr(plate_img)
        output = self.pipeline.predict(processed)
        
        all_texts = []
        all_scores = []
        for res in output:
            texts = res.get('rec_texts', [])
            scores = res.get('rec_scores', [])
            all_texts.extend(texts)
            all_scores.extend(scores)
        
        if all_texts:
            print(f"PLATE TEXT : {' '.join(all_texts)}")
            for text, score in zip(all_texts, all_scores):
                print(f"  '{text}' — confidence: {score:.2%}")
        else:
            print("Could not read plate text")
        
        return ' '.join(all_texts) if all_texts else None


        
     
