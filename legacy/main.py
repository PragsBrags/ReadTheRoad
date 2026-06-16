import os
from ingestion.video_feed import frame
from ocr.licensePlate import LicensePlateDetection, PaddleInference
from ocr.vehicleDetection import VehicleDetection
from ultralytics import YOLO
import cv2

script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, 'models/license_plate.pt')

cap = cv2.VideoCapture(os.path.join(script_dir,'test_video.mp4'))
LPD = LicensePlateDetection(model_path)
PI = PaddleInference()

for frames in frame(cap):
    coords = LPD.license_coordinates(frames)
    if coords is None:
        continue  # Skip this frame if no plate detected

    x1, y1, x2, y2 = coords
   
    plate_img = LPD.crop_into_plate(frames, x1, y1, x2, y2)
    print(f"Plate crop size: {plate_img.shape[1]}x{plate_img.shape[0]}px")
    res = PI.ocr_inference(plate_img)
    if res:
        print("OCR result:\n",res)
    else:
        print("OCR returned None")
