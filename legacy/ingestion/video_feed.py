import cv2
import time


object_detector = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=16,detectShadows=False)

def calculate_sharpness(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())

def frame(cap) :
    # Threshold for motion detection
    THRESHOLD = 100
    best_frame = None
    max_score = -1
    frames_since_motion = 0
    COOLDOWN_LIMIT = 10
    frame_count = 0
    WARMUP_FRAMES = 100

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Video ended.")
            return
        
        frame_count += 1
        mask = object_detector.apply(frame)
        if frame_count < WARMUP_FRAMES:
            continue
        
        contours, _ =cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        active_motion = False

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > THRESHOLD:
                active_motion = True
                score = calculate_sharpness(frame)
                final_score = score * area

                if final_score > max_score:
                    max_score = final_score
                    best_frame = frame.copy()
        
        if not active_motion and best_frame is not None:
            frames_since_motion = frames_since_motion + 1

            if frames_since_motion > COOLDOWN_LIMIT:
                yield best_frame
                best_frame = None
                max_score = -1
                frames_since_motion=0



         