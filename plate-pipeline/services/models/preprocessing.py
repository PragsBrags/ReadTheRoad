from __future__ import annotations
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

def crop_resize_plate (plate_img: np.ndarray, min_width: int = 100) -> np.ndarray:
        # Upscaling
        h, w = plate_img.shape[:2]
        if w < min_width:
            scale = min_width / w
            new_w = int(w * scale)
            new_h = int(h * scale)
            plate_img = cv2.resize(plate_img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

        return plate_img
        
def preprocess_for_ocr(
              plate_img: np.ndarray,
              save_debug: bool = False,
              debug_dir: Optional[Path] = None,
              debug_prefix: str = "plate",
              ) -> np.ndarray:

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
        
        if save_debug and debug_dir:
               out = Path(debug_dir)
               out.mkdir(parents=True, exist_ok=True)
               # Save debug images
               cv2.imwrite(str(out / f"{debug_prefix}_upscaled.jpg"), plate_img)
               cv2.imwrite(str(out / f"{debug_prefix}_processed.jpg"), processed)
 
        return processed


        
     
