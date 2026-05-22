import cv2
import os
import logging
import numpy as np
from ultralytics import YOLO
from paddleocr import PaddleOCR

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MICRExtractor:
    def __init__(self, model_path="Models/micr.pt", confidence=0.25):
        logger.info("🔄 Initializing MICRExtractor...")
        self.model_path = model_path
        self.confidence = confidence
        
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"MICR model not found at {self.model_path}")
            
        self.model = YOLO(self.model_path)
        self.ocr = PaddleOCR(use_angle_cls=True, lang="en", show_log=False)
        logger.info("✅ Initialization Complete.")

    def extract_micr(self, image_path):
        """
        Extracts MICR text from a cheque image.
        """
        img = cv2.imread(image_path)
        if img is None:
            logger.error(f"Could not read image: {image_path}")
            return None

        # Convert grayscale to BGR if necessary
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.shape[2] == 1:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        # 1. Try finding with YOLO
        results = self.model(img, conf=self.confidence, verbose=False)
        best_box = None
        max_conf = -1

        for r in results:
            if not r.boxes: continue
            for box in r.boxes:
                conf = float(box.conf[0])
                if conf > max_conf:
                    max_conf = conf
                    best_box = box

        if best_box:
            logger.info(f"📍 MICR region found by YOLO (confidence: {max_conf:.2f})")
            x1, y1, x2, y2 = map(int, best_box.xyxy[0])
            crop = img[y1:y2, x1:x2]
        else:
            logger.warning("⚠️ YOLO failed to detect MICR region. Using fallback (bottom crop).")
            # Fallback: Crop bottom 15% of the image
            h, w = img.shape[:2]
            y1 = int(h * 0.85)
            crop = img[y1:h, 0:w]

        # 2. OCR the crop
        extracted_text = ""
        if crop.size > 0:
            res = self.ocr.ocr(crop, cls=True)
            if res:
                text_parts = []
                for line in res:
                    if line:
                        for word_info in line:
                            text_parts.append(word_info[1][0])
                extracted_text = " ".join(text_parts).strip()

        return extracted_text

if __name__ == "__main__":
    # Test with a sample image
    sample_image = "Dataset/images/P_71231925000000838.tiff"
    
    if os.path.exists(sample_image):
        extractor = MICRExtractor()
        micr_text = extractor.extract_micr(sample_image)
        print(f"\n--- Result ---")
        print(f"Image: {sample_image}")
        print(f"Extracted MICR: {micr_text}")
        print(f"--------------\n")
    else:
        print(f"Sample image not found: {sample_image}")
