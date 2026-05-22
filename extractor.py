import cv2
import os
import logging
from ultralytics import YOLO
from paddleocr import PaddleOCR
from text_normalizer import normalize_date, normalize_amount_numeric, normalize_amount_in_words
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ChequeExtractor:
    def __init__(self, models_dir="Models", confidence=0.25):
        print("🔄 [ChequeExtractor] Initializing...")
        self.models_dir = models_dir
        self.confidence = confidence
        self.models = {}
        self.ocr = None
        self._load_models()
        print("✅ [ChequeExtractor] Initialization Complete.")

    def _load_models(self):
        """Loads YOLO models and PaddleOCR."""
        if not os.path.exists(self.models_dir):
            raise FileNotFoundError(f"Models directory '{self.models_dir}' not found.")
        
        # Load YOLO models
        logger.info("Loading YOLO models...")
        for f in os.listdir(self.models_dir):
            if f.endswith(".pt"):
                name = os.path.splitext(f)[0]
                model_path = os.path.join(self.models_dir, f)
                self.models[name] = YOLO(model_path)
                logger.info(f"✅ Loaded model: {name}")

        # Load OCR
        logger.info("Loading PaddleOCR...")
        self.ocr = PaddleOCR(use_angle_cls=True, lang="en", show_log=False)

    def process_image(self, image_source):
        """
        Process an image (path or array) and return extracted data.
        Returns: (extracted_data_dict, annotated_image_bgr)
        """
        # Load Image
        if isinstance(image_source, str):
            img = cv2.imread(image_source, cv2.IMREAD_UNCHANGED)
            if img is None:
                raise ValueError(f"Could not read image: {image_source}")
        elif isinstance(image_source, np.ndarray):
             img = image_source
        else:
             raise ValueError("image_source must be a path string or numpy array")

        # Handle grayscale
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.shape[2] == 1:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        extracted_data = {}
        annotated_img = img.copy()

        FIELD_MAP = {
            "pay": "Pay Name",
            "amt_word": "Amount in Word",
            "amt_digit": "Amount in Digit",
            "date": "Date",
            "acc_holder": "Account Holder Name",
            "sign": "Sign",
            "acc_no": "Account Number" 
        }

        # Initialize all fields with defaults
        for key in FIELD_MAP.values():
            extracted_data[key] = "[NO TEXT FOUND]"
        extracted_data["Sign"] = "No" # Default sign to No

        for field, model in self.models.items():
            match_field_name = field # default
            # mapping filename to simple field name if needed, but current setup seems to allow direct match
            # models key names are filenames without .pt. 
            # E.g. acc_holder, acc_no, amt_digit, amt_word, date, pay, sign
            
            results = model(img, conf=self.confidence, verbose=False)
            
            extracted_text = None
            max_conf = -1
            best_box = None

            for r in results:
                if not r.boxes: continue
                for box in r.boxes:
                    conf = float(box.conf[0])
                    if conf > max_conf:
                        max_conf = conf
                        best_box = box

            if best_box:
                x1, y1, x2, y2 = map(int, best_box.xyxy[0])
                
                # Draw on annotated image
                display_name = FIELD_MAP.get(field, field)
                cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(annotated_img, display_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                if field == "sign":
                    extracted_text = "Yes"
                else:
                    crop = img[y1:y2, x1:x2]
                    if crop.size > 0:
                        res = self.ocr.ocr(crop, cls=True)
                        text = []
                        if res:
                            for line in res:
                                if line: # Check if line is not None
                                    for w in line:
                                        if w and len(w) > 1: # check structure
                                             text.append(w[1][0])
                        extracted_text = " ".join(text).strip()

            extract_key = FIELD_MAP.get(field, field)
            
            # Save raw if needed, but we go straight to normalized logic
            
            # Normalization
            if extracted_text and extracted_text != "Yes": # Don't normalize "Yes" for sign
                 # Basic check
                 pass

            # Apply Logic
            final_val = extracted_text
            
            if field == "sign":
                if not extracted_text:
                    final_val = "No"
            elif not extracted_text:
                final_val = "[NO TEXT FOUND]"
            else:
                # Normalize
                if field == "date":
                    norm = normalize_date(extracted_text)
                    if norm: final_val = norm
                elif field == "amt_digit":
                    norm = normalize_amount_numeric(extracted_text)
                    if norm is not None: final_val = str(norm)
                elif field == "amt_word":
                    norm = normalize_amount_in_words(extracted_text)
                    if norm is not None: final_val = str(norm)
            
            # Update data
            extracted_data[extract_key] = final_val

        return extracted_data, annotated_img
