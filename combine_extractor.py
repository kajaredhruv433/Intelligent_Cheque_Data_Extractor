import cv2
import os
from ultralytics import YOLO
from paddleocr import PaddleOCR
from text_normalizer import normalize_date, normalize_amount_numeric, normalize_amount_in_words

# ===============================
# CONFIG
# ===============================
MODELS_DIR = "Models"
SOURCE = "Dataset/images/P_71231925000000851.tiff"
CONFIDENCE = 0.25

# ===============================
# OCR (CLASSIC, STABLE)
# ===============================
OCR_EN = PaddleOCR(
    use_angle_cls=True,
    lang="en",
    show_log=False
)

# ===============================
# LOAD YOLO MODELS
# ===============================
models = {}
for f in os.listdir(MODELS_DIR):
    if f.endswith(".pt"):
        name = os.path.splitext(f)[0]
        models[name] = YOLO(os.path.join(MODELS_DIR, f))
        print(f"✅ Loaded model: {name}")

# ===============================
# READ IMAGE
# ===============================
img = cv2.imread(SOURCE, cv2.IMREAD_UNCHANGED)
if img is None:
    raise ValueError("Image not found")

if len(img.shape) == 2:
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
elif img.shape[2] == 1:
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

print("✅ Image shape:", img.shape)

# ===============================
# YOLO + OCR
# ===============================
print("\n📄 OCR RESULTS")
print("=" * 60)

# ===============================
# FIELD MAPPING
# ===============================
FIELD_MAP = {
    "pay": "Pay Name",
    "amt_word": "Amount in Word",
    "amt_digit": "Amount in Digit",
    "date": "Date",
    "acc_holder": "Account Holder Name",
    "sign": "Sign"
}

extracted_data = {}
annotated_img = img.copy() # Create a copy for visualization

for field, model in models.items():
    results = model(img, conf=CONFIDENCE, verbose=False) # Use clean img
    
    # Default values
    extracted_text = None
    confidence = 0.0
    box_coords = None

    for r in results:
        if not r.boxes:
            continue

        # Get the highest confidence box if multiple are found (or just the first one)
        # simplistic approach: verify checks max conf
        best_box = None
        max_conf = -1

        for box in r.boxes:
            conf = float(box.conf[0])
            if conf > max_conf:
                max_conf = conf
                best_box = box

        if best_box:
            x1, y1, x2, y2 = map(int, best_box.xyxy[0])
            confidence = max_conf
            box_coords = (x1, y1, x2, y2)
            
            # Special handling for SIGN
            if field == "sign":
                extracted_text = "Yes" # Logic: if detected, it exists
            else:
                crop = img[y1:y2, x1:x2] # Crop from clean img
                if crop.size > 0:
                    res = OCR_EN.ocr(crop, cls=True)
                    text = []
                    if res:
                        for line in res:
                            for w in line:
                                text.append(w[1][0])
                    extracted_text = " ".join(text).strip()

    # Store result
    display_name = FIELD_MAP.get(field, field)
    
    # Normalize sign if not found
    if field == "sign" and not extracted_text:
        extracted_text = "No"
    
    # ===============================
    # NORMALIZATION
    # ===============================
    if extracted_text and extracted_text != "No" and extracted_text != "[NO TEXT FOUND]":
        if field == "date":
            norm_date = normalize_date(extracted_text)
            if norm_date:
                extracted_text = norm_date
                
        elif field == "amt_digit":
            norm_amt = normalize_amount_numeric(extracted_text)
            if norm_amt is not None:
                extracted_text = str(norm_amt) # Convert back to string for display
                
        elif field == "amt_word":
            norm_word_amt = normalize_amount_in_words(extracted_text)
            if norm_word_amt is not None:
                extracted_text = str(norm_word_amt)

    # Store data
    extracted_data[display_name] = extracted_text if extracted_text else "[NO TEXT FOUND]"
    
    # DRAWING (Optional, keeping it for testing)
    if box_coords:
        x1, y1, x2, y2 = box_coords
        cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (0, 255, 0), 2) # Draw on copy
        label = f"{display_name}"
        cv2.putText(annotated_img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2) # Draw on copy

# ===============================
# FINAL OUTPUT
# ===============================
print("\n📄 FINAL EXTRACTED DATA")
print("=" * 60)
for key, value in extracted_data.items():
    print(f"{key:<25} : {value}")
print("-" * 60)

# ===============================
# SAVE RESULT
# ===============================
OUTPUT_FILE = "output_with_boxes.jpg"
cv2.imwrite(OUTPUT_FILE, annotated_img) # Save the copy
print(f"\n✅ Saved annotated image to: {OUTPUT_FILE}")
