import os
import cv2
from ultralytics import YOLO

# ===============================
# CONFIG
# ===============================
MODELS_DIR = "Models"
SOURCE = "Dataset/images/P_71231925000000841.tiff"
OUTPUT_PATH = "output.jpg"
CONFIDENCE = 0.25

# ===============================
# LOAD MODELS
# ===============================
models = {}

for file in os.listdir(MODELS_DIR):
    if file.lower().endswith(".pt"):
        model_name = os.path.splitext(file)[0]
        model_path = os.path.join(MODELS_DIR, file)
        models[model_name] = YOLO(model_path)
        print(f"✅ Loaded model: {model_name}")

# ===============================
# READ IMAGE (GRAYSCALE SAFE)
# ===============================
image = cv2.imread(SOURCE, cv2.IMREAD_UNCHANGED)

if image is None:
    raise ValueError("❌ Image not found!")

if len(image.shape) == 2 or image.shape[2] == 1:
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

print("✅ Image shape:", image.shape)

# ===============================
# RUN EACH MODEL → 1 BOX ONLY
# ===============================
for model_name, model in models.items():

    best_box = None  # (conf, x1, y1, x2, y2)

    results = model(image, conf=CONFIDENCE, verbose=False)

    for r in results:
        if r.boxes is None:
            continue

        for box in r.boxes:
            conf = float(box.conf[0])

            if best_box is None or conf > best_box[0]:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                best_box = (conf, x1, y1, x2, y2)

    # 🔒 DRAW ONLY BEST BOX FOR THIS MODEL
    if best_box is not None:
        conf, x1, y1, x2, y2 = best_box
        label = f"{model_name} {conf:.2f}"

        cv2.rectangle(
            image,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        cv2.putText(
            image,
            label,
            (x1, max(y1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

        print(f"✅ {model_name}: drawn 1 box ({conf:.2f})")
    else:
        print(f"⚠️ {model_name}: no detection")

# ===============================
# SAVE & DISPLAY
# ===============================
cv2.imwrite(OUTPUT_PATH, image)
print(f"✅ Output saved as {OUTPUT_PATH}")

cv2.imshow("YOLOv8 One Box Per Model", image)
cv2.waitKey(0)
cv2.destroyAllWindows()