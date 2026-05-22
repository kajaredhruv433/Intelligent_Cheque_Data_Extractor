import os
import cv2
from ultralytics import YOLO

# ===============================
# CONFIG
# ===============================
MODELS_DIR = "Models"
SOURCE = "unnamed.jpg"
OUTPUT_PATH = "output.jpg"
CONFIDENCE = 0.25

# ===============================
# LOAD MODELS (KEY = PT FILE NAME)
# ===============================
models = {}

for file in os.listdir(MODELS_DIR):
    if file.lower().endswith(".pt"):
        model_name = os.path.splitext(file)[0]  # exact pt filename
        model_path = os.path.join(MODELS_DIR, file)
        models[model_name] = YOLO(model_path)
        print(f"✅ Loaded model: {model_name}")

# ===============================
# READ IMAGE (TIFF / GRAYSCALE SAFE)
# ===============================
image = cv2.imread(SOURCE, cv2.IMREAD_UNCHANGED)

if image is None:
    raise ValueError("❌ Image not found!")

# Convert grayscale → BGR
if len(image.shape) == 2:
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
elif image.shape[2] == 1:
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

print("✅ Image shape:", image.shape)  # (H, W, 3)

# ===============================
# RUN INFERENCE (LABEL = PT NAME)
# ===============================
for model_name, model in models.items():

    results = model(
        image,
        conf=CONFIDENCE,
        verbose=False
    )

    for r in results:
        if r.boxes is None:
            continue

        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])

            # 🔒 LABEL STRICTLY FROM PT FILE NAME
            label = f"{model_name} {conf:.2f}"

            # Draw bounding box
            cv2.rectangle(
                image,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            # Draw label
            cv2.putText(
                image,
                label,
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

# ===============================
# SAVE & DISPLAY
# ===============================
cv2.imwrite(OUTPUT_PATH, image)
print(f"✅ Output saved as {OUTPUT_PATH}")

cv2.imshow("YOLOv8 Multi-Model Detection", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
