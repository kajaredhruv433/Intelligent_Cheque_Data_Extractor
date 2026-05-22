import cv2
import os
import ctypes

# -------- PATHS --------
IMAGE_DIR = r"images"
LABEL_DIR = r"C:\Users\dhruv\Music\Skill\Python\Majorpro\Majorpro2\Amt_word\1\labels"
os.makedirs(LABEL_DIR, exist_ok=True)

# -------- SCREEN SIZE (Windows) --------
user32 = ctypes.windll.user32
SCREEN_W = user32.GetSystemMetrics(0)
SCREEN_H = user32.GetSystemMetrics(1)

def resize_fit_height(img):
    h, w = img.shape[:2]
    scale = (SCREEN_H * 0.90) / h
    new_w = int(w * scale)
    new_h = int(h * scale)
    return cv2.resize(img, (new_w, new_h)), scale

image_files = sorted([
    f for f in os.listdir(IMAGE_DIR)
    if f.lower().endswith((".jpg", ".jpeg", ".png", ".tiff"))
])

print("\nINSTRUCTIONS:")
print(" - Draw up to 2 boxes")
print(" - c = cancel last box")
print(" - ENTER = save")
print(" - ESC = skip")
print(" - Class ID = 0\n")

for img_name in image_files:
    img_path = os.path.join(IMAGE_DIR, img_name)
    original = cv2.imread(img_path)

    if original is None:
        print(f"❌ Cannot open {img_name}")
        continue

    H, W = original.shape[:2]
    display_img, scale = resize_fit_height(original)

    boxes = []

    while True:
        img_copy = display_img.copy()

        # Draw existing boxes
        for (x, y, w, h) in boxes:
            cv2.rectangle(img_copy, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.namedWindow("YOLO Label Tool", cv2.WINDOW_NORMAL)
        cv2.imshow("YOLO Label Tool", img_copy)

        key = cv2.waitKey(1) & 0xFF

        # Draw new box (max 2)
        if key == ord('/') and len(boxes) < 2:
            roi = cv2.selectROI(

                "YOLO Label Tool",
                display_img,
                showCrosshair=True,
                fromCenter=False
            )
            if roi[2] > 0 and roi[3] > 0:
                boxes.append(roi)

        # Cancel last box
        elif key == ord('c') and boxes:
            boxes.pop()
            print("↩ Last box removed")

        # Save
        elif key == 13 and boxes:  # ENTER
            label_path = os.path.join(
                LABEL_DIR,
                os.path.splitext(img_name)[0] + ".txt"
            )

            with open(label_path, "w") as f:
                for (x, y, w, h) in boxes:
                    x = int(x / scale)
                    y = int(y / scale)
                    w = int(w / scale)
                    h = int(h / scale)

                    x_center = (x + w / 2) / W
                    y_center = (y + h / 2) / H
                    w_norm = w / W
                    h_norm = h / H

                    f.write(
                        f"0 {x_center:.6f} {y_center:.6f} "
                        f"{w_norm:.6f} {h_norm:.6f}\n"
                    )

            print(f"✅ Labeled: {img_name} ({len(boxes)} boxes)")
            break

        # Skip
        elif key == 27:  # ESC
            print(f"⏭ Skipped {img_name}")
            break

    cv2.destroyAllWindows()

print("\n🎉 All images labeled. Ready for YOLO training.")
