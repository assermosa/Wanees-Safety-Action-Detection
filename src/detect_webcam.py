"""
detect_webcam.py
================
Quick desktop test of the trained YOLOv8n model (PyTorch weights) on a webcam.
Used to validate the model on a laptop before deploying the ONNX version to
the Raspberry Pi 5 (see detector_pi5.py).

Usage:
    python src/detect_webcam.py                      # defaults below
    python src/detect_webcam.py --model models/best.pt --conf 0.7 --camera 0
Press Q to quit.
"""
import argparse
import time
import cv2
from ultralytics import YOLO

# ============================================================
# Configuration
# ============================================================
parser = argparse.ArgumentParser(description="YOLOv8 webcam test")
parser.add_argument("--model", default="models/best.pt", help="Path to .pt weights")
parser.add_argument("--conf", type=float, default=0.7, help="Confidence threshold")
parser.add_argument("--camera", type=int, default=0, help="Webcam index")
args = parser.parse_args()

MODEL_PATH = args.model
CONFIDENCE = args.conf
CAMERA_ID = args.camera

# ============================================================
# Load Model
# ============================================================
print("[INFO] Loading model...")
model = YOLO(MODEL_PATH)
print("[INFO] Model loaded successfully!")

# ============================================================
# Open Camera
# ============================================================
cap = cv2.VideoCapture(CAMERA_ID)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

cap.set(cv2.CAP_PROP_FPS, 30)

if not cap.isOpened():
    raise RuntimeError("Could not open webcam.")

prev_time = time.time()

# ============================================================
# Inference Loop
# ============================================================
while True:

    ret, frame = cap.read()
    if not ret:
        break

    # YOLO Inference
    results = model.predict(
        source=frame,
        conf=CONFIDENCE,
        verbose=False
    )

    annotated = frame.copy()
    detections = 0

    for result in results:

        boxes = result.boxes

        if boxes is None:
            continue

        detections = len(boxes)

        for box in boxes:

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            cls = int(box.cls[0])
            conf = float(box.conf[0])

            label = f"{model.names[cls]} {conf:.2f}"

            cv2.rectangle(
                annotated,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            cv2.putText(
                annotated,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

    # FPS
    current = time.time()
    fps = 1 / (current - prev_time)
    prev_time = current

    cv2.putText(
        annotated,
        f"FPS: {fps:.1f}",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 0, 0),
        2
    )

    cv2.putText(
        annotated,
        f"Objects: {detections}",
        (20, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 255),
        2
    )

    cv2.imshow("YOLOv8 Detection", annotated)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
