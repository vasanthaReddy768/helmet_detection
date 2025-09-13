import cv2
import os
import csv
from ultralytics import YOLO
import numpy as np

# --- Configuration Section ---
MODEL_PATH = r"C:\Users\vasan\runs\detect\training_with_augmentation12\weights\best.pt"
VIDEO_PATH = "detector.mp4"
VIOLATIONS_DIR = "violations"
LOG_FILE_NAME = "violation_log.csv"
CONFIDENCE_THRESHOLD = 0.25  # Adjust confidence as needed

# --- Load YOLO Model ---
try:
    model = YOLO(MODEL_PATH)
    print("✅ Model loaded successfully.")
    print("✅ MODEL CLASSES:", model.names)
except Exception as e:
    print(f"❌ Error loading YOLO model: {e}")
    exit()

# --- Ensure Violations Directory Exists ---
os.makedirs(VIOLATIONS_DIR, exist_ok=True)

# --- Load Input Video ---
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print(f"❌ Could not open video '{VIDEO_PATH}'")
    exit()

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
print(f"Video Resolution: {frame_width}x{frame_height}, FPS: {fps:.2f}")

# --- Prepare CSV Logging ---
log_file = open(LOG_FILE_NAME, mode="w", newline="")
csv_writer = csv.writer(log_file)
csv_writer.writerow(["Frame", "Time (s)", "Label", "Confidence", "Bounding Box (x1,y1,x2,y2)"])

frame_number = 0

# --- Label Mapping for Colors and Display ---
label_map = {
    "with helmet": ("With Helmet", (0, 255, 0), (0, 0, 0)),
    "without helmet": ("No Helmet", (0, 0, 255), (255, 255, 255))
}

print("\nStarting video processing. Press 'q' to quit.")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("End of video or failed to read frame.")
        break

    frame_number += 1
    time_stamp = frame_number / fps
    annotated_frame = frame.copy()

    # --- Run YOLO Prediction ---
    results = model.predict(source=annotated_frame, conf=CONFIDENCE_THRESHOLD, imgsz=640, verbose=False)

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls_id = int(box.cls[0])
            confidence = float(box.conf[0])
            
            # Get label string
            label = model.names[cls_id].lower()
            
            # Debug: print detections
            print(f"[Frame {frame_number:05d}] Detected: {label} (Conf: {confidence:.2f}) at [{x1},{y1},{x2},{y2}]")

            # Get display name, color, and text color
            display_name, color, text_color = label_map.get(label, ("Unknown", (0, 0, 0), (255, 255, 255)))
            display_text = f"{display_name} ({confidence:.2f})"

            # Save violation images only for 'No Helmet'
            if label == "without helmet":
                filename = os.path.join(VIOLATIONS_DIR, f"frame_{frame_number:05d}.jpg")
                cv2.imwrite(filename, annotated_frame)

            # Log all detections
            csv_writer.writerow([frame_number, round(time_stamp, 2), display_name, f"{confidence:.2f}", f"({x1},{y1},{x2},{y2})"])

            # --- Drawing Bounding Box ---
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)

            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.7
            font_thickness = 2

            (text_width, text_height), baseline = cv2.getTextSize(display_text, font, font_scale, font_thickness)
            text_y_position = max(y1 - 10, text_height + 5)

            # Draw text background rectangle
            cv2.rectangle(annotated_frame, (x1, text_y_position - text_height - baseline),
                          (x1 + text_width, text_y_position + baseline), color, cv2.FILLED)

            # Draw label text
            cv2.putText(annotated_frame, display_text, (x1, text_y_position),
                        font, font_scale, text_color, font_thickness, cv2.LINE_AA)

    # --- Display Frame ---
    cv2.imshow("Helmet Detection", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("\n'q' pressed. Exiting video stream.")
        break

# --- Cleanup ---
cap.release()
log_file.close()
cv2.destroyAllWindows()
print("\nProcessing complete. Video released and log file closed.")
