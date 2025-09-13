from ultralytics import YOLO

model = YOLO('yolov8n.pt')  # This will download the file from the internet automatically


model.train(
    data=r'C:\Users\vasan\OneDrive\Desktop\helmet_detection\data.yaml',
    epochs=50, # Or 100 if you want to try, but 50 is fine for diagnosis
    imgsz=640,
    name='training_with_augmentation', # NEW folder name for this run
    verbose=True,
    patience=50, # Ensures it runs for full epochs if not overfitting heavily

    # --- Data Augmentation Parameters (adjust these!) ---
    # These are some common ones you can explicitly set or adjust their intensity
    fliplr=0.5,       # Random horizontal flip (0.5 means 50% chance). Common.
    flipud=0.0,       # Random vertical flip (0.0 means no vertical flip). Often keep 0.
    hsv_h=0.015,      # Hue augmentation (range 0-1)
    hsv_s=0.7,        # Saturation augmentation (range 0-1)
    hsv_v=0.4,        # Value (brightness) augmentation (range 0-1)
    degrees=0.0,      # Random rotation (degrees)
    translate=0.1,    # Random translation (fraction of image size)
    scale=0.5,        # Random scaling (min/max scale factor)
    shear=0.0,        # Random shear (degrees)
    perspective=0.0,  # Random perspective distortion (0.0 to 0.001)
    mixup=0.0,        # MixUp augmentation (0.0 to 1.0, typically 0.0 or 0.1)
    copy_paste=0.0,   # Copy-Paste augmentation (0.0 to 1.0, typically 0.0 or 0.1)
    # ---------------------------------------------------
)

print("Training process initiated.")