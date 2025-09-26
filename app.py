# test_yolo_safe.py
from ultralytics import YOLO
from PIL import Image

# Load pretrained YOLOv8 nano model
model = YOLO("yolov8n.pt")

# Run detection on a sample image (bus with people)
results = model("https://ultralytics.com/images/bus.jpg")

# Save first result with bounding boxes
results[0].save("result.jpg")

# Load with PIL just to verify we got an image
img = Image.open("runs/detect/predict/result.jpg")
print("✅ YOLO detection ran successfully. Output image saved at:", img.filename)
