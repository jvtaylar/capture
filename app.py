from ultralytics import YOLO

# Load pretrained YOLOv8 nano model (small & fast)
model = YOLO("yolov8n.pt")

# Run detection on an image from the web (YOLO has sample images)
results = model("https://ultralytics.com/images/bus.jpg")

# Print results summary
print(results)

# Show annotated image (this will pop up a window locally)
results[0].show()
