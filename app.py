import streamlit as st
import numpy as np
from ultralytics import YOLO
from PIL import Image, ImageDraw

st.title("👤 Person Detection using YOLOv8 (no OpenCV)")

# Load pretrained YOLOv8 model (nano version is fast & small)
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Open uploaded image
    img = Image.open(uploaded_file).convert("RGB")

    # Run YOLO detection
    results = model.predict(np.array(img))

    # Draw bounding boxes using PIL
    draw = ImageDraw.Draw(img)
    person_count = 0

    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            label = model.names[cls_id]
            if label == "person":
                person_count += 1
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                draw.rectangle([x1, y1, x2, y2], outline="green", width=3)
                draw.text((x1, y1 - 10), label, fill="green")

    # Show image and result
    st.image(img, caption=f"Persons detected: {person_count}")
    st.success(f"✅ Number of persons detected: {person_count}")
