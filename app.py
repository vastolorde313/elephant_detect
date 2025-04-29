import streamlit as st
import cv2
import torch
import tempfile
from ultralytics import YOLO
import numpy as np

# Load YOLOv8 model (custom trained model)
model = YOLO('best.pt')  # Make sure 'best1.pt' is correctly placed

# Function to detect animals in video
def detect_animals(video_path):
    cap = cv2.VideoCapture(video_path)
    stframe = st.empty()
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Run YOLOv8 model
        results = model.predict(frame, conf=0.1)  # Set confidence threshold if needed
        
        # Process results
        result = results[0]  # Only one result since we passed one frame
        if result.boxes is not None:
            for box in result.boxes:
                cls = int(box.cls.cpu().numpy())  # Convert tensor to int
                label = model.names[cls]
                if label in ['elephant']:  # You can add more animal classes if needed
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    conf = box.conf.item()
                    
                    # Draw bounding box and label
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Convert frame to RGB for Streamlit
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Display frame in Streamlit
        stframe.image(frame, channels="RGB")
        
    cap.release()

# Streamlit UI
st.title("Animal Detection in Video using YOLOv8")
uploaded_video = st.file_uploader("Upload a Video", type=["mp4", "avi", "mov", "mkv"])

if uploaded_video is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_video.read())
    detect_animals(tfile.name)
