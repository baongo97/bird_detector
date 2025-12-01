# Bird Detection POC - Streamlit Frontend
# Run with: streamlit run app.py

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import requests
import json
from datetime import datetime
import time

# Page configuration
st.set_page_config(
    page_title="Bird Detection",
    page_icon="🐦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 0;
    }
    .stMetric {
        background-color: #f0f8f9;
        padding: 10px;
        border-radius: 8px;
    }
    .detection-box {
        background-color: #e8f4f8;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #208091;
        margin: 5px 0;
    }
    h1 {
        color: #134252;
    }
    .info-box {
        background-color: #fff3cd;
        padding: 12px;
        border-radius: 6px;
        border-left: 4px solid #a84d2f;
        margin: 10px 0;
    }
    </style>
""", unsafe_allow_html=True)

# API Configuration
API_URL = st.secrets.get("API_URL", "http://localhost:8000")

# Session state initialization
if "tracked_birds" not in st.session_state:
    st.session_state.tracked_birds = {}
if "detection_history" not in st.session_state:
    st.session_state.detection_history = []

# Header
st.markdown("# 🐦 Bird Detection POC")
st.markdown("Real-time bird detection, tracking, and counting using YOLO11n (CPU-optimized)")


# Helper Functions
def call_detection_api(image, confidence, iou_threshold):
    """Call backend API for bird detection"""
    try:
        # Convert PIL image to bytes
        img_bytes = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        _, buffer = cv2.imencode('.jpg', img_bytes)
        
        files = {'file': buffer.tobytes()}
        params = {
            'confidence': confidence,
            'iou': iou_threshold
        }
        
        response = requests.post(
            f"{API_URL}/detect",
            files={'file': ('image.jpg', buffer.tobytes(), 'image/jpeg')},
            params=params,
            timeout=30
        )
        
        if response.status_code == 200:
            return response.json().get("detections", [])
        else:
            st.error(f"API Error: {response.status_code}")
            return []
    
    except requests.exceptions.ConnectionError:
        st.error("❌ Cannot connect to backend. Make sure the Python API is running on http://localhost:8000")
        return []
    except Exception as e:
        st.error(f"Error calling API: {str(e)}")
        return []

def draw_detections_on_image(image_array, detections, draw_boxes=True, draw_labels=True):
    """Draw bounding boxes and labels on image"""
    image = image_array.copy()
    
    for det in detections:
        x1 = int(det["x"])
        y1 = int(det["y"])
        x2 = int(det["x"] + det["width"])
        y2 = int(det["y"] + det["height"])
        confidence = det["confidence"]
        
        # Draw box
        if draw_boxes:
            cv2.rectangle(image, (x1, y1), (x2, y2), (32, 128, 145), 2)
        
        # Draw label
        if draw_labels:
            label = f"Bird {confidence*100:.0f}%"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            
            # Draw background for text
            cv2.rectangle(
                image,
                (x1, y1 - label_size[1] - 4),
                (x1 + label_size[0] + 4, y1),
                (32, 128, 145),
                -1
            )
            
            # Draw text
            cv2.putText(
                image,
                label,
                (x1 + 2, y1 - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1
            )
    
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def run_webcam_detection(frame_ph, status_ph, confidence, iou, tracking, counting):
    """Process webcam stream with real-time detection"""
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        st.error("❌ Cannot access webcam")
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    frame_count = 0
    frame_times = []
    
    stop_button = st.button("⏹️ Stop Webcam")
    
    while not stop_button and st.session_state.get("webcam_running", True):
        ret, frame = cap.read()
        
        if not ret:
            st.error("Failed to read from webcam")
            break
        
        frame_count += 1
        start_time = time.time()
        
        # Detect
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        detections = call_detection_api(pil_image, confidence, iou)
        
        # Draw
        annotated = draw_detections_on_image(frame_rgb, detections)
        
        # Update display
        frame_ph.image(annotated, use_container_width=True)
        
        # Update metrics
        processing_ms = (time.time() - start_time) * 1000
        frame_times.append(processing_ms)
        
        if len(frame_times) > 30:
            frame_times.pop(0)
            avg_fps = 1000 / np.mean(frame_times)
        else:
            avg_fps = 0
        
        status_ph.write(f"Frame: {frame_count} | FPS: {avg_fps:.1f} | Processing: {processing_ms:.1f}ms | Detections: {len(detections)}")
    
    cap.release()

def process_video(video_file, frame_ph, status_ph, confidence, iou, tracking, counting):
    """Process uploaded video file"""
    # Save uploaded video temporarily
    video_path = f"/tmp/video_{datetime.now().timestamp()}.mp4"
    with open(video_path, "wb") as f:
        f.write(video_file.getbuffer())
    
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    frame_count = 0
    detections_per_frame = []
    
    progress_bar = st.progress(0)
    
    while frame_count < total_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Convert frame to RGB (needed for both detection and display)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detect every 2 frames for speed
        if frame_count % 2 == 0:
            pil_image = Image.fromarray(frame_rgb)
            detections = call_detection_api(pil_image, confidence, iou)
        else:
            detections = []
        
        detections_per_frame.append(len(detections))
        
        # Draw
        annotated = draw_detections_on_image(frame_rgb, detections)
        frame_ph.image(annotated, use_container_width=True)
        
        progress = frame_count / total_frames
        progress_bar.progress(progress)
        status_ph.write(f"Frame {frame_count}/{total_frames} | Birds: {len(detections)}")
    
    cap.release()
    
    # Summary
    st.success("✅ Video processing complete!")
    st.markdown(f"""
    **Video Summary:**
    - Total Frames: {total_frames}
    - Duration: {total_frames/fps:.1f}s
    - Average Birds per Frame: {np.mean(detections_per_frame):.1f}
    - Max Birds in Frame: {max(detections_per_frame)}
    """)

# Sidebar Configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Model Parameters
    st.subheader("Detection Settings")
    confidence = st.slider("Confidence Threshold", 0.1, 1.0, 0.5, 0.05)
    iou_threshold = st.slider("IoU Threshold", 0.1, 0.95, 0.45, 0.05)
    
    # Features
    st.subheader("Features")
    enable_tracking = st.checkbox("Enable Tracking", value=True)
    enable_counting = st.checkbox("Enable Counting", value=True)
    draw_boxes = st.checkbox("Draw Bounding Boxes", value=True)
    draw_labels = st.checkbox("Draw Labels", value=True)
    
    # Input Source Selection
    st.subheader("Input Source")
    input_source = st.radio("Select Source", ["Webcam", "Upload Image", "Upload Video"])
    
    st.divider()
    st.info(
        "💡 **Tip**: Adjust confidence to filter weak detections. "
        "Lower values catch more birds but may have false positives."
    )

# Main Content
col1, = st.columns([3])

with col1:
    st.subheader("📹 Detection Stream")
    frame_placeholder = st.empty()
    status_placeholder = st.empty()
    stats_placeholder = st.empty()

# Process based on input source
if input_source == "Webcam":
    st.subheader("🎥 Live Webcam Feed")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("📹 Start Webcam Detection", key="start_webcam"):
            st.session_state.webcam_running = True
    
    with col2:
        if st.button("⏹️ Stop Webcam", key="stop_webcam"):
            st.session_state.webcam_running = False
    
    if st.session_state.get("webcam_running", False):
        run_webcam_detection(
            frame_placeholder,
            status_placeholder,
            confidence,
            iou_threshold,
            enable_tracking,
            enable_counting
        )

elif input_source == "Upload Image":
    st.subheader("🖼️ Upload Image for Detection")
    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png", "bmp"])
    
    if uploaded_file and st.button("🔍 Detect Birds"):
        with st.spinner("Processing..."):
            image = Image.open(uploaded_file)
            image_array = np.array(image)
            
            start_time = time.time()
            detections = call_detection_api(image, confidence, iou_threshold)
            processing_ms = (time.time() - start_time) * 1000
            
            # Draw detections
            annotated_image = draw_detections_on_image(
                image_array, detections, draw_boxes, draw_labels
            )
            
            frame_placeholder.image(annotated_image, use_container_width=True)
            
            # Update stats (summary-style)
            avg_conf_text = "0%" if not detections else f"{np.mean([d['confidence'] for d in detections])*100:.1f}%"
            stats_placeholder.markdown(f"""
            **Detection Summary:**
            - Birds Detected: {len(detections)}
            - Avg Confidence: {avg_conf_text}
            - Processing: {processing_ms:.1f} ms
            """)
            
            # Display detections
            st.markdown("### Detected Birds")
            for i, det in enumerate(detections, 1):
                st.markdown(f"""
                <div class="detection-box">
                <b>Bird #{i}</b><br>
                Confidence: {det['confidence']*100:.1f}% | 
                Position: ({det['x']:.0f}, {det['y']:.0f}) | 
                Size: {det['width']:.0f}×{det['height']:.0f}
                </div>
                """, unsafe_allow_html=True)

elif input_source == "Upload Video":
    st.subheader("🎬 Upload Video for Detection")
    uploaded_video = st.file_uploader("Choose a video", type=["mp4", "avi", "mov"])
    
    if uploaded_video and st.button("🔍 Detect in Video"):
        with st.spinner("Processing video..."):
            process_video(
                uploaded_video,
                frame_placeholder,
                status_placeholder,
                confidence,
                iou_threshold,
                enable_tracking,
                enable_counting
            )

# Footer
st.divider()
st.markdown("""
---
**🔧 Backend Setup Required:**

This Streamlit app requires a Python backend running YOLO11n inference.

1. Run the backend API:
   ```bash
   python bird_detector_api.py
   ```

2. Backend should be accessible at: `http://localhost:8000`

For detailed setup, see the README in the project root.
""")