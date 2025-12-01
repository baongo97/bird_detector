# 🐦 Bird Detection POC - Streamlit Edition

A complete bird detection, tracking, and counting application using YOLO11n on CPU with Streamlit frontend and FastAPI backend.

## 🎯 Features

- **Real-time Bird Detection** - YOLO11n optimized for CPU inference
- **Live Webcam Stream** - Direct webcam input with live detection
- **File Upload** - Process images and video files
- **Object Tracking** - Track individual birds across frames
- **Bird Counting** - Count total and unique birds detected
- **Adjustable Parameters** - Real-time confidence and IoU threshold control
- **Performance Metrics** - FPS, processing time, and detection statistics
- **Web-based UI** - Easy-to-use Streamlit interface

## 📋 System Requirements

- Python 3.9+
- CPU-based laptop (8GB+ RAM recommended)
- Webcam (for live mode)

## 🚀 Quick Start

### Step 1: Install Dependencies

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install requirements
pip install -r requirements.txt
```

### Step 2: Create Directory Structure

```
bird_poc/
├── app.py                       # Streamlit frontend
├── bird_detector_api.py        # FastAPI backend
├── requirements.txt            # Python dependencies
├── .streamlit/
│   └── secrets.toml           # Configuration (create this file)
└── README.md
```

### Step 3: Run Backend API

Open **Terminal 1**:

```bash
python bird_detector_api.py
```

Expected output:
```
Loading YOLO11n model...
ONNX export complete
Uvicorn running on http://0.0.0.0:8000
```

### Step 4: Run Streamlit Frontend

Open **Terminal 2**:

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`

## 📊 Expected Performance

| Hardware | FPS | Latency | Notes |
|----------|-----|---------|-------|
| i7/i9 (11th+) | 10-15 fps | 65-100 ms | Best performance |
| i5 (11th gen) | 8-12 fps | 85-125 ms | Acceptable |
| MacBook M2/M3 | 18-25 fps | 40-55 ms | Excellent |

## 🎮 Usage Guide

### Webcam Detection
1. Click **"Start Webcam Detection"** button
2. Allow camera access when prompted
3. Adjust confidence slider to filter detections
4. Live detections with bounding boxes and labels
5. Click **"Stop Webcam"** to stop

### Image Upload
1. Select **"Upload Image"** from sidebar
2. Choose an image file (JPG, PNG, BMP)
3. Click **"Detect Birds"**
4. View detections with statistics

### Video Upload
1. Select **"Upload Video"** from sidebar
2. Choose a video file (MP4, AVI, MOV)
3. Click **"Detect in Video"**
4. Watch progress as frames are processed
5. View summary statistics

## ⚙️ Configuration

Edit sidebar settings:

- **Confidence Threshold** (0.1-1.0): Filter weak detections
  - Lower = More detections (may have false positives)
  - Higher = Fewer detections (more confident)
  
- **IoU Threshold** (0.1-0.95): Non-maximum suppression parameter
  - Lower = More boxes per bird
  - Higher = Fewer overlapping boxes
  
- **Enable Tracking**: Track individual birds across frames
- **Enable Counting**: Count unique birds
- **Draw Bounding Boxes**: Show detection boxes
- **Draw Labels**: Show confidence percentages

## 🔧 Backend API Reference

### POST /detect

Detect birds in an image.

**Request:**
```json
{
  "file": <binary image data>,
  "confidence": 0.5,
  "iou": 0.45
}
```

**Response:**
```json
{
  "detections": [
    {
      "x": 100,
      "y": 150,
      "width": 50,
      "height": 60,
      "confidence": 0.92,
      "class": "bird"
    }
  ],
  "count": 1,
  "processing_time_ms": 45.2
}
```

### GET /health

Health check endpoint.

**Response:**
```json
{
  "status": "ok"
}
```

## 📈 Performance Optimization Tips

### For CPU
1. **Reduce Input Resolution**: Modify backend to use 416×416 instead of 640×640
   ```python
   model(image, imgsz=416)  # 2x speedup
   ```

2. **Use OpenVINO** (Intel CPUs only):
   ```bash
   pip install openvino
   model.export(format='openvino')
   ```

3. **Skip Frames**: Process every 2nd or 3rd frame for faster throughput
   ```python
   if frame_count % 2 == 0:
       detections = detect(frame)
   ```

4. **Batch Processing**: Process multiple frames together
   ```python
   model([frame1, frame2, frame3], device='cpu')
   ```

### For GPU (Future)
When deploying to GPU (e.g., Jetson Orin):
1. Switch to YOLOv10 model
2. Use TensorRT optimization
3. Use NVIDIA DeepStream SDK

## 🐛 Troubleshooting

### "Cannot connect to backend"
- Ensure `bird_detector_api.py` is running
- Check that API is on `http://localhost:8000`
- Verify no firewall blocking port 8000

### "Cannot access webcam"
- Grant camera permissions to Python
- Check webcam is not in use by another app
- Try camera index 1 instead of 0 in backend

### Slow Performance
- Reduce confidence threshold (faster but more false positives)
- Lower input resolution (modify backend `imgsz=416`)
- Reduce FPS by skipping frames
- Close other CPU-intensive applications

### Out of Memory
- Reduce batch size in backend
- Lower resolution input images
- Use smaller YOLO model (yolo11s instead of yolo11m)

## 📚 Project Structure

```
bird_poc/
├── app.py                          # Streamlit frontend
├── bird_detector_api.py           # FastAPI backend
├── requirements.txt               # Dependencies
├── .streamlit/
│   └── secrets.toml              # API configuration
└── README.md                      # This file
```

## 🔄 Workflow

```
User Input (Webcam/File)
    ↓
Streamlit Frontend (app.py)
    ↓
FastAPI Backend (bird_detector_api.py)
    ↓
YOLO11n Model (ONNX on CPU)
    ↓
Detections JSON
    ↓
Streamlit UI (Draw & Display)
```

## 🎓 Next Steps

1. **Fine-tune on Custom Birds**: Train YOLO11n on your specific bird species
2. **Add Audio Classification**: Detect bird calls alongside visual detection
3. **Deploy to Jetson**: Move to edge device with GPU acceleration
4. **Build Pipeline**: Integrate with bird conservation databases
5. **Real-time Alerts**: Send notifications when rare species detected

## 📖 Resources

- [Ultralytics YOLO11 Docs](https://docs.ultralytics.com/models/yolo11/)
- [Streamlit Docs](https://docs.streamlit.io/)
- [FastAPI Docs](https://fastapi.tiangolo.com/)
- [OpenCV Docs](https://docs.opencv.org/)

## 📝 License

This POC is provided as-is for educational and research purposes.

## 🤝 Support

For issues or questions:
1. Check troubleshooting section above
2. Verify backend is running
3. Check Python version (3.9+)
4. Ensure all dependencies installed: `pip install -r requirements.txt`
