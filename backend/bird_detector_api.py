from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
from ultralytics import YOLO
import io
from PIL import Image
import time

app = FastAPI()

# Enable CORS for browser requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model (downloads automatically on first run)
print("Loading YOLO11n model...")
model = YOLO('yolo11n.pt')

# Export to ONNX for CPU optimization (run once)
try:
    model.export(format='onnx', imgsz=640, optimize=True)
    print("ONNX export complete")
except:
    print("ONNX export skipped (may already exist)")

@app.post("/detect")
async def detect_birds(
    file: UploadFile = File(...),
    confidence: float = 0.5,
    iou: float = 0.45
):
    """
    Detect birds in an image
    """
    start_time = time.time()
    
    # Read image
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Run inference
    results = model(image, conf=confidence, iou=iou, device='cpu', verbose=False)
    result = results[0]
    
    # Extract detections
    detections = []
    if result.boxes is not None:
        boxes = result.boxes.xyxy.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()
        
        for box, conf in zip(boxes, confidences):
            x1, y1, x2, y2 = box
            detections.append({
                "x": float(x1),
                "y": float(y1),
                "width": float(x2 - x1),
                "height": float(y2 - y1),
                "confidence": float(conf),
                "class": "bird"
            })
    
    processing_time = (time.time() - start_time) * 1000
    
    return {
        "detections": detections,
        "count": len(detections),
        "processing_time_ms": processing_time
    }

@app.get("/health")
def health():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
