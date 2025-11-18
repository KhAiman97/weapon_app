from fastapi import FastAPI, File, UploadFile, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import io
import os
import base64
import tempfile
import asyncio
from typing import List
import json

app = FastAPI(title="YOLOv11 Detection API")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = os.getenv("MODEL_PATH", "model.pt")
model = None

@app.on_event("startup")
async def load_model():
    global model
    try:
        model = YOLO(MODEL_PATH)
        print(f"Model loaded successfully from {MODEL_PATH}")
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        raise

@app.get("/")
async def root():
    return {
        "message": "YOLOv11 Detection API",
        "status": "running",
        "endpoints": {
            "detect_image": "/detect/image - POST image for detection",
            "detect_video": "/detect/video - POST video for detection",
            "realtime": "/detect/realtime - WebSocket for real-time detection",
            "health": "/health - Check API health"
        }
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None
    }

# 1. IMAGE DETECTION
@app.post("/detect/image")
async def detect_image(file: UploadFile = File(...), confidence: float = 0.25):
    """
    Detect objects in an uploaded image
    
    Args:
        file: Image file (jpg, png, etc.)
        confidence: Confidence threshold (default: 0.25)
    
    Returns:
        JSON with detection results
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        img_array = np.array(image)
        
        # Run inference with confidence threshold
        results = model(img_array, conf=confidence)
        
        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                detection = {
                    "class_id": int(box.cls[0]),
                    "class_name": model.names[int(box.cls[0])],
                    "confidence": float(box.conf[0]),
                    "bbox": {
                        "x1": float(box.xyxy[0][0]),
                        "y1": float(box.xyxy[0][1]),
                        "x2": float(box.xyxy[0][2]),
                        "y2": float(box.xyxy[0][3])
                    }
                }
                detections.append(detection)
        
        return JSONResponse(content={
            "success": True,
            "detections": detections,
            "count": len(detections),
            "image_size": {
                "width": image.width,
                "height": image.height
            }
        })
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Detection failed: {str(e)}")

# 2. VIDEO DETECTION
@app.post("/detect/video")
async def detect_video(
    file: UploadFile = File(...), 
    confidence: float = 0.25,
    skip_frames: int = 1
):
    """
    Detect objects in an uploaded video
    
    Args:
        file: Video file (mp4, avi, etc.)
        confidence: Confidence threshold (default: 0.25)
        skip_frames: Process every Nth frame (default: 1 = all frames)
    
    Returns:
        JSON with detection results per frame
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if not file.content_type.startswith("video/"):
        raise HTTPException(status_code=400, detail="File must be a video")
    
    try:
        # Save uploaded video to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_video:
            contents = await file.read()
            temp_video.write(contents)
            temp_video_path = temp_video.name
        
        # Open video
        cap = cv2.VideoCapture(temp_video_path)
        
        if not cap.isOpened():
            raise HTTPException(status_code=400, detail="Could not open video file")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        frame_results = []
        frame_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Skip frames if specified
            if frame_count % skip_frames != 0:
                frame_count += 1
                continue
            
            # Run detection
            results = model(frame, conf=confidence)
            
            detections = []
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    detection = {
                        "class_id": int(box.cls[0]),
                        "class_name": model.names[int(box.cls[0])],
                        "confidence": float(box.conf[0]),
                        "bbox": {
                            "x1": float(box.xyxy[0][0]),
                            "y1": float(box.xyxy[0][1]),
                            "x2": float(box.xyxy[0][2]),
                            "y2": float(box.xyxy[0][3])
                        }
                    }
                    detections.append(detection)
            
            frame_results.append({
                "frame_number": frame_count,
                "timestamp": frame_count / fps,
                "detections": detections,
                "count": len(detections)
            })
            
            frame_count += 1
        
        cap.release()
        os.unlink(temp_video_path)  # Delete temp file
        
        return JSONResponse(content={
            "success": True,
            "video_info": {
                "fps": fps,
                "total_frames": total_frames,
                "processed_frames": len(frame_results),
                "width": width,
                "height": height,
                "duration_seconds": total_frames / fps if fps > 0 else 0
            },
            "frames": frame_results
        })
    
    except Exception as e:
        if 'temp_video_path' in locals():
            try:
                os.unlink(temp_video_path)
            except:
                pass
        raise HTTPException(status_code=500, detail=f"Video detection failed: {str(e)}")

# 3. REAL-TIME DETECTION (WebSocket)
@app.websocket("/detect/realtime")
async def detect_realtime(websocket: WebSocket):
    """
    Real-time object detection via WebSocket
    
    Client should send base64-encoded images
    Server responds with detection results
    
    Message format (from client):
    {
        "image": "base64_encoded_image_string",
        "confidence": 0.25  // optional
    }
    
    Response format (from server):
    {
        "success": true,
        "detections": [...],
        "count": 5,
        "processing_time": 0.123
    }
    """
    await websocket.accept()
    
    if model is None:
        await websocket.send_json({
            "success": False,
            "error": "Model not loaded"
        })
        await websocket.close()
        return
    
    try:
        while True:
            # Receive message from client
            data = await websocket.receive_text()
            
            try:
                import time
                start_time = time.time()
                
                message = json.loads(data)
                
                # Decode base64 image
                image_data = base64.b64decode(message.get("image", ""))
                confidence = message.get("confidence", 0.25)
                
                # Convert to numpy array
                nparr = np.frombuffer(image_data, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if img is None:
                    await websocket.send_json({
                        "success": False,
                        "error": "Invalid image data"
                    })
                    continue
                
                # Run detection
                results = model(img, conf=confidence)
                
                detections = []
                for result in results:
                    boxes = result.boxes
                    for box in boxes:
                        detection = {
                            "class_id": int(box.cls[0]),
                            "class_name": model.names[int(box.cls[0])],
                            "confidence": float(box.conf[0]),
                            "bbox": {
                                "x1": float(box.xyxy[0][0]),
                                "y1": float(box.xyxy[0][1]),
                                "x2": float(box.xyxy[0][2]),
                                "y2": float(box.xyxy[0][3])
                            }
                        }
                        detections.append(detection)
                
                processing_time = time.time() - start_time
                
                # Send results back
                await websocket.send_json({
                    "success": True,
                    "detections": detections,
                    "count": len(detections),
                    "processing_time": round(processing_time, 3),
                    "image_size": {
                        "width": img.shape[1],
                        "height": img.shape[0]
                    }
                })
                
            except json.JSONDecodeError:
                await websocket.send_json({
                    "success": False,
                    "error": "Invalid JSON format"
                })
            except Exception as e:
                await websocket.send_json({
                    "success": False,
                    "error": f"Processing error: {str(e)}"
                })
    
    except WebSocketDisconnect:
        print("WebSocket client disconnected")
    except Exception as e:
        print(f"WebSocket error: {str(e)}")
        try:
            await websocket.close()
        except:
            pass

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000)),
        reload=False
    )