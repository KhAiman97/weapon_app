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
import warnings
# Suppress NNPACK warnings
warnings.filterwarnings('ignore', category=UserWarning)

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

# NEW: Default configuration
DEFAULT_CONFIDENCE = 0.55  # Increased from 0.25 to 0.55 (55%)
DEFAULT_MAX_DETECTIONS = 50  # Maximum detections per frame/image

@app.on_event("startup")
async def load_model():
    global model
    try:
        model = YOLO(MODEL_PATH)
        print(f"Model loaded successfully from {MODEL_PATH}")
        print(f"Default confidence threshold: {DEFAULT_CONFIDENCE}")
        print(f"Default max detections: {DEFAULT_MAX_DETECTIONS}")
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        raise

@app.get("/")
async def root():
    return {
        "message": "YOLOv11 Detection API",
        "status": "running",
        "default_config": {
            "confidence_threshold": DEFAULT_CONFIDENCE,
            "max_detections": DEFAULT_MAX_DETECTIONS
        },
        "endpoints": {
            "detect_image": "/detect/image - POST image for detection",
            "detect_video": "/detect/video - POST video for detection",
            "realtime": "/detect/realtime - WebSocket for real-time detection",
            "health": "/health - Check API health",
            "model_classes": "/model/classes - GET list of detectable classes"
        }
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "config": {
            "confidence_threshold": DEFAULT_CONFIDENCE,
            "max_detections": DEFAULT_MAX_DETECTIONS
        }
    }

def filter_top_detections(detections: list, max_detections: int) -> list:
    """
    Filter detections to keep only top N by confidence score
    
    Args:
        detections: List of detection dictionaries
        max_detections: Maximum number of detections to keep
    
    Returns:
        Filtered list of detections
    """
    if len(detections) <= max_detections:
        return detections
    
    # Sort by confidence (descending) and take top N
    sorted_detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)
    return sorted_detections[:max_detections]

# 1. IMAGE DETECTION
@app.post("/detect/image")
async def detect_image(
    file: UploadFile = File(...), 
    confidence: float = DEFAULT_CONFIDENCE,
    max_detections: int = DEFAULT_MAX_DETECTIONS
):
    """
    Detect objects in an uploaded image
    
    Args:
        file: Image file (jpg, png, etc.)
        confidence: Confidence threshold (default: 0.55)
        max_detections: Maximum number of detections to return (default: 50)
    
    Returns:
        JSON with detection results
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    # Validate parameters
    if confidence < 0.0 or confidence > 1.0:
        raise HTTPException(status_code=400, detail="Confidence must be between 0.0 and 1.0")
    
    if max_detections < 1:
        raise HTTPException(status_code=400, detail="max_detections must be at least 1")
    
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
        
        # Apply max detections limit
        original_count = len(detections)
        detections = filter_top_detections(detections, max_detections)
        
        return JSONResponse(content={
            "success": True,
            "detections": detections,
            "count": len(detections),
            "original_count": original_count,
            "truncated": original_count > max_detections,
            "image_size": {
                "width": image.width,
                "height": image.height
            },
            "parameters": {
                "confidence_threshold": confidence,
                "max_detections": max_detections
            }
        })
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Detection failed: {str(e)}")

# 2. VIDEO DETECTION (Streaming Response)
@app.post("/detect/video")
async def detect_video(
    file: UploadFile = File(...), 
    confidence: float = DEFAULT_CONFIDENCE,
    max_detections: int = DEFAULT_MAX_DETECTIONS,
    skip_frames: int = 5,
    max_frames: int = 300
):
    """
    Detect objects in an uploaded video with streaming response
    
    Args:
        file: Video file (mp4, avi, etc.)
        confidence: Confidence threshold (default: 0.55)
        max_detections: Maximum detections per frame (default: 50)
        skip_frames: Process every Nth frame (default: 5 for faster processing)
        max_frames: Maximum frames to process (default: 300)
    
    Returns:
        Streaming JSON with detection results per frame
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if not file.content_type.startswith("video/"):
        raise HTTPException(status_code=400, detail="File must be a video")
    
    # Validate parameters
    if confidence < 0.0 or confidence > 1.0:
        raise HTTPException(status_code=400, detail="Confidence must be between 0.0 and 1.0")
    
    if max_detections < 1:
        raise HTTPException(status_code=400, detail="max_detections must be at least 1")
    
    temp_video_path = None
    
    try:
        # Save uploaded video to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_video:
            contents = await file.read()
            temp_video.write(contents)
            temp_video_path = temp_video.name
        
        # Open video
        cap = cv2.VideoCapture(temp_video_path)
        
        if not cap.isOpened():
            if temp_video_path and os.path.exists(temp_video_path):
                os.unlink(temp_video_path)
            raise HTTPException(status_code=400, detail="Could not open video file")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        async def generate_results():
            frame_count = 0
            processed_count = 0
            
            # Send video info first
            yield json.dumps({
                "type": "video_info",
                "fps": fps,
                "total_frames": total_frames,
                "width": width,
                "height": height,
                "duration_seconds": total_frames / fps if fps > 0 else 0,
                "parameters": {
                    "confidence_threshold": confidence,
                    "max_detections": max_detections,
                    "skip_frames": skip_frames
                }
            }) + "\n"
            
            while cap.isOpened() and processed_count < max_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Skip frames if specified
                if frame_count % skip_frames != 0:
                    frame_count += 1
                    continue
                
                # Run detection
                results = model(frame, conf=confidence, verbose=False)
                
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
                
                # Apply max detections limit
                original_count = len(detections)
                detections = filter_top_detections(detections, max_detections)
                
                # Only return frames with detections
                if len(detections) > 0:
                    # Encode frame as base64 JPEG
                    _, buffer = cv2.imencode('.jpg', frame)
                    frame_base64 = base64.b64encode(buffer).decode('utf-8')
                    
                    # Stream frame result with snapshot
                    yield json.dumps({
                        "type": "frame",
                        "frame_number": frame_count,
                        "timestamp": frame_count / fps if fps > 0 else 0,
                        "detections": detections,
                        "count": len(detections),
                        "original_count": original_count,
                        "truncated": original_count > max_detections,
                        "snapshot": frame_base64
                    }) + "\n"
                    
                    processed_count += 1
                
                frame_count += 1
            
            # Send completion message
            yield json.dumps({
                "type": "complete",
                "total_processed": processed_count,
                "success": True
            }) + "\n"
            
            cap.release()
            if temp_video_path and os.path.exists(temp_video_path):
                os.unlink(temp_video_path)
        
        return StreamingResponse(
            generate_results(),
            media_type="application/x-ndjson"
        )
    
    except Exception as e:
        if temp_video_path and os.path.exists(temp_video_path):
            try:
                os.unlink(temp_video_path)
            except:
                pass
        raise HTTPException(status_code=500, detail=f"Video detection failed: {str(e)}")

# 3. REAL-TIME DETECTION (WebSocket) - WITH SNAPSHOTS AND LIMITS!
@app.websocket("/detect/realtime")
async def detect_realtime(websocket: WebSocket):
    """
    Real-time object detection via WebSocket with annotated snapshots
    
    Client should send base64-encoded images
    Server responds with detection results AND annotated snapshot
    
    Message format (from client):
    {
        "image": "base64_encoded_image_string",
        "confidence": 0.55,  // optional, default 0.55
        "max_detections": 50,  // optional, default 50
        "include_snapshot": true  // optional, default true
    }
    
    Response format (from server):
    {
        "success": true,
        "detections": [...],
        "count": 5,
        "original_count": 8,
        "truncated": true,
        "processing_time": 0.123,
        "snapshot": "base64_encoded_annotated_image"  // if include_snapshot=true
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
                confidence = message.get("confidence", DEFAULT_CONFIDENCE)
                max_detections = message.get("max_detections", DEFAULT_MAX_DETECTIONS)
                include_snapshot = message.get("include_snapshot", True)
                
                # Validate parameters
                confidence = max(0.0, min(1.0, confidence))
                max_detections = max(1, max_detections)
                
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
                results = model(img, conf=confidence, verbose=False)
                
                detections = []
                annotated_img = img.copy()
                
                for result in results:
                    boxes = result.boxes
                    for box in boxes:
                        # Extract detection info
                        x1, y1, x2, y2 = box.xyxy[0]
                        class_id = int(box.cls[0])
                        class_name = model.names[class_id]
                        conf = float(box.conf[0])
                        
                        detection = {
                            "class_id": class_id,
                            "class_name": class_name,
                            "confidence": conf,
                            "bbox": {
                                "x1": float(x1),
                                "y1": float(y1),
                                "x2": float(x2),
                                "y2": float(y2)
                            }
                        }
                        detections.append(detection)
                
                # Apply max detections limit
                original_count = len(detections)
                detections = filter_top_detections(detections, max_detections)
                
                # Draw bounding boxes for filtered detections
                if include_snapshot:
                    for detection in detections:
                        bbox = detection['bbox']
                        x1, y1 = int(bbox['x1']), int(bbox['y1'])
                        x2, y2 = int(bbox['x2']), int(bbox['y2'])
                        
                        # Draw rectangle
                        cv2.rectangle(
                            annotated_img,
                            (x1, y1),
                            (x2, y2),
                            (0, 255, 0),  # Green color
                            2
                        )
                        
                        # Prepare label text
                        label = f"{detection['class_name']} {detection['confidence']:.2f}"
                        
                        # Get text size for background
                        (text_width, text_height), baseline = cv2.getTextSize(
                            label,
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            1
                        )
                        
                        # Draw label background
                        cv2.rectangle(
                            annotated_img,
                            (x1, y1 - text_height - 10),
                            (x1 + text_width, y1),
                            (0, 255, 0),
                            -1
                        )
                        
                        # Draw label text
                        cv2.putText(
                            annotated_img,
                            label,
                            (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (0, 0, 0),
                            1
                        )
                
                processing_time = time.time() - start_time
                
                # Prepare response
                response = {
                    "success": True,
                    "detections": detections,
                    "count": len(detections),
                    "original_count": original_count,
                    "truncated": original_count > max_detections,
                    "processing_time": round(processing_time, 3),
                    "image_size": {
                        "width": img.shape[1],
                        "height": img.shape[0]
                    },
                    "parameters": {
                        "confidence_threshold": confidence,
                        "max_detections": max_detections
                    }
                }
                
                # Add annotated snapshot if requested
                if include_snapshot:
                    _, buffer = cv2.imencode('.jpg', annotated_img)
                    snapshot_base64 = base64.b64encode(buffer).decode('utf-8')
                    response["snapshot"] = snapshot_base64
                
                # Send results back
                await websocket.send_json(response)
                
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

@app.get("/model/classes")
async def get_model_classes():
    """
    Get list of classes the model can detect
    
    Returns:
        JSON with class names and IDs
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "classes": model.names,
        "total_classes": len(model.names),
        "class_list": [{"id": k, "name": v} for k, v in model.names.items()]
    }

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 7860)),
        reload=False
    )