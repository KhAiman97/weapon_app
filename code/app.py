import streamlit as st
import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image
import time

# Page configuration
st.set_page_config(
    page_title="YOLOv11 Weapon Detection",
    page_icon="🔫",
    layout="wide"
)

# Custom CSS for fixed header
st.markdown("""
    <style>
    .fixed-header {
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        background-color: #0E1117;
        z-index: 999;
        padding: 1rem 3rem;
        border-bottom: 2px solid #FF4B4B;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .fixed-header h1 {
        margin: 0;
        color: #FAFAFA;
        font-size: 2rem;
    }
    .main-content {
        margin-top: 5rem;
    }
    </style>
    <div class="fixed-header">
        <h1>🔫 Real-Time Weapon Detection with YOLOv11</h1>
    </div>
    <div class="main-content"></div>
""", unsafe_allow_html=True)
st.markdown("---")

# Sidebar configuration
st.sidebar.header("Configuration")
confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.05)
iou_threshold = st.sidebar.slider("IOU Threshold", 0.0, 1.0, 0.45, 0.05)
model_path = st.sidebar.text_input("Model Path", "models/weapon_model.onnx")

# Class names - adjust these based on your model
CLASS_NAMES = ["gun", "knife"]
    
def load_model(model_path):
    """Load ONNX model"""
    try:
        session = ort.InferenceSession(model_path)
        return session
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def preprocess_image(img, input_shape=(640, 640)):
    """Preprocess image for YOLOv11"""
    # Resize image
    img_resized = cv2.resize(img, input_shape)
    
    # Convert BGR to RGB
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    
    # Normalize to [0, 1] and transpose to (1, 3, H, W)
    img_normalized = img_rgb.astype(np.float32) / 255.0
    img_transposed = np.transpose(img_normalized, (2, 0, 1))
    img_batch = np.expand_dims(img_transposed, axis=0)
    
    return img_batch

def postprocess(outputs, img_shape, input_shape=(640, 640), conf_threshold=0.5, iou_threshold=0.45):
    """Post-process YOLOv11 outputs"""
    # YOLOv11 output shape: (1, 84, 8400) for COCO or (1, num_classes+4, num_predictions)
    output = outputs[0]
    
    # Transpose to (num_predictions, num_classes+4)
    predictions = np.squeeze(output).T
    
    # Extract boxes and scores
    num_classes = predictions.shape[1] - 4
    boxes = predictions[:, :4]
    scores = predictions[:, 4:]
    
    # Get class with highest score for each prediction
    class_ids = np.argmax(scores, axis=1)
    confidences = np.max(scores, axis=1)
    
    # Filter by confidence
    mask = confidences > conf_threshold
    boxes = boxes[mask]
    confidences = confidences[mask]
    class_ids = class_ids[mask]
    
    # Convert boxes from center format to corner format and scale to original image
    h, w = img_shape[:2]
    input_h, input_w = input_shape
    scale_x = w / input_w
    scale_y = h / input_h
    
    boxes_corner = np.zeros_like(boxes)
    boxes_corner[:, 0] = (boxes[:, 0] - boxes[:, 2] / 2) * scale_x  # x1
    boxes_corner[:, 1] = (boxes[:, 1] - boxes[:, 3] / 2) * scale_y  # y1
    boxes_corner[:, 2] = (boxes[:, 0] + boxes[:, 2] / 2) * scale_x  # x2
    boxes_corner[:, 3] = (boxes[:, 1] + boxes[:, 3] / 2) * scale_y  # y2
    
    # Apply NMS
    indices = cv2.dnn.NMSBoxes(
        boxes_corner.tolist(),
        confidences.tolist(),
        conf_threshold,
        iou_threshold
    )
    
    final_boxes = []
    final_confidences = []
    final_class_ids = []
    
    if len(indices) > 0:
        indices = indices.flatten()
        for i in indices:
            final_boxes.append(boxes_corner[i])
            final_confidences.append(confidences[i])
            final_class_ids.append(class_ids[i])
    
    return final_boxes, final_confidences, final_class_ids

def draw_detections(img, boxes, confidences, class_ids, class_names):
    """Draw bounding boxes and labels on image"""
    for box, conf, class_id in zip(boxes, confidences, class_ids):
        x1, y1, x2, y2 = map(int, box)
        
        # Draw rectangle
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        
        # Prepare label
        if class_id < len(class_names):
            label = f"{class_names[class_id]}: {conf:.2f}"
        else:
            label = f"Class {class_id}: {conf:.2f}"
        
        # Draw label background
        (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(img, (x1, y1 - label_h - 10), (x1 + label_w, y1), (0, 0, 255), -1)
        
        # Draw label text
        cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    return img

# Main application
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Camera Feed")
    frame_placeholder = st.empty()
    
with col2:
    st.subheader("Detection Statistics")
    stats_placeholder = st.empty()

# Start/Stop buttons
start_button = st.button("Start Detection")
stop_button = st.button("Stop Detection")

# Initialize session state
if 'run' not in st.session_state:
    st.session_state.run = False

if start_button:
    st.session_state.run = True
    
if stop_button:
    st.session_state.run = False

# Load model
if st.session_state.run:
    model = load_model(model_path)
    
    if model is not None:
        # Open camera
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            st.error("Cannot access camera. Please check your camera connection.")
            st.session_state.run = False
        else:
            fps_list = []
            
            while st.session_state.run:
                start_time = time.time()
                
                ret, frame = cap.read()
                if not ret:
                    st.error("Failed to read from camera")
                    break
                
                # Preprocess
                input_tensor = preprocess_image(frame)
                
                # Inference
                input_name = model.get_inputs()[0].name
                outputs = model.run(None, {input_name: input_tensor})
                
                # Postprocess
                boxes, confidences, class_ids = postprocess(
                    outputs,
                    frame.shape,
                    conf_threshold=confidence_threshold,
                    iou_threshold=iou_threshold
                )
                
                # Draw detections
                frame_with_detections = draw_detections(
                    frame.copy(),
                    boxes,
                    confidences,
                    class_ids,
                    CLASS_NAMES
                )
                
                # Calculate FPS
                end_time = time.time()
                fps = 1 / (end_time - start_time)
                fps_list.append(fps)
                if len(fps_list) > 30:
                    fps_list.pop(0)
                avg_fps = np.mean(fps_list)
                
                # Add FPS to frame
                cv2.putText(frame_with_detections, f"FPS: {avg_fps:.2f}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Convert to RGB for display
                frame_rgb = cv2.cvtColor(frame_with_detections, cv2.COLOR_BGR2RGB)
                
                # Display frame
                frame_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
                
                # Display statistics
                with stats_placeholder.container():
                    st.metric("Detections", len(boxes))
                    st.metric("Average FPS", f"{avg_fps:.2f}")
                    
                    if len(boxes) > 0:
                        st.warning("⚠️ Weapon Detected!")
                        for i, (conf, class_id) in enumerate(zip(confidences, class_ids)):
                            if class_id < len(CLASS_NAMES):
                                st.write(f"**{CLASS_NAMES[class_id]}**: {conf:.2%}")
                    else:
                        st.success("✅ No weapons detected")
            
            cap.release()
    else:
        st.session_state.run = False

st.markdown("---")
st.info("📝 **Note**: Make sure your ONNX model file is in the same directory or provide the correct path in the sidebar.")