import streamlit as st
import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image
import tempfile
import time
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import av
import queue
import threading

# Page configuration
st.set_page_config(
    page_title="YOLOv11 Weapon Detection",
    page_icon="🔫",
    layout="wide"
)

# Title and description
st.title("🔫 Real-Time Weapon Detection with YOLOv11")
st.markdown("---")

# Sidebar configuration
st.sidebar.header("Configuration")
confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.05)
iou_threshold = st.sidebar.slider("IOU Threshold", 0.0, 1.0, 0.45, 0.05)
model_path = st.sidebar.text_input("Model Path", "models/weapon_model.onnx")

# Input mode selection
input_mode = st.sidebar.radio("Input Mode", ["Real-Time Camera", "Upload Image", "Upload Video"])

# Class names - adjust these based on your model
CLASS_NAMES = ["gun", "knife"]

# RTC Configuration for STUN/TURN servers
RTC_CONFIGURATION = RTCConfiguration(
    {
        "iceServers": [
            {"urls": ["stun:stun.l.google.com:19302"]},
            {"urls": ["stun:stun1.l.google.com:19302"]},
            {"urls": ["stun:stun2.l.google.com:19302"]},
        ]
    }
)

@st.cache_resource
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

def process_image(img, model, conf_thresh, iou_thresh):
    """Process a single image"""
    # Preprocess
    input_tensor = preprocess_image(img)
    
    # Inference
    input_name = model.get_inputs()[0].name
    outputs = model.run(None, {input_name: input_tensor})
    
    # Postprocess
    boxes, confidences, class_ids = postprocess(
        outputs,
        img.shape,
        conf_threshold=conf_thresh,
        iou_threshold=iou_thresh
    )
    
    # Draw detections
    img_with_detections = draw_detections(
        img.copy(),
        boxes,
        confidences,
        class_ids,
        CLASS_NAMES
    )
    
    return img_with_detections, boxes, confidences, class_ids

# Callback function for video processing (new API)
def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")
    
    # Get model from session state
    if 'model' in st.session_state and st.session_state.model is not None:
        try:
            # Process frame
            img_with_detections, boxes, confidences, class_ids = process_image(
                img, 
                st.session_state.model, 
                st.session_state.get('conf_threshold', 0.5),
                st.session_state.get('iou_threshold', 0.45)
            )
            
            # Store detection info in session state
            st.session_state.detection_count = len(boxes)
            st.session_state.detections = list(zip(boxes, confidences, class_ids))
            
            return av.VideoFrame.from_ndarray(img_with_detections, format="bgr24")
        except Exception as e:
            print(f"Error processing frame: {e}")
            return frame
    
    return frame

# Main application
if input_mode == "Real-Time Camera":
    st.subheader("Real-Time Camera Detection")
    
    # Configuration options
    use_turn = st.sidebar.checkbox("Use TURN Server (for difficult networks)", value=False)
    
    # Update RTC configuration based on selection
    if use_turn:
        rtc_config = RTCConfiguration(
            {
                "iceServers": [
                    {"urls": ["stun:stun.l.google.com:19302"]},
                    # Free public TURN server (limited capacity)
                    {
                        "urls": ["turn:openrelay.metered.ca:80"],
                        "username": "openrelayproject",
                        "credential": "openrelayproject"
                    },
                    {
                        "urls": ["turn:openrelay.metered.ca:443"],
                        "username": "openrelayproject",
                        "credential": "openrelayproject"
                    },
                ]
            }
        )
        st.info("🔄 Using TURN server for better connectivity through firewalls")
    else:
        rtc_config = RTC_CONFIGURATION
    
    st.info("🎥 Allow camera access when prompted by your browser. This works in both local and cloud deployments!")
    
    with st.expander("📡 Troubleshooting Connection Issues"):
        st.markdown("""
        **If the camera connection fails:**
        1. ✅ Make sure you allowed camera access in your browser
        2. ✅ Try enabling "Use TURN Server" in the sidebar
        3. ✅ Check if your network/firewall blocks WebRTC
        4. ✅ Try a different browser (Chrome/Edge usually work best)
        5. ✅ If on corporate network, it might block WebRTC ports
        6. ✅ Try refreshing the page and allowing camera again
        
        **Browser Compatibility:**
        - ✅ Chrome/Edge: Excellent support
        - ✅ Firefox: Good support
        - ⚠️ Safari: Limited support
        - ❌ Mobile browsers: May have issues
        """)
    
    # Load model and store in session state
    if 'model' not in st.session_state:
        st.session_state.model = load_model(model_path)
        st.session_state.conf_threshold = confidence_threshold
        st.session_state.iou_threshold = iou_threshold
        st.session_state.detection_count = 0
        st.session_state.detections = []
    
    # Update thresholds in session state
    st.session_state.conf_threshold = confidence_threshold
    st.session_state.iou_threshold = iou_threshold
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # WebRTC streamer with new API
        ctx = webrtc_streamer(
            key="weapon-detection",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=rtc_config,
            media_stream_constraints={
                "video": {
                    "width": {"ideal": 1280},
                    "height": {"ideal": 720},
                },
                "audio": False
            },
            video_frame_callback=video_frame_callback,
            async_processing=True,
        )
    
    with col2:
        st.subheader("Detection Statistics")
        stats_placeholder = st.empty()
        
        # Display live stats
        with stats_placeholder.container():
            detection_count = st.session_state.get('detection_count', 0)
            detections = st.session_state.get('detections', [])
            
            st.metric("Detections", detection_count)
            
            if detection_count > 0:
                st.warning("⚠️ Weapon Detected!")
                for box, conf, class_id in detections:
                    if class_id < len(CLASS_NAMES):
                        st.write(f"**{CLASS_NAMES[class_id]}**: {conf:.2%}")
            else:
                st.success("✅ No weapons detected")

elif input_mode == "Upload Image":
    st.subheader("Upload Image for Detection")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Load model
        model = load_model(model_path)
        
        if model is not None:
            # Read image
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            
            # Process image
            with st.spinner("Processing..."):
                img_with_detections, boxes, confidences, class_ids = process_image(
                    img, model, confidence_threshold, iou_threshold
                )
            
            # Display results
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.subheader("Detection Result")
                img_rgb = cv2.cvtColor(img_with_detections, cv2.COLOR_BGR2RGB)
                st.image(img_rgb, use_container_width=True)
            
            with col2:
                st.subheader("Detection Statistics")
                st.metric("Detections", len(boxes))
                
                if len(boxes) > 0:
                    st.warning("⚠️ Weapon Detected!")
                    for i, (conf, class_id) in enumerate(zip(confidences, class_ids)):
                        if class_id < len(CLASS_NAMES):
                            st.write(f"**{CLASS_NAMES[class_id]}**: {conf:.2%}")
                else:
                    st.success("✅ No weapons detected")

else:  # Upload Video
    st.subheader("Upload Video for Detection")
    uploaded_file = st.file_uploader("Choose a video...", type=["mp4", "avi", "mov"])
    
    if uploaded_file is not None:
        # Load model
        model = load_model(model_path)
        
        if model is not None:
            # Save uploaded video to temp file
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            tfile.write(uploaded_file.read())
            
            # Get video properties
            cap = cv2.VideoCapture(tfile.name)
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            
            # Process every nth frame for faster processing
            frame_skip = st.sidebar.slider("Process every N frames", 1, 10, 1)
            
            # Start processing button
            process_button = st.button("🎬 Start Processing Video")
            
            if process_button:
                # Create output video file
                output_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(output_path, fourcc, fps // frame_skip, (width, height))
                
                # Reopen video for processing
                cap = cv2.VideoCapture(tfile.name)
                
                # Create placeholders
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    frame_placeholder = st.empty()
                
                with col2:
                    stats_placeholder = st.empty()
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                frame_count = 0
                processed_count = 0
                detection_summary = []
                
                # Process video
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    frame_count += 1
                    
                    # Skip frames for faster processing
                    if frame_count % frame_skip != 0:
                        continue
                    
                    processed_count += 1
                    
                    # Process frame
                    frame_with_detections, boxes, confidences, class_ids = process_image(
                        frame, model, confidence_threshold, iou_threshold
                    )
                    
                    # Write to output video
                    out.write(frame_with_detections)
                    
                    # Store detection info
                    if len(boxes) > 0:
                        detection_summary.append({
                            'frame': frame_count,
                            'time': frame_count / fps,
                            'detections': len(boxes),
                            'classes': [CLASS_NAMES[cid] if cid < len(CLASS_NAMES) else f"Class {cid}" 
                                       for cid in class_ids]
                        })
                    
                    # Display current frame every 10 processed frames
                    if processed_count % 10 == 0:
                        frame_rgb = cv2.cvtColor(frame_with_detections, cv2.COLOR_BGR2RGB)
                        frame_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
                        
                        # Display statistics
                        with stats_placeholder.container():
                            st.metric("Current Frame", f"{frame_count}/{total_frames}")
                            st.metric("Detections in Frame", len(boxes))
                            st.metric("Total Detections", len(detection_summary))
                            
                            if len(boxes) > 0:
                                st.warning("⚠️ Weapon Detected!")
                                for i, (conf, class_id) in enumerate(zip(confidences, class_ids)):
                                    if class_id < len(CLASS_NAMES):
                                        st.write(f"**{CLASS_NAMES[class_id]}**: {conf:.2%}")
                    
                    # Update progress
                    progress = frame_count / total_frames
                    progress_bar.progress(progress)
                    status_text.text(f"Processing: {frame_count}/{total_frames} frames ({progress*100:.1f}%)")
                
                # Release resources
                cap.release()
                out.release()
                
                # Show completion
                progress_bar.progress(1.0)
                status_text.text("✅ Processing complete!")
                st.success("🎉 Video processing complete!")
                
                # Display final processed video
                st.subheader("📹 Processed Video")
                
                # Read the processed video file
                with open(output_path, 'rb') as video_file:
                    video_bytes = video_file.read()
                    st.video(video_bytes)
                
                # Download button for processed video
                st.download_button(
                    label="⬇️ Download Processed Video",
                    data=video_bytes,
                    file_name="weapon_detection_output.mp4",
                    mime="video/mp4"
                )
                
                # Display detection summary
                if len(detection_summary) > 0:
                    st.subheader("📊 Detection Summary")
                    st.warning(f"⚠️ Weapons detected in {len(detection_summary)} frames")
                    
                    # Create expandable section for details
                    with st.expander("View Detailed Detection Log"):
                        for det in detection_summary:
                            time_str = time.strftime('%M:%S', time.gmtime(det['time']))
                            st.write(f"**Frame {det['frame']}** (at {time_str}): {det['detections']} detection(s) - {', '.join(det['classes'])}")
                else:
                    st.success("✅ No weapons detected in the entire video")

st.markdown("---")
st.info("""
📝 **Usage Notes**:
- **Real-Time Camera**: Works on both local and cloud deployments using WebRTC
- **Upload Image/Video**: Process pre-recorded media
- Make sure your ONNX model file is in the correct path
- Adjust confidence and IOU thresholds in the sidebar for better results
""")