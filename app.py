import cv2
import numpy as np
from gtts import gTTS
import os
import time
import tensorflow as tf
import torch
import pygame
import streamlit as st
import tempfile
import uuid
from PIL import Image
import io

# Initialize pygame for audio playback
pygame.init()

# App state management
if 'running' not in st.session_state:
    st.session_state.running = False
if 'last_speak_time' not in st.session_state:
    st.session_state.last_speak_time = time.time()
if 'models_loaded' not in st.session_state:
    st.session_state.models_loaded = False
if 'midas' not in st.session_state:
    st.session_state.midas = None
if 'transform' not in st.session_state:
    st.session_state.transform = None
if 'device' not in st.session_state:
    st.session_state.device = None
if 'detect_fn' not in st.session_state:
    st.session_state.detect_fn = None
if 'classes' not in st.session_state:
    st.session_state.classes = []

# Streamlit page configuration
st.set_page_config(page_title="Object Detection with Depth Estimation", layout="wide")
st.title("Object Detection with Depth Estimation")

# Fixed parameters - no user controls
depth_calibration = 10.0
confidence_threshold = 0.5
speech_interval = 5
show_depth_map = False
model_path_input = "D:\\VSCODE\\Mini project\\efficientdet-tensorflow2-d0-v1"
coco_names_path = "coco.names"

# Create control buttons
col_control1, col_control2 = st.columns(2)
with col_control1:
    start_button = st.button("Start Detection")
with col_control2:
    stop_button = st.button("Stop Detection")

# Status indicator
status_text = st.empty()

# Set up the layout for video display
main_view = st.empty()

def load_models():
    """Load all required models"""
    status_text.text("Loading models, please wait...")
    
    # Load MiDaS model
    try:
        midas_model_type = "MiDaS_small"
        midas = torch.hub.load("intel-isl/MiDaS", midas_model_type)
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        midas.to(device)
        midas.eval()
        
        # Load appropriate transforms for the model
        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        transform = midas_transforms.small_transform
        
        st.success(f"MiDaS model loaded successfully on {device}")
    except Exception as e:
        st.error(f"Error loading MiDaS model: {e}")
        return None, None, None, None, None
    
    # Load EfficientDet model
    try:
        detect_fn = tf.saved_model.load(model_path_input)
        st.success("EfficientDet model loaded successfully")
    except Exception as e:
        st.error(f"Error loading EfficientDet model: {e}")
        return midas, transform, device, None, None
    
    # Load COCO class labels
    try:
        with open(coco_names_path, "r") as f:
            classes = [line.strip() for line in f.readlines()]
        st.success(f"Loaded {len(classes)} class labels")
    except Exception as e:
        st.error(f"Error loading class labels: {e}")
        classes = []
    
    status_text.text("Models loaded successfully!")
    return midas, transform, device, detect_fn, classes

def process_depth(frame, midas, transform, device):
    """Process frame with MiDaS to get depth estimation"""
    # Transform input for MiDaS
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    input_batch = transform(img).to(device)
    
    # Make prediction
    with torch.no_grad():
        prediction = midas(input_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=frame.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()
    
    # Convert to numpy and normalize
    depth_map = prediction.cpu().numpy()
    depth_map = cv2.normalize(depth_map, None, 0, 1, cv2.NORM_MINMAX)
    
    return depth_map

def speak(text):
    """Generate and play speech using pygame with robust error handling"""
    try:
        # Initialize pygame mixer if not already initialized
        if not pygame.mixer.get_init():
            pygame.mixer.init()
        
        # Create a temporary directory with a unique name to avoid conflicts
        temp_dir = os.path.join(tempfile.gettempdir(), f"speech_app_{uuid.uuid4().hex[:8]}")
        os.makedirs(temp_dir, exist_ok=True)
        
        # Use a unique filename each time to avoid file locks
        output_file = os.path.join(temp_dir, f"speech_{uuid.uuid4().hex[:8]}.mp3")
        
        # Generate speech
        tts = gTTS(text=text, lang='en')
        tts.save(output_file)
        
        # Play the audio file
        pygame.mixer.music.load(output_file)
        pygame.mixer.music.play()
        
        # Print as fallback
        st.info(f"SPEAKING: {text}")
        
        # Optional: Clean up old files
        try:
            for file in os.listdir(temp_dir):
                if file != os.path.basename(output_file):  # Don't delete the file we're using
                    try:
                        os.remove(os.path.join(temp_dir, file))
                    except:
                        pass
        except:
            pass
            
    except Exception as e:
        st.error(f"Error generating speech: {e}")
        st.info(f"SPEECH MESSAGE: {text}")

def run_detection():
    """Main function to run the detection loop"""
    # Use the pre-loaded models from session state
    midas = st.session_state.midas
    transform = st.session_state.transform
    device = st.session_state.device
    detect_fn = st.session_state.detect_fn
    classes = st.session_state.classes
    
    # Initialize video capture
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        status_text.error("Error: Could not open camera")
        st.session_state.running = False
        return
    
    status_text.text("Detection is running... Press 'Stop Detection' to end.")
    
    # Main detection loop - runs in the main thread
    while st.session_state.running:
        ret, frame = cap.read()
        if not ret:
            status_text.error("Error: Failed to grab frame")
            break
        
        # Get depth estimation using MiDaS
        try:
            depth_map = process_depth(frame, midas, transform, device)
            depth_map_display = (depth_map * 255).astype(np.uint8)  # Convert for display
            depth_colored = cv2.applyColorMap(depth_map_display, cv2.COLORMAP_JET)
        except Exception as e:
            status_text.error(f"Error processing depth: {e}")
            continue
        
        # Object detection with EfficientDet
        try:
            input_tensor = tf.convert_to_tensor(np.expand_dims(frame, 0), dtype=tf.uint8)
            detections = detect_fn(input_tensor)
            
            boxes = detections['detection_boxes'][0].numpy()
            scores = detections['detection_scores'][0].numpy()
            classes_detected = detections['detection_classes'][0].numpy().astype(np.int32)
            
            # Display frame with detected objects
            frame_with_objects = frame.copy()
            h, w, _ = frame.shape
            
            # Filter detections based on confidence threshold
            threshold = confidence_threshold
            valid_indices = scores > threshold
            
            detected_objects = []  # To collect objects for speech
            
            for i in range(len(boxes)):
                if valid_indices[i]:
                    # Get coordinates
                    y_min, x_min, y_max, x_max = boxes[i]
                    x_min, x_max = int(x_min * w), int(x_max * w)
                    y_min, y_max = int(y_min * h), int(y_max * h)
                    
                    # Ensure coordinates are within frame bounds
                    x_min, y_min = max(0, x_min), max(0, y_min)
                    x_max, y_max = min(w, x_max), min(h, y_max)
                    
                    # Get average depth in detection region
                    if x_min < x_max and y_min < y_max:
                        object_region = depth_map[y_min:y_max, x_min:x_max]
                        if object_region.size > 0:
                            # Apply the calibration factor
                            avg_depth = np.mean(object_region) * depth_calibration
                            
                            # Get class name
                            class_id = classes_detected[i] - 1  # EfficientDet uses 1-based indexing
                            if 0 <= class_id < len(classes):
                                label = classes[class_id]
                            else:
                                label = f"Unknown ({class_id})"
                            
                            # Draw bounding box with BLACK color
                            cv2.rectangle(frame_with_objects, (x_min, y_min), (x_max, y_max), (0, 0, 0), 2)
                            cv2.putText(frame_with_objects, f"{label}: {avg_depth:.2f}m", 
                                      (x_min, y_min-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                            
                            # Add to detected objects
                            detected_objects.append((label, avg_depth))
                            
            # Voice announcement (only for the closest object, every X seconds)
            current_time = time.time()
            if detected_objects and current_time - st.session_state.last_speak_time > speech_interval:
                # Sort by distance and announce closest object
                closest_object = min(detected_objects, key=lambda x: x[1])
                speak(f"{closest_object[0]} detected at {closest_object[1]:.2f} meters")
                st.session_state.last_speak_time = current_time
                
        except Exception as e:
            status_text.error(f"Error in object detection: {e}")
        
        # Display results in Streamlit
        frame_with_objects_rgb = cv2.cvtColor(frame_with_objects, cv2.COLOR_BGR2RGB)
        main_view.image(frame_with_objects_rgb, channels="RGB", use_column_width=True)
        
        # Short sleep to reduce CPU usage and allow Streamlit to update
        time.sleep(0.05)
    
    # Cleanup
    cap.release()
    status_text.text("Detection stopped.")

# Handle button clicks
if start_button:
    if not st.session_state.models_loaded:
        # Load models in the main thread
        midas, transform, device, detect_fn, classes = load_models()
        if midas is not None and detect_fn is not None:
            st.session_state.midas = midas
            st.session_state.transform = transform
            st.session_state.device = device
            st.session_state.detect_fn = detect_fn
            st.session_state.classes = classes
            st.session_state.models_loaded = True
        else:
            st.error("Failed to load models. Please check paths and try again.")
    
    # Start detection
    st.session_state.running = True
    # Manually trigger the run_detection function instead of using experimental_rerun
    if st.session_state.models_loaded:
        run_detection()

if stop_button:
    st.session_state.running = False
    status_text.text("Stopping detection...")

# Run detection if it's active (this is needed when page refreshes)
if st.session_state.running and st.session_state.models_loaded:
    run_detection()

# Footer
st.markdown("---")
st.markdown("© 2025 Object Detection with Depth Estimation")
