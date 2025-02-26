import streamlit as st
import numpy as np
import pandas as pd
import os
import cv2
import torch
import matplotlib.pyplot as plt
import tempfile
from PIL import Image
from ultralytics import YOLO
from additional_func import (map_to_lead_names, CNNLSTMModel, normalize_signal,
                            resample_to_ptb_xl_format)

# Set fixed paths for models
YOLO_MODEL_PATH = "models/yolo_ecg_model.pt"  # Update with your actual fixed path
ML_MODEL_PATH = "models/cnn_lstm_model.pt"    # Update with your actual fixed path

def detect_ecg_leads(image_data, yolo_model, conf_threshold=0.25):
    """
    Detect ECG leads using YOLOv8 model.
    """
    # Run detection
    results = yolo_model(image_data, conf=conf_threshold)
    
    # Process results
    detections = []
    
    for result in results:
        boxes = result.boxes
        
        for i in range(len(boxes)):
            box = boxes[i]
            x1, y1, x2, y2 = box.xyxyn[0].cpu().numpy()  # Normalized coordinates
            conf = box.conf[0].item()
            class_id = int(box.cls[0].item())
            
            # Calculate center and dimensions (normalized)
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            width = x2 - x1
            height = y2 - y1
            
            detections.append({
                'class_id': class_id,
                'confidence': conf,
                'center_x': center_x,
                'center_y': center_y,
                'width': width,
                'height': height,
                'x1': x1,
                'y1': y1,
                'x2': x2,
                'y2': y2
            })
    
    return detections

def process_lead(image, bbox):
    """
    Process a lead from the ECG image and extract its signal.
    """
    import cv2
    # Extract ROI
    height, width = image.shape[:2]
    x1 = max(0, int(bbox['x1'] * width))
    y1 = max(0, int(bbox['y1'] * height))
    x2 = min(width, int(bbox['x2'] * width))
    y2 = min(height, int(bbox['y2'] * height))
    roi = image[y1:y2, x1:x2]
    
    # Extract color channels
    b, g, r = cv2.split(roi)
    
    # Create grid mask
    grid_mask = np.zeros_like(r)
    grid_mask[(r > 150) & (b < 100) & (g < 100)] = 255
    
    # Convert to grayscale
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    
    # Remove grid
    gray_no_grid = gray.copy()
    gray_no_grid[grid_mask > 0] = 255
    
    # Apply threshold
    _, binary = cv2.threshold(gray_no_grid, 100, 255, cv2.THRESH_BINARY_INV)
    
    # Connected component analysis
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    
    # Sort components by area
    sorted_components = sorted([(i, stats[i, cv2.CC_STAT_AREA]) for i in range(1, num_labels)],
                              key=lambda x: x[1], reverse=True)
    
    # Create clean mask, keep only the longest component in terms of width
    clean_mask = np.zeros_like(binary)
    max_trace_area = roi.shape[0] * roi.shape[1] * 0.05
    min_trace_area = 10
    
    max_width = -1
    max_label = None
    
    for label_id, area in sorted_components:
        if area > max_trace_area or area < min_trace_area:
            continue
        
        # Extract the coordinates of the current component
        component_mask = labels == label_id
        
        # Find the bounding box of the component
        rows, cols = np.where(component_mask)
        if len(cols) == 0:
            continue
            
        min_col = np.min(cols)
        max_col = np.max(cols)
        
        # Calculate the width of the component (difference in x-axis)
        width = max_col - min_col + 1
        
        # Check if this component is the longest
        if width > max_width:
            max_width = width
            max_label = label_id
    
    # Keep only the longest component (in terms of width)
    if max_label is not None:
        clean_mask[labels == max_label] = 255
    else:
        # If no component is kept, keep all components
        for label_id, area in sorted_components[:1]:  # Keep only the largest by area
            clean_mask[labels == label_id] = 255
    
    # Skeletonize
    try:
        import cv2.ximgproc
        skeleton = cv2.ximgproc.thinning(clean_mask)
    except ImportError:
        # Fallback if ximgproc is not available
        kernel = np.ones((3,3), np.uint8)
        skeleton = clean_mask.copy()
        while True:
            eroded = cv2.erode(skeleton, kernel)
            temp = cv2.dilate(eroded, kernel)
            temp = cv2.subtract(skeleton, temp)
            skeleton = eroded
            if cv2.countNonZero(temp) == 0:
                break
    
    # Extract trace points
    trace_points = []
    for x in range(skeleton.shape[1]):
        y_values = np.where(skeleton[:, x] > 0)[0]
        if len(y_values) > 0:
            avg_y = sum(y_values) / len(y_values)
            trace_points.append((x, avg_y))
    
    # Sort points by x-coordinate
    trace_points.sort(key=lambda p: p[0])
    
    return trace_points

def extract_ecg_signals(image, detections):
    """
    Extract ECG signals from an image using YOLOv8 detections.
    """
    # Map class IDs to lead names
    mapped_boxes = map_to_lead_names(detections)
    
    # Process each lead to extract signals
    lead_signals = {}
    
    for box in mapped_boxes:
        if box['lead_name'] == 'Unknown' or box['width'] > 0.5:
            continue  # Skip unknown leads or the rhythm strip
        
        # Extract signal points
        signal_points = process_lead(image, box)
        
        # Store the signal
        lead_signals[box['lead_name']] = signal_points
    
    return lead_signals

def load_model(model_path, device=None):
    """
    Load the CNN-LSTM model.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = CNNLSTMModel(sequence_length=1000)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    return model

def predict_ecg(model, ecg_signal, device=None):
    """
    Make a prediction on an ECG signal.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Normalize the signal
    normalized_signal = normalize_signal(ecg_signal)
    
    # Prepare input for model (add batch dimension)
    X = np.expand_dims(normalized_signal, axis=0)
    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
    
    # Make prediction
    with torch.no_grad():
        output = model(X_tensor)
        probability = output.item()
        prediction = 1 if probability > 0.5 else 0
    
    return {
        "probability": probability,
        "prediction": prediction,
        "label": "MI" if prediction == 1 else "NORM"
    }

def visualize_detections(image, detections):
    """
    Visualize the detected ECG leads on the image.
    """
    height, width = image.shape[:2]
    
    # Map class IDs to lead names
    mapped_boxes = map_to_lead_names(detections)
    
    # Create a copy for visualization
    vis_image = image.copy()
    
    # Draw bounding boxes
    for box in mapped_boxes:
        # Convert normalized coordinates to pixel coordinates
        x1 = int(box['x1'] * width)
        y1 = int(box['y1'] * height)
        x2 = int(box['x2'] * width)
        y2 = int(box['y2'] * height)
        
        # Draw rectangle (green color)
        cv2.rectangle(vis_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Add lead name and confidence
        text = f"{box['lead_name']} ({box['confidence']:.2f})"
        cv2.putText(vis_image, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
    return vis_image

def plot_ecg_signals(signals, fs=100):
    """
    Plot ECG signals.
    """
    lead_names = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    num_leads = signals.shape[1]
    
    # Create time axis
    time = np.arange(signals.shape[0]) / fs
    
    # Create plot grid 3x4 for 12 leads
    fig, axes = plt.subplots(3, 4, figsize=(12, 8))
    axes = axes.flatten()
    
    # Plot each lead
    for i in range(min(num_leads, 12)):
        ax = axes[i]
        ax.plot(time, signals[:, i], 'b-')
        ax.set_title(f'Lead {lead_names[i]}')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Amplitude')
        ax.grid(True)
    
    # Hide empty subplots if less than 12 leads
    for i in range(num_leads, 12):
        axes[i].axis('off')
    
    plt.tight_layout()
    return fig

def process_ecg_image(uploaded_file):
    """Process the uploaded ECG image and make a prediction"""
    
    # Create a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
        temp_file.write(uploaded_file.getvalue())
        temp_file_path = temp_file.name
    
    try:
        # Load the image for processing
        image = cv2.imread(temp_file_path)
        if image is None:
            return {"success": False, "error": "Failed to load image"}
        
        # Set device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load YOLO model
        yolo_model = YOLO(YOLO_MODEL_PATH)
        
        # Step 1: Detect ECG leads
        detections = detect_ecg_leads(image, yolo_model)
        if not detections:
            return {"success": False, "error": "No ECG leads detected"}
        
        # Step 2: Extract signals
        lead_signals = extract_ecg_signals(image, detections)
        detected_leads = list(lead_signals.keys())
        if not detected_leads:
            return {"success": False, "error": "Failed to extract ECG signals"}
        
        # Step 3: Resample to PTB-XL format
        signals, fs = resample_to_ptb_xl_format(lead_signals)
        
        # Step 4: Load ML model
        ml_model = load_model(ML_MODEL_PATH, device)
        
        # Step 5: Make prediction
        result = predict_ecg(ml_model, signals, device)
        
        # Create visualization image with bounding boxes
        vis_image = visualize_detections(image, detections)
        
        # Return results
        return {
            "success": True,
            "image": image,
            "vis_image": vis_image,
            "signals": signals,
            "fs": fs,
            "prediction": result["label"],
            "probability": result["probability"],
            "detected_leads": detected_leads
        }
    
    except Exception as e:
        return {"success": False, "error": str(e)}
    
    finally:
        # Clean up the temporary file
        if os.path.exists(temp_file_path):
            os.unlink(temp_file_path)

# Main Streamlit app
def main():
    st.set_page_config(
        page_title="PulseGuard: ECG MI Detection",
        page_icon="❤️",
        layout="wide"
    )
    
    # Header with logo and title
    col1, col2 = st.columns([1, 5])
    
    # You can add a logo here if you have one
    # with col1:
    #     st.image("logo.png", width=100)
    
    with col2:
        st.title("PulseGuard: Protecting Hearts, One Scan At a Time")
    
    st.markdown("""
    This application analyzes ECG images to detect myocardial infarction (MI).
    Upload an ECG image to get started.
    """)
    
    # File uploader
    uploaded_file = st.file_uploader("Upload an ECG image", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Display uploaded image
        st.subheader("Uploaded ECG Image")
        original_image = Image.open(uploaded_file)
        st.image(original_image, caption="Uploaded ECG Image", use_column_width=True)
        
        # Process button
        if st.button("Analyze ECG"):
            with st.spinner("Processing ECG image..."):
                # Process the image
                result = process_ecg_image(uploaded_file)
                
                if result["success"]:
                    # Create two columns
                    col1, col2 = st.columns(2)
                    
                    # Display detection visualization
                    with col1:
                        st.subheader("Lead Detection")
                        st.image(
                            cv2.cvtColor(result["vis_image"], cv2.COLOR_BGR2RGB),
                            caption="Detected ECG Leads",
                            use_column_width=True
                        )
                        st.write(f"Detected leads: {', '.join(result['detected_leads'])}")
                    
                    # Display the reconstructed ECG signals
                    with col2:
                        st.subheader("Reconstructed ECG Signals")
                        fig = plot_ecg_signals(result["signals"], result["fs"])
                        st.pyplot(fig)
                    
                    # Display prediction with more detailed information
                    st.subheader("Analysis Result")
                    
                    # Use columns for prediction display
                    pred_col1, pred_col2 = st.columns([1, 2])
                    
                    with pred_col1:
                        # Display a gauge or probability meter
                        if result["prediction"] == "MI":
                            st.error(f"Diagnosis: **Myocardial Infarction (MI)**")
                            st.metric("MI Probability", f"{result['probability']*100:.1f}%")
                        else:
                            st.success(f"Diagnosis: **Normal ECG**")
                            st.metric("Normal Probability", f"{(1-result['probability'])*100:.1f}%")
                    
                    with pred_col2:
                        # Display additional information or recommendations
                        if result["prediction"] == "MI":
                            st.markdown("""
                            ### Findings
                            The analysis indicates signs of myocardial infarction. This suggests possible heart muscle damage 
                            due to reduced blood flow.
                            
                            ### Recommendations
                            - Immediate medical attention is advised
                            - This is an automated analysis and should be confirmed by a medical professional
                            """)
                        else:
                            st.markdown("""
                            ### Findings
                            No significant signs of myocardial infarction detected in the ECG.
                            
                            ### Note
                            - This is an automated analysis and should be confirmed by a medical professional
                            - Regular cardiac check-ups are still recommended
                            """)
                else:
                    st.error(f"Error: {result['error']}")
                    st.markdown("""
                    ### Troubleshooting Tips:
                    - Ensure the uploaded image is a clear ECG scan
                    - Check that the image contains standard ECG leads (I, II, III, aVR, aVL, aVF, V1-V6)
                    - Try uploading a different ECG image
                    """)
    
    # Add footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center">
        <p>PulseGuard ECG Analysis System | Not for clinical use | © 2025</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
