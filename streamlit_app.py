import streamlit as st
import numpy as np
import pandas as pd
import os
import cv2
import torch
import matplotlib.pyplot as plt
import tempfile
import sys
from PIL import Image

# Add import for YOLO model
from ultralytics import YOLO
from additional_func import *

# Import functions from the original script
# In a real application, you would import these from a separate module
# For this example, we'll include the necessary functions directly

def map_to_lead_names(detections):
    """Map YOLO class IDs to ECG lead names"""
    lead_mapping = [
        {'yolo_class_id': 0, 'lead_name': 'I'},
        {'yolo_class_id': 3, 'lead_name': 'aVR'},
        {'yolo_class_id': 6, 'lead_name': 'V1'},
        {'yolo_class_id': 9, 'lead_name': 'V4'},
        {'yolo_class_id': 1, 'lead_name': 'II'},
        {'yolo_class_id': 4, 'lead_name': 'aVL'},
        {'yolo_class_id': 7, 'lead_name': 'V2'},
        {'yolo_class_id': 10, 'lead_name': 'V5'},
        {'yolo_class_id': 2, 'lead_name': 'III'},
        {'yolo_class_id': 5, 'lead_name': 'aVF'},
        {'yolo_class_id': 8, 'lead_name': 'V3'},
        {'yolo_class_id': 11, 'lead_name': 'V6'},
    ]
    
    leads_by_class_id = {item['yolo_class_id']: item['lead_name'] for item in lead_mapping}
    
    mapped_boxes = []
    for box in detections:
        box_copy = box.copy()
        box_copy['lead_name'] = leads_by_class_id.get(box['class_id'], 'Unknown')
        mapped_boxes.append(box_copy)
    
    return mapped_boxes

# Define model classes
class TransformerBlock(torch.nn.Module):
    """Transformer block with multi-head attention and feed-forward network."""
    
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.attention = torch.nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = torch.nn.LayerNorm(embed_dim)
        self.norm2 = torch.nn.LayerNorm(embed_dim)
        self.ff = torch.nn.Sequential(
            torch.nn.Linear(embed_dim, ff_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(ff_dim, embed_dim)
        )
        self.dropout = torch.nn.Dropout(dropout)
        
    def forward(self, x):
        # Self-attention
        attn_output, _ = self.attention(x, x, x)
        x = x + self.dropout(attn_output)
        x = self.norm1(x)
        
        # Feed-forward network
        ff_output = self.ff(x)
        x = x + self.dropout(ff_output)
        x = self.norm2(x)
        
        return x

class TransformerModel(torch.nn.Module):
    """Transformer model for ECG classification."""
    
    def __init__(self, sequence_length=1000, num_leads=12, embed_dim=64, num_heads=8, 
                 ff_dim=128, num_transformer_blocks=4, dropout=0.2):
        super(TransformerModel, self).__init__()
        
        # Initial projection
        self.projection = torch.nn.Sequential(
            torch.nn.Conv1d(num_leads, embed_dim, kernel_size=5, stride=2, padding=2),
            torch.nn.BatchNorm1d(embed_dim),
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(kernel_size=2, stride=2)
        )
        
        # Calculate sequence length after projection
        projected_seq_len = sequence_length // 4  # After Conv1d with stride=2 and MaxPool1d with stride=2
        
        # Transformer blocks
        self.transformer_blocks = torch.nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, ff_dim, dropout)
            for _ in range(num_transformer_blocks)
        ])
        
        # Classification head
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(embed_dim, 64),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(64, 1),
            torch.nn.Sigmoid()
        )
        
    def forward(self, x):
        # x shape: [batch_size, sequence_length, num_leads]
        # Transpose for 1D convolution [batch_size, num_leads, sequence_length]
        x = x.transpose(1, 2)
        
        # Apply initial projection
        x = self.projection(x)
        
        # Transpose back for transformer [batch_size, sequence_length, embed_dim]
        x = x.transpose(1, 2)
        
        # Apply transformer blocks
        for block in self.transformer_blocks:
            x = block(x)
            
        # Global average pooling
        x = torch.mean(x, dim=1)
        
        # Classification
        x = self.classifier(x)
        
        return x

class CNNLSTMModel(torch.nn.Module):
    """CNN-LSTM hybrid model for ECG classification."""
    
    def __init__(self, sequence_length=1000, num_leads=12, dropout=0.3):
        super(CNNLSTMModel, self).__init__()
        
        # CNN feature extraction
        self.cnn_layers = torch.nn.Sequential(
            # First CNN block
            torch.nn.Conv1d(num_leads, 64, kernel_size=5, stride=1, padding=2),
            torch.nn.BatchNorm1d(64),
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(kernel_size=2, stride=2),
            
            # Second CNN block
            torch.nn.Conv1d(64, 128, kernel_size=5, stride=1, padding=2),
            torch.nn.BatchNorm1d(128),
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(kernel_size=2, stride=2),
            
            # Third CNN block
            torch.nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm1d(256),
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(kernel_size=2, stride=2)
        )
        
        # Calculate sequence length after CNN layers
        self.lstm_input_size = 256
        self.lstm_hidden_size = 128
        self.lstm_seq_len = sequence_length // 8  # After 3 MaxPool layers with stride=2
        
        # Bidirectional LSTM layers
        self.lstm_layers = torch.nn.Sequential(
            torch.nn.LSTM(input_size=self.lstm_input_size,
                    hidden_size=self.lstm_hidden_size,
                    num_layers=2,
                    dropout=dropout,
                    bidirectional=True,
                    batch_first=True)
        )
        
        # Classification head
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(self.lstm_hidden_size * 2, 64),  # * 2 for bidirectional
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(64, 1),
            torch.nn.Sigmoid()
        )
        
    def forward(self, x):
        # x shape: [batch_size, sequence_length, num_leads]
        batch_size = x.shape[0]
        
        # Transpose for 1D convolution [batch_size, num_leads, sequence_length]
        x = x.transpose(1, 2)
        
        # Apply CNN layers
        x = self.cnn_layers(x)
        
        # Transpose back for LSTM [batch_size, sequence_length, features]
        x = x.transpose(1, 2)
        
        # Apply LSTM layers
        lstm_out, (h_n, c_n) = self.lstm_layers[0](x)
        
        # Get the final hidden state from both directions
        h_n = h_n.view(2, 2, batch_size, self.lstm_hidden_size)  # [num_layers, num_directions, batch_size, hidden_size]
        final_hidden = torch.cat((h_n[-1, 0, :, :], h_n[-1, 1, :, :]), dim=1)
        
        # Classification
        x = self.classifier(final_hidden)
        
        return x

def detect_ecg_leads(image_data, yolo_model, conf_threshold=0.25):
    """
    Detect ECG leads using YOLOv8 model.
    
    Args:
        image_data: Image data (numpy array)
        yolo_model: Loaded YOLOv8 model
        conf_threshold: Confidence threshold for detections
        
    Returns:
        List of dictionaries with detected lead information
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
    
    Args:
        image: ECG image
        bbox: Bounding box information for the lead
        
    Returns:
        List of (x, y) points representing the signal
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
    
    Args:
        image: ECG image
        detections: List of detected bounding boxes
    
    Returns:
        Dictionary with lead signals
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

def resample_to_ptb_xl_format(lead_signals, target_fs=100, record_id='record'):
    """
    Resample the extracted ECG signals to PTB-XL format (100Hz).
    
    Args:
        lead_signals: Dictionary of lead name to signal points
        target_fs: Target sampling frequency (100Hz for PTB-XL)
        record_id: Record ID for the output file
        
    Returns:
        Resampled signals in PTB-XL format
    """
    # Standard lead order in PTB-XL
    ptb_xl_leads = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    
    # Estimate original sampling rate based on signal length
    # Assuming 10 seconds recording like PTB-XL standard
    signal_lengths = [len(lead_signals.get(lead, [])) for lead in ptb_xl_leads if lead in lead_signals]
    if not signal_lengths:
        raise ValueError("No valid signals found")
        
    # Estimate original sampling rate (points per second)
    orig_fs = max(signal_lengths) / 10  # Assuming 10 second recording
    
    # Prepare resampled signals
    resampled_signals = []
    
    for lead in ptb_xl_leads:
        if lead in lead_signals and lead_signals[lead]:
            # Extract x and y values
            x_values = np.array([p[0] for p in lead_signals[lead]])
            y_values = np.array([p[1] for p in lead_signals[lead]])
            
            # Invert y-values (since image coordinates have origin at top-left)
            y_values = -y_values
            
            # Center around zero by subtracting the median
            y_values = y_values - np.median(y_values)
            
            # Create a time array based on x_values
            orig_times = x_values / orig_fs
            
            # Create a new time array for target sampling rate
            num_samples = int(10 * target_fs)  # 10 seconds at target_fs
            new_times = np.linspace(0, 10, num_samples)
            
            # Resample signal using interpolation
            if len(orig_times) > 1:
                resampled = np.interp(new_times, orig_times, y_values)
            else:
                # Handle empty or single-point signal
                resampled = np.zeros(num_samples)
        else:
            # Create an empty signal if lead not found
            resampled = np.zeros(int(10 * target_fs))
        
        resampled_signals.append(resampled)
    
    # Convert list of arrays to 2D numpy array (samples × leads)
    signals = np.column_stack(resampled_signals)
    
    return signals, target_fs

def normalize_signal(signal):
    """
    Normalize ECG signal lead-wise.
    
    Args:
        signal (numpy.ndarray): ECG signal with shape (sequence_length, 12)
        
    Returns:
        numpy.ndarray: Normalized ECG signal
    """
    normalized = np.zeros_like(signal)
    
    for i in range(signal.shape[1]):  # For each lead
        lead_data = signal[:, i]
        mean = np.mean(lead_data)
        std = np.std(lead_data)
        # Avoid division by zero
        normalized[:, i] = (lead_data - mean) / (std if std > 0 else 1)
        
    return normalized

def load_model(model_path, model_type="transformer", device=None):
    """
    Load a trained model from a checkpoint file.
    
    Args:
        model_path (str): Path to the model checkpoint
        model_type (str): Type of model ("transformer" or "cnn_lstm")
        device (torch.device): Device to load the model on
        
    Returns:
        torch.nn.Module: Loaded model
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if model_type.lower() == "transformer":
        model = TransformerModel(sequence_length=1000)  # 10 seconds at 100Hz
    elif model_type.lower() == "cnn_lstm":
        model = CNNLSTMModel(sequence_length=1000)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    return model

def predict_ecg(model, ecg_signal, device=None):
    """
    Make a prediction on an ECG signal.
    
    Args:
        model (torch.nn.Module): Trained model
        ecg_signal (numpy.ndarray): ECG signal with shape (sequence_length, 12)
        device (torch.device): Device to run inference on
        
    Returns:
        dict: Prediction results including probability and class
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
    
    Args:
        image: ECG image
        detections (list): List of detection dictionaries
        
    Returns:
        numpy.ndarray: Visualization image
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
    
    Args:
        signals (numpy.ndarray): ECG signals with shape (samples, leads)
        fs (int): Sampling frequency
        
    Returns:
        matplotlib.figure.Figure: Figure with plotted signals
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

def process_ecg_image(uploaded_file, yolo_model_path, ml_model_path, model_type='cnn_lstm'):
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
        yolo_model = YOLO(yolo_model_path)
        
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
        ml_model = load_model(ml_model_path, model_type, device)
        
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
        page_title="ECG Digitization and MI Prediction",
        page_icon="❤️",
        layout="wide"
    )
    
    st.title("PulseGaurd: Protecting Hearts, One Scan At a Time")
    st.markdown("""
    This app allows you to upload an ECG image and:
    1. Digitize the ECG signals
    2. Predict whether the ECG shows myocardial infarction (MI)
    """)
    
    # Sidebar
    st.sidebar.title("Configuration")
    
    # Model paths
    yolo_model_path = st.sidebar.text_input(
        "YOLOv8 Model Path",
        value="path/to/yolo/model.pt"
    )
    
    ml_model_path = st.sidebar.text_input(
        "ML Model Path",
        value="path/to/ml/model.pt"
    )
    
    model_type = st.sidebar.selectbox(
        "Model Type",
        options=["transformer", "cnn_lstm"],
        index=1  # Default to CNN-LSTM
    )
    
    # File uploader
    uploaded_file = st.file_uploader("Upload an ECG image", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Display uploaded image
        st.subheader("Uploaded ECG Image")
        original_image = Image.open(uploaded_file)
        st.image(original_image, caption="Uploaded ECG Image", use_column_width=True)
        
        # Process button
        if st.button("Process ECG"):
            with st.spinner("Processing ECG image..."):
                # Process the image
                result = process_ecg_image(
                    uploaded_file,
                    yolo_model_path,
                    ml_model_path,
                    model_type
                )
                
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
                    
                    # Display prediction
                    st.subheader("Prediction Result")
                    if result["prediction"] == "MI":
                        st.error(f"Prediction: **{result['prediction']}** (Myocardial Infarction)")
                    else:
                        st.success(f"Prediction: **{result['prediction']}** (Normal)")
                else:
                    st.error(f"Error: {result['error']}")

if __name__ == "__main__":
    main()