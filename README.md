# PulseGuard: ECG Analysis System Deployment Guide

This guide will help you deploy the PulseGuard ECG analysis system, which analyzes ECG images to detect myocardial infarction (MI).

## System Requirements

- Python 3.8 or higher
- 4GB RAM minimum (8GB recommended)
- GPU support optional but recommended for faster inference

## Directory Structure

Set up your project with the following structure:

```
pulse-guard/
├── additional_func.py     # Core functions (from your original code)
├── streamlit_app.py       # Streamlit application
├── requirements.txt       # Package dependencies
├── models/
│   ├── yolo_ecg_model.pt  # Your trained YOLO model
│   └── cnn_lstm_model.pt  # Your trained CNN-LSTM model
└── README.md              # Project documentation
```

## Installation Steps

1. **Create a virtual environment:**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

3. **Place your model files:**

   Place your trained YOLO and CNN-LSTM model files in the `models/` directory:
   - YOLO model at `models/yolo_ecg_model.pt`
   - CNN-LSTM model at `models/cnn_lstm_model.pt`

4. **Update model paths (if needed):**

   If your model files are in different locations, update the `YOLO_MODEL_PATH` and `ML_MODEL_PATH` variables in `streamlit_app.py`.

## Running the Application Locally

Start the Streamlit application:

```bash
streamlit run streamlit_app.py
```

The application will be available at `http://localhost:8501` by default.

## Deployment Options

### Option 1: Streamlit Community Cloud

1. Create a GitHub repository with your project files
2. Sign up for [Streamlit Community Cloud](https://streamlit.io/cloud)
3. Deploy directly from your GitHub repository

### Option 2: Docker Deployment

1. Create a Dockerfile:

```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

2. Build and run the Docker container:

```bash
docker build -t pulse-guard .
docker run -p 8501:8501 pulse-guard
```

### Option 3: Platform as a Service (PaaS)

You can deploy to platforms like Heroku, Google Cloud Run, or AWS Elastic Beanstalk following their respective deployment guides.

## Production Considerations

1. **Model Size:** The YOLOv8 and CNN-LSTM models can be large. Consider model optimization techniques like quantization for production.

2. **Memory Management:** Implement proper cleanup of temporary files and resources.

3. **Authentication:** Add user authentication for production deployment if dealing with sensitive data.

4. **Monitoring:** Implement logging and monitoring to track usage and detect errors.

## Customization

To customize the application:

1. **Logo:** Add your own logo by uncommenting the logo section in `streamlit_app.py` and providing a logo file.

2. **UI Colors:** Customize the Streamlit theme by creating a `.streamlit/config.toml` file.

3. **Additional Features:** Extend the application with features like report generation, patient database integration, or multi-model ensemble predictions.

## Support

For any issues or questions regarding deployment, please refer to the following resources:

- [Streamlit Documentation](https://docs.streamlit.io/)
- [PyTorch Deployment Guide](https://pytorch.org/tutorials/recipes/deployment_with_flask.html)
- [YOLOv8 Documentation](https://docs.ultralytics.com/)
