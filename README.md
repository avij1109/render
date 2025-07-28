# PPG Server for Render

This is a FastAPI WebSocket server for real-time PPG (photoplethysmography) signal processing.

## Features
- Real-time heart rate calculation using FFT analysis
- WebSocket communication for live data streaming
- Green signal extraction and filtering
- Support for mobile camera frame processing

## Deployment on Render

1. Connect your GitHub repository to Render
2. Deploy as a Web Service
3. Use these settings:
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `uvicorn ppg_server:app --host 0.0.0.0 --port $PORT`
   - Environment: Python 3.11

## API Endpoints

- `/` - Root endpoint with server status
- `/health` - Health check endpoint
- `/ws` - WebSocket endpoint for real-time PPG processing

## Usage

Send camera frames as Base64-encoded images to the WebSocket endpoint to receive real-time PPG analysis results.
