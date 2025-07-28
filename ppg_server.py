from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import base64
import time
import json
import numpy as np
import cv2
from scipy import signal
from scipy.signal import find_peaks
from scipy.fft import fft, fftfreq
import asyncio
from typing import List, Dict, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="PPG Signal Processing Server", version="1.0.0")

# Add CORS middleware for mobile app connections
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for WebSocket connections
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PPGProcessor:
    def __init__(self):
        self.green_signal = []
        self.red_signal = []
        self.blue_signal = []
        self.timestamps = []
        self.frame_count = 0
        self.start_time = None
        
        # Signal processing parameters (HealthWatcher inspired)
        self.sampling_rate = 30  # Assuming 30 FPS
        self.window_size = 150   # 5 seconds at 30 FPS
        self.hr_freq_range = (0.7, 4.0)  # 42-240 BPM
        self.resp_freq_range = (0.1, 0.5)  # 6-30 breaths per minute
        
    def reset(self):
        """Reset all signals for new measurement"""
        self.green_signal = []
        self.red_signal = []
        self.blue_signal = []
        self.timestamps = []
        self.frame_count = 0
        self.start_time = None
        logger.info("PPG processor reset")
    
    def extract_rgb_from_frame(self, frame_data: bytes) -> Dict[str, float]:
        """Extract average RGB values from frame data"""
        try:
            # Decode base64 image
            nparr = np.frombuffer(frame_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if img is None:
                return {"red": 0, "green": 0, "blue": 0, "error": "Failed to decode image"}
            
            # Convert BGR to RGB (OpenCV uses BGR)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Calculate average RGB values for the entire frame
            # In a real implementation, you might want to focus on a specific ROI
            avg_red = np.mean(img_rgb[:, :, 0])
            avg_green = np.mean(img_rgb[:, :, 1])
            avg_blue = np.mean(img_rgb[:, :, 2])
            
            return {
                "red": float(avg_red),
                "green": float(avg_green), 
                "blue": float(avg_blue),
                "width": img.shape[1],
                "height": img.shape[0]
            }
            
        except Exception as e:
            logger.error(f"Error extracting RGB: {str(e)}")
            return {"red": 0, "green": 0, "blue": 0, "error": str(e)}
    
    def apply_bandpass_filter(self, data: List[float], low_freq: float, high_freq: float) -> np.ndarray:
        """Apply bandpass filter to signal"""
        try:
            if len(data) < 10:  # Need minimum data points
                return np.array(data)
            
            nyquist = self.sampling_rate / 2
            low = low_freq / nyquist
            high = high_freq / nyquist
            
            # Ensure frequencies are valid
            low = max(0.01, min(low, 0.99))
            high = max(low + 0.01, min(high, 0.99))
            
            b, a = signal.butter(4, [low, high], btype='band')
            filtered = signal.filtfilt(b, a, data)
            return filtered
            
        except Exception as e:
            logger.error(f"Error in bandpass filter: {str(e)}")
            return np.array(data)
    
    def calculate_heart_rate(self, green_signal: List[float]) -> Dict[str, Any]:
        """Calculate heart rate using improved PPG analysis"""
        try:
            if len(green_signal) < 60:  # Need at least 2 seconds
                return {"heart_rate": 0, "confidence": 0, "method": "insufficient_data"}
            
            # Convert to numpy array and normalize
            signal = np.array(green_signal)
            
            # Remove DC component (mean)
            signal = signal - np.mean(signal)
            
            # Apply more aggressive bandpass filter for heart rate (0.5-3.0 Hz = 30-180 BPM)
            filtered_signal = self.apply_bandpass_filter(signal.tolist(), 0.5, 3.0)
            
            if len(filtered_signal) < 60:
                return {"heart_rate": 0, "confidence": 0, "method": "filter_failed"}
            
            # Apply window function to reduce spectral leakage
            windowed_signal = filtered_signal * np.hanning(len(filtered_signal))
            
            # Perform FFT with zero-padding for better frequency resolution
            n_fft = max(512, len(windowed_signal) * 2)
            fft_result = fft(windowed_signal, n=n_fft)
            freqs = fftfreq(n_fft, 1/self.sampling_rate)
            
            # Get magnitude and focus on positive frequencies
            magnitude = np.abs(fft_result[:n_fft//2])
            freqs = freqs[:n_fft//2]
            
            # Define realistic heart rate range: 0.8-2.5 Hz (48-150 BPM)
            hr_min, hr_max = 0.8, 2.5
            hr_mask = (freqs >= hr_min) & (freqs <= hr_max)
            
            if not np.any(hr_mask):
                return {"heart_rate": 0, "confidence": 0, "method": "no_valid_frequencies"}
            
            hr_freqs = freqs[hr_mask]
            hr_magnitudes = magnitude[hr_mask]
            
            # Find multiple peaks to avoid harmonics
            from scipy.signal import find_peaks
            
            # Find peaks with minimum prominence
            peaks, properties = find_peaks(hr_magnitudes, 
                                         prominence=np.max(hr_magnitudes) * 0.1,
                                         distance=5)  # Minimum distance between peaks
            
            if len(peaks) == 0:
                # Fallback: use simple max if no peaks found
                peak_idx = np.argmax(hr_magnitudes)
                peak_freq = hr_freqs[peak_idx]
                peak_magnitude = hr_magnitudes[peak_idx]
            else:
                # Choose the most prominent peak
                peak_prominences = properties['prominences']
                best_peak_idx = peaks[np.argmax(peak_prominences)]
                peak_freq = hr_freqs[best_peak_idx]
                peak_magnitude = hr_magnitudes[best_peak_idx]
            
            # Convert to BPM
            heart_rate = int(round(peak_freq * 60))
            
            # Ensure heart rate is in realistic range
            heart_rate = max(45, min(180, heart_rate))
            
            # Calculate confidence based on signal quality
            noise_level = np.std(hr_magnitudes)
            signal_to_noise = peak_magnitude / (noise_level + 1e-10)
            confidence = min(95, int(signal_to_noise * 10))
            confidence = max(20, confidence)  # Minimum confidence
            
            # Determine signal quality
            if confidence > 70:
                quality = "excellent"
            elif confidence > 50:
                quality = "good"
            elif confidence > 30:
                quality = "fair"
            else:
                quality = "poor"
            
            logger.info(f"Heart rate: {heart_rate} BPM, confidence: {confidence}%, quality: {quality}")
            logger.info(f"Peak freq: {peak_freq:.3f} Hz, S/N ratio: {signal_to_noise:.2f}")
            
            return {
                "heart_rate": heart_rate,
                "confidence": confidence,
                "method": "improved_fft_analysis",
                "peak_frequency": float(peak_freq),
                "signal_quality": quality,
                "signal_to_noise_ratio": float(signal_to_noise)
            }
            
        except Exception as e:
            logger.error(f"Error calculating heart rate: {str(e)}")
            return {"heart_rate": 0, "confidence": 0, "method": "error", "error": str(e)}
    
    def calculate_respiration_rate(self, red_signal: List[float]) -> Dict[str, Any]:
        """Calculate respiration rate from red channel variations"""
        try:
            if len(red_signal) < 90:  # Need at least 3 seconds
                return {"respiration_rate": 0, "confidence": 0}
            
            # Apply bandpass filter for respiration frequency range
            filtered_signal = self.apply_bandpass_filter(red_signal, 
                                                       self.resp_freq_range[0], 
                                                       self.resp_freq_range[1])
            
            # Perform FFT
            fft_result = fft(filtered_signal)
            freqs = fftfreq(len(filtered_signal), 1/self.sampling_rate)
            
            # Get magnitude and find peak in respiration range
            magnitude = np.abs(fft_result)
            
            # Find frequencies in respiration range
            resp_mask = (freqs >= self.resp_freq_range[0]) & (freqs <= self.resp_freq_range[1])
            if not np.any(resp_mask):
                return {"respiration_rate": 0, "confidence": 0}
            
            resp_freqs = freqs[resp_mask]
            resp_magnitudes = magnitude[resp_mask]
            
            # Find peak frequency
            peak_idx = np.argmax(resp_magnitudes)
            peak_freq = resp_freqs[peak_idx]
            
            # Convert to breaths per minute
            respiration_rate = int(peak_freq * 60)
            
            # Calculate confidence
            mean_magnitude = np.mean(resp_magnitudes)
            peak_magnitude = resp_magnitudes[peak_idx]
            confidence = min(100, int((peak_magnitude / mean_magnitude) * 15))
            
            return {
                "respiration_rate": respiration_rate,
                "confidence": confidence
            }
            
        except Exception as e:
            logger.error(f"Error calculating respiration rate: {str(e)}")
            return {"respiration_rate": 0, "confidence": 0}
    
    def calculate_spo2_estimate(self, red_signal: List[float], infrared_signal: List[float]) -> Dict[str, Any]:
        """Estimate SpO2 using AC/DC ratio method (simplified)"""
        try:
            if len(red_signal) < 30 or len(infrared_signal) < 30:
                return {"spo2": 0, "confidence": 0}
            
            # Calculate AC (standard deviation) and DC (mean) components
            red_ac = np.std(red_signal)
            red_dc = np.mean(red_signal)
            ir_ac = np.std(infrared_signal)
            ir_dc = np.mean(infrared_signal)
            
            if red_dc == 0 or ir_dc == 0:
                return {"spo2": 0, "confidence": 0}
            
            # Calculate ratio of ratios
            red_ratio = red_ac / red_dc
            ir_ratio = ir_ac / ir_dc
            
            if ir_ratio == 0:
                return {"spo2": 0, "confidence": 0}
            
            ratio = red_ratio / ir_ratio
            
            # Simplified SpO2 calculation (calibration needed for accuracy)
            spo2 = int(100 - (ratio * 25))  # Simplified formula
            spo2 = max(70, min(100, spo2))  # Clamp to reasonable range
            
            confidence = 60 if 85 <= spo2 <= 100 else 30
            
            return {
                "spo2": spo2,
                "confidence": confidence,
                "ratio": float(ratio)
            }
            
        except Exception as e:
            logger.error(f"Error calculating SpO2: {str(e)}")
            return {"spo2": 0, "confidence": 0}
    
    def process_frame(self, frame_data: str, timestamp: float) -> Dict[str, Any]:
        """Process a single frame and update signals"""
        try:
            if self.start_time is None:
                self.start_time = timestamp
            
            # Decode base64 frame
            image_data = base64.b64decode(frame_data)
            rgb_values = self.extract_rgb_from_frame(image_data)
            
            if "error" in rgb_values:
                return {"status": "error", "error": rgb_values["error"]}
            
            # Add to signal buffers
            self.green_signal.append(rgb_values["green"])
            self.red_signal.append(rgb_values["red"])
            self.blue_signal.append(rgb_values["blue"])
            self.timestamps.append(timestamp)
            self.frame_count += 1
            
            # Maintain sliding window
            if len(self.green_signal) > self.window_size:
                self.green_signal.pop(0)
                self.red_signal.pop(0)
                self.blue_signal.pop(0)
                self.timestamps.pop(0)
            
            # Calculate vital signs if we have enough data
            results = {
                "status": "processing",
                "frame_count": self.frame_count,
                "elapsed_time": timestamp - self.start_time,
                "rgb_values": rgb_values,
                "buffer_size": len(self.green_signal),
                "green_signal_value": rgb_values["green"]  # Current green value for real-time display
            }
            
            # Calculate heart rate every 15 frames (twice per second) for responsive display
            if len(self.green_signal) >= 60 and self.frame_count % 15 == 0:
                hr_result = self.calculate_heart_rate(self.green_signal)
                
                results.update({
                    "heart_rate": hr_result,
                    "green_signal_history": self.green_signal[-30:] if len(self.green_signal) >= 30 else self.green_signal,  # Last 1 second of data
                    "signal_processing": "active"
                })
            
            return results
            
        except Exception as e:
            logger.error(f"Error processing frame: {str(e)}")
            return {"status": "error", "error": str(e)}

# Global PPG processor instance
ppg_processor = PPGProcessor()

@app.get("/")
async def root():
    return {"message": "PPG Signal Processing Server", "status": "running"}

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "server_time": time.time(),
        "processor_status": {
            "frame_count": ppg_processor.frame_count,
            "buffer_size": len(ppg_processor.green_signal),
            "is_active": ppg_processor.start_time is not None
        }
    }

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    logger.info("Client connected to WebSocket")
    
    try:
        while True:
            data = await websocket.receive_text()
            
            try:
                # Parse incoming message
                message = json.loads(data)
                message_type = message.get("type", "frame")
                
                if message_type == "reset":
                    ppg_processor.reset()
                    response = {
                        "type": "reset_ack",
                        "status": "success",
                        "timestamp": time.time()
                    }
                
                elif message_type == "frame":
                    frame_data = message.get("frame", "")
                    timestamp = message.get("timestamp", time.time())
                    
                    if not frame_data:
                        response = {
                            "type": "error",
                            "error": "No frame data provided",
                            "timestamp": time.time()
                        }
                    else:
                        # Process the frame
                        result = ppg_processor.process_frame(frame_data, timestamp)
                        response = {
                            "type": "result",
                            "data": result,
                            "timestamp": time.time()
                        }
                
                else:
                    response = {
                        "type": "error",
                        "error": f"Unknown message type: {message_type}",
                        "timestamp": time.time()
                    }
                
                await websocket.send_text(json.dumps(response))
                
            except json.JSONDecodeError as e:
                error_response = {
                    "type": "error",
                    "error": f"Invalid JSON: {str(e)}",
                    "timestamp": time.time()
                }
                await websocket.send_text(json.dumps(error_response))
                
            except Exception as e:
                logger.error(f"Error processing WebSocket message: {str(e)}")
                error_response = {
                    "type": "error",
                    "error": f"Processing error: {str(e)}",
                    "timestamp": time.time()
                }
                await websocket.send_text(json.dumps(error_response))
    
    except WebSocketDisconnect:
        logger.info("Client disconnected from WebSocket")
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    import os
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
