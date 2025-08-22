#!/usr/bin/env python3
"""
Real-time Radar Action Recognition Test Script
Read radar data from serial port, perform 6-frame fusion, and input to PETer model for real-time action recognition
"""

import torch
import numpy as np
import json
import time
import threading
import queue
from collections import deque
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
import serial
import serial.tools.list_ports
from peter_network import PETerNetwork
import warnings
warnings.filterwarnings('ignore')

class RealTimeRadarReader:
    """Real-time radar data reader"""
    
    def __init__(self, cli_port='COM4', data_port='COM6', baudrate=921600):
        self.cli_port = cli_port
        self.data_port = data_port
        self.baudrate = baudrate
        self.cli_serial = None
        self.data_serial = None
        self.is_running = False
        self.data_queue = queue.Queue()
        self.frame_buffer = deque(maxlen=100)  # Store recent 100 frames
        
        # Magic word
        self.MAGIC_WORD = b'\x02\x01\x04\x03\x06\x05\x08\x07'
        
    def connect_radar(self):
        """Connect radar device"""
        try:
            print(f"\nConnecting serial ports...")
            
            # Connect CLI serial port
            print(f"   Connecting CLI serial port: {self.cli_port}")
            self.cli_serial = serial.Serial(
                self.cli_port, 
                115200,  # CLI serial port usually uses 115200 baud rate
                timeout=1
            )
            print(f"   CLI serial port connected successfully")
            
            # Connect data port
            print(f"   Connecting data serial port: {self.data_port}")
            self.data_serial = serial.Serial(
                self.data_port, 
                self.baudrate,  # Data serial port uses 921600 baud rate
                timeout=1
            )
            print(f"   Data serial port connected successfully")
            
            print(f"All serial ports connected successfully")
            
            # Send configuration commands
            self.send_config()
            
            # Wait for radar to start
            print("Waiting for radar to start...")
            time.sleep(2)
            
            # Check if data serial port has data
            print("Checking data serial port status...")
            if self.data_serial.in_waiting > 0:
                print(f"Data serial port has {self.data_serial.in_waiting} bytes of data")
                # Read first few bytes to see
                test_data = self.data_serial.read(min(16, self.data_serial.in_waiting))
                print(f"First 16 bytes: {test_data.hex()}")
            else:
                print("Data serial port has no data yet")
            
            # Wait longer for radar to start sending data
            print("Waiting for radar to start sending data...")
            time.sleep(3)
            
            # Check data serial port again
            if self.data_serial.in_waiting > 0:
                print(f"After 3 seconds, data serial port has {self.data_serial.in_waiting} bytes of data")
            else:
                print("After 3 seconds, data serial port still has no data, radar may not have started correctly")
            
            return True
            
        except Exception as e:
            print(f"Connection failed: {e}")
            return False
    
    def load_config_file(self, config_path):
        """Load configuration file"""
        try:
            with open(config_path, 'r') as f:
                return [line.strip() for line in f.readlines()]
        except Exception as e:
            print(f"Failed to load configuration file: {e}")
            return []
    
    def send_config(self):
        """Send radar configuration - using radar_reader.py method"""
        try:
            # Load configuration file
            config_path = 'decoding/3m.cfg'
            config_lines = self.load_config_file(config_path)
            
            if not config_lines:
                print("Configuration file loading failed")
                return False
            
            print(f"\nSending configuration file...")
            
            # Remove empty lines and comment lines
            cfg = [line for line in config_lines if line.strip() and not line.startswith('%')]
            # Ensure each line ends with \n
            cfg = [line + '\n' if not line.endswith('\n') else line for line in cfg]
            
            for line in cfg:
                time.sleep(0.03)  # Line delay
                
                # Send command
                self.cli_serial.write(line.encode())
                
                # Read acknowledgment
                ack = self.cli_serial.readline()
                if len(ack) == 0:
                    print("Serial port timeout, device may be in flash mode")
                    return False
                
                # Safe decode
                try:
                    ack_text = ack.decode('utf-8').strip()
                except UnicodeDecodeError:
                    ack_text = ack.decode('utf-8', errors='ignore').strip()
                
                print(f"   {line.strip()} -> {ack_text}")
                
                # Read second line acknowledgment
                ack = self.cli_serial.readline()
                if ack:
                    try:
                        ack_text = ack.decode('utf-8').strip()
                    except UnicodeDecodeError:
                        ack_text = ack.decode('utf-8', errors='ignore').strip()
                    print(f"   Acknowledgment: {ack_text}")
            
            # Give buffer some time to clear
            time.sleep(0.03)
            self.cli_serial.reset_input_buffer()
            
            print(f"Configuration file sent successfully")
            return True
            
        except Exception as e:
            print(f"Configuration sending failed: {e}")
            return False
    
    def read_frame(self):
        """Read one frame of data - using radar_reader.py method"""
        try:
            # Find magic word
            index = 0
            magicByte = self.data_serial.read(1)
            frameData = bytearray(b'')
            
            while True:
                # Check if there is data
                if len(magicByte) < 1:
                    return None  # Timeout, no data
                
                # Find matching byte
                if magicByte[0] == self.MAGIC_WORD[index]:
                    index += 1
                    frameData.append(magicByte[0])
                    if index == 8:  # Found complete magic word
                        break
                    magicByte = self.data_serial.read(1)
                else:
                    # Reset index
                    if index == 0:
                        magicByte = self.data_serial.read(1)
                    index = 0
                    frameData = bytearray(b'')
            
            # Read version number
            versionBytes = self.data_serial.read(4)
            frameData += bytearray(versionBytes)
            
            # Read length
            lengthBytes = self.data_serial.read(4)
            frameData += bytearray(lengthBytes)
            frameLength = int.from_bytes(lengthBytes, byteorder='little')
            
            # Subtract already read bytes (magic word, version, length)
            frameLength -= 16
            
            # Read the rest of the frame
            frameData += bytearray(self.data_serial.read(frameLength))
            
            return frameData
            
        except Exception as e:
            print(f"Failed to read frame: {e}")
            return None
    
    def parse_point_cloud(self, frame_data):
        """Parse point cloud data"""
        try:
            # Import parsing module
            import sys
            sys.path.append('decoding')
            from parseFrame import parseStandardFrame
            
            # Parse frame
            parsed_data = parseStandardFrame(frame_data)
            
            if 'pointCloud' in parsed_data:
                point_cloud = parsed_data['pointCloud']
                if len(point_cloud) > 0:
                    # Extract xyz coordinates
                    points = []
                    for point in point_cloud:
                        if len(point) >= 3:
                            x, y, z = point[0], point[1], point[2]
                            points.append([x, y, z])
                    
                    return np.array(points)
            
            return np.array([])
            
        except Exception as e:
            print(f"Failed to parse point cloud: {e}")
            return np.array([])
    
    def start_reading(self):
        """Start reading data"""
        self.is_running = True
        self.read_thread = threading.Thread(target=self._read_loop)
        self.read_thread.daemon = True
        self.read_thread.start()
        print("Starting real-time data reading...")
    
    def _read_loop(self):
        """Data reading loop"""
        print("Starting data reading loop...")
        frame_count = 0
        last_status_time = time.time()
        
        while self.is_running:
            try:
                # Check if serial port has data
                if self.data_serial.in_waiting > 0:
                    print(f"Detected data: {self.data_serial.in_waiting} bytes")
                
                frame_data = self.read_frame()
                if frame_data:
                    frame_count += 1
                    print(f"Successfully read frame {frame_count}")
                    
                    points = self.parse_point_cloud(frame_data)
                    if len(points) > 0:
                        timestamp = time.time()
                        self.frame_buffer.append({
                            'timestamp': timestamp,
                            'points': points
                        })
                        self.data_queue.put({
                            'timestamp': timestamp,
                            'points': points
                        })
                        print(f"Parsed {len(points)} point cloud points")
                    else:
                        print("No point cloud information in frame data")
                else:
                    # Show status every 5 seconds
                    current_time = time.time()
                    if current_time - last_status_time > 5:
                        print(f"Waiting for data... (Read {frame_count} frames)")
                        last_status_time = current_time
                
                time.sleep(0.01)  # 10ms interval
                
            except Exception as e:
                print(f"Reading loop error: {e}")
                time.sleep(0.1)
    
    def stop_reading(self):
        """Stop reading data"""
        self.is_running = False
        if self.cli_serial:
            self.cli_serial.close()
        if self.data_serial:
            self.data_serial.close()
        print("Stopped data reading")

class RealTimeActionRecognizer:
    """Real-time action recognizer"""
    
    def __init__(self, model_path='best_peter_model.pth', num_frames=6, num_points=100):
        self.num_frames = num_frames  # Number of frames to fuse each time
        self.num_points = num_points
        self.frame_buffer = deque(maxlen=num_frames)  # Only need to store 6 frames
        self.action_labels = ['sit', 'squat', 'stand']
        
        # Stability mechanism
        self.prediction_history = deque(maxlen=10)  # Save recent 10 predictions
        self.stability_threshold = 0.6  # Confidence threshold
        self.min_consistent_predictions = 3  # Minimum consecutive prediction count
        self.last_stable_prediction = None  # Last stable prediction result
        
        # Class balance mechanism
        self.class_weights = [2.0, 2.0, 0.5]  # Weights for sit, squat, stand
        self.prediction_counts = {'sit': 0, 'squat': 0, 'stand': 0}  # Prediction counts
        self.max_consecutive_same = 5  # Maximum consecutive same prediction count
        
        # Load model
        self.model = PETerNetwork(num_classes=3, num_points=num_points, num_frames=25, k=10)
        if torch.cuda.is_available():
            self.model.load_state_dict(torch.load(model_path, map_location='cuda'))
            self.model = self.model.cuda()
        else:
            self.model.load_state_dict(torch.load(model_path, map_location='cpu'))
        
        self.model.eval()
        print(f"Model loaded successfully: {model_path}")
    
    def preprocess_points(self, points):
        """Preprocess point cloud data"""
        if len(points) == 0:
            return np.zeros((self.num_points, 3))
        
        # Standardize
        centroid = np.mean(points, axis=0)
        points = points - centroid
        
        # Scale to unit sphere
        max_dist = np.max(np.linalg.norm(points, axis=1))
        if max_dist > 0:
            points = points / max_dist
        
        # Sample fixed number of points
        if len(points) >= self.num_points:
            indices = np.random.choice(len(points), self.num_points, replace=False)
            return points[indices]
        else:
            indices = np.random.choice(len(points), self.num_points, replace=True)
            return points[indices]
    
    def fuse_frames(self, frame_group):
        """
        Fuse point cloud data from a group of frames
        Args:
            frame_group: List of point cloud data from a group of frames
        Returns:
            fused_points: Fused point cloud data
        """
        all_points = []
        
        for frame in frame_group:
            points = frame['points']
            if len(points) > 0:
                all_points.append(points)
        
        if not all_points:
            return np.empty((0, 3))
        
        # Merge point clouds from all frames
        fused_points = np.vstack(all_points)
        return fused_points
    
    def create_sequence(self):
        """Create time sequence - directly fuse 6 frames then repeat to 25 frames"""
        if len(self.frame_buffer) < self.num_frames:
            return None
        
        # Get recent 6 frames
        recent_frames = list(self.frame_buffer)[-self.num_frames:]
        
        # Fuse 6 frames of point cloud data
        fused_points = self.fuse_frames(recent_frames)
        
        # Preprocess fused point cloud
        processed_points = self.preprocess_points(fused_points)
        
        # Add intensity information
        intensity = np.ones((len(processed_points), 1))
        points_with_intensity = np.hstack([processed_points, intensity])
        
        # Create 25-frame sequence (repeat fused point cloud)
        sequence = np.tile(points_with_intensity, (25, 1, 1))  # Repeat 25 times
        
        return torch.FloatTensor(sequence).unsqueeze(0)  # Add batch dimension
    
    def predict_action(self, sequence):
        """Predict action (with class balancing)"""
        try:
            with torch.no_grad():
                if torch.cuda.is_available():
                    sequence = sequence.cuda()
                
                output = self.model(sequence)
                
                # Apply class weights
                weighted_output = output.clone()
                for i, weight in enumerate(self.class_weights):
                    weighted_output[0, i] *= weight
                
                probabilities = torch.softmax(weighted_output, dim=1)
                predicted_class = torch.argmax(weighted_output, dim=1).item()
                confidence = probabilities[0][predicted_class].item()
                
                # Update prediction counts
                predicted_action = self.action_labels[predicted_class]
                self.prediction_counts[predicted_action] += 1
                
                return {
                    'action': predicted_action,
                    'confidence': confidence,
                    'probabilities': probabilities[0].cpu().numpy(),
                    'raw_probabilities': torch.softmax(output, dim=1)[0].cpu().numpy()  # Raw probabilities
                }
                
        except Exception as e:
            print(f"Prediction failed: {e}")
            return None
    
    def add_frame(self, points):
        """Add new frame"""
        self.frame_buffer.append({
            'timestamp': time.time(),
            'points': points
        })
    
    def get_stable_prediction(self):
        """Get stable prediction result (with class balancing)"""
        sequence = self.create_sequence()
        if sequence is not None:
            current_prediction = self.predict_action(sequence)
            if current_prediction is None:
                return self.last_stable_prediction
            
            # Add to prediction history
            self.prediction_history.append(current_prediction)
            
            # Check consecutive prediction count to prevent over-biasing towards a certain class
            if len(self.prediction_history) >= self.max_consecutive_same:
                recent_actions = [pred['action'] for pred in list(self.prediction_history)[-self.max_consecutive_same:]]
                if len(set(recent_actions)) == 1:  # Consecutive same action predictions
                    # Reduce weight for this action
                    action_idx = self.action_labels.index(recent_actions[0])
                    self.class_weights[action_idx] = max(0.1, self.class_weights[action_idx] * 0.8)
                    print(f"Consecutive {recent_actions[0]} predictions, reducing weight to {self.class_weights[action_idx]:.2f}")
            
            # If confidence is high enough, return directly
            if current_prediction['confidence'] > 0.8:
                self.last_stable_prediction = current_prediction
                return current_prediction
            
            # Check consistency of historical predictions
            if len(self.prediction_history) >= self.min_consistent_predictions:
                recent_predictions = list(self.prediction_history)[-self.min_consistent_predictions:]
                
                # Check if consecutive same action predictions
                actions = [pred['action'] for pred in recent_predictions]
                if len(set(actions)) == 1:  # All predictions are the same action
                    avg_confidence = np.mean([pred['confidence'] for pred in recent_predictions])
                    if avg_confidence > self.stability_threshold:
                        stable_prediction = {
                            'action': actions[0],
                            'confidence': avg_confidence,
                            'probabilities': np.mean([pred['probabilities'] for pred in recent_predictions], axis=0),
                            'raw_probabilities': np.mean([pred['raw_probabilities'] for pred in recent_predictions], axis=0)
                        }
                        self.last_stable_prediction = stable_prediction
                        return stable_prediction
            
            # If current prediction confidence is high, update stable prediction
            if current_prediction['confidence'] > self.stability_threshold:
                self.last_stable_prediction = current_prediction
            
            # Return last stable prediction or current prediction
            return self.last_stable_prediction if self.last_stable_prediction else current_prediction
        
        return self.last_stable_prediction
    
    def get_prediction(self):
        """Get prediction result (maintain backward compatibility)"""
        return self.get_stable_prediction()

class RealTimeVisualizer:
    """Real-time visualizer"""
    
    def __init__(self):
        self.fig = plt.figure(figsize=(15, 6))
        self.fig.suptitle('Real-time Radar Action Recognition', fontsize=16)
        
        # Point cloud display (3D)
        self.ax1 = self.fig.add_subplot(121, projection='3d')
        self.ax1.set_xlabel('X')
        self.ax1.set_ylabel('Y')
        self.ax1.set_zlabel('Z')
        self.ax1.set_title('Real-time Point Cloud')
        
        # Action probability display
        self.ax2 = self.fig.add_subplot(122)
        self.ax2.set_title('Action Recognition Probability')
        self.ax2.set_ylim(0, 1)
        
        # Action probability display
        self.ax2.set_title('Action Recognition Probability')
        self.ax2.set_ylim(0, 1)
        
        self.action_labels = ['sit', 'squat', 'stand']
        self.prob_bars = None
        self.point_cloud_plot = None
        
        plt.ion()  # Enable interactive mode
    
    def update_visualization(self, points, prediction):
        """Update visualization"""
        # Clear old plots
        self.ax1.clear()
        self.ax2.clear()
        
        # Draw point cloud
        if len(points) > 0:
            self.ax1.scatter(points[:, 0], points[:, 1], points[:, 2], c='b', alpha=0.6)
        
        self.ax1.set_xlabel('X')
        self.ax1.set_ylabel('Y')
        self.ax1.set_zlabel('Z')
        self.ax1.set_title('Real-time Point Cloud')
        self.ax1.set_xlim(-2, 2)
        self.ax1.set_ylim(-2, 2)
        self.ax1.set_zlim(-2, 2)
        self.ax1.view_init(elev=20, azim=45)  # Set 3D view angle
        
        # Draw action probabilities
        if prediction:
            probabilities = prediction['probabilities']
            bars = self.ax2.bar(self.action_labels, probabilities, color=['red', 'green', 'blue'])
            
            # Add probability value labels
            for bar, prob in zip(bars, probabilities):
                self.ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                             f'{prob:.3f}', ha='center', va='bottom')
            
            # Highlight predicted action
            predicted_action = prediction['action']
            predicted_idx = self.action_labels.index(predicted_action)
            bars[predicted_idx].set_color('orange')
            
            self.ax2.set_title(f'Action Recognition: {predicted_action} (Confidence: {prediction["confidence"]:.3f})')
        
        self.ax2.set_ylim(0, 1)
        self.ax2.set_ylabel('Probability')
        
        plt.tight_layout()
        plt.pause(0.01)

def main():
    """Main function"""
    print("Real-time Radar Action Recognition Test")
    print("=" * 60)
    
    # Check model file
    if not os.path.exists('best_peter_model.pth'):
        print("Model file does not exist, please run training script first")
        return
    
    # Create components
    radar_reader = RealTimeRadarReader()
    action_recognizer = RealTimeActionRecognizer()
    visualizer = RealTimeVisualizer()
    
    # Connect radar
    if not radar_reader.connect_radar():
        print("Radar connection failed")
        return
    
    try:
        # Start reading data
        radar_reader.start_reading()
        
        print("Starting real-time recognition...")
        print("Need to collect 6 frames of data to start prediction (6-frame fusion)")
        print("Press Ctrl+C to stop")
        
        last_prediction_time = 0
        prediction_interval = 0.5  # Predict every 0.5 seconds
        
        while True:
            try:
                # Get latest data
                if not radar_reader.data_queue.empty():
                    data = radar_reader.data_queue.get()
                    points = data['points']
                    
                    # Add to recognizer
                    action_recognizer.add_frame(points)
                    
                    # Show current frame count status
                    current_frames = len(action_recognizer.frame_buffer)
                    if current_frames <= 6:  # Show status for first 6 frames
                        print(f"Collected {current_frames}/6 frames of data")
                    
                    # Perform prediction periodically
                    current_time = time.time()
                    if current_time - last_prediction_time >= prediction_interval:
                        prediction = action_recognizer.get_prediction()
                        if prediction:
                            print(f"Recognition result: {prediction['action']} (Confidence: {prediction['confidence']:.3f})")
                            
                            # Update visualization
                            visualizer.update_visualization(points, prediction)
                        else:
                            print(f"Waiting for more data... (Current: {current_frames}/6 frames)")
                        
                        last_prediction_time = current_time
                
                time.sleep(0.01)
                
            except KeyboardInterrupt:
                print("\nUser stopped")
                break
            except Exception as e:
                print(f"Runtime error: {e}")
                time.sleep(0.1)
    
    finally:
        # Clean up resources
        radar_reader.stop_reading()
        plt.ioff()
        plt.close()
        print("Test completed")

if __name__ == "__main__":
    import os
    main() 