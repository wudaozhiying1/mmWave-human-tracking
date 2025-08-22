#!/usr/bin/env python3
"""
Real-time 3D Point Cloud and Human Action Recognition Display Script
Specifically designed for real-time display of point cloud data and action recognition results
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
import threading
import queue
from collections import deque
from peter_network import PETerNetwork
import torch
import warnings
warnings.filterwarnings('ignore')

class RealTime3DDisplay:
    """Real-time 3D point cloud and action recognition display"""
    
    def __init__(self, model_path='best_peter_model.pth', num_points=100, num_frames=6):
        self.num_points = num_points
        self.num_frames = num_frames
        self.action_labels = ['sit', 'squat', 'stand']
        
        # Frame buffer
        self.frame_buffer = deque(maxlen=num_frames)
        self.data_queue = queue.Queue()
        self.is_running = False
        
        # Load model
        self.model = PETerNetwork(num_classes=3, num_points=num_points, num_frames=25, k=10)
        if torch.cuda.is_available():
            self.model.load_state_dict(torch.load(model_path, map_location='cuda'))
            self.model = self.model.cuda()
        else:
            self.model.load_state_dict(torch.load(model_path, map_location='cpu'))
        
        self.model.eval()
        print(f"Model loaded successfully: {model_path}")
        
        # Action color mapping
        self.action_colors = {
            'sit': 'red',
            'squat': 'green', 
            'stand': 'blue'
        }
        
        # Action descriptions
        self.action_descriptions = {
            'sit': 'Sitting',
            'squat': 'Squatting',
            'stand': 'Standing'
        }
        
        # Prediction history
        self.prediction_history = deque(maxlen=10)
        self.last_prediction = None
        
        # Create visualization interface
        self.create_display()
    
    def create_display(self):
        """Create display interface"""
        self.fig = plt.figure(figsize=(16, 10))
        self.fig.suptitle('Real-time 3D Point Cloud and Human Action Recognition', fontsize=16, fontweight='bold')
        
        # 3D point cloud display
        self.ax1 = self.fig.add_subplot(221, projection='3d')
        self.ax1.set_xlabel('X-axis')
        self.ax1.set_ylabel('Y-axis')
        self.ax1.set_zlabel('Z-axis')
        self.ax1.set_title('Real-time 3D Point Cloud')
        self.ax1.grid(True, alpha=0.3)
        
        # Action probability display
        self.ax2 = self.fig.add_subplot(222)
        self.ax2.set_title('Action Recognition Probability')
        self.ax2.set_ylim(0, 1)
        self.ax2.set_ylabel('Probability')
        self.ax2.grid(True, alpha=0.3)
        
        # Status information display
        self.ax3 = self.fig.add_subplot(223)
        self.ax3.set_title('System Status')
        self.ax3.axis('off')
        
        # Prediction history display
        self.ax4 = self.fig.add_subplot(224)
        self.ax4.set_title('Prediction History')
        self.ax4.set_ylim(0, 1)
        self.ax4.set_ylabel('Confidence')
        self.ax4.grid(True, alpha=0.3)
        
        plt.ion()  # Enable interactive mode
        plt.tight_layout()
    
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
        """Fuse point cloud data from a group of frames"""
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
    
    def predict_action(self, points):
        """Predict action"""
        try:
            # Preprocess point cloud
            processed_points = self.preprocess_points(points)
            
            # Add intensity information
            intensity = np.ones((len(processed_points), 1))
            points_with_intensity = np.hstack([processed_points, intensity])
            
            # Create 25-frame sequence (repeat single frame)
            sequence = np.tile(points_with_intensity, (25, 1, 1))
            sequence = torch.FloatTensor(sequence).unsqueeze(0)
            
            with torch.no_grad():
                if torch.cuda.is_available():
                    sequence = sequence.cuda()
                
                output = self.model(sequence)
                probabilities = torch.softmax(output, dim=1)
                predicted_class = torch.argmax(output, dim=1).item()
                confidence = probabilities[0][predicted_class].item()
                
                return {
                    'action': self.action_labels[predicted_class],
                    'confidence': confidence,
                    'probabilities': probabilities[0].cpu().numpy()
                }
                
        except Exception as e:
            print(f"Prediction failed: {e}")
            return None
    
    def add_frame(self, points):
        """Add new frame to buffer"""
        self.frame_buffer.append({
            'timestamp': time.time(),
            'points': points
        })
    
    def get_prediction(self):
        """Get prediction result (fuse multiple frames)"""
        if len(self.frame_buffer) < self.num_frames:
            return None
        
        # Get recent 6 frames
        recent_frames = list(self.frame_buffer)[-self.num_frames:]
        
        # Fuse 6 frames of point cloud data
        fused_points = self.fuse_frames(recent_frames)
        
        if len(fused_points) == 0:
            return None
        
        # Predict action
        prediction = self.predict_action(fused_points)
        
        if prediction:
            # Add to prediction history
            self.prediction_history.append(prediction)
            self.last_prediction = prediction
            
            # Calculate stability (number of consecutive same predictions)
            if len(self.prediction_history) >= 3:
                recent_actions = [pred['action'] for pred in list(self.prediction_history)[-3:]]
                if len(set(recent_actions)) == 1:  # 3 consecutive same predictions
                    prediction['stable'] = True
                else:
                    prediction['stable'] = False
            else:
                prediction['stable'] = False
        
        return prediction
    
    def update_display(self, points, prediction):
        """Update display"""
        # Clear old plots
        self.ax1.clear()
        self.ax2.clear()
        self.ax3.clear()
        self.ax4.clear()
        
        # Update 3D point cloud
        if len(points) > 0:
            # Choose color based on prediction result
            if prediction:
                color = self.action_colors[prediction['action']]
            else:
                color = 'gray'
            
            self.ax1.scatter(points[:, 0], points[:, 1], points[:, 2], 
                           c=color, alpha=0.7, s=20, edgecolors='black', linewidth=0.5)
            
            # Calculate point cloud center
            center = np.mean(points, axis=0)
            self.ax1.scatter(center[0], center[1], center[2], 
                           c='yellow', s=100, marker='*', label='Human Center')
            
            # Set coordinate axis range
            x_range = np.max(points[:, 0]) - np.min(points[:, 0])
            y_range = np.max(points[:, 1]) - np.min(points[:, 1])
            z_range = np.max(points[:, 2]) - np.min(points[:, 2])
            
            max_range = max(x_range, y_range, z_range)
            if max_range > 0:
                self.ax1.set_xlim(-max_range/2, max_range/2)
                self.ax1.set_ylim(-max_range/2, max_range/2)
                self.ax1.set_zlim(-max_range/2, max_range/2)
        
        self.ax1.set_xlabel('X-axis')
        self.ax1.set_ylabel('Y-axis')
        self.ax1.set_zlabel('Z-axis')
        self.ax1.set_title('Real-time 3D Point Cloud')
        self.ax1.grid(True, alpha=0.3)
        self.ax1.legend()
        
        # Update action probabilities
        if prediction:
            probabilities = prediction['probabilities']
            bars = self.ax2.bar(self.action_labels, probabilities, 
                              color=[self.action_colors[label] for label in self.action_labels],
                              alpha=0.7, edgecolor='black', linewidth=1)
            
            # Add probability value labels
            for bar, prob in zip(bars, probabilities):
                self.ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                            f'{prob:.3f}', ha='center', va='bottom', fontweight='bold')
            
            # Highlight predicted action
            predicted_action = prediction['action']
            predicted_idx = self.action_labels.index(predicted_action)
            bars[predicted_idx].set_alpha(1.0)
            bars[predicted_idx].set_edgecolor('red')
            bars[predicted_idx].set_linewidth(3)
            
            # Display prediction result
            stability_text = "Stable" if prediction.get('stable', False) else "Unstable"
            self.ax2.set_title(f'Action Recognition Result\nPrediction: {self.action_descriptions[predicted_action]} ({predicted_action})\nConfidence: {prediction["confidence"]:.3f} ({stability_text})')
        else:
            self.ax2.text(0.5, 0.5, 'Waiting for data...', ha='center', va='center', 
                        transform=self.ax2.transAxes, fontsize=14, color='gray')
            self.ax2.set_title('Action Recognition Result')
        
        self.ax2.set_ylim(0, 1)
        self.ax2.set_ylabel('Probability')
        self.ax2.grid(True, alpha=0.3)
        
        # Update status information
        status_text = f'System Status:\n'
        status_text += f'Frame Buffer: {len(self.frame_buffer)}/{self.num_frames}\n'
        status_text += f'Point Cloud Points: {len(points) if len(points) > 0 else 0}\n'
        if prediction:
            status_text += f'Current Action: {self.action_descriptions[prediction["action"]]}\n'
            status_text += f'Confidence: {prediction["confidence"]:.3f}\n'
            status_text += f'Stability: {stability_text}'
        else:
            status_text += f'Current Action: Waiting for recognition\n'
            status_text += f'Confidence: --\n'
            status_text += f'Stability: --'
        
        self.ax3.text(0.1, 0.5, status_text, fontsize=12, 
                     bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8),
                     transform=self.ax3.transAxes, verticalalignment='center')
        self.ax3.set_title('System Status')
        
        # Update prediction history
        if len(self.prediction_history) > 0:
            history_actions = [pred['action'] for pred in self.prediction_history]
            history_confidences = [pred['confidence'] for pred in self.prediction_history]
            
            # Draw historical confidence
            for i, (action, conf) in enumerate(zip(history_actions, history_confidences)):
                color = self.action_colors[action]
                self.ax4.scatter(i, conf, c=color, s=50, alpha=0.7)
                self.ax4.text(i, conf + 0.02, action, ha='center', va='bottom', fontsize=8)
        
        self.ax4.set_title('Prediction History')
        self.ax4.set_ylim(0, 1)
        self.ax4.set_ylabel('Confidence')
        self.ax4.set_xlabel('Time')
        self.ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.pause(0.01)
    
    def start_simulation(self):
        """Start simulated real-time data"""
        print("Starting real-time 3D point cloud display")
        print("Using simulated data for real-time testing")
        print("Press Ctrl+C to stop")
        print("=" * 60)
        
        # Create simulated data
        def create_simulated_points(action_type):
            """Create simulated point cloud data"""
            if action_type == 'sit':
                # Sitting: lower point cloud distribution
                points = np.random.normal(0, 0.5, (100, 3))
                points[:, 2] = points[:, 2] * 0.3 - 0.5  # Reduce height
            elif action_type == 'squat':
                # Squatting: medium height point cloud distribution
                points = np.random.normal(0, 0.4, (100, 3))
                points[:, 2] = points[:, 2] * 0.5 - 0.2
            else:  # stand
                # Standing: higher point cloud distribution
                points = np.random.normal(0, 0.3, (100, 3))
                points[:, 2] = points[:, 2] * 0.8 + 0.3  # Increase height
            
            return points
        
        # Simulate different actions in a loop
        actions = ['sit', 'squat', 'stand']
        action_duration = 20  # Each action lasts 20 frames
        
        try:
            frame_count = 0
            while True:
                # Determine current action
                current_action_idx = (frame_count // action_duration) % len(actions)
                current_action = actions[current_action_idx]
                
                # Create simulated point cloud
                current_points = create_simulated_points(current_action)
                
                # Add some random variation
                noise = np.random.normal(0, 0.02, current_points.shape)
                current_points = current_points + noise
                
                # Add to frame buffer
                self.add_frame(current_points)
                
                # Get prediction result
                prediction = self.get_prediction()
                
                # Update display
                self.update_display(current_points, prediction)
                
                # Display status
                if frame_count % 10 == 0:
                    if prediction:
                        print(f"Frame {frame_count}: Simulated action={current_action}, Recognition={prediction['action']} (Confidence: {prediction['confidence']:.3f})")
                    else:
                        print(f"Frame {frame_count}: Waiting for data...")
                
                frame_count += 1
                time.sleep(0.3)  # Update every 0.3 seconds
                
        except KeyboardInterrupt:
            print("\nUser stopped")
        except Exception as e:
            print(f"Runtime error: {e}")
        finally:
            plt.ioff()
            plt.close()
            print("Real-time display ended")

def main():
    """Main function"""
    print("Real-time 3D Point Cloud and Human Action Recognition Display")
    print("=" * 60)
    
    # Check model file
    import os
    if not os.path.exists('best_peter_model.pth'):
        print("Model file does not exist, please run training script first")
        return
    
    # Create display
    display = RealTime3DDisplay()
    
    # Start simulation
    display.start_simulation()

if __name__ == "__main__":
    main() 