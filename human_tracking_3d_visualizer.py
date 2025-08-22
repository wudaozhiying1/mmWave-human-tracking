#!/usr/bin/env python3
"""
Human Tracking 3D Animation Visualizer
Uses PyQtGraph OpenGL to display radar data human action recognition in real-time
Integrates action recognition results from demo_action_recognition.py
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'decoding'))

import numpy as np
import time
import threading
import torch
from collections import deque
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QWidget, QLabel, QPushButton, QComboBox, QSlider, QCheckBox, QGroupBox, QGridLayout
from PyQt5.QtCore import QTimer, Qt, pyqtSignal as Signal, QThread, QMutex
import pyqtgraph as pg
import pyqtgraph.opengl as gl
from PyQt5 import QtCore
import json
from gl_text import GLTextItem

# Import utility functions
from graph_utilities import getBoxLinesCoords, get_trackColors
# from human_model_3d import HumanModelFactory  
# Import radar reader and action recognition system
from real_time_radar_reader import RealTimeRadarReader

USE_IMPROVED_SYSTEM = True  

# Action color mapping for 4D point cloud balanced model
ACTION_COLORS = {
    'stand': (0, 0, 1, 1),      # Blue - Stand
    'squat': (1, 1, 0, 1),      # Yellow - Squat
    'sit': (1, 0, 0, 1),        # Red - Sit
    'unknown': (0.5, 0.5, 0.5, 1)  # Gray - Unknown
}

class HumanTrackingVisualizer(QMainWindow):
    def __init__(self, cli_port='COM4', data_port='COM6'):
        super().__init__()
        
        self.cli_port = cli_port
        self.data_port = data_port
        
        # Radar reader - use directly
        self.radar_reader = None
        
        # Action recognition system - for getting classification results
        self.action_recognition = None
        
        # Data buffers
        self.track_history = {}  # Track historical trajectories
        self.max_history_length = 50  # Maximum history length
        
        # 3D visualization components
        self.gl_widget = None
        self.scatter_plot = None
        self.track_boxes = {}  # 3D boxes for tracked targets
        self.track_trails = {}  # Track trajectory lines
        self.human_models = {}  # Human models
        self.action_text_items = {}  # track_id: GLTextItem
        
        # Control parameters
        self.show_trails = True
        self.show_human_models = False  
        self.trail_length = 20
        self.point_size = 5
        self.human_model_type = "simple"  # "simple" or "detailed"
        
        # Color mapping
        self.track_colors = get_trackColors(20)
        
        # Current action state
        self.current_action = 'unknown'
        self.current_conf = 0.0
        
        # Statistics
        self.frame_count = 0
        self.frame_rate = 0
        self.last_update_time = time.time()
        
        # Thread safety
        self.data_mutex = QMutex()
        self.is_updating = False
        self.last_debug_time = 0
        self.debug_interval = 2.0  # Debug info interval (seconds)
        
        # Performance monitoring
        self.update_times = deque(maxlen=100)
        self.last_performance_check = time.time()
        
        # Initialize UI
        self.init_ui()
        
        # Start update timer - reduce frequency to avoid freezing
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_visualization)
        self.update_timer.start(100)  # 10 FPS, reduce frequency
        
    def init_ui(self):
        """Initialize user interface"""
        self.setWindowTitle("Radar Human Action Recognition 3D Visualizer")
        self.setGeometry(100, 100, 1200, 800)
        
        # Main window widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QHBoxLayout(central_widget)
        
        # Left control panel
        control_panel = self.create_control_panel()
        main_layout.addWidget(control_panel, 1)
        
        # Right 3D view
        self.gl_widget = gl.GLViewWidget()
        self.gl_widget.setCameraPosition(distance=10, elevation=20, azimuth=45)
        
        # Add coordinate axes
        axis = gl.GLAxisItem()
        axis.setSize(5, 5, 5)
        self.gl_widget.addItem(axis)
        
        # Add grid
        grid = gl.GLGridItem()
        grid.setSize(10, 10)
        grid.setSpacing(1, 1)
        self.gl_widget.addItem(grid)
        
        # Create scatter plot for point cloud display - use valid initial data to avoid OpenGL errors
        initial_pos = np.array([[0, 0, 0]])  # Use one initial point instead of empty array
        self.scatter_plot = gl.GLScatterPlotItem(pos=initial_pos, size=self.point_size, color=(0, 1, 0, 0.5))
        self.gl_widget.addItem(self.scatter_plot)
        
        # Create action label display
        self.action_text_item = None
        
        main_layout.addWidget(self.gl_widget, 4)
        
        # Status bar
        self.status_label = QLabel("Not connected")
        self.statusBar().addWidget(self.status_label)
        
    def create_control_panel(self):
        """Create control panel"""
        panel = QGroupBox("Control Panel")
        layout = QVBoxLayout(panel)
        
        # Connection control
        connection_group = QGroupBox("Connection Control")
        connection_layout = QGridLayout(connection_group)
        
        self.connect_btn = QPushButton("Connect Radar")
        self.connect_btn.clicked.connect(self.toggle_connection)
        connection_layout.addWidget(self.connect_btn, 0, 0)
        
        self.port_label = QLabel(f"CLI: {self.cli_port}, DATA: {self.data_port}")
        connection_layout.addWidget(self.port_label, 1, 0)
        
        layout.addWidget(connection_group)
        
        # Display control
        display_group = QGroupBox("Display Control")
        display_layout = QGridLayout(display_group)
        
        self.show_trails_cb = QCheckBox("Show Trails")
        self.show_trails_cb.setChecked(self.show_trails)
        self.show_trails_cb.toggled.connect(self.toggle_trails)
        display_layout.addWidget(self.show_trails_cb, 0, 0)
        
        self.show_models_cb = QCheckBox("Show Human Models")
        self.show_models_cb.setChecked(self.show_human_models)
        self.show_models_cb.toggled.connect(self.toggle_human_models)
        display_layout.addWidget(self.show_models_cb, 1, 0)
        
        # Trail length slider
        display_layout.addWidget(QLabel("Trail Length:"), 2, 0)
        self.trail_slider = QSlider(Qt.Horizontal)
        self.trail_slider.setRange(5, 100)
        self.trail_slider.setValue(self.trail_length)
        self.trail_slider.valueChanged.connect(self.change_trail_length)
        display_layout.addWidget(self.trail_slider, 3, 0)
        
        # Point size slider
        display_layout.addWidget(QLabel("Point Size:"), 4, 0)
        self.point_slider = QSlider(Qt.Horizontal)
        self.point_slider.setRange(1, 20)
        self.point_slider.setValue(self.point_size)
        self.point_slider.valueChanged.connect(self.change_point_size)
        display_layout.addWidget(self.point_slider, 5, 0)
        
        # Human model type selection
        display_layout.addWidget(QLabel("Human Model:"), 6, 0)
        self.model_type_combo = QComboBox()
        self.model_type_combo.addItems(["Simple Model", "Detailed Model"])
        self.model_type_combo.setCurrentText("Simple Model")
        self.model_type_combo.currentTextChanged.connect(self.change_model_type)
        display_layout.addWidget(self.model_type_combo, 7, 0)
        
        layout.addWidget(display_group)
        
        # Statistics
        stats_group = QGroupBox("Statistics")
        stats_layout = QGridLayout(stats_group)
        
        self.frame_count_label = QLabel("Frames: 0")
        stats_layout.addWidget(self.frame_count_label, 0, 0)
        
        self.frame_rate_label = QLabel("Frame Rate: 0 Hz")
        stats_layout.addWidget(self.frame_rate_label, 1, 0)
        
        # 添加醒目的动作显示区域
        action_display_group = QGroupBox("Action Recognition")
        action_display_layout = QGridLayout(action_display_group)
        
        self.action_label = QLabel("Current Action: UNKNOWN")
        self.action_label.setStyleSheet("font-size: 14px; font-weight: bold; color: red;")
        action_display_layout.addWidget(self.action_label, 0, 0)
        
        self.confidence_label = QLabel("Confidence: 0.000")
        self.confidence_label.setStyleSheet("font-size: 12px; color: orange;")
        action_display_layout.addWidget(self.confidence_label, 1, 0)
        
        # 添加动作状态指示器
        self.action_status_label = QLabel("Status: IDLE")
        self.action_status_label.setStyleSheet("font-size: 10px; color: gray;")
        action_display_layout.addWidget(self.action_status_label, 2, 0)
        
        layout.addWidget(action_display_group)
        
        self.radar_frame_label = QLabel("Radar Frames: 0")
        stats_layout.addWidget(self.radar_frame_label, 4, 0)
        
        self.radar_rate_label = QLabel("Radar Rate: 0 Hz")
        stats_layout.addWidget(self.radar_rate_label, 5, 0)
        
        # Performance monitoring
        self.performance_label = QLabel("Update Time: 0ms")
        stats_layout.addWidget(self.performance_label, 6, 0)
        
        layout.addWidget(stats_group)
        
        # Clear button
        self.clear_btn = QPushButton("Clear Display")
        self.clear_btn.clicked.connect(self.clear_display)
        layout.addWidget(self.clear_btn)
        
        # Debug button
        self.debug_btn = QPushButton("Debug Radar")
        self.debug_btn.clicked.connect(self.debug_radar)
        layout.addWidget(self.debug_btn)
        
        layout.addStretch()
        return panel
    
    def init_systems(self):
        """Initialize radar reader and action recognition system"""
        try:
            # 1. Create radar reader
            print("📡 Creating radar reader...")
            self.radar_reader = RealTimeRadarReader(self.cli_port, self.data_port)
            
            # 2. Create action recognition system (use same radar reader)
            print("🤖 Creating action recognition system...")
            
            # Get correct model file paths - use current directory model
            script_dir = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(script_dir, 'best_peter_model.pth')
            
            print(f"📁 PyTorch Model path: {model_path}")
            
            # Check if model file exists
            if not os.path.exists(model_path):
                print(f"❌ PyTorch model file does not exist: {model_path}")
                return False
            
            print(f"✅ Using PyTorch model: {model_path}")
            print(f"💡 模型特点: 使用融合点云数据训练，支持实时动作识别")
            
            # Create action recognition system using PyTorch model
            try:
                # Import the model class
                from peter_network import PETerNetwork
                import torch
                
                # Load the model
                self.model = PETerNetwork(num_classes=3, num_points=100, num_frames=25, k=10)
                if torch.cuda.is_available():
                    self.model.load_state_dict(torch.load(model_path, map_location='cuda'))
                    self.model = self.model.cuda()
                else:
                    self.model.load_state_dict(torch.load(model_path, map_location='cpu'))
                self.model.eval()
                
                print("✅ PyTorch model loaded successfully")
                print(f"  Supported actions: ['sit', 'squat', 'stand']")
                print(f"  Sequence length: 25 frames")
                print(f"  Data format: 3D radar point cloud frames [x, y, z]")
                
            except Exception as e:
                print(f"❌ Failed to load PyTorch model: {e}")
                return False
            
            # 3. Connect to radar
            print("🔌 Connecting to radar...")
            if not self.radar_reader.connect():
                self.status_label.setText("Radar connection failed")
                print("✗ Radar connection failed")
                return False
            
            # 4. Send configuration
            try:
                config_path = os.path.join(os.path.dirname(__file__), 'decoding', '3m.cfg')
                with open(config_path, 'r') as f:
                    config_lines = f.readlines()
                self.radar_reader.send_config(config_lines)
                print("✓ Radar configuration sent successfully")
            except FileNotFoundError:
                print("⚠ Configuration file not found, using default configuration")
            
            # 5. Start reading radar data
            print("📡 Starting radar data reading...")
            if not self.radar_reader.start_reading():
                self.status_label.setText("Radar data reading startup failed")
                print("✗ Radar data reading startup failed")
                return False
            
            # 6. Start action recognition system (using shared radar reader)
            print("🤖 Starting action recognition system...")
            
            # Initialize action recognition with PyTorch model
            self.action_labels = ['sit', 'squat', 'stand']
            self.current_action = 'unknown'
            self.current_conf = 0.0
            
            # Start the action recognition thread
            self.action_recognition_thread = threading.Thread(target=self.action_recognition_loop, daemon=True)
            self.action_recognition_thread.start()
            
            print(f"✓ Action recognition thread started")
            print(f"  Thread alive status: {self.action_recognition_thread.is_alive()}")
            
            # 7. Wait a few seconds for radar to start working
            print("⏳ Waiting for radar initialization...")
            time.sleep(3)
            
            # 8. Check initial status
            initial_stats = self.radar_reader.get_statistics()
            print(f"📊 Initial status: frame_count={initial_stats['frame_count']}, buffer_size={initial_stats['buffer_size']}")
            
            self.status_label.setText("System startup successful")
            print("✓ Radar data reading and action recognition system startup successful")
            return True
            
        except Exception as e:
            self.status_label.setText(f"System initialization failed: {e}")
            print(f"✗ System initialization failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def toggle_connection(self):
        """Toggle connection status"""
        if self.radar_reader and self.radar_reader.is_running:
            # System is running, show status
            self.status_label.setText("System running")
            self.connect_btn.setText("System Running")
            self.connect_btn.setEnabled(False)
        else:
            # System not running, try to connect
            self.status_label.setText("Initializing system...")
            self.connect_btn.setEnabled(False)
            
            try:
                # Initialize system
                if self.init_systems():
                    self.connect_btn.setText("System Running")
                    self.status_label.setText("System startup successful")
                    self.connect_btn.setEnabled(False)
                else:
                    self.connect_btn.setText("Connect System")
                    self.status_label.setText("Connection failed")
                    self.connect_btn.setEnabled(True)
            except Exception as e:
                self.status_label.setText(f"Connection exception: {e}")
                self.connect_btn.setText("Connect System")
                self.connect_btn.setEnabled(True)
                print(f"✗ System connection failed: {e}")
    
    def toggle_trails(self, checked):
        """Toggle trail display"""
        self.show_trails = checked
        if not checked:
            # Hide all trails
            for trail in self.track_trails.values():
                if trail in self.gl_widget.items:
                    self.gl_widget.removeItem(trail)
    
    def toggle_human_models(self, checked):
        """Toggle human model display"""
        self.show_human_models = checked
        if not checked:
            # Hide all human models
            for model in self.human_models.values():
                model.set_visible(False)
    
    def change_trail_length(self, value):
        """Change trail length"""
        self.trail_length = value
        # Update existing trails
        for track_id in list(self.track_history.keys()):
            if len(self.track_history[track_id]) > value:
                self.track_history[track_id] = self.track_history[track_id][-value:]
    
    def change_point_size(self, value):
        """Change point size"""
        self.point_size = value
        if self.scatter_plot:
            self.scatter_plot.setData(size=value)
    
    def change_model_type(self, text):
        """Change human model type"""
        if text == "Simple Model":
            self.human_model_type = "simple"
        else:
            self.human_model_type = "detailed"
        
        # Recreate all human models
        self.recreate_human_models()
    
    def recreate_human_models(self):
        """Recreate all human models"""

        print(" 3D Ban")
        return
        
        # Save current tracked target positions
        current_positions = {}
        for track_id, model in self.human_models.items():
            current_positions[track_id] = (model.x, model.y, model.z)
        
        # Remove all existing models
        for model in self.human_models.values():
            model.remove_from_widget(self.gl_widget)
        self.human_models.clear()
        
        # Recreate models
        for track_id, (x, y, z) in current_positions.items():
            color = self.track_colors[track_id % len(self.track_colors)]
            new_model = HumanModelFactory.create_model(self.human_model_type, track_id, color)
            new_model.update_position(x, y, z)
            new_model.add_to_widget(self.gl_widget)
            self.human_models[track_id] = new_model
    
    def clear_display(self):
        """Clear display"""
        # Clear trail history
        self.track_history.clear()
        
        # Remove all track boxes
        for box in self.track_boxes.values():
            if box in self.gl_widget.items:
                self.gl_widget.removeItem(box)
        self.track_boxes.clear()
        
        # Remove all trail lines
        for trail in self.track_trails.values():
            if trail in self.gl_widget.items:
                self.gl_widget.removeItem(trail)
        self.track_trails.clear()
        
        # Remove all human models
        for model in self.human_models.values():
            model.remove_from_widget(self.gl_widget)
        self.human_models.clear()
        
        # Remove all action labels
        for text_item in self.action_text_items.values():
            if text_item in self.gl_widget.items:
                self.gl_widget.removeItem(text_item)
        self.action_text_items.clear()
        
        # Clear scatter plot - use valid empty data to avoid OpenGL errors
        self.scatter_plot.setData(pos=np.array([[0, 0, 0]]))
    
    def create_human_model(self, x, y, z, track_id):
        """Create human 3D model"""
        print("⚠️ 3D Ban")
        return None
        
        color = ACTION_COLORS.get(self.current_action, (0.5, 0.5, 0.5, 1))
        model = HumanModelFactory.create_model(self.human_model_type, track_id, color)
        model.update_position(x, y, z)
        return model
    
    def debug_radar(self):
        """Debug radar status"""
        if not self.radar_reader:
            print("❌ Radar reader not initialized")
            return
        
        print("\n" + "="*60)
        print(" Radar Debug Information")
        print("="*60)
        
        # Get statistics
        stats = self.radar_reader.get_statistics()
        print(f" Radar Statistics:")
        print(f"  Frame count: {stats['frame_count']}")
        print(f"  Error count: {stats['error_count']}")
        print(f"  Frame rate: {stats['frame_rate']:.1f} Hz")
        print(f"  Queue size: {stats['queue_size']}")
        print(f"  Buffer size: {stats['buffer_size']}")
        print(f"  Connection status: {stats['is_connected']}")
        print(f"  Running status: {stats['is_running']}")
        
        # Check serial port status
        if self.radar_reader.cli_com:
            print(f" CLI port: {self.radar_reader.cli_com.port} - {'Connected' if self.radar_reader.cli_com.is_open else 'Not connected'}")
        if self.radar_reader.data_com:
            print(f" Data port: {self.radar_reader.data_com.port} - {'Connected' if self.radar_reader.data_com.is_open else 'Not connected'}")
        
        # Try multiple methods to get data
        print(f"\n🔍 Data retrieval test:")
        
        # Method 1: peek method
        frame_data = self.radar_reader.get_latest_frame_peek()
        if frame_data:
            print(f" Peek method successful: {list(frame_data.keys())}")
        else:
            print(f" Peek method failed")
        
        # Method 2: queue method
        try:
            frame_data = self.radar_reader.get_latest_frame()
            if frame_data:
                print(f" Queue method successful: {list(frame_data.keys())}")
            else:
                print(f" Queue method failed")
        except Exception as e:
            print(f" Queue method exception: {e}")
        
        # Method 3: buffer method
        frame_buffer = self.radar_reader.get_frame_buffer()
        if frame_buffer:
            print(f" Buffer method successful: {len(frame_buffer)} frames")
            if len(frame_buffer) > 0:
                latest_frame = frame_buffer[-1]
                print(f"  Latest frame keys: {list(latest_frame.keys())}")
        else:
            print(f" Buffer method failed")
        
        # Check feature sequence
        feature_sequence = self.radar_reader.get_latest_feature_sequence()
        if feature_sequence is not None:
            print(f" Feature sequence available: {feature_sequence.shape}")
            if feature_sequence.shape == (30, 6):
                print(f"    6D data format correct: [x, y, z, vx, vy, vz]")
                # Show feature ranges
                feature_ranges = np.ptp(feature_sequence, axis=0)
                print(f"   Feature ranges: x={feature_ranges[0]:.3f}, y={feature_ranges[1]:.3f}, z={feature_ranges[2]:.3f}")
                print(f"                    vx={feature_ranges[3]:.3f}, vy={feature_ranges[4]:.3f}, vz={feature_ranges[5]:.3f}")
            elif feature_sequence.shape[1] == 4:
                print(f"     4D data format: [y, z, vy, vz] - need to expand to 6D")
            else:
                print(f"    Unknown data format: {feature_sequence.shape}")
        else:
            print(f" Feature sequence not available")
        
        # Check data reading thread status
        if hasattr(self.radar_reader, 'reader_thread') and self.radar_reader.reader_thread:
            print(f" Data reading thread: {self.radar_reader.reader_thread.is_alive()}")
        else:
            print(f" Data reading thread: Not started")
        
        print("="*60)
    
    def update_visualization(self):
        """Update visualization"""
        # Prevent duplicate calls
        if self.is_updating:
            return
        
        self.is_updating = True
        start_time = time.time()
        
        try:
            # Performance monitoring
            current_time = time.time()
            if current_time - self.last_performance_check > 5.0:  # Check performance every 5 seconds
                if len(self.update_times) > 0:
                    avg_time = np.mean(self.update_times) * 1000
                    max_time = np.max(self.update_times) * 1000
                    self.performance_label.setText(f"Update time: {avg_time:.1f}ms (max: {max_time:.1f}ms)")
                self.last_performance_check = current_time
            
            # Check system status
            if not self.radar_reader or not self.radar_reader.is_running:
                self.status_label.setText("System not running")
                return

            # Get radar statistics
            radar_stats = self.radar_reader.get_statistics()
            
            # self.current_action 和 self.current_conf 由 action_recognition_loop 更新
            
            # Debug info - reduce frequency
            if current_time - self.last_debug_time > self.debug_interval:
                print(f"UI update status:")
                print(f"   Current action: {self.current_action}")
                print(f"   confidence level: {self.current_conf:.3f}")
                self.last_debug_time = current_time
            
            # Get radar data
            frame_data = None
            try:
                frame_data = self.radar_reader.get_latest_frame()
            except Exception as e:
                if current_time - self.last_debug_time > self.debug_interval:
                    print(f" Data retrieval exception: {e}")
                    self.last_debug_time = current_time
            
            # Debug info - reduce frequency
            if current_time - self.last_debug_time > self.debug_interval:
                print(f" 3D Visualizer status: frame_data={frame_data is not None}, action={self.current_action}, frames={self.frame_count}")
                self.last_debug_time = current_time
            
            if not frame_data:
                self.status_label.setText("Waiting for radar data...")
                return

            # Update statistics
            self.frame_count += 1
            now = time.time()
            if now - self.last_update_time > 0:
                self.frame_rate = 1.0 / (now - self.last_update_time)
            self.last_update_time = now
            
            # Update UI labels
            self.frame_count_label.setText(f"Frames: {self.frame_count}")
            self.frame_rate_label.setText(f"Frame Rate: {self.frame_rate:.1f} Hz")
            

            action_text = f"Current Action: {self.current_action.upper()}"
            if self.current_action == 'stand':
                self.action_label.setStyleSheet("font-size: 14px; font-weight: bold; color: blue;")
                self.action_status_label.setText("Status: STANDING")
                self.action_status_label.setStyleSheet("font-size: 10px; color: blue;")
            elif self.current_action == 'sit':
                self.action_label.setStyleSheet("font-size: 14px; font-weight: bold; color: red;")
                self.action_status_label.setText("Status: SITTING")
                self.action_status_label.setStyleSheet("font-size: 10px; color: red;")
            elif self.current_action == 'squat':
                self.action_label.setStyleSheet("font-size: 14px; font-weight: bold; color: yellow;")
                self.action_status_label.setText("Status: SQUATTING")
                self.action_status_label.setStyleSheet("font-size: 10px; color: yellow;")
            else:
                self.action_label.setStyleSheet("font-size: 14px; font-weight: bold; color: gray;")
                self.action_status_label.setText("Status: UNKNOWN")
                self.action_status_label.setStyleSheet("font-size: 10px; color: gray;")
            
            self.action_label.setText(action_text)
            

            confidence_text = f"Confidence: {self.current_conf:.3f}"
            if self.current_conf > 0.8:
                self.confidence_label.setStyleSheet("font-size: 12px; color: green; font-weight: bold;")
            elif self.current_conf > 0.6:
                self.confidence_label.setStyleSheet("font-size: 12px; color: orange; font-weight: bold;")
            else:
                self.confidence_label.setStyleSheet("font-size: 12px; color: red;")
            
            self.confidence_label.setText(confidence_text)
            
            self.radar_frame_label.setText(f"Radar Frames: {radar_stats['frame_count']}")
            self.radar_rate_label.setText(f"Radar Rate: {radar_stats['frame_rate']:.1f} Hz")

            # 1. Display point cloud (always display)
            try:
                if 'pointCloud' in frame_data and frame_data.get('numDetectedPoints', 0) > 0:
                    points = np.array(frame_data['pointCloud'][:frame_data.get('numDetectedPoints', 0)])
                    if len(points) > 0:
                        self.scatter_plot.setData(pos=points[:, :3], size=10, color=(0, 1, 0, 0.5))
                        center = np.mean(points[:, :3], axis=0)
                        self.status_label.setText(f"Point Cloud: {len(points)} | Center: ({center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f})")
                    else:
                        self.scatter_plot.setData(pos=np.array([[0,0,0]]), size=1)
                        self.status_label.setText("Point cloud empty")
                else:
                    # Show default status when no point cloud data
                    self.scatter_plot.setData(pos=np.array([[0,0,0]]), size=1)
                    self.status_label.setText("Waiting for point cloud data...")
            except Exception as e:
                if current_time - self.last_debug_time > self.debug_interval:
                    print(f"⚠️ Point cloud update exception: {e}")
                    self.last_debug_time = current_time
            
            # 2. Display trackData skeleton/box + action labels
            active_track_ids = set()
            try:
                if 'trackData' in frame_data and frame_data.get('numDetectedTracks', 0) > 0:
                    tracks = frame_data['trackData'][:frame_data.get('numDetectedTracks', 0)]
                    for track in tracks:
                        if len(track) >= 4:
                            track_id = int(track[0])
                            x, y, z = track[1], track[2], track[3]
                            active_track_ids.add(track_id)
                            
                            # --- Skeleton --- 
                            if self.show_human_models and track_id not in self.human_models:
                                try:
                                    
                                    print(" 3D Ban")
                                    # color = (0, 1, 1, 1)  # Cyan skeleton
                                    # model = HumanModelFactory.create_model(self.human_model_type, track_id, color)
                                    # model.add_to_widget(self.gl_widget)
                                    # self.human_models[track_id] = model
                                except Exception as e:
                                    print(f" 3D faild: {e}")
                                    self.show_human_models = False
                            
                            if self.show_human_models and track_id in self.human_models:
                                try:
                                    
                                    # self.human_models[track_id].update_position(x, y, z)
                                    # self.human_models[track_id].set_visible(True)
                                    pass
                                except Exception as e:
                                    print(f" 3D update faild: {e}")
                            
                            # --- Box ---
                            if track_id not in self.track_boxes:
                                box_lines = getBoxLinesCoords(x, y, z, 0.5, 1.7, 0.5)
                                box_item = gl.GLLinePlotItem(pos=box_lines, color=(1, 1, 0, 1), width=2, mode='lines')
                                self.gl_widget.addItem(box_item)
                                self.track_boxes[track_id] = box_item
                            else:
                                box_lines = getBoxLinesCoords(x, y, z, 0.5, 1.7, 0.5)
                                self.track_boxes[track_id].setData(pos=box_lines)
                            
                            # --- Action Label ---
                            if track_id == 1:  # Assume first tracked target is main target
                                # Set color based on action (use global ACTION_COLORS)
                                action_colors = ACTION_COLORS
                                label_color = action_colors.get(self.current_action, (0.5, 0.5, 0.5, 1))
                                
                                # Update status bar to show detailed information
                                status_text = f"Action: {self.current_action.upper()} | Confidence: {self.current_conf:.3f} | Position: ({x:.2f}, {y:.2f}, {z:.2f})"
                                self.status_label.setText(status_text)
                                
                                # Display action label in 3D scene (use large colored point marker)
                                label_pos = np.array([[x, y, z + 2.0]])  # 2 meters above skeleton head
                                
                                # Remove old label
                                if self.action_text_item is not None:
                                    self.gl_widget.removeItem(self.action_text_item)
                                
                                # Create new label (use large colored point)
                                self.action_text_item = gl.GLScatterPlotItem(
                                    pos=label_pos, 
                                    size=30,  # Larger point for better visibility
                                    color=label_color
                                )
                                self.gl_widget.addItem(self.action_text_item)
                                
                                # Add text label above the point
                                text_item = gl.GLTextItem(
                                    pos=(x, y, z + 2.5),  # Above the colored point
                                    text=f"{self.current_action.upper()}\n{self.current_conf:.3f}",
                                    color=label_color,
                                    font=pg.QtGui.QFont('Arial', 12)
                                )
                                self.gl_widget.addItem(text_item)
                                
                                print(f" Show action tags: {self.current_action.upper()} (confidence level: {self.current_conf:.3f})")
                        else:
                            if int(track[0]) in self.human_models:
                                self.human_models[int(track[0])].set_visible(False)
                            if int(track[0]) in self.track_boxes:
                                self.track_boxes[int(track[0])].setVisible(False)
                            if int(track[0]) in self.action_text_items:
                                self.action_text_items[int(track[0])].setVisible(False)
            except Exception as e:
                if current_time - self.last_debug_time > self.debug_interval:
                    print(f"⚠️ Track data update exception: {e}")
                    self.last_debug_time = current_time
            
            # Clean up disappeared target skeletons, boxes, labels
            try:
                for track_id in list(self.human_models.keys()):
                    if track_id not in active_track_ids:
                        self.human_models[track_id].set_visible(False)
                for track_id in list(self.track_boxes.keys()):
                    if track_id not in active_track_ids:
                        self.track_boxes[track_id].setVisible(False)
                for track_id in list(self.action_text_items.keys()):
                    if track_id not in active_track_ids:
                        text_item, _ = self.action_text_items[track_id]
                        if text_item in self.gl_widget.items:
                            self.gl_widget.removeItem(text_item)
                        del self.action_text_items[track_id]
            except Exception as e:
                if current_time - self.last_debug_time > self.debug_interval:
                    print(f" Cleanup operation exception: {e}")
                    self.last_debug_time = current_time
            
            # Record update time
            update_time = time.time() - start_time
            self.update_times.append(update_time)
            
        except Exception as e:
            print(f" Visualization update exception: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.is_updating = False
    
    def action_recognition_loop(self):

        print(" Action recognition loop started")
        
        while self.radar_reader and self.radar_reader.is_running:
            try:
                # Get latest frame data
                frame_data = self.radar_reader.get_latest_frame()
                if frame_data and 'pointCloud' in frame_data:
                    points = np.array(frame_data['pointCloud'][:frame_data.get('numDetectedPoints', 0)])
                    
                    print(f"🔍 Shape of raw point cloud data: {points.shape}")
                    
                    if len(points) > 0:

                        if points.shape[1] > 3:
                            print(f" Detected{points.shape[1]}Pointcloud,Use only the first 3 dimensions(xyz)")
                            points = points[:, :3]
                        
                        print(f" Processing point cloud data: {len(points)} poind")
                        
             
                        sequence = self.create_sequence(points)
                        
                        if sequence is not None:
                            print(f" Input sequence shape: {sequence.shape}")
                            
                            # Predict action
                            with torch.no_grad():
                                if torch.cuda.is_available():
                                    sequence = sequence.cuda()
                                
                                output = self.model(sequence)
                                probabilities = torch.softmax(output, dim=1)
                                predicted_class = torch.argmax(output, dim=1).item()
                                confidence = probabilities[0][predicted_class].item()
                                
                                # Update current action
                                self.current_action = self.action_labels[predicted_class]
                                self.current_conf = confidence
                                
                                print(f" Action recognition results: {self.current_action} (confidence level: {confidence:.3f})")
                        else:
                            print(" Unable to create a valid sequence")
                            self.current_action = 'unknown'
                            self.current_conf = 0.0
                    else:
                        print(" No point cloud data detected")
                        self.current_action = 'unknown'
                        self.current_conf = 0.0
                else:
                    print(" No radar data obtained")
                    self.current_action = 'unknown'
                    self.current_conf = 0.0
                
                time.sleep(0.1)  # 10 FPS
                
            except Exception as e:
                print(f"⚠️ Action recognition exception: {e}")
                import traceback
                traceback.print_exc()
                self.current_action = 'unknown'
                self.current_conf = 0.0
                time.sleep(0.1)
    
    def preprocess_points(self, points):

        if len(points) == 0:
            return np.zeros((100, 4)) 
        

        centroid = np.mean(points, axis=0)
        points = points - centroid
        

        max_dist = np.max(np.linalg.norm(points, axis=1))
        if max_dist > 0:
            points = points / max_dist
        

        if len(points) >= 100:
            indices = np.random.choice(len(points), 100, replace=False)
        else:
            indices = np.random.choice(len(points), 100, replace=True)
        
        sampled = points[indices]
        

        intensity = np.ones((100, 1))
        final_points = np.hstack([sampled, intensity])  
        
        return final_points
    
    def create_sequence(self, points):

        try:

            processed_points = self.preprocess_points(points)
            
 
            sequence = np.tile(processed_points, (25, 1, 1))  # (25, 100, 4)
            

            sequence = torch.FloatTensor(sequence).unsqueeze(0)  # (1, 25, 100, 4)
            
            return sequence
            
        except Exception as e:
            print(f" Failed to create sequence: {e}")
            return None
    
    def closeEvent(self, event):
        """Close event"""
        try:
            if hasattr(self, 'action_recognition_thread') and self.action_recognition_thread.is_alive():
                print(" Stopping action recognition thread...")
                # Note: Thread will stop automatically when radar_reader.is_running becomes False
            if self.radar_reader:
                self.radar_reader.stop_reading()
                self.radar_reader.disconnect()
        except Exception as e:
            print(f"Exception during close: {e}")
        event.accept()


def main():
    """Main function"""
    CLI_PORT = 'COM4'
    DATA_PORT = 'COM6'
    
    print("🚀 Starting 3D Action Recognition Visualization System")
    print(f"CLI port: {CLI_PORT}")
    print(f"Data port: {DATA_PORT}")
    
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    visualizer = HumanTrackingVisualizer(cli_port=CLI_PORT, data_port=DATA_PORT)
    visualizer.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main() 