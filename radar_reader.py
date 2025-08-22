#!/usr/bin/env python3
"""
Radar Data Reader
Read IWR6843 radar data from COM4 and COM6 serial ports
"""

import serial
import time
import struct
import threading
import json
import os
import sys
from datetime import datetime

# Add decoding directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
decoding_dir = os.path.join(current_dir, 'decoding')
if decoding_dir not in sys.path:
    sys.path.insert(0, decoding_dir)

try:
    from parseFrame import parseStandardFrame
    from demo_defines import *
    from tlv_defines import *
except ImportError as e:
    print(f"Import error: {e}")
    print(f"Current directory: {current_dir}")
    print(f"Decoding directory: {decoding_dir}")
    raise

class RadarReader:
    def __init__(self, cli_com='COM4', data_com='COM6', baud_rate=921600):
        """
        Initialize radar reader
        Args:
            cli_com: CLI serial port name (COM4)
            data_com: Data serial port name (COM6)
            baud_rate: Baud rate
        """
        self.cli_com = cli_com
        self.data_com = data_com
        self.baud_rate = baud_rate
        
        # Serial connections
        self.cli_serial = None
        self.data_serial = None
        
        # Control flags
        self.is_running = False
        
        # Data storage
        self.data4 = []
        self.raw_frames = []  # Store raw frame data
        
        # Magic word
        self.MAGIC_WORD = b'\x02\x01\x04\x03\x06\x05\x08\x07'
        
        # Create data save folder
        self.create_data_folder()
        
        print(f"Radar reader initialization completed")
        print(f"CLI serial port: {cli_com}")
        print(f"Data serial port: {data_com}")
        print(f"Baud rate: {baud_rate}")
        
    def create_data_folder(self):
        """Create data save folder"""
        try:
            # Create main folder
            self.data_folder = "radar_data"
            if not os.path.exists(self.data_folder):
                os.makedirs(self.data_folder)
                print(f"Created data folder: {self.data_folder}")
            
            # Create timestamp subfolder
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.session_folder = os.path.join(self.data_folder, f"session_{timestamp}")
            os.makedirs(self.session_folder)
            print(f"Created session folder: {self.session_folder}")
            
            # Create subfolders
            self.pointcloud_folder = os.path.join(self.session_folder, "pointcloud")
            self.tracking_folder = os.path.join(self.session_folder, "tracking")
            self.raw_folder = os.path.join(self.session_folder, "raw")
            
            os.makedirs(self.pointcloud_folder)
            os.makedirs(self.tracking_folder)
            os.makedirs(self.raw_folder)
            
            print(f"Created subfolders:")
            print(f"   Point cloud data: {self.pointcloud_folder}")
            print(f"   Tracking data: {self.tracking_folder}")
            print(f"   Raw data: {self.raw_folder}")
            
        except Exception as e:
            print(f"Failed to create folders: {e}")
            # Use current directory as fallback
            self.session_folder = "."
            self.pointcloud_folder = "."
            self.tracking_folder = "."
            self.raw_folder = "."
        
    def connect_ports(self):
        """Connect serial ports"""
        try:
            print(f"\nConnecting serial ports...")
            
            # Connect CLI serial port
            print(f"   Connecting CLI serial port: {self.cli_com}")
            self.cli_serial = serial.Serial(
                self.cli_com, 
                115200,  # CLI serial port usually uses 115200 baud rate
                timeout=1
            )
            print(f"   CLI serial port connected successfully")
            
            # Connect data serial port
            print(f"   Connecting data serial port: {self.data_com}")
            self.data_serial = serial.Serial(
                self.data_com, 
                self.baud_rate,  # Data serial port uses 921600 baud rate
                timeout=1
            )
            print(f"   Data serial port connected successfully")
            
            print(f"All serial ports connected successfully")
            return True
            
        except Exception as e:
            print(f"Serial port connection failed: {e}")
            return False
    
    def load_config_file(self, config_path):
        """Load configuration file"""
        try:
            with open(config_path, 'r') as f:
                return [line.strip() for line in f.readlines()]
        except Exception as e:
            print(f"Failed to load configuration file: {e}")
            return []
    
    def send_config(self, config_lines):
        """Send configuration commands to CLI serial port - using gui_parser.py method"""
        try:
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
                
                print(f"   {line.strip()} -> {ack.decode().strip()}")
                
                # Read second line acknowledgment
                ack = self.cli_serial.readline()
                if ack:
                    print(f"   Acknowledgment: {ack.decode().strip()}")
            
            # Give buffer some time to clear
            time.sleep(0.03)
            self.cli_serial.reset_input_buffer()
            
            print(f"Configuration file sent successfully")
            return True
            
        except Exception as e:
            print(f"Failed to send configuration file: {e}")
            return False
    
    def disconnect_ports(self):
        """Disconnect serial ports"""
        if self.cli_serial and self.cli_serial.is_open:
            self.cli_serial.close()
        if self.data_serial and self.data_serial.is_open:
            self.data_serial.close()
        print("Serial ports disconnected")
    
    def read_frame(self, serial_port):
        """Read one frame of data - using gui_parser.py method"""
        try:
            # Find magic word
            index = 0
            magicByte = serial_port.read(1)
            frameData = bytearray(b'')
            
            while True:
                # Check if there is data
                if len(magicByte) < 1:
                    print("Serial port timeout, no data")
                    return None
                
                # Find matching byte
                if magicByte[0] == self.MAGIC_WORD[index]:
                    index += 1
                    frameData.append(magicByte[0])
                    if index == 8:  # Found complete magic word
                        break
                    magicByte = serial_port.read(1)
                else:
                    # Reset index
                    if index == 0:
                        magicByte = serial_port.read(1)
                    index = 0
                    frameData = bytearray(b'')
            
            # Read version number
            versionBytes = serial_port.read(4)
            frameData += bytearray(versionBytes)
            
            # Read length
            lengthBytes = serial_port.read(4)
            frameData += bytearray(lengthBytes)
            frameLength = int.from_bytes(lengthBytes, byteorder='little')
            
            # Subtract already read bytes (magic word, version, length)
            frameLength -= 16
            
            # Read the rest of the frame
            frameData += bytearray(serial_port.read(frameLength))
            
            return frameData
            
        except Exception as e:
            print(f"Failed to read frame data: {e}")
            return None
    
    def parse_frame(self, frame_data):
        """Parse frame data"""
        try:
            parsed_data = parseStandardFrame(frame_data)
            return parsed_data
        except Exception as e:
            print(f"Failed to parse frame data: {e}")
            return None
    
    def read_port_data(self, serial_port, port_name, data_list):
        """Read data from single serial port"""
        print(f"Starting to read {port_name} data...")
        
        frame_count = 0
        last_debug_time = time.time()
        
        while self.is_running:
            try:
                frame_data = self.read_frame(serial_port)
                if frame_data:
                    frame_count += 1
                    print(f"Read frame {frame_count}, length: {len(frame_data)} bytes")
                    
                    # Save raw frame data
                    raw_frame_info = {
                        'timestamp': time.time(),
                        'frame_number': frame_count,
                        'frame_length': len(frame_data),
                        'frame_data_hex': frame_data.hex()
                    }
                    self.raw_frames.append(raw_frame_info)
                    
                    # Parse data
                    parsed_data = self.parse_frame(frame_data)
                    if parsed_data:
                        print(f"Parsed successfully, contains {len(parsed_data)} TLVs")
                        
                        # Extract point cloud data
                        if 'pointCloud' in parsed_data:
                            points = parsed_data['pointCloud']
                            num_points = parsed_data.get('numDetectedPoints', 0)
                            print(f"Detected {num_points} point cloud points")
                            
                            if num_points > 0:
                                print(f"Point cloud data example:")
                                for i in range(min(3, num_points)):  # Only show first 3 points
                                    if i < len(points):
                                        point = points[i]
                                        if len(point) >= 3:
                                            print(f"   Point{i+1}: x={point[0]:.2f}, y={point[1]:.2f}, z={point[2]:.2f}")
                            
                            for i in range(num_points):
                                if i < len(points):
                                    point = points[i]
                                    if len(point) >= 3:
                                        coord = {
                                            'timestamp': time.time(),
                                            'port': port_name,
                                            'x': float(point[0]),
                                            'y': float(point[1]),
                                            'z': float(point[2]),
                                            'doppler': float(point[3]) if len(point) > 3 else 0.0,
                                            'snr': float(point[4]) if len(point) > 4 else 0.0,
                                            'noise': float(point[5]) if len(point) > 5 else 0.0
                                        }
                                        data_list.append(coord)
                        else:
                            print(f"Point cloud data not found")
                            print(f"Available data keys: {list(parsed_data.keys())}")
                        
                        # Extract tracking data
                        if 'trackData' in parsed_data:
                            tracks = parsed_data['trackData']
                            num_tracks = parsed_data.get('numDetectedTracks', 0)
                            print(f"Detected {num_tracks} tracking targets")
                            
                            for i in range(num_tracks):
                                if i < len(tracks):
                                    track = tracks[i]
                                    track_info = {
                                        'timestamp': time.time(),
                                        'port': port_name,
                                        'track_id': int(track[0]),
                                        'pos_x': float(track[1]),
                                        'pos_y': float(track[2]),
                                        'pos_z': float(track[3]),
                                        'vel_x': float(track[4]),
                                        'vel_y': float(track[5]),
                                        'vel_z': float(track[6])
                                    }
                                    print(f"Tracking target {track_info['track_id']}: position=({track_info['pos_x']:.2f}, {track_info['pos_y']:.2f}, {track_info['pos_z']:.2f})")
                    else:
                        print(f"Frame data parsing failed")
                else:
                    # Print debug info every 10 seconds
                    current_time = time.time()
                    if current_time - last_debug_time >= 10:
                        print(f"Waiting for data... Serial port status: {serial_port.in_waiting} bytes available")
                        last_debug_time = current_time
                
                time.sleep(0.01)  # Brief sleep
                
            except Exception as e:
                print(f"{port_name} read exception: {e}")
                time.sleep(0.1)
        
        print(f"{port_name} read thread stopped")
    
    def start_reading(self):
        """Start reading data"""
        if not self.data_serial:
            print("Data serial port not connected")
            return False
        
        self.is_running = True
        
        # Start data reading thread
        thread = threading.Thread(
            target=self.read_port_data,
            args=(self.data_serial, 'RADAR', self.data4)  # Use data4 to store all data
        )
        thread.daemon = True
        thread.start()
        
        print("Data reading started")
        return True
    
    def stop_reading(self):
        """Stop reading data"""
        self.is_running = False
        print("Stopping data reading...")
        time.sleep(2)  # Wait for thread to end
    
    def save_data(self, filename_prefix="radar_data"):
        """Save data to file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Separate point cloud data and tracking data
        pointcloud_data = []
        tracking_data = []
        
        for item in self.data4:
            if 'track_id' in item:
                tracking_data.append(item)
            else:
                pointcloud_data.append(item)
        
        # Save point cloud data
        if pointcloud_data:
            filename = os.path.join(self.pointcloud_folder, f"pointcloud_{timestamp}.json")
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(pointcloud_data, f, indent=2, ensure_ascii=False)
            print(f"Point cloud data saved: {filename} ({len(pointcloud_data)} points)")
        
        # Save tracking data
        if tracking_data:
            filename = os.path.join(self.tracking_folder, f"tracking_{timestamp}.json")
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(tracking_data, f, indent=2, ensure_ascii=False)
            print(f"Tracking data saved: {filename} ({len(tracking_data)} targets)")
        
        # Save complete data
        if self.data4:
            filename = os.path.join(self.session_folder, f"complete_data_{timestamp}.json")
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(self.data4, f, indent=2, ensure_ascii=False)
            print(f"Complete data saved: {filename} ({len(self.data4)} data points)")
        
        # Save raw frame data
        if self.raw_frames:
            filename = os.path.join(self.raw_folder, f"raw_frames_{timestamp}.json")
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(self.raw_frames, f, indent=2, ensure_ascii=False)
            print(f"Raw frame data saved: {filename} ({len(self.raw_frames)} frames)")
    
    def print_statistics(self):
        """Print statistics"""
        print("\n=== Data Statistics ===")
        
        # Separate point cloud data and tracking data
        pointcloud_count = sum(1 for item in self.data4 if 'track_id' not in item)
        tracking_count = sum(1 for item in self.data4 if 'track_id' in item)
        
        print(f"Total data points: {len(self.data4)}")
        print(f"Point cloud data: {pointcloud_count} points")
        print(f"Tracking data: {tracking_count} targets")
        print(f"Raw frames: {len(self.raw_frames)} frames")
        print("=" * 30)

def main():
    """Main function"""
    print("Radar Data Reader")
    print("=" * 50)
    
    # Create reader
    reader = RadarReader(
        cli_com='COM4',
        data_com='COM6',
        baud_rate=921600
    )
    
    try:
        # Connect serial ports
        if not reader.connect_ports():
            print("Serial port connection failed, program exiting")
            return
        
        # Load and send configuration file
        config_path = os.path.join(decoding_dir, '3m.cfg')
        config_lines = reader.load_config_file(config_path)
        if config_lines:
            if not reader.send_config(config_lines):
                print("Configuration file sending failed")
                return
        else:
            print("Configuration file loading failed")
            return
        
        # Wait for radar to start
        print("Waiting for radar to start...")
        time.sleep(2)
        
        # Check if data serial port has data
        print("Checking data serial port status...")
        if reader.data_serial.in_waiting > 0:
            print(f"Data serial port has {reader.data_serial.in_waiting} bytes of data")
            # Read first few bytes to see
            test_data = reader.data_serial.read(min(16, reader.data_serial.in_waiting))
            print(f"First 16 bytes: {test_data.hex()}")
        else:
            print("Data serial port has no data yet")
        
        # Wait longer for radar to start sending data
        print("Waiting for radar to start sending data...")
        time.sleep(5)
        
        # Check data serial port again
        if reader.data_serial.in_waiting > 0:
            print(f"After 5 seconds, data serial port has {reader.data_serial.in_waiting} bytes of data")
        else:
            print("After 5 seconds, data serial port still has no data, radar may not have started correctly")
        
        # Start reading data
        if not reader.start_reading():
            print("Failed to start data reading")
            return
        
        print("\nReading data... (Press Ctrl+C to stop)")
        print("=" * 50)
        
        # Main loop
        start_time = time.time()
        last_stats_time = start_time
        
        while True:
            try:
                time.sleep(5)  # Check every 5 seconds
                current_time = time.time()
                
                # Print statistics every 30 seconds
                if current_time - last_stats_time >= 30:
                    reader.print_statistics()
                    last_stats_time = current_time
                
                # Save data every 60 seconds
                if current_time - start_time >= 60:
                    reader.save_data()
                    start_time = current_time
                    
            except KeyboardInterrupt:
                print("\nReceived stop signal...")
                break
            except Exception as e:
                print(f"Main loop exception: {e}")
                break
    
    except Exception as e:
        print(f"Program exception: {e}")
    
    finally:
        # Clean up resources
        reader.stop_reading()
        reader.save_data()  # Final save
        reader.disconnect_ports()
        print("Program exited")

if __name__ == "__main__":
    main() 