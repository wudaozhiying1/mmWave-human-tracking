# Radar Point Cloud Action Recognition System

A real-time human action recognition system using TI IWR6843 radar sensor and PETer (Point cloud Edge Transformer) neural network architecture. This system can classify three human actions: sitting, squatting, and standing.

## Demo

Watch the system in action: [YouTube Demo](https://youtu.be/M7bTbZctuMI)

## Project Overview

This project implements a complete pipeline for radar-based human action recognition:

1. **Real-time radar data acquisition** from TI IWR6843 sensor
2. **Multi-frame point cloud fusion** and clustering-based cleaning
3. **PETer neural network** for action classification
4. **Real-time visualization** and monitoring

## Project Structure

```
├── radar_reader.py              # Radar data acquisition from serial ports
├── pointcloud_fusion_cleaner.py # Multi-frame fusion and DBSCAN clustering
├── train_peter.py              # PETer model training script
├── real_time_test.py           # Real-time action recognition testing
├── realtime_3d_display.py      # Real-time 3D visualization
├── test_serial.py              # Serial port testing utility
├── visualize_data.py           # Data visualization and analysis
├── visualize_stand_improved.py # Improved fused data visualization
├── peter_network.py            # PETer neural network architecture
├── decoding/                   # Radar data parsing modules
│   ├── parseFrame.py
│   ├── demo_defines.py
│   ├── tlv_defines.py
│   └── 3m.cfg                  # Radar configuration file
├── radar_data/                 # Data storage directory
│   ├── sit_improved_fused/
│   ├── squat_improved_fused/
│   └── stand_improved_fused/
└── README.md                   # This file
```

## Quick Start

### Prerequisites

1. **Hardware Requirements:**
   - TI IWR6843 radar sensor
   - USB-to-UART converter (for CLI port)
   - USB-to-UART converter (for data port)

2. **Software Requirements:**
   ```bash
   pip install torch torchvision
   pip install numpy matplotlib seaborn
   pip install scikit-learn
   pip install pyserial
   pip install pandas
   ```

### Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd radar-action-recognition
   ```

2. **Connect the radar sensor:**
   - CLI port: COM4 (115200 baud rate)
   - Data port: COM6 (921600 baud rate)

3. **Test serial connection:**
   ```bash
   python test_serial.py
   ```

## Data Collection and Processing

### 1. Collect Raw Radar Data

```bash
python radar_reader.py
```

This script will:
- Connect to radar sensor via serial ports
- Send configuration commands
- Collect point cloud and tracking data
- Save data in organized folders with timestamps

### 2. Process and Clean Data

```bash
python pointcloud_fusion_cleaner.py
```

This script implements:
- **Multi-frame fusion**: Combines 6 consecutive frames
- **DBSCAN clustering**: Separates target points from noise
- **Feature extraction**: Calculates density and spatial features
- **Data organization**: Creates improved fused datasets

### 3. Visualize Processed Data

```bash
python visualize_stand_improved.py
python visualize_data.py
```

These scripts provide:
- 3D point cloud visualization
- Data statistics and distributions
- Temporal sequence analysis

## Model Training

### Train PETer Model

```bash
python train_peter.py
```

**Training Features:**
- **Data augmentation**: Rotation, noise, scaling
- **Class balancing**: Weighted loss function
- **Early stopping**: Prevents overfitting
- **Learning rate scheduling**: Optimizes convergence
- **Cross-validation**: Ensures model robustness

**Model Architecture:**
- **EdgeConv layers**: Local feature extraction
- **Transformer encoder**: Global feature learning
- **Classification head**: Action prediction

**Training Output:**
- `best_peter_model.pth`: Trained model weights
- `training_history.png`: Training curves
- `confusion_matrix.png`: Model performance

## Real-time Recognition

### 1. Real-time Testing

```bash
python real_time_test.py
```

**Features:**
- Real-time radar data reading
- 6-frame fusion for stability
- Live action prediction
- Confidence scoring
- Category balancing

### 2. 3D Visualization

```bash
python realtime_3d_display.py
```

**Features:**
- Real-time 3D point cloud display
- Action probability visualization
- System status monitoring
- Prediction history tracking

## Performance Metrics

The system achieves:
- **Accuracy**: >90% on test set
- **Latency**: <500ms for real-time prediction
- **Stability**: Robust to noise and variations
- **Scalability**: Supports multiple action classes

## Configuration

### Radar Configuration (`decoding/3m.cfg`)

Key parameters:
- **Range**: 0.5-3.0 meters
- **Frame rate**: 10 FPS
- **Point cloud density**: ~100 points per frame
- **Detection sensitivity**: Optimized for human targets

### Model Configuration

```python
# PETer Network Parameters
num_classes = 3          # sit, squat, stand
num_points = 100         # Points per frame
num_frames = 25          # Temporal sequence length
k = 10                  # K-nearest neighbors for EdgeConv
```

## Usage Examples

### Basic Data Collection

```python
from radar_reader import RadarReader

# Initialize reader
reader = RadarReader(cli_port='COM4', data_port='COM6')

# Connect and collect data
if reader.connect_radar():
    reader.start_reading()
    # Data will be saved automatically
```

### Real-time Recognition

```python
from real_time_test import RealTimeRadarReader, RealTimeActionRecognizer

# Initialize components
radar_reader = RealTimeRadarReader()
action_recognizer = RealTimeActionRecognizer()

# Start recognition
radar_reader.connect_radar()
radar_reader.start_reading()

# Get predictions
prediction = action_recognizer.get_prediction()
print(f"Action: {prediction['action']}, Confidence: {prediction['confidence']:.3f}")
```

### Data Visualization

```python
from visualize_data import visualize_action_samples, plot_data_statistics

# Visualize sample point clouds
visualize_action_samples()

# Plot data statistics
plot_data_statistics()
```

## Troubleshooting

### Common Issues

1. **Serial Port Connection Failed**
   - Check COM port numbers
   - Verify USB drivers are installed
   - Ensure radar sensor is powered on

2. **No Data Received**
   - Check radar configuration file
   - Verify sensor is in correct mode
   - Monitor serial port status

3. **Model Training Issues**
   - Ensure sufficient training data
   - Check class balance
   - Verify data preprocessing

4. **Real-time Performance Issues**
   - Reduce frame rate if needed
   - Optimize model parameters
   - Check system resources

### Debug Commands

```bash
# Test serial ports
python test_serial.py

# Check data files
python visualize_stand_improved.py

# Monitor system performance
python realtime_3d_display.py
```

## Technical Details

### Data Processing Pipeline

1. **Raw Data Acquisition**
   - Serial port communication
   - Frame parsing and validation
   - Point cloud extraction

2. **Multi-frame Fusion**
   - Temporal alignment
   - Point cloud merging
   - Feature preservation

3. **Clustering and Cleaning**
   - DBSCAN clustering
   - Noise removal
   - Target point extraction

4. **Feature Engineering**
   - Spatial features
   - Density features
   - Temporal features

### Model Architecture

The PETer network combines:
- **EdgeConv**: Local geometric feature learning
- **Transformer**: Global temporal dependencies
- **Attention mechanisms**: Focus on relevant features
- **Multi-head classification**: Robust prediction

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- TI for IWR6843 radar sensor
- PETer paper authors for the network architecture
- Open source community for supporting libraries

## Support

For questions and support:
- Create an issue on GitHub
- Check the troubleshooting section
- Review the documentation

---

**Note**: This system is designed for research and educational purposes. Ensure proper safety measures when using radar sensors in real-world applications. 