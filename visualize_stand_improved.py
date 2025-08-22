#!/usr/bin/env python3
"""
Visualize point cloud data in radar_data directory (including improved_fused directory)
"""

import json
import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import glob

def load_pointcloud_data(file_path):
    """Load point cloud data file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"Failed to load file {file_path}: {e}")
        return None

def visualize_pointcloud_samples(data_dir, category_name, max_samples=3):
    """Visualize point cloud samples"""
    print(f"Visualizing directory: {data_dir}")
    
    # Find all JSON files
    json_files = glob.glob(os.path.join(data_dir, "*.json"))
    
    if not json_files:
        print(f"No JSON files found: {data_dir}")
        return
    
    print(f"Found {len(json_files)} files")
    
    # Limit number of samples to display
    json_files = json_files[:max_samples]
    
    for i, json_file in enumerate(json_files):
        print(f"\nProcessing file {i+1}/{len(json_files)}: {os.path.basename(json_file)}")
        
        # Load data
        data = load_pointcloud_data(json_file)
        if data is None:
            continue
        
        # Check data format
        if isinstance(data, list):
            print(f"  Data format: List, contains {len(data)} samples")
            
            # Visualize first few samples
            for j, sample in enumerate(data[:3]):  # Only show first 3 samples
                if isinstance(sample, dict) and 'target_points' in sample:
                    target_points = np.array(sample['target_points'])
                    noise_points = np.array(sample.get('noise_points', []))
                    
                    print(f"    Sample {j+1}: Target points {len(target_points)}, Noise points {len(noise_points)}")
                    
                    # Create visualization
                    fig = plt.figure(figsize=(15, 5))
                    
                    # Target point cloud
                    ax1 = fig.add_subplot(131, projection='3d')
                    if len(target_points) > 0:
                        ax1.scatter(target_points[:, 0], target_points[:, 1], target_points[:, 2], 
                                   c='red', marker='o', alpha=0.8, s=20, label='Target Points')
                    ax1.set_xlabel('X')
                    ax1.set_ylabel('Y')
                    ax1.set_zlabel('Z')
                    ax1.set_title(f'Target Points ({len(target_points)})')
                    ax1.legend()
                    
                    # Noise point cloud
                    ax2 = fig.add_subplot(132, projection='3d')
                    if len(noise_points) > 0:
                        ax2.scatter(noise_points[:, 0], noise_points[:, 1], noise_points[:, 2], 
                                   c='black', marker='x', alpha=0.6, s=10, label='Noise Points')
                    ax2.set_xlabel('X')
                    ax2.set_ylabel('Y')
                    ax2.set_zlabel('Z')
                    ax2.set_title(f'Noise Points ({len(noise_points)})')
                    ax2.legend()
                    
                    # Combined point cloud
                    ax3 = fig.add_subplot(133, projection='3d')
                    if len(target_points) > 0:
                        ax3.scatter(target_points[:, 0], target_points[:, 1], target_points[:, 2], 
                                   c='red', marker='o', alpha=0.8, s=20, label='Target')
                    if len(noise_points) > 0:
                        ax3.scatter(noise_points[:, 0], noise_points[:, 1], noise_points[:, 2], 
                                   c='black', marker='x', alpha=0.6, s=10, label='Noise')
                    ax3.set_xlabel('X')
                    ax3.set_ylabel('Y')
                    ax3.set_zlabel('Z')
                    ax3.set_title(f'Combined ({len(target_points) + len(noise_points)})')
                    ax3.legend()
                    
                    plt.suptitle(f'{category_name.capitalize()} Improved Fused - {os.path.basename(json_file)} - Sample {j+1}')
                    plt.tight_layout()
                    
                    # Save image
                    save_path = f"{category_name}_improved_visualization_{os.path.basename(json_file).replace('.json', '')}_sample{j+1}.png"
                    plt.savefig(save_path, dpi=300, bbox_inches='tight')
                    print(f"    Visualization saved: {save_path}")
                    
                    plt.show()
                    
                    # Display feature information
                    if 'target_features' in sample:
                        features = sample['target_features']
                        print(f"    Features: Points={features.get('num_points', 0)}, "
                              f"Density={features.get('point_density', 0):.2f}, "
                              f"Volume={features.get('spatial_volume', 0):.2f}")
                else:
                    print(f"    Sample {j+1}: Incorrect data format")
        else:
            print(f"  Data format: Non-list format")
            print(f"  Data type: {type(data)}")
            if isinstance(data, dict):
                print(f"  Dictionary keys: {list(data.keys())}")

def main():
    """Main function"""
    print("Visualizing Improved Fused Point Cloud Data")
    print("=" * 60)
    
    # Define improved_fused data directories
    improved_data_dirs = {
        'stand': 'radar_data/stand_improved_fused/improved_fused_pointcloud',
        'squat': 'radar_data/squat_improved_fused/improved_fused_pointcloud', 
        'sit': 'radar_data/sit_improved_fused/improved_fused_pointcloud'
    }
    
    for category, data_dir in improved_data_dirs.items():
        print(f"\n{'='*20} {category.upper()} IMPROVED FUSED {'='*20}")
        
        if not os.path.exists(data_dir):
            print(f"Directory does not exist: {data_dir}")
            continue
        
        # Visualize point cloud samples
        visualize_pointcloud_samples(data_dir, category, max_samples=2)  # Show 2 files per category

if __name__ == "__main__":
    main() 