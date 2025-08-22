#!/usr/bin/env python3
"""
Multi-frame Fusion and Clustering Cleaner for TI Radar Point Cloud
Reference Paper: Human Action Recognition Dataset and Method Based on Millimeter Wave Radar 3D Point Cloud
"""

import json
import os
import numpy as np
from datetime import datetime
import glob
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class PointCloudFusionCleaner:
    def __init__(self, data_folder="radar_data"):
        """
        Initialize point cloud fusion cleaner
        Args:
            data_folder: Data folder path
        """
        self.data_folder = data_folder
        self.frames_per_sample = 6  # 6 frames fused into one sample
        
    def load_pointcloud_data(self, file_path):
        """Load point cloud data file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return data
        except Exception as e:
            print(f"Failed to load file {file_path}: {e}")
            return []
    
    def extract_xyz_points(self, point_data):
        """
        Extract x,y,z coordinates from point cloud data
        Args:
            point_data: Point cloud data list
        Returns:
            points: numpy array with shape (n_points, 3)
        """
        points = []
        for item in point_data:
            if 'x' in item and 'y' in item and 'z' in item:
                points.append([item['x'], item['y'], item['z']])
        
        return np.array(points) if points else np.empty((0, 3))
    
    def group_frames_by_time(self, point_data, time_window=0.1):
        """
        Group frame data by time window (shorter time window to maintain frame continuity)
        Args:
            point_data: Point cloud data list
            time_window: Time window (seconds)
        Returns:
            grouped_frames: Grouped frame data
        """
        if not point_data:
            return []
        
        # Sort by timestamp
        sorted_data = sorted(point_data, key=lambda x: x.get('timestamp', 0))
        
        grouped_frames = []
        current_group = []
        last_timestamp = None
        
        for item in sorted_data:
            timestamp = item.get('timestamp', 0)
            
            if last_timestamp is None or (timestamp - last_timestamp) <= time_window:
                current_group.append(item)
            else:
                if current_group:
                    grouped_frames.append(current_group)
                current_group = [item]
            
            last_timestamp = timestamp
        
        # Add the last group
        if current_group:
            grouped_frames.append(current_group)
        
        return grouped_frames
    
    def multi_frame_fusion(self, frame_groups):
        """
        Multi-frame fusion: fuse 6 frames of point cloud data
        Args:
            frame_groups: Grouped frame data
        Returns:
            fused_points: Fused point cloud data
        """
        all_points = []
        
        for frame_group in frame_groups:
            points = self.extract_xyz_points(frame_group)
            if len(points) > 0:
                all_points.append(points)
        
        if not all_points:
            return np.empty((0, 3))
        
        # Merge point clouds from all frames
        fused_points = np.vstack(all_points)
        return fused_points
    
    def dbscan_clustering(self, points, eps=0.5, min_samples=3):
        """
        DBSCAN clustering to separate targets and noise points
        Args:
            points: Point cloud data with shape (n_points, 3)
            eps: Neighborhood radius
            min_samples: Minimum number of samples
        Returns:
            target_points: Target point cloud
            noise_points: Noise point cloud
        """
        if len(points) == 0:
            return np.empty((0, 3)), np.empty((0, 3))
        
        # Standardize data
        scaler = StandardScaler()
        points_scaled = scaler.fit_transform(points)
        
        # DBSCAN clustering
        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(points_scaled)
        labels = clustering.labels_
        
        # Separate targets and noise points
        target_mask = labels != -1  # Non-noise points
        noise_mask = labels == -1   # Noise points
        
        target_points = points[target_mask]
        noise_points = points[noise_mask]
        
        return target_points, noise_points
    
    def calculate_density_features(self, points):
        """
        Calculate point cloud density features
        Args:
            points: Point cloud data
        Returns:
            features: Density feature dictionary
        """
        if len(points) == 0:
            return {
                'num_points': 0,
                'point_density': 0.0,
                'spatial_volume': 0.0,
                'concentration_ratio': 0.0
            }
        
        # Basic statistics
        num_points = len(points)
        
        # Calculate spatial range
        x_range = np.max(points[:, 0]) - np.min(points[:, 0])
        y_range = np.max(points[:, 1]) - np.min(points[:, 1])
        z_range = np.max(points[:, 2]) - np.min(points[:, 2])
        
        # Spatial volume
        spatial_volume = x_range * y_range * z_range
        
        # Point cloud density (points/volume)
        point_density = num_points / (spatial_volume + 1e-6)
        
        # Calculate point cloud center
        center = np.mean(points, axis=0)
        
        # Calculate distance to center
        distances = np.linalg.norm(points - center, axis=1)
        
        # Concentration ratio (proportion of close points)
        concentration_ratio = np.sum(distances < np.mean(distances)) / num_points
        
        return {
            'num_points': num_points,
            'point_density': float(point_density),
            'spatial_volume': float(spatial_volume),
            'concentration_ratio': float(concentration_ratio),
            'x_range': float(x_range),
            'y_range': float(y_range),
            'z_range': float(z_range),
            'center_x': float(center[0]),
            'center_y': float(center[1]),
            'center_z': float(center[2])
        }
    
    def create_fused_sample(self, frame_groups):
        """
        Create fused sample
        Args:
            frame_groups: Grouped frame data
        Returns:
            sample: Fused sample
        """
        if len(frame_groups) < self.frames_per_sample:
            return None
        
        # Take first 6 frames for fusion
        sample_frames = frame_groups[:self.frames_per_sample]
        
        # Multi-frame fusion
        fused_points = self.multi_frame_fusion(sample_frames)
        
        if len(fused_points) == 0:
            return None
        
        # DBSCAN clustering
        target_points, noise_points = self.dbscan_clustering(fused_points)
        
        # Calculate features
        target_features = self.calculate_density_features(target_points)
        noise_features = self.calculate_density_features(noise_points)
        
        # Create sample
        sample = {
            'sample_id': 0,  # Will be set externally
            'start_timestamp': sample_frames[0][0].get('timestamp', 0),
            'end_timestamp': sample_frames[-1][-1].get('timestamp', 0),
            'num_frames': len(sample_frames),
            'target_points': target_points.tolist() if len(target_points) > 0 else [],
            'noise_points': noise_points.tolist() if len(noise_points) > 0 else [],
            'target_features': target_features,
            'noise_features': noise_features,
            'fusion_ratio': len(target_points) / (len(target_points) + len(noise_points) + 1e-6)
        }
        
        return sample
    
    def visualize_fusion_result(self, original_points, fused_points, target_points, noise_points, save_path=None):
        """
        Visualize fusion results (similar to Figure 10 in the paper)
        Args:
            original_points: Original single frame point cloud
            fused_points: Fused point cloud
            target_points: Target point cloud
            noise_points: Noise point cloud
            save_path: Save path
        """
        fig = plt.figure(figsize=(15, 6))
        
        # Original single frame point cloud
        ax1 = fig.add_subplot(121, projection='3d')
        if len(original_points) > 0:
            ax1.scatter(original_points[:, 0], original_points[:, 1], original_points[:, 2], 
                       c='blue', marker='o', alpha=0.6, label='Original Points')
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')
        ax1.set_title('(a) Single Frame Point Cloud')
        ax1.legend()
        
        # Fused and clustered point cloud
        ax2 = fig.add_subplot(122, projection='3d')
        if len(target_points) > 0:
            ax2.scatter(target_points[:, 0], target_points[:, 1], target_points[:, 2], 
                       c='red', marker='o', alpha=0.8, label='Target')
        if len(noise_points) > 0:
            ax2.scatter(noise_points[:, 0], noise_points[:, 1], noise_points[:, 2], 
                       c='black', marker='x', alpha=0.6, label='Noise')
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.set_zlabel('Z')
        ax2.set_title('(b) Fused and Clustered Point Cloud')
        ax2.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Visualization result saved: {save_path}")
        
        plt.show()
    
    def clean_and_save_data(self, input_file, output_file, visualize=False):
        """
        Clean and save data
        Args:
            input_file: Input file path
            output_file: Output file path
            visualize: Whether to generate visualization
        """
        print(f"Cleaning data: {input_file}")
        
        # Load original data
        raw_data = self.load_pointcloud_data(input_file)
        if not raw_data:
            print(f"File is empty or failed to load: {input_file}")
            return
        
        print(f"Original data: {len(raw_data)} data points")
        
        # Group by time
        frame_groups = self.group_frames_by_time(raw_data)
        print(f"After grouping: {len(frame_groups)} time windows")
        
        # Create fused samples
        samples = []
        for i in range(0, len(frame_groups), self.frames_per_sample):
            sample_frames = frame_groups[i:i + self.frames_per_sample]
            
            if len(sample_frames) == self.frames_per_sample:
                # Get original single frame point cloud for comparison
                original_points = self.extract_xyz_points(sample_frames[0])
                
                # Create fused sample
                sample = self.create_fused_sample(sample_frames)
                if sample:
                    sample['sample_id'] = len(samples)
                    samples.append(sample)
                    
                    # Visualize first sample
                    if visualize and len(samples) == 1:
                        fused_points = np.array(sample['target_points'] + sample['noise_points'])
                        target_points = np.array(sample['target_points'])
                        noise_points = np.array(sample['noise_points'])
                        
                        viz_path = output_file.replace('.json', '_visualization.png')
                        self.visualize_fusion_result(original_points, fused_points, target_points, noise_points, viz_path)
        
        print(f"Created samples: {len(samples)} fused samples")
        
        # Save cleaned data
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(samples, f, indent=2, ensure_ascii=False)
        
        print(f"Cleaned data saved: {output_file}")
        
        return samples
    
    def process_all_data(self, visualize=False):
        """Process all data folders"""
        print("Starting to clean all radar data based on paper method")
        print("=" * 60)
        
        # Find all data folders
        data_dirs = ['sit', 'squat', 'stand']
        
        for data_dir in data_dirs:
            dir_path = os.path.join(self.data_folder, data_dir)
            if not os.path.exists(dir_path):
                print(f"Folder does not exist: {dir_path}")
                continue
            
            print(f"\nProcessing folder: {data_dir}")
            
            # Find all JSON files
            json_files = glob.glob(os.path.join(dir_path, "**/*.json"), recursive=True)
            
            if not json_files:
                print(f"No JSON files found: {dir_path}")
                continue
            
            # Create output folder
            output_dir = os.path.join(self.data_folder, f"{data_dir}_fused")
            os.makedirs(output_dir, exist_ok=True)
            
            total_samples = 0
            
            for json_file in json_files:
                # Generate output filename
                rel_path = os.path.relpath(json_file, dir_path)
                output_file = os.path.join(output_dir, f"fused_{rel_path}")
                
                # Ensure output directory exists
                os.makedirs(os.path.dirname(output_file), exist_ok=True)
                
                # Clean data
                samples = self.clean_and_save_data(json_file, output_file, visualize)
                if samples:
                    total_samples += len(samples)
            
            print(f"{data_dir} processing completed, generated {total_samples} fused samples")
        
        print("\nAll data fusion cleaning completed!")

    def process_stand_data(self, visualize=False):
        """Specifically process stand data"""
        print("Starting to process stand point cloud data")
        print("=" * 60)
        
        # Stand original data directory
        stand_dir = os.path.join(self.data_folder, 'stand', 'pointcloud')
        if not os.path.exists(stand_dir):
            print(f"Stand directory does not exist: {stand_dir}")
            return
        
        # Output directory
        output_dir = os.path.join(self.data_folder, 'stand_data', 'fused_pointcloud')
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"Input directory: {stand_dir}")
        print(f"Output directory: {output_dir}")
        
        # Find all JSON files
        json_files = glob.glob(os.path.join(stand_dir, "*.json"))
        
        if not json_files:
            print(f"No JSON files found: {stand_dir}")
            return
        
        print(f"Found {len(json_files)} point cloud files")
        
        total_samples = 0
        
        for json_file in json_files:
            # Generate output filename
            base_name = os.path.basename(json_file)
            output_file = os.path.join(output_dir, base_name)
            
            print(f"\nProcessing file: {base_name}")
            
            # Clean data
            samples = self.clean_and_save_data(json_file, output_file, visualize)
            if samples:
                total_samples += len(samples)
                print(f"Generated {len(samples)} fused samples")
        
        print(f"\nStand data processing completed!")
        print(f"Total generated {total_samples} fused samples")
        print(f"Output directory: {output_dir}")
        
        return total_samples

def main():
    """Main function"""
    print("Point Cloud Fusion Cleaner Based on Paper Method")
    print("=" * 60)
    
    # Create cleaner
    cleaner = PointCloudFusionCleaner("radar_data")
    
    # Specifically process stand data
    cleaner.process_stand_data(visualize=True)

if __name__ == "__main__":
    main() 