#!/usr/bin/env python3
"""
Improved TI radar point cloud multi-frame fusion and clustering cleaner
Solves data loss issues and increases sample quantity
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

class ImprovedPointCloudFusionCleaner:
    def __init__(self, data_folder="radar_data"):

        self.data_folder = data_folder
        self.frames_per_sample = 6  
        self.min_frames_per_sample = 4  
        
    def load_pointcloud_data(self, file_path):

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return data
        except Exception as e:
            print(f" Failed to load file {file_path}: {e}")
            return []
    
    def extract_xyz_points(self, point_data):

        points = []
        for item in point_data:
            if 'x' in item and 'y' in item and 'z' in item:
                points.append([item['x'], item['y'], item['z']])
        
        return np.array(points) if points else np.empty((0, 3))
    
    def group_frames_by_time(self, point_data, time_window=0.3):

        if not point_data:
            return []

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
        

        if current_group:
            grouped_frames.append(current_group)
        
        return grouped_frames
    
    def multi_frame_fusion(self, frame_groups):

        all_points = []
        
        for frame_group in frame_groups:
            points = self.extract_xyz_points(frame_group)
            if len(points) > 0:
                all_points.append(points)
        
        if not all_points:
            return np.empty((0, 3))
        

        fused_points = np.vstack(all_points)
        return fused_points
    
    def dbscan_clustering(self, points, eps=1.0, min_samples=2):

        if len(points) == 0:
            return np.empty((0, 3)), np.empty((0, 3))
        

        scaler = StandardScaler()
        points_scaled = scaler.fit_transform(points)
        

        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(points_scaled)
        labels = clustering.labels_

        target_mask = labels != -1  
        noise_mask = labels == -1   
        
        target_points = points[target_mask]
        noise_points = points[noise_mask]
        
        return target_points, noise_points
    
    def calculate_density_features(self, points):

        if len(points) == 0:
            return {
                'num_points': 0,
                'point_density': 0.0,
                'spatial_volume': 0.0,
                'concentration_ratio': 0.0
            }
        

        num_points = len(points)
        

        x_range = np.max(points[:, 0]) - np.min(points[:, 0])
        y_range = np.max(points[:, 1]) - np.min(points[:, 1])
        z_range = np.max(points[:, 2]) - np.min(points[:, 2])
        

        spatial_volume = x_range * y_range * z_range
        

        point_density = num_points / (spatial_volume + 1e-6)
        

        center = np.mean(points, axis=0)
        

        distances = np.linalg.norm(points - center, axis=1)
        

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

        if len(frame_groups) < self.min_frames_per_sample:
            return None
        

        sample_frames = frame_groups[:min(len(frame_groups), self.frames_per_sample)]
        

        fused_points = self.multi_frame_fusion(sample_frames)
        
        if len(fused_points) == 0:
            return None
        

        target_points, noise_points = self.dbscan_clustering(fused_points)
        

        target_features = self.calculate_density_features(target_points)
        noise_features = self.calculate_density_features(noise_points)
        

        sample = {
            'sample_id': 0, 
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
    
    def create_improved_samples(self, frame_groups):

        samples = []
        

        for i in range(0, len(frame_groups) - self.min_frames_per_sample + 1):
            sample_frames = frame_groups[i:i + self.frames_per_sample]
            

            if len(sample_frames) >= self.min_frames_per_sample:
                sample = self.create_fused_sample(sample_frames)
                if sample:
                    sample['sample_id'] = len(samples)
                    samples.append(sample)
        
        return samples
    
    def clean_and_save_data(self, input_file, output_file, visualize=False):
  
        print(f" clean data: {input_file}")
        
        
        raw_data = self.load_pointcloud_data(input_file)
        if not raw_data:
            print(f"  The file is empty or failed to load.: {input_file}")
            return
        
        print(f"Raw data: {len(raw_data)} point")
        
       
        frame_groups = self.group_frames_by_time(raw_data, time_window=0.3)
        print(f" After grouping: {len(frame_groups)} time window")
        

        samples = self.create_improved_samples(frame_groups)
        print(f" Create sample: {len(samples)}  fusion sample")

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(samples, f, indent=2, ensure_ascii=False)
        
        print(f" Data saved after cleaning: {output_file}")
        
        return samples
    
    def process_all_data(self, visualize=False):
        print(" Start improved data fusion cleaning")
        print("=" * 60)
        

        data_dirs = ['sit', 'squat', 'stand']
        
        for data_dir in data_dirs:
            dir_path = os.path.join(self.data_folder, data_dir)
            if not os.path.exists(dir_path):
                print(f"  Folder does not exist: {dir_path}")
                continue
            
            print(f"\n Process folder: {data_dir}")
            

            pointcloud_dir = os.path.join(dir_path, "pointcloud")
            if os.path.exists(pointcloud_dir):
                json_files = glob.glob(os.path.join(pointcloud_dir, "*.json"))
            else:
                json_files = []
            
            if not json_files:
                print(f"  JSON file not found: {dir_path}")
                continue
            

            output_dir = os.path.join(self.data_folder, f"{data_dir}_improved_fused")
            os.makedirs(output_dir, exist_ok=True)
            
            total_samples = 0
            
            for json_file in json_files:

                rel_path = os.path.relpath(json_file, dir_path)
                output_file = os.path.join(output_dir, f"improved_fused_{rel_path}")
                

                os.makedirs(os.path.dirname(output_file), exist_ok=True)
                

                samples = self.clean_and_save_data(json_file, output_file, visualize)
                if samples:
                    total_samples += len(samples)
            
            print(f" {data_dir} Processing complete, total generated {total_samples} Improved fusion sample")
        
        print("\n All data improvements and integration cleaning completed")

def main():

    print(" Improved Point Cloud Fusion Cleaner")
    print("=" * 60)
    

    cleaner = ImprovedPointCloudFusionCleaner("radar_data")
    

    cleaner.process_all_data(visualize=False)

if __name__ == "__main__":
    main() 