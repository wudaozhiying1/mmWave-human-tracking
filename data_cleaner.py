#!/usr/bin/env python3
"""
雷达数据清洗器
清洗点云数据，保留x,y,z和密度信息，6帧融合为一个样本
"""

import json
import os
import numpy as np
from datetime import datetime
import glob

class RadarDataCleaner:
    def __init__(self, data_folder="radar_data"):
        """
        初始化数据清洗器
        Args:
            data_folder: 数据文件夹路径
        """
        self.data_folder = data_folder
        self.frames_per_sample = 6  # 6帧融合为一个样本
        
    def load_pointcloud_data(self, file_path):
        """加载点云数据文件"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return data
        except Exception as e:
            print(f"❌ 加载文件失败 {file_path}: {e}")
            return []
    
    def extract_pointcloud_features(self, point_data):
        """
        从点云数据中提取特征
        Args:
            point_data: 单帧点云数据
        Returns:
            features: 包含x,y,z和密度信息的特征字典
        """
        if not point_data:
            return None
        
        # 提取所有点的坐标
        points = []
        for item in point_data:
            if 'x' in item and 'y' in item and 'z' in item:
                points.append([item['x'], item['y'], item['z']])
        
        if not points:
            return None
        
        points = np.array(points)
        
        # 计算特征
        features = {
            'timestamp': point_data[0].get('timestamp', 0),
            'num_points': len(points),  # 点云密度
            'x_mean': float(np.mean(points[:, 0])),
            'y_mean': float(np.mean(points[:, 1])),
            'z_mean': float(np.mean(points[:, 2])),
            'x_std': float(np.std(points[:, 0])),
            'y_std': float(np.std(points[:, 1])),
            'z_std': float(np.std(points[:, 2])),
            'x_min': float(np.min(points[:, 0])),
            'y_min': float(np.min(points[:, 1])),
            'z_min': float(np.min(points[:, 2])),
            'x_max': float(np.max(points[:, 0])),
            'y_max': float(np.max(points[:, 1])),
            'z_max': float(np.max(points[:, 2])),
            'spatial_range': float(np.max(points) - np.min(points)),  # 空间范围
            'point_density': len(points) / (np.max(points) - np.min(points) + 1e-6)  # 点云密度
        }
        
        return features
    
    def group_frames_by_time(self, point_data, time_window=1.0):
        """
        按时间窗口分组帧数据
        Args:
            point_data: 点云数据列表
            time_window: 时间窗口（秒）
        Returns:
            grouped_frames: 分组后的帧数据
        """
        if not point_data:
            return []
        
        # 按时间戳排序
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
        
        # 添加最后一组
        if current_group:
            grouped_frames.append(current_group)
        
        return grouped_frames
    
    def create_6_frame_sample(self, frame_groups):
        """
        创建6帧融合样本
        Args:
            frame_groups: 分组后的帧数据
        Returns:
            samples: 6帧融合的样本列表
        """
        samples = []
        
        for i in range(0, len(frame_groups), self.frames_per_sample):
            sample_frames = frame_groups[i:i + self.frames_per_sample]
            
            if len(sample_frames) == self.frames_per_sample:
                # 提取每帧的特征
                frame_features = []
                for frame_group in sample_frames:
                    features = self.extract_pointcloud_features(frame_group)
                    if features:
                        frame_features.append(features)
                
                if len(frame_features) == self.frames_per_sample:
                    # 创建融合样本
                    sample = {
                        'sample_id': len(samples),
                        'start_timestamp': frame_features[0]['timestamp'],
                        'end_timestamp': frame_features[-1]['timestamp'],
                        'frames': frame_features,
                        # 计算整体特征
                        'total_points': sum(f['num_points'] for f in frame_features),
                        'avg_density': np.mean([f['point_density'] for f in frame_features]),
                        'avg_spatial_range': np.mean([f['spatial_range'] for f in frame_features])
                    }
                    samples.append(sample)
        
        return samples
    
    def clean_and_save_data(self, input_file, output_file):
        """
        清洗并保存数据
        Args:
            input_file: 输入文件路径
            output_file: 输出文件路径
        """
        print(f"🧹 清洗数据: {input_file}")
        
        # 加载原始数据
        raw_data = self.load_pointcloud_data(input_file)
        if not raw_data:
            print(f"⚠️  文件为空或加载失败: {input_file}")
            return
        
        print(f"📊 原始数据: {len(raw_data)} 个数据点")
        
        # 按时间分组
        frame_groups = self.group_frames_by_time(raw_data)
        print(f"📦 分组后: {len(frame_groups)} 个时间窗口")
        
        # 创建6帧样本
        samples = self.create_6_frame_sample(frame_groups)
        print(f"🎯 创建样本: {len(samples)} 个6帧样本")
        
        # 保存清洗后的数据
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(samples, f, indent=2, ensure_ascii=False)
        
        print(f"💾 清洗后数据已保存: {output_file}")
        
        return samples
    
    def process_all_data(self):
        """处理所有数据文件夹"""
        print("🚀 开始清洗所有雷达数据")
        print("=" * 50)
        
        # 查找所有数据文件夹
        data_dirs = ['sit', 'squat', 'stand']
        
        for data_dir in data_dirs:
            dir_path = os.path.join(self.data_folder, data_dir)
            if not os.path.exists(dir_path):
                print(f"⚠️  文件夹不存在: {dir_path}")
                continue
            
            print(f"\n📁 处理文件夹: {data_dir}")
            
            # 查找所有JSON文件
            json_files = glob.glob(os.path.join(dir_path, "**/*.json"), recursive=True)
            
            if not json_files:
                print(f"⚠️  未找到JSON文件: {dir_path}")
                continue
            
            # 创建输出文件夹
            output_dir = os.path.join(self.data_folder, f"{data_dir}_cleaned")
            os.makedirs(output_dir, exist_ok=True)
            
            total_samples = 0
            
            for json_file in json_files:
                # 生成输出文件名
                rel_path = os.path.relpath(json_file, dir_path)
                output_file = os.path.join(output_dir, f"cleaned_{rel_path}")
                
                # 确保输出目录存在
                os.makedirs(os.path.dirname(output_file), exist_ok=True)
                
                # 清洗数据
                samples = self.clean_and_save_data(json_file, output_file)
                if samples:
                    total_samples += len(samples)
            
            print(f"✅ {data_dir} 处理完成，共生成 {total_samples} 个样本")
        
        print("\n🎉 所有数据清洗完成！")

def main():
    """主函数"""
    print("🧹 雷达数据清洗器")
    print("=" * 50)
    
    # 创建清洗器
    cleaner = RadarDataCleaner("radar_data")
    
    # 处理所有数据
    cleaner.process_all_data()

if __name__ == "__main__":
    main() 