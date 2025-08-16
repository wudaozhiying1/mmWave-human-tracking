#!/usr/bin/env python3
"""
基于TI雷达点云的多帧融合与聚类清洗器
参考论文：基于毫米波雷达三维点云的人体动作识别数据集与方法
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
        初始化点云融合清洗器
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
    
    def extract_xyz_points(self, point_data):
        """
        从点云数据中提取x,y,z坐标
        Args:
            point_data: 点云数据列表
        Returns:
            points: numpy数组，形状为(n_points, 3)
        """
        points = []
        for item in point_data:
            if 'x' in item and 'y' in item and 'z' in item:
                points.append([item['x'], item['y'], item['z']])
        
        return np.array(points) if points else np.empty((0, 3))
    
    def group_frames_by_time(self, point_data, time_window=0.1):
        """
        按时间窗口分组帧数据（更短的时间窗口以保持帧的连续性）
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
    
    def multi_frame_fusion(self, frame_groups):
        """
        多帧融合：将6帧的点云数据融合
        Args:
            frame_groups: 分组后的帧数据
        Returns:
            fused_points: 融合后的点云数据
        """
        all_points = []
        
        for frame_group in frame_groups:
            points = self.extract_xyz_points(frame_group)
            if len(points) > 0:
                all_points.append(points)
        
        if not all_points:
            return np.empty((0, 3))
        
        # 将所有帧的点云合并
        fused_points = np.vstack(all_points)
        return fused_points
    
    def dbscan_clustering(self, points, eps=0.5, min_samples=3):
        """
        DBSCAN聚类，分离目标和噪声点
        Args:
            points: 点云数据，形状为(n_points, 3)
            eps: 邻域半径
            min_samples: 最小样本数
        Returns:
            target_points: 目标点云
            noise_points: 噪声点云
        """
        if len(points) == 0:
            return np.empty((0, 3)), np.empty((0, 3))
        
        # 标准化数据
        scaler = StandardScaler()
        points_scaled = scaler.fit_transform(points)
        
        # DBSCAN聚类
        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(points_scaled)
        labels = clustering.labels_
        
        # 分离目标和噪声点
        target_mask = labels != -1  # 非噪声点
        noise_mask = labels == -1   # 噪声点
        
        target_points = points[target_mask]
        noise_points = points[noise_mask]
        
        return target_points, noise_points
    
    def calculate_density_features(self, points):
        """
        计算点云密度特征
        Args:
            points: 点云数据
        Returns:
            features: 密度特征字典
        """
        if len(points) == 0:
            return {
                'num_points': 0,
                'point_density': 0.0,
                'spatial_volume': 0.0,
                'concentration_ratio': 0.0
            }
        
        # 基本统计
        num_points = len(points)
        
        # 计算空间范围
        x_range = np.max(points[:, 0]) - np.min(points[:, 0])
        y_range = np.max(points[:, 1]) - np.min(points[:, 1])
        z_range = np.max(points[:, 2]) - np.min(points[:, 2])
        
        # 空间体积
        spatial_volume = x_range * y_range * z_range
        
        # 点云密度（点数/体积）
        point_density = num_points / (spatial_volume + 1e-6)
        
        # 计算点云中心
        center = np.mean(points, axis=0)
        
        # 计算到中心的距离
        distances = np.linalg.norm(points - center, axis=1)
        
        # 集中度比率（近距离点的比例）
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
        创建融合样本
        Args:
            frame_groups: 分组后的帧数据
        Returns:
            sample: 融合后的样本
        """
        if len(frame_groups) < self.frames_per_sample:
            return None
        
        # 取前6帧进行融合
        sample_frames = frame_groups[:self.frames_per_sample]
        
        # 多帧融合
        fused_points = self.multi_frame_fusion(sample_frames)
        
        if len(fused_points) == 0:
            return None
        
        # DBSCAN聚类
        target_points, noise_points = self.dbscan_clustering(fused_points)
        
        # 计算特征
        target_features = self.calculate_density_features(target_points)
        noise_features = self.calculate_density_features(noise_points)
        
        # 创建样本
        sample = {
            'sample_id': 0,  # 将在外部设置
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
        可视化融合结果（类似论文Figure 10）
        Args:
            original_points: 原始单帧点云
            fused_points: 融合后的点云
            target_points: 目标点云
            noise_points: 噪声点云
            save_path: 保存路径
        """
        fig = plt.figure(figsize=(15, 6))
        
        # 原始单帧点云
        ax1 = fig.add_subplot(121, projection='3d')
        if len(original_points) > 0:
            ax1.scatter(original_points[:, 0], original_points[:, 1], original_points[:, 2], 
                       c='blue', marker='o', alpha=0.6, label='Original Points')
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')
        ax1.set_title('(a) 单帧点云 (Single Frame Point Cloud)')
        ax1.legend()
        
        # 融合聚类后的点云
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
        ax2.set_title('(b) 融合聚类后的点云 (Fused and Clustered Point Cloud)')
        ax2.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 可视化结果已保存: {save_path}")
        
        plt.show()
    
    def clean_and_save_data(self, input_file, output_file, visualize=False):
        """
        清洗并保存数据
        Args:
            input_file: 输入文件路径
            output_file: 输出文件路径
            visualize: 是否生成可视化
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
        
        # 创建融合样本
        samples = []
        for i in range(0, len(frame_groups), self.frames_per_sample):
            sample_frames = frame_groups[i:i + self.frames_per_sample]
            
            if len(sample_frames) == self.frames_per_sample:
                # 获取原始单帧点云用于对比
                original_points = self.extract_xyz_points(sample_frames[0])
                
                # 创建融合样本
                sample = self.create_fused_sample(sample_frames)
                if sample:
                    sample['sample_id'] = len(samples)
                    samples.append(sample)
                    
                    # 可视化第一个样本
                    if visualize and len(samples) == 1:
                        fused_points = np.array(sample['target_points'] + sample['noise_points'])
                        target_points = np.array(sample['target_points'])
                        noise_points = np.array(sample['noise_points'])
                        
                        viz_path = output_file.replace('.json', '_visualization.png')
                        self.visualize_fusion_result(original_points, fused_points, target_points, noise_points, viz_path)
        
        print(f"🎯 创建样本: {len(samples)} 个融合样本")
        
        # 保存清洗后的数据
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(samples, f, indent=2, ensure_ascii=False)
        
        print(f"💾 清洗后数据已保存: {output_file}")
        
        return samples
    
    def process_all_data(self, visualize=False):
        """处理所有数据文件夹"""
        print("🚀 开始基于论文方法清洗所有雷达数据")
        print("=" * 60)
        
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
            output_dir = os.path.join(self.data_folder, f"{data_dir}_fused")
            os.makedirs(output_dir, exist_ok=True)
            
            total_samples = 0
            
            for json_file in json_files:
                # 生成输出文件名
                rel_path = os.path.relpath(json_file, dir_path)
                output_file = os.path.join(output_dir, f"fused_{rel_path}")
                
                # 确保输出目录存在
                os.makedirs(os.path.dirname(output_file), exist_ok=True)
                
                # 清洗数据
                samples = self.clean_and_save_data(json_file, output_file, visualize)
                if samples:
                    total_samples += len(samples)
            
            print(f"✅ {data_dir} 处理完成，共生成 {total_samples} 个融合样本")
        
        print("\n🎉 所有数据融合清洗完成！")

    def process_stand_data(self, visualize=False):
        """专门处理stand数据"""
        print("🚀 开始处理stand点云数据")
        print("=" * 60)
        
        # stand原始数据目录
        stand_dir = os.path.join(self.data_folder, 'stand', 'pointcloud')
        if not os.path.exists(stand_dir):
            print(f"⚠️  stand目录不存在: {stand_dir}")
            return
        
        # 输出目录
        output_dir = os.path.join(self.data_folder, 'stand_data', 'fused_pointcloud')
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"📁 输入目录: {stand_dir}")
        print(f"📁 输出目录: {output_dir}")
        
        # 查找所有JSON文件
        json_files = glob.glob(os.path.join(stand_dir, "*.json"))
        
        if not json_files:
            print(f"⚠️  未找到JSON文件: {stand_dir}")
            return
        
        print(f"📊 找到 {len(json_files)} 个点云文件")
        
        total_samples = 0
        
        for json_file in json_files:
            # 生成输出文件名
            base_name = os.path.basename(json_file)
            output_file = os.path.join(output_dir, base_name)
            
            print(f"\n📄 处理文件: {base_name}")
            
            # 清洗数据
            samples = self.clean_and_save_data(json_file, output_file, visualize)
            if samples:
                total_samples += len(samples)
                print(f"✅ 生成 {len(samples)} 个融合样本")
        
        print(f"\n🎉 stand数据处理完成！")
        print(f"📊 总共生成 {total_samples} 个融合样本")
        print(f"📁 输出目录: {output_dir}")
        
        return total_samples

def main():
    """主函数"""
    print("🧹 基于论文方法的点云融合清洗器")
    print("=" * 60)
    
    # 创建清洗器
    cleaner = PointCloudFusionCleaner("radar_data")
    
    # 专门处理stand数据
    cleaner.process_stand_data(visualize=True)

if __name__ == "__main__":
    main() 