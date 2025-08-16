#!/usr/bin/env python3
"""
可视化radar_data目录下的点云数据（包括improved_fused目录）
"""

import json
import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import glob

def load_pointcloud_data(file_path):
    """加载点云数据文件"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"❌ 加载文件失败 {file_path}: {e}")
        return None

def visualize_pointcloud_samples(data_dir, category_name, max_samples=3):
    """可视化点云样本"""
    print(f"📁 可视化目录: {data_dir}")
    
    # 查找所有JSON文件
    json_files = glob.glob(os.path.join(data_dir, "*.json"))
    
    if not json_files:
        print(f"⚠️  未找到JSON文件: {data_dir}")
        return
    
    print(f"📊 找到 {len(json_files)} 个文件")
    
    # 限制显示的样本数量
    json_files = json_files[:max_samples]
    
    for i, json_file in enumerate(json_files):
        print(f"\n📄 处理文件 {i+1}/{len(json_files)}: {os.path.basename(json_file)}")
        
        # 加载数据
        data = load_pointcloud_data(json_file)
        if data is None:
            continue
        
        # 检查数据格式
        if isinstance(data, list):
            print(f"  数据格式: 列表，包含 {len(data)} 个样本")
            
            # 可视化前几个样本
            for j, sample in enumerate(data[:3]):  # 只显示前3个样本
                if isinstance(sample, dict) and 'target_points' in sample:
                    target_points = np.array(sample['target_points'])
                    noise_points = np.array(sample.get('noise_points', []))
                    
                    print(f"    样本 {j+1}: 目标点 {len(target_points)}, 噪声点 {len(noise_points)}")
                    
                    # 创建可视化
                    fig = plt.figure(figsize=(15, 5))
                    
                    # 目标点云
                    ax1 = fig.add_subplot(131, projection='3d')
                    if len(target_points) > 0:
                        ax1.scatter(target_points[:, 0], target_points[:, 1], target_points[:, 2], 
                                   c='red', marker='o', alpha=0.8, s=20, label='Target Points')
                    ax1.set_xlabel('X')
                    ax1.set_ylabel('Y')
                    ax1.set_zlabel('Z')
                    ax1.set_title(f'Target Points ({len(target_points)})')
                    ax1.legend()
                    
                    # 噪声点云
                    ax2 = fig.add_subplot(132, projection='3d')
                    if len(noise_points) > 0:
                        ax2.scatter(noise_points[:, 0], noise_points[:, 1], noise_points[:, 2], 
                                   c='black', marker='x', alpha=0.6, s=10, label='Noise Points')
                    ax2.set_xlabel('X')
                    ax2.set_ylabel('Y')
                    ax2.set_zlabel('Z')
                    ax2.set_title(f'Noise Points ({len(noise_points)})')
                    ax2.legend()
                    
                    # 合并点云
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
                    
                    # 保存图片
                    save_path = f"{category_name}_improved_visualization_{os.path.basename(json_file).replace('.json', '')}_sample{j+1}.png"
                    plt.savefig(save_path, dpi=300, bbox_inches='tight')
                    print(f"    📊 可视化已保存: {save_path}")
                    
                    plt.show()
                    
                    # 显示特征信息
                    if 'target_features' in sample:
                        features = sample['target_features']
                        print(f"    特征: 点数={features.get('num_points', 0)}, "
                              f"密度={features.get('point_density', 0):.2f}, "
                              f"体积={features.get('spatial_volume', 0):.2f}")
                else:
                    print(f"    样本 {j+1}: 数据格式不正确")
        else:
            print(f"  数据格式: 非列表格式")
            print(f"  数据类型: {type(data)}")
            if isinstance(data, dict):
                print(f"  字典键: {list(data.keys())}")

def main():
    """主函数"""
    print("🎨 可视化Improved Fused点云数据")
    print("=" * 60)
    
    # 定义improved_fused数据目录
    improved_data_dirs = {
        'stand': 'radar_data/stand_improved_fused/improved_fused_pointcloud',
        'squat': 'radar_data/squat_improved_fused/improved_fused_pointcloud', 
        'sit': 'radar_data/sit_improved_fused/improved_fused_pointcloud'
    }
    
    for category, data_dir in improved_data_dirs.items():
        print(f"\n{'='*20} {category.upper()} IMPROVED FUSED {'='*20}")
        
        if not os.path.exists(data_dir):
            print(f"❌ 目录不存在: {data_dir}")
            continue
        
        # 可视化点云样本
        visualize_pointcloud_samples(data_dir, category, max_samples=2)  # 每个类别显示2个文件

if __name__ == "__main__":
    main() 