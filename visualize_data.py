import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

def visualize_action_samples():
    """可视化每个动作的样本点云"""
    # 加载数据
    df = pd.read_csv('extracted_3d_data.csv')
    
    # 为每个动作选择一个代表性样本
    fig = plt.figure(figsize=(15, 5))
    
    actions = ['sit', 'squat', 'stand']
    colors = ['red', 'green', 'blue']
    
    for i, (action, color) in enumerate(zip(actions, colors)):
        ax = fig.add_subplot(1, 3, i+1, projection='3d')
        
        # 选择该动作的一个文件的一帧数据
        action_data = df[df['action'] == action]
        sample_file = action_data['file_name'].iloc[0]
        sample_frame = action_data[
            (action_data['file_name'] == sample_file) & 
            (action_data['frame_num'] == action_data['frame_num'].iloc[0])
        ]
        
        if len(sample_frame) > 0:
            x = sample_frame['x'].values
            y = sample_frame['y'].values  
            z = sample_frame['z'].values
            
            ax.scatter(x, y, z, c=color, alpha=0.6, s=20)
            ax.set_title(f'{action.upper()} - {len(sample_frame)} points')
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
    
    plt.tight_layout()
    plt.savefig('action_samples.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_data_statistics():
    """绘制数据统计信息"""
    df = pd.read_csv('extracted_3d_data.csv')
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 动作分布
    action_counts = df['action'].value_counts()
    axes[0, 0].pie(action_counts.values, labels=action_counts.index, autopct='%1.1f%%')
    axes[0, 0].set_title('Action Distribution')
    
    # XYZ坐标分布
    axes[0, 1].hist(df['x'], bins=50, alpha=0.7, label='X', color='red')
    axes[0, 1].hist(df['y'], bins=50, alpha=0.7, label='Y', color='green')
    axes[0, 1].hist(df['z'], bins=50, alpha=0.7, label='Z', color='blue')
    axes[0, 1].set_title('Coordinate Distributions')
    axes[0, 1].set_xlabel('Value')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].legend()
    
    # 每个动作的点数分布
    action_point_counts = df.groupby(['action', 'file_name', 'frame_num']).size()
    for action in df['action'].unique():
        action_data = action_point_counts[action_point_counts.index.get_level_values(0) == action]
        axes[1, 0].hist(action_data.values, bins=30, alpha=0.5, label=action)
    axes[1, 0].set_title('Points per Frame by Action')
    axes[1, 0].set_xlabel('Number of Points')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].legend()
    
    # 空间分布（XY平面投影）
    for action, color in zip(['sit', 'squat', 'stand'], ['red', 'green', 'blue']):
        action_data = df[df['action'] == action].sample(1000)  # 采样减少点数
        axes[1, 1].scatter(action_data['x'], action_data['y'], 
                          alpha=0.5, s=1, label=action, c=color)
    axes[1, 1].set_title('Spatial Distribution (XY Projection)')
    axes[1, 1].set_xlabel('X')
    axes[1, 1].set_ylabel('Y')
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig('data_statistics.png', dpi=300, bbox_inches='tight')
    plt.show()

def visualize_temporal_sequence():
    """可视化时间序列数据"""
    df = pd.read_csv('extracted_3d_data.csv')
    
    # 选择一个文件的时间序列
    sample_file = df['file_name'].iloc[0]
    sample_action = df[df['file_name'] == sample_file]['action'].iloc[0]
    file_data = df[df['file_name'] == sample_file].sort_values('frame_num')
    
    # 计算每帧的质心
    frame_centroids = file_data.groupby('frame_num')[['x', 'y', 'z']].mean()
    
    fig = plt.figure(figsize=(12, 8))
    
    # 3D轨迹
    ax1 = fig.add_subplot(2, 2, 1, projection='3d')
    ax1.plot(frame_centroids['x'], frame_centroids['y'], frame_centroids['z'], 'b-o', alpha=0.7)
    ax1.set_title(f'3D Trajectory - {sample_action}')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    
    # XY轨迹
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.plot(frame_centroids['x'], frame_centroids['y'], 'r-o', alpha=0.7)
    ax2.set_title('XY Trajectory')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.grid(True)
    
    # 时间序列
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.plot(frame_centroids.index, frame_centroids['x'], label='X', alpha=0.7)
    ax3.plot(frame_centroids.index, frame_centroids['y'], label='Y', alpha=0.7)
    ax3.plot(frame_centroids.index, frame_centroids['z'], label='Z', alpha=0.7)
    ax3.set_title('Centroid Coordinates over Time')
    ax3.set_xlabel('Frame Number')
    ax3.set_ylabel('Coordinate Value')
    ax3.legend()
    ax3.grid(True)
    
    # 每帧点数
    points_per_frame = file_data.groupby('frame_num').size()
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.plot(points_per_frame.index, points_per_frame.values, 'g-o', alpha=0.7)
    ax4.set_title('Points per Frame')
    ax4.set_xlabel('Frame Number')
    ax4.set_ylabel('Number of Points')
    ax4.grid(True)
    
    plt.tight_layout()
    plt.savefig('temporal_sequence.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """主函数"""
    print("=== Visualizing 3D Point Cloud Data ===")
    
    print("1. Creating action sample visualizations...")
    visualize_action_samples()
    
    print("2. Creating data statistics plots...")
    plot_data_statistics()
    
    print("3. Creating temporal sequence visualization...")
    visualize_temporal_sequence()
    
    print("All visualizations saved as PNG files!")

if __name__ == "__main__":
    main() 