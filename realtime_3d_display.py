#!/usr/bin/env python3
"""
实时3D点云与人体动作识别显示脚本
专门用于实时显示点云数据和动作识别结果
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
import threading
import queue
from collections import deque
from peter_network import PETerNetwork
import torch
import warnings
warnings.filterwarnings('ignore')

class RealTime3DDisplay:
    """实时3D点云与动作识别显示器"""
    
    def __init__(self, model_path='best_peter_model.pth', num_points=100, num_frames=6):
        self.num_points = num_points
        self.num_frames = num_frames
        self.action_labels = ['sit', 'squat', 'stand']
        
        # 帧缓冲区
        self.frame_buffer = deque(maxlen=num_frames)
        self.data_queue = queue.Queue()
        self.is_running = False
        
        # 加载模型
        self.model = PETerNetwork(num_classes=3, num_points=num_points, num_frames=25, k=10)
        if torch.cuda.is_available():
            self.model.load_state_dict(torch.load(model_path, map_location='cuda'))
            self.model = self.model.cuda()
        else:
            self.model.load_state_dict(torch.load(model_path, map_location='cpu'))
        
        self.model.eval()
        print(f"✅ 模型加载完成: {model_path}")
        
        # 动作颜色映射
        self.action_colors = {
            'sit': 'red',
            'squat': 'green', 
            'stand': 'blue'
        }
        
        # 动作描述
        self.action_descriptions = {
            'sit': '坐姿',
            'squat': '蹲姿',
            'stand': '站姿'
        }
        
        # 预测历史
        self.prediction_history = deque(maxlen=10)
        self.last_prediction = None
        
        # 创建可视化界面
        self.create_display()
    
    def create_display(self):
        """创建显示界面"""
        self.fig = plt.figure(figsize=(16, 10))
        self.fig.suptitle('实时3D点云与人体动作识别', fontsize=16, fontweight='bold')
        
        # 3D点云显示
        self.ax1 = self.fig.add_subplot(221, projection='3d')
        self.ax1.set_xlabel('X轴')
        self.ax1.set_ylabel('Y轴')
        self.ax1.set_zlabel('Z轴')
        self.ax1.set_title('实时3D点云')
        self.ax1.grid(True, alpha=0.3)
        
        # 动作概率显示
        self.ax2 = self.fig.add_subplot(222)
        self.ax2.set_title('动作识别概率')
        self.ax2.set_ylim(0, 1)
        self.ax2.set_ylabel('概率')
        self.ax2.grid(True, alpha=0.3)
        
        # 状态信息显示
        self.ax3 = self.fig.add_subplot(223)
        self.ax3.set_title('系统状态')
        self.ax3.axis('off')
        
        # 预测历史显示
        self.ax4 = self.fig.add_subplot(224)
        self.ax4.set_title('预测历史')
        self.ax4.set_ylim(0, 1)
        self.ax4.set_ylabel('置信度')
        self.ax4.grid(True, alpha=0.3)
        
        plt.ion()  # 开启交互模式
        plt.tight_layout()
    
    def preprocess_points(self, points):
        """预处理点云数据"""
        if len(points) == 0:
            return np.zeros((self.num_points, 3))
        
        # 标准化
        centroid = np.mean(points, axis=0)
        points = points - centroid
        
        # 缩放到单位球
        max_dist = np.max(np.linalg.norm(points, axis=1))
        if max_dist > 0:
            points = points / max_dist
        
        # 采样固定数量点
        if len(points) >= self.num_points:
            indices = np.random.choice(len(points), self.num_points, replace=False)
            return points[indices]
        else:
            indices = np.random.choice(len(points), self.num_points, replace=True)
            return points[indices]
    
    def fuse_frames(self, frame_group):
        """融合一组帧的点云数据"""
        all_points = []
        for frame in frame_group:
            points = frame['points']
            if len(points) > 0:
                all_points.append(points)
        
        if not all_points:
            return np.empty((0, 3))
        
        # 将所有帧的点云合并
        fused_points = np.vstack(all_points)
        return fused_points
    
    def predict_action(self, points):
        """预测动作"""
        try:
            # 预处理点云
            processed_points = self.preprocess_points(points)
            
            # 添加强度信息
            intensity = np.ones((len(processed_points), 1))
            points_with_intensity = np.hstack([processed_points, intensity])
            
            # 创建25帧序列（重复单帧）
            sequence = np.tile(points_with_intensity, (25, 1, 1))
            sequence = torch.FloatTensor(sequence).unsqueeze(0)
            
            with torch.no_grad():
                if torch.cuda.is_available():
                    sequence = sequence.cuda()
                
                output = self.model(sequence)
                probabilities = torch.softmax(output, dim=1)
                predicted_class = torch.argmax(output, dim=1).item()
                confidence = probabilities[0][predicted_class].item()
                
                return {
                    'action': self.action_labels[predicted_class],
                    'confidence': confidence,
                    'probabilities': probabilities[0].cpu().numpy()
                }
                
        except Exception as e:
            print(f"❌ 预测失败: {e}")
            return None
    
    def add_frame(self, points):
        """添加新帧到缓冲区"""
        self.frame_buffer.append({
            'timestamp': time.time(),
            'points': points
        })
    
    def get_prediction(self):
        """获取预测结果（融合多帧）"""
        if len(self.frame_buffer) < self.num_frames:
            return None
        
        # 获取最近6帧
        recent_frames = list(self.frame_buffer)[-self.num_frames:]
        
        # 融合6帧点云数据
        fused_points = self.fuse_frames(recent_frames)
        
        if len(fused_points) == 0:
            return None
        
        # 预测动作
        prediction = self.predict_action(fused_points)
        
        if prediction:
            # 添加到预测历史
            self.prediction_history.append(prediction)
            self.last_prediction = prediction
            
            # 计算稳定性（连续相同预测的次数）
            if len(self.prediction_history) >= 3:
                recent_actions = [pred['action'] for pred in list(self.prediction_history)[-3:]]
                if len(set(recent_actions)) == 1:  # 连续3次相同预测
                    prediction['stable'] = True
                else:
                    prediction['stable'] = False
            else:
                prediction['stable'] = False
        
        return prediction
    
    def update_display(self, points, prediction):
        """更新显示"""
        # 清除旧图
        self.ax1.clear()
        self.ax2.clear()
        self.ax3.clear()
        self.ax4.clear()
        
        # 更新3D点云
        if len(points) > 0:
            # 根据预测结果选择颜色
            if prediction:
                color = self.action_colors[prediction['action']]
            else:
                color = 'gray'
            
            self.ax1.scatter(points[:, 0], points[:, 1], points[:, 2], 
                           c=color, alpha=0.7, s=20, edgecolors='black', linewidth=0.5)
            
            # 计算点云中心
            center = np.mean(points, axis=0)
            self.ax1.scatter(center[0], center[1], center[2], 
                           c='yellow', s=100, marker='*', label='人体中心')
            
            # 设置坐标轴范围
            x_range = np.max(points[:, 0]) - np.min(points[:, 0])
            y_range = np.max(points[:, 1]) - np.min(points[:, 1])
            z_range = np.max(points[:, 2]) - np.min(points[:, 2])
            
            max_range = max(x_range, y_range, z_range)
            if max_range > 0:
                self.ax1.set_xlim(-max_range/2, max_range/2)
                self.ax1.set_ylim(-max_range/2, max_range/2)
                self.ax1.set_zlim(-max_range/2, max_range/2)
        
        self.ax1.set_xlabel('X轴')
        self.ax1.set_ylabel('Y轴')
        self.ax1.set_zlabel('Z轴')
        self.ax1.set_title('实时3D点云')
        self.ax1.grid(True, alpha=0.3)
        self.ax1.legend()
        
        # 更新动作概率
        if prediction:
            probabilities = prediction['probabilities']
            bars = self.ax2.bar(self.action_labels, probabilities, 
                              color=[self.action_colors[label] for label in self.action_labels],
                              alpha=0.7, edgecolor='black', linewidth=1)
            
            # 添加概率值标签
            for bar, prob in zip(bars, probabilities):
                self.ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                            f'{prob:.3f}', ha='center', va='bottom', fontweight='bold')
            
            # 高亮预测的动作
            predicted_action = prediction['action']
            predicted_idx = self.action_labels.index(predicted_action)
            bars[predicted_idx].set_alpha(1.0)
            bars[predicted_idx].set_edgecolor('red')
            bars[predicted_idx].set_linewidth(3)
            
            # 显示预测结果
            stability_text = "稳定" if prediction.get('stable', False) else "不稳定"
            self.ax2.set_title(f'动作识别结果\n预测: {self.action_descriptions[predicted_action]} ({predicted_action})\n置信度: {prediction["confidence"]:.3f} ({stability_text})')
        else:
            self.ax2.text(0.5, 0.5, '等待数据...', ha='center', va='center', 
                        transform=self.ax2.transAxes, fontsize=14, color='gray')
            self.ax2.set_title('动作识别结果')
        
        self.ax2.set_ylim(0, 1)
        self.ax2.set_ylabel('概率')
        self.ax2.grid(True, alpha=0.3)
        
        # 更新状态信息
        status_text = f'系统状态:\n'
        status_text += f'帧缓冲区: {len(self.frame_buffer)}/{self.num_frames}\n'
        status_text += f'点云点数: {len(points) if len(points) > 0 else 0}\n'
        if prediction:
            status_text += f'当前动作: {self.action_descriptions[prediction["action"]]}\n'
            status_text += f'置信度: {prediction["confidence"]:.3f}\n'
            status_text += f'稳定性: {stability_text}'
        else:
            status_text += f'当前动作: 等待识别\n'
            status_text += f'置信度: --\n'
            status_text += f'稳定性: --'
        
        self.ax3.text(0.1, 0.5, status_text, fontsize=12, 
                     bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8),
                     transform=self.ax3.transAxes, verticalalignment='center')
        self.ax3.set_title('系统状态')
        
        # 更新预测历史
        if len(self.prediction_history) > 0:
            history_actions = [pred['action'] for pred in self.prediction_history]
            history_confidences = [pred['confidence'] for pred in self.prediction_history]
            
            # 绘制历史置信度
            for i, (action, conf) in enumerate(zip(history_actions, history_confidences)):
                color = self.action_colors[action]
                self.ax4.scatter(i, conf, c=color, s=50, alpha=0.7)
                self.ax4.text(i, conf + 0.02, action, ha='center', va='bottom', fontsize=8)
        
        self.ax4.set_title('预测历史')
        self.ax4.set_ylim(0, 1)
        self.ax4.set_ylabel('置信度')
        self.ax4.set_xlabel('时间')
        self.ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.pause(0.01)
    
    def start_simulation(self):
        """启动模拟实时数据"""
        print("🚀 启动实时3D点云显示")
        print("📡 使用模拟数据进行实时测试")
        print("按 Ctrl+C 停止")
        print("=" * 60)
        
        # 创建模拟数据
        def create_simulated_points(action_type):
            """创建模拟点云数据"""
            if action_type == 'sit':
                # 坐姿：较低的点云分布
                points = np.random.normal(0, 0.5, (100, 3))
                points[:, 2] = points[:, 2] * 0.3 - 0.5  # 降低高度
            elif action_type == 'squat':
                # 蹲姿：中等高度的点云分布
                points = np.random.normal(0, 0.4, (100, 3))
                points[:, 2] = points[:, 2] * 0.5 - 0.2
            else:  # stand
                # 站姿：较高的点云分布
                points = np.random.normal(0, 0.3, (100, 3))
                points[:, 2] = points[:, 2] * 0.8 + 0.3  # 增加高度
            
            return points
        
        # 模拟不同动作的循环
        actions = ['sit', 'squat', 'stand']
        action_duration = 20  # 每个动作持续20帧
        
        try:
            frame_count = 0
            while True:
                # 确定当前动作
                current_action_idx = (frame_count // action_duration) % len(actions)
                current_action = actions[current_action_idx]
                
                # 创建模拟点云
                current_points = create_simulated_points(current_action)
                
                # 添加一些随机变化
                noise = np.random.normal(0, 0.02, current_points.shape)
                current_points = current_points + noise
                
                # 添加到帧缓冲区
                self.add_frame(current_points)
                
                # 获取预测结果
                prediction = self.get_prediction()
                
                # 更新显示
                self.update_display(current_points, prediction)
                
                # 显示状态
                if frame_count % 10 == 0:
                    if prediction:
                        print(f"📊 帧 {frame_count}: 模拟动作={current_action}, 识别={prediction['action']} (置信度: {prediction['confidence']:.3f})")
                    else:
                        print(f"📊 帧 {frame_count}: 等待数据...")
                
                frame_count += 1
                time.sleep(0.3)  # 每0.3秒更新一次
                
        except KeyboardInterrupt:
            print("\n⏹️ 用户停止")
        except Exception as e:
            print(f"❌ 运行错误: {e}")
        finally:
            plt.ioff()
            plt.close()
            print("✅ 实时显示结束")

def main():
    """主函数"""
    print("🎨 实时3D点云与人体动作识别显示")
    print("=" * 60)
    
    # 检查模型文件
    import os
    if not os.path.exists('best_peter_model.pth'):
        print("❌ 模型文件不存在，请先运行训练脚本")
        return
    
    # 创建显示器
    display = RealTime3DDisplay()
    
    # 启动模拟
    display.start_simulation()

if __name__ == "__main__":
    main() 