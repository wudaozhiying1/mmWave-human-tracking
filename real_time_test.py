#!/usr/bin/env python3
"""
实时雷达动作识别测试脚本
从雷达串口读取数据，进行6帧融合，输入到PETer模型进行实时动作识别
"""

import torch
import numpy as np
import json
import time
import threading
import queue
from collections import deque
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
import serial
import serial.tools.list_ports
from peter_network import PETerNetwork
import warnings
warnings.filterwarnings('ignore')

class RealTimeRadarReader:
    """实时雷达数据读取器"""
    
    def __init__(self, cli_port='COM4', data_port='COM6', baudrate=921600):
        self.cli_port = cli_port
        self.data_port = data_port
        self.baudrate = baudrate
        self.cli_serial = None
        self.data_serial = None
        self.is_running = False
        self.data_queue = queue.Queue()
        self.frame_buffer = deque(maxlen=100)  # 存储最近100帧
        
        # 魔法字
        self.MAGIC_WORD = b'\x02\x01\x04\x03\x06\x05\x08\x07'
        
    def connect_radar(self):
        """连接雷达设备"""
        try:
            print(f"\n🔌 正在连接串口...")
            
            # 连接CLI串口
            print(f"   📡 连接CLI串口: {self.cli_port}")
            self.cli_serial = serial.Serial(
                self.cli_port, 
                115200,  # CLI串口通常使用115200波特率
                timeout=1
            )
            print(f"   ✅ CLI串口连接成功")
            
            # 连接数据端口
            print(f"   📡 连接数据串口: {self.data_port}")
            self.data_serial = serial.Serial(
                self.data_port, 
                self.baudrate,  # 数据串口使用921600波特率
                timeout=1
            )
            print(f"   ✅ 数据串口连接成功")
            
            print(f"✅ 所有串口连接成功")
            
            # 发送配置命令
            self.send_config()
            
            # 等待雷达启动
            print("⏳ 等待雷达启动...")
            time.sleep(2)
            
            # 检查数据串口是否有数据
            print("🔍 检查数据串口状态...")
            if self.data_serial.in_waiting > 0:
                print(f"📡 数据串口有 {self.data_serial.in_waiting} 字节数据")
                # 读取前几个字节看看
                test_data = self.data_serial.read(min(16, self.data_serial.in_waiting))
                print(f"🔍 前16字节: {test_data.hex()}")
            else:
                print("📡 数据串口暂无数据")
            
            # 等待更长时间让雷达开始发送数据
            print("⏳ 等待雷达开始发送数据...")
            time.sleep(3)
            
            # 再次检查数据串口
            if self.data_serial.in_waiting > 0:
                print(f"📡 3秒后数据串口有 {self.data_serial.in_waiting} 字节数据")
            else:
                print("⚠️ 3秒后数据串口仍无数据，可能雷达未正确启动")
            
            return True
            
        except Exception as e:
            print(f"❌ 连接失败: {e}")
            return False
    
    def load_config_file(self, config_path):
        """加载配置文件"""
        try:
            with open(config_path, 'r') as f:
                return [line.strip() for line in f.readlines()]
        except Exception as e:
            print(f"❌ 加载配置文件失败: {e}")
            return []
    
    def send_config(self):
        """发送雷达配置 - 使用radar_reader.py的方法"""
        try:
            # 加载配置文件
            config_path = 'decoding/3m.cfg'
            config_lines = self.load_config_file(config_path)
            
            if not config_lines:
                print("❌ 配置文件加载失败")
                return False
            
            print(f"\n📋 正在发送配置文件...")
            
            # 移除空行和注释行
            cfg = [line for line in config_lines if line.strip() and not line.startswith('%')]
            # 确保每行以\n结尾
            cfg = [line + '\n' if not line.endswith('\n') else line for line in cfg]
            
            for line in cfg:
                time.sleep(0.03)  # 行延迟
                
                # 发送命令
                self.cli_serial.write(line.encode())
                
                # 读取确认
                ack = self.cli_serial.readline()
                if len(ack) == 0:
                    print("❌ 串口超时，设备可能处于闪烁模式")
                    return False
                
                # 安全解码
                try:
                    ack_text = ack.decode('utf-8').strip()
                except UnicodeDecodeError:
                    ack_text = ack.decode('utf-8', errors='ignore').strip()
                
                print(f"   📤 {line.strip()} -> {ack_text}")
                
                # 读取第二行确认
                ack = self.cli_serial.readline()
                if ack:
                    try:
                        ack_text = ack.decode('utf-8').strip()
                    except UnicodeDecodeError:
                        ack_text = ack.decode('utf-8', errors='ignore').strip()
                    print(f"   📤 确认: {ack_text}")
            
            # 给缓冲区一些时间清理
            time.sleep(0.03)
            self.cli_serial.reset_input_buffer()
            
            print(f"✅ 配置文件发送完成")
            return True
            
        except Exception as e:
            print(f"❌ 配置发送失败: {e}")
            return False
    
    def read_frame(self):
        """读取一帧数据 - 使用radar_reader.py的方法"""
        try:
            # 查找魔法字
            index = 0
            magicByte = self.data_serial.read(1)
            frameData = bytearray(b'')
            
            while True:
                # 检查是否有数据
                if len(magicByte) < 1:
                    return None  # 超时，没有数据
                
                # 找到匹配的字节
                if magicByte[0] == self.MAGIC_WORD[index]:
                    index += 1
                    frameData.append(magicByte[0])
                    if index == 8:  # 找到完整的魔法字
                        break
                    magicByte = self.data_serial.read(1)
                else:
                    # 重置索引
                    if index == 0:
                        magicByte = self.data_serial.read(1)
                    index = 0
                    frameData = bytearray(b'')
            
            # 读取版本号
            versionBytes = self.data_serial.read(4)
            frameData += bytearray(versionBytes)
            
            # 读取长度
            lengthBytes = self.data_serial.read(4)
            frameData += bytearray(lengthBytes)
            frameLength = int.from_bytes(lengthBytes, byteorder='little')
            
            # 减去已经读取的字节（魔法字、版本、长度）
            frameLength -= 16
            
            # 读取帧的其余部分
            frameData += bytearray(self.data_serial.read(frameLength))
            
            return frameData
            
        except Exception as e:
            print(f"❌ 读取帧失败: {e}")
            return None
    
    def parse_point_cloud(self, frame_data):
        """解析点云数据"""
        try:
            # 导入解析模块
            import sys
            sys.path.append('decoding')
            from parseFrame import parseStandardFrame
            
            # 解析帧
            parsed_data = parseStandardFrame(frame_data)
            
            if 'pointCloud' in parsed_data:
                point_cloud = parsed_data['pointCloud']
                if len(point_cloud) > 0:
                    # 提取xyz坐标
                    points = []
                    for point in point_cloud:
                        if len(point) >= 3:
                            x, y, z = point[0], point[1], point[2]
                            points.append([x, y, z])
                    
                    return np.array(points)
            
            return np.array([])
            
        except Exception as e:
            print(f"❌ 解析点云失败: {e}")
            return np.array([])
    
    def start_reading(self):
        """开始读取数据"""
        self.is_running = True
        self.read_thread = threading.Thread(target=self._read_loop)
        self.read_thread.daemon = True
        self.read_thread.start()
        print("🚀 开始实时数据读取...")
    
    def _read_loop(self):
        """数据读取循环"""
        print("🔄 开始数据读取循环...")
        frame_count = 0
        last_status_time = time.time()
        
        while self.is_running:
            try:
                # 检查串口是否有数据
                if self.data_serial.in_waiting > 0:
                    print(f"📡 检测到数据: {self.data_serial.in_waiting} 字节")
                
                frame_data = self.read_frame()
                if frame_data:
                    frame_count += 1
                    print(f"✅ 成功读取第 {frame_count} 帧数据")
                    
                    points = self.parse_point_cloud(frame_data)
                    if len(points) > 0:
                        timestamp = time.time()
                        self.frame_buffer.append({
                            'timestamp': timestamp,
                            'points': points
                        })
                        self.data_queue.put({
                            'timestamp': timestamp,
                            'points': points
                        })
                        print(f"📊 解析到 {len(points)} 个点云点")
                    else:
                        print("⚠️ 帧数据中没有点云信息")
                else:
                    # 每5秒显示一次状态
                    current_time = time.time()
                    if current_time - last_status_time > 5:
                        print(f"⏳ 等待数据... (已读取 {frame_count} 帧)")
                        last_status_time = current_time
                
                time.sleep(0.01)  # 10ms间隔
                
            except Exception as e:
                print(f"❌ 读取循环错误: {e}")
                time.sleep(0.1)
    
    def stop_reading(self):
        """停止读取数据"""
        self.is_running = False
        if self.cli_serial:
            self.cli_serial.close()
        if self.data_serial:
            self.data_serial.close()
        print("⏹️ 停止数据读取")

class RealTimeActionRecognizer:
    """实时动作识别器"""
    
    def __init__(self, model_path='best_peter_model.pth', num_frames=6, num_points=100):
        self.num_frames = num_frames  # 每次融合的帧数
        self.num_points = num_points
        self.frame_buffer = deque(maxlen=num_frames)  # 只需要存储6帧
        self.action_labels = ['sit', 'squat', 'stand']
        
        # 稳定性机制
        self.prediction_history = deque(maxlen=10)  # 保存最近10次预测
        self.stability_threshold = 0.6  # 置信度阈值
        self.min_consistent_predictions = 3  # 最少连续预测次数
        self.last_stable_prediction = None  # 上次稳定的预测结果
        
        # 类别平衡机制
        self.class_weights = [2.0, 2.0, 0.5]  # sit, squat, stand的权重
        self.prediction_counts = {'sit': 0, 'squat': 0, 'stand': 0}  # 预测计数
        self.max_consecutive_same = 5  # 最大连续相同预测次数
        
        # 加载模型
        self.model = PETerNetwork(num_classes=3, num_points=num_points, num_frames=25, k=10)
        if torch.cuda.is_available():
            self.model.load_state_dict(torch.load(model_path, map_location='cuda'))
            self.model = self.model.cuda()
        else:
            self.model.load_state_dict(torch.load(model_path, map_location='cpu'))
        
        self.model.eval()
        print(f"✅ 模型加载完成: {model_path}")
    
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
        """
        融合一组帧的点云数据
        Args:
            frame_group: 一组帧的点云数据列表
        Returns:
            fused_points: 融合后的点云数据
        """
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
    
    def create_sequence(self):
        """创建时序序列 - 直接融合6帧后重复到25帧"""
        if len(self.frame_buffer) < self.num_frames:
            return None
        
        # 获取最近6帧
        recent_frames = list(self.frame_buffer)[-self.num_frames:]
        
        # 融合6帧点云数据
        fused_points = self.fuse_frames(recent_frames)
        
        # 预处理融合后的点云
        processed_points = self.preprocess_points(fused_points)
        
        # 添加强度信息
        intensity = np.ones((len(processed_points), 1))
        points_with_intensity = np.hstack([processed_points, intensity])
        
        # 创建25帧序列（重复融合后的点云）
        sequence = np.tile(points_with_intensity, (25, 1, 1))  # 重复25次
        
        return torch.FloatTensor(sequence).unsqueeze(0)  # 添加batch维度
    
    def predict_action(self, sequence):
        """预测动作（带类别平衡）"""
        try:
            with torch.no_grad():
                if torch.cuda.is_available():
                    sequence = sequence.cuda()
                
                output = self.model(sequence)
                
                # 应用类别权重
                weighted_output = output.clone()
                for i, weight in enumerate(self.class_weights):
                    weighted_output[0, i] *= weight
                
                probabilities = torch.softmax(weighted_output, dim=1)
                predicted_class = torch.argmax(weighted_output, dim=1).item()
                confidence = probabilities[0][predicted_class].item()
                
                # 更新预测计数
                predicted_action = self.action_labels[predicted_class]
                self.prediction_counts[predicted_action] += 1
                
                return {
                    'action': predicted_action,
                    'confidence': confidence,
                    'probabilities': probabilities[0].cpu().numpy(),
                    'raw_probabilities': torch.softmax(output, dim=1)[0].cpu().numpy()  # 原始概率
                }
                
        except Exception as e:
            print(f"❌ 预测失败: {e}")
            return None
    
    def add_frame(self, points):
        """添加新帧"""
        self.frame_buffer.append({
            'timestamp': time.time(),
            'points': points
        })
    
    def get_stable_prediction(self):
        """获取稳定的预测结果（带类别平衡）"""
        sequence = self.create_sequence()
        if sequence is not None:
            current_prediction = self.predict_action(sequence)
            if current_prediction is None:
                return self.last_stable_prediction
            
            # 添加到预测历史
            self.prediction_history.append(current_prediction)
            
            # 检查连续预测次数，防止过度偏向某个类别
            if len(self.prediction_history) >= self.max_consecutive_same:
                recent_actions = [pred['action'] for pred in list(self.prediction_history)[-self.max_consecutive_same:]]
                if len(set(recent_actions)) == 1:  # 连续预测相同动作
                    # 降低该动作的权重
                    action_idx = self.action_labels.index(recent_actions[0])
                    self.class_weights[action_idx] = max(0.1, self.class_weights[action_idx] * 0.8)
                    print(f"⚠️ 连续预测{recent_actions[0]}，降低权重至{self.class_weights[action_idx]:.2f}")
            
            # 如果置信度足够高，直接返回
            if current_prediction['confidence'] > 0.8:
                self.last_stable_prediction = current_prediction
                return current_prediction
            
            # 检查历史预测的一致性
            if len(self.prediction_history) >= self.min_consistent_predictions:
                recent_predictions = list(self.prediction_history)[-self.min_consistent_predictions:]
                
                # 检查是否连续预测相同动作
                actions = [pred['action'] for pred in recent_predictions]
                if len(set(actions)) == 1:  # 所有预测都是同一动作
                    avg_confidence = np.mean([pred['confidence'] for pred in recent_predictions])
                    if avg_confidence > self.stability_threshold:
                        stable_prediction = {
                            'action': actions[0],
                            'confidence': avg_confidence,
                            'probabilities': np.mean([pred['probabilities'] for pred in recent_predictions], axis=0),
                            'raw_probabilities': np.mean([pred['raw_probabilities'] for pred in recent_predictions], axis=0)
                        }
                        self.last_stable_prediction = stable_prediction
                        return stable_prediction
            
            # 如果当前预测置信度较高，更新稳定预测
            if current_prediction['confidence'] > self.stability_threshold:
                self.last_stable_prediction = current_prediction
            
            # 返回上次的稳定预测或当前预测
            return self.last_stable_prediction if self.last_stable_prediction else current_prediction
        
        return self.last_stable_prediction
    
    def get_prediction(self):
        """获取预测结果（保持向后兼容）"""
        return self.get_stable_prediction()

class RealTimeVisualizer:
    """实时可视化器"""
    
    def __init__(self):
        self.fig = plt.figure(figsize=(15, 6))
        self.fig.suptitle('实时雷达动作识别', fontsize=16)
        
        # 点云显示（3D）
        self.ax1 = self.fig.add_subplot(121, projection='3d')
        self.ax1.set_xlabel('X')
        self.ax1.set_ylabel('Y')
        self.ax1.set_zlabel('Z')
        self.ax1.set_title('实时点云')
        
        # 动作概率显示
        self.ax2 = self.fig.add_subplot(122)
        self.ax2.set_title('动作识别概率')
        self.ax2.set_ylim(0, 1)
        
        # 动作概率显示
        self.ax2.set_title('动作识别概率')
        self.ax2.set_ylim(0, 1)
        
        self.action_labels = ['sit', 'squat', 'stand']
        self.prob_bars = None
        self.point_cloud_plot = None
        
        plt.ion()  # 开启交互模式
    
    def update_visualization(self, points, prediction):
        """更新可视化"""
        # 清除旧图
        self.ax1.clear()
        self.ax2.clear()
        
        # 绘制点云
        if len(points) > 0:
            self.ax1.scatter(points[:, 0], points[:, 1], points[:, 2], c='b', alpha=0.6)
        
        self.ax1.set_xlabel('X')
        self.ax1.set_ylabel('Y')
        self.ax1.set_zlabel('Z')
        self.ax1.set_title('实时点云')
        self.ax1.set_xlim(-2, 2)
        self.ax1.set_ylim(-2, 2)
        self.ax1.set_zlim(-2, 2)
        self.ax1.view_init(elev=20, azim=45)  # 设置3D视角
        
        # 绘制动作概率
        if prediction:
            probabilities = prediction['probabilities']
            bars = self.ax2.bar(self.action_labels, probabilities, color=['red', 'green', 'blue'])
            
            # 添加概率值标签
            for bar, prob in zip(bars, probabilities):
                self.ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                             f'{prob:.3f}', ha='center', va='bottom')
            
            # 高亮预测的动作
            predicted_action = prediction['action']
            predicted_idx = self.action_labels.index(predicted_action)
            bars[predicted_idx].set_color('orange')
            
            self.ax2.set_title(f'动作识别: {predicted_action} (置信度: {prediction["confidence"]:.3f})')
        
        self.ax2.set_ylim(0, 1)
        self.ax2.set_ylabel('概率')
        
        plt.tight_layout()
        plt.pause(0.01)

def main():
    """主函数"""
    print("🎯 实时雷达动作识别测试")
    print("=" * 60)
    
    # 检查模型文件
    if not os.path.exists('best_peter_model.pth'):
        print("❌ 模型文件不存在，请先运行训练脚本")
        return
    
    # 创建组件
    radar_reader = RealTimeRadarReader()
    action_recognizer = RealTimeActionRecognizer()
    visualizer = RealTimeVisualizer()
    
    # 连接雷达
    if not radar_reader.connect_radar():
        print("❌ 雷达连接失败")
        return
    
    try:
        # 开始读取数据
        radar_reader.start_reading()
        
        print("📊 开始实时识别...")
        print("📋 需要收集6帧数据才能开始预测（6帧融合）")
        print("按 Ctrl+C 停止")
        
        last_prediction_time = 0
        prediction_interval = 0.5  # 每0.5秒预测一次
        
        while True:
            try:
                # 获取最新数据
                if not radar_reader.data_queue.empty():
                    data = radar_reader.data_queue.get()
                    points = data['points']
                    
                    # 添加到识别器
                    action_recognizer.add_frame(points)
                    
                    # 显示当前帧数状态
                    current_frames = len(action_recognizer.frame_buffer)
                    if current_frames <= 6:  # 显示前6帧的状态
                        print(f"📊 已收集 {current_frames}/6 帧数据")
                    
                    # 定期进行预测
                    current_time = time.time()
                    if current_time - last_prediction_time >= prediction_interval:
                        prediction = action_recognizer.get_prediction()
                        if prediction:
                            print(f"🎯 识别结果: {prediction['action']} (置信度: {prediction['confidence']:.3f})")
                            
                            # 更新可视化
                            visualizer.update_visualization(points, prediction)
                        else:
                            print(f"⏳ 等待更多数据... (当前: {current_frames}/6 帧)")
                        
                        last_prediction_time = current_time
                
                time.sleep(0.01)
                
            except KeyboardInterrupt:
                print("\n⏹️ 用户停止")
                break
            except Exception as e:
                print(f"❌ 运行错误: {e}")
                time.sleep(0.1)
    
    finally:
        # 清理资源
        radar_reader.stop_reading()
        plt.ioff()
        plt.close()
        print("✅ 测试完成")

if __name__ == "__main__":
    import os
    main() 