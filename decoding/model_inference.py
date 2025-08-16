#!/usr/bin/env python3
"""
模型推理示例
使用实时雷达数据进行模型预测
"""

import numpy as np
import time
import threading
from collections import deque
from real_time_radar_reader import RealTimeRadarReader

class RadarModelInference:
    def __init__(self, cli_port, data_port, sequence_length=30, model_path=None):
        """
        初始化雷达模型推理
        
        Args:
            cli_port: CLI串口
            data_port: 数据串口
            sequence_length: 序列长度
            model_path: 模型文件路径（可选）
        """
        self.sequence_length = sequence_length
        self.model_path = model_path
        
        # 创建雷达读取器
        self.radar_reader = RealTimeRadarReader(cli_port, data_port)
        
        # 数据缓冲区
        self.frame_buffer = deque(maxlen=sequence_length)
        
        # 控制标志
        self.is_running = False
        self.inference_thread = None
        
        # 加载模型（如果有）
        self.model = None
        if model_path:
            self._load_model()
    
    def _load_model(self):
        """加载模型"""
        try:
            # 这里可以加载你的模型
            # 例如：self.model = load_model(self.model_path)
            print(f"模型加载成功: {self.model_path}")
        except Exception as e:
            print(f"模型加载失败: {e}")
            self.model = None
    
    def connect_and_start(self, config_file=None):
        """连接雷达并开始推理"""
        # 连接雷达
        if not self.radar_reader.connect():
            print("雷达连接失败")
            return False
        
        # 发送配置
        if config_file:
            try:
                with open(config_file, 'r') as f:
                    config_lines = f.readlines()
                self.radar_reader.send_config(config_lines)
                print(f"配置发送完成: {config_file}")
            except Exception as e:
                print(f"配置发送失败: {e}")
                return False
        
        # 开始数据读取
        if not self.radar_reader.start_reading():
            print("数据读取启动失败")
            return False
        
        # 开始推理线程
        self.is_running = True
        self.inference_thread = threading.Thread(target=self._inference_loop, daemon=True)
        self.inference_thread.start()
        
        print("模型推理已启动")
        return True
    
    def stop_inference(self):
        """停止推理"""
        self.is_running = False
        
        if self.inference_thread:
            self.inference_thread.join(timeout=2.0)
        
        self.radar_reader.stop_reading()
        self.radar_reader.disconnect()
        
        print("模型推理已停止")
    
    def _extract_features(self, frame_data):
        """从帧数据中提取特征"""
        try:
            # 优先使用跟踪数据
            if ('trackData' in frame_data and 
                'numDetectedTracks' in frame_data and 
                frame_data['numDetectedTracks'] > 0):
                
                tracks = frame_data['trackData'][:frame_data['numDetectedTracks']]
                
                # 选择最接近的目标
                best_track = None
                min_distance = float('inf')
                
                for track in tracks:
                    if len(track) >= 6:
                        x, y, z, vx, vy, vz = track[1:7]
                        distance = np.sqrt(x**2 + y**2 + z**2)
                        
                        if distance < min_distance:
                            min_distance = distance
                            best_track = [x, y, z, vx, vy, vz]
                
                return best_track
            
            # 使用点云数据
            elif ('pointCloud' in frame_data and 
                  'numDetectedPoints' in frame_data and 
                  frame_data['numDetectedPoints'] > 0):
                
                points = frame_data['pointCloud'][:frame_data['numDetectedPoints']]
                
                if len(points) > 0:
                    best_point = None
                    min_distance = float('inf')
                    
                    for point in points:
                        if len(point) >= 4:
                            x, y, z, doppler = point[:4]
                            distance = np.sqrt(x**2 + y**2 + z**2)
                            
                            if distance < min_distance:
                                min_distance = distance
                                speed = doppler * 0.1
                                best_point = [x, y, z, speed, 0, 0]
                    
                    return best_point
            
            return None
            
        except Exception as e:
            print(f"特征提取失败: {e}")
            return None
    
    def _preprocess_sequence(self, sequence):
        """预处理序列数据"""
        # 转换为numpy数组
        sequence = np.array(sequence, dtype=np.float64)
        
        # 标准化（可选）
        # sequence = (sequence - np.mean(sequence, axis=0)) / np.std(sequence, axis=0)
        
        # 添加批次维度
        sequence = sequence.reshape(1, sequence.shape[0], sequence.shape[1])
        
        return sequence
    
    def _predict(self, sequence):
        """模型预测"""
        if self.model is None:
            # 模拟预测（实际使用时替换为真实模型）
            return self._mock_predict(sequence)
        
        try:
            # 实际模型预测
            # prediction = self.model.predict(sequence)
            # return prediction
            return self._mock_predict(sequence)
        except Exception as e:
            print(f"模型预测失败: {e}")
            return None
    
    def _mock_predict(self, sequence):
        """模拟预测（用于测试）"""
        # 基于位置和速度的简单分类
        avg_x = np.mean(sequence[0, :, 0])
        avg_y = np.mean(sequence[0, :, 1])
        avg_vx = np.mean(sequence[0, :, 3])
        avg_vy = np.mean(sequence[0, :, 4])
        
        # 简单的运动模式识别
        speed = np.sqrt(avg_vx**2 + avg_vy**2)
        distance = np.sqrt(avg_x**2 + avg_y**2)
        
        if speed < 0.1:
            return "静止"
        elif speed < 0.5:
            return "慢走"
        elif speed < 1.0:
            return "正常行走"
        else:
            return "快走/跑步"
    
    def _inference_loop(self):
        """推理主循环"""
        print("推理线程启动")
        
        while self.is_running:
            try:
                # 获取最新帧数据
                frame_data = self.radar_reader.get_latest_frame()
                
                if frame_data:
                    # 提取特征
                    features = self._extract_features(frame_data)
                    
                    if features is not None:
                        # 添加到帧缓冲区
                        self.frame_buffer.append(features)
                        
                        # 检查是否有完整的序列
                        if len(self.frame_buffer) == self.sequence_length:
                            # 预处理序列
                            sequence = self._preprocess_sequence(list(self.frame_buffer))
                            
                            # 模型预测
                            prediction = self._predict(sequence)
                            
                            if prediction is not None:
                                print(f"预测结果: {prediction}")
                                print(f"序列形状: {sequence.shape}")
                                
                                # 显示特征统计
                                seq_data = sequence[0]
                                print(f"位置范围: X[{seq_data[:,0].min():.2f}, {seq_data[:,0].max():.2f}], "
                                      f"Y[{seq_data[:,1].min():.2f}, {seq_data[:,1].max():.2f}]")
                                print(f"速度范围: VX[{seq_data[:,3].min():.2f}, {seq_data[:,3].max():.2f}], "
                                      f"VY[{seq_data[:,4].min():.2f}, {seq_data[:,4].max():.2f}]")
                                print("-" * 50)
                
                time.sleep(0.01)  # 10ms延迟
                
            except Exception as e:
                print(f"推理错误: {e}")
                time.sleep(0.1)
    
    def get_statistics(self):
        """获取统计信息"""
        stats = self.radar_reader.get_statistics()
        stats.update({
            'frame_buffer_size': len(self.frame_buffer),
            'is_running': self.is_running
        })
        return stats


def main():
    """主函数示例"""
    import threading
    
    # 配置
    CLI_PORT = 'COM4'
    DATA_PORT = 'COM6'
    CONFIG_FILE = '3m.cfg'
    
    # 创建推理器
    inference = RadarModelInference(
        cli_port=CLI_PORT,
        data_port=DATA_PORT,
        sequence_length=30
    )
    
    try:
        # 连接并开始推理
        if not inference.connect_and_start(CONFIG_FILE):
            return
        
        print("开始实时模型推理，按 Ctrl+C 停止...")
        
        # 主循环
        while True:
            # 显示统计信息
            stats = inference.get_statistics()
            if stats['frame_count'] % 50 == 0 and stats['frame_count'] > 0:
                print(f"统计: 帧数={stats['frame_count']}, "
                      f"帧率={stats['frame_rate']:.1f}Hz, "
                      f"缓冲区={stats['frame_buffer_size']}")
            
            time.sleep(1)  # 1秒更新一次
            
    except KeyboardInterrupt:
        print("\n正在停止推理...")
    finally:
        inference.stop_inference()
        print("程序已退出")


if __name__ == "__main__":
    main() 