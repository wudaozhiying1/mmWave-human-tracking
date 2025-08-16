#!/usr/bin/env python3
"""
模型数据收集器
将实时雷达数据转换为模型输入格式
"""

import numpy as np
import time
import threading
import queue
from collections import deque
from real_time_radar_reader import RealTimeRadarReader

class ModelDataCollector:
    def __init__(self, cli_port, data_port, sequence_length=30, save_interval=100):
        """
        初始化模型数据收集器
        
        Args:
            cli_port: CLI串口
            data_port: 数据串口
            sequence_length: 序列长度（帧数）
            save_interval: 保存间隔（多少个序列保存一次）
        """
        self.sequence_length = sequence_length
        self.save_interval = save_interval
        
        # 创建雷达读取器
        self.radar_reader = RealTimeRadarReader(cli_port, data_port)
        
        # 数据缓冲区
        self.frame_buffer = deque(maxlen=sequence_length)
        self.sequence_buffer = []
        
        # 统计信息
        self.sequence_count = 0
        self.saved_count = 0
        
        # 控制标志
        self.is_collecting = False
        self.collection_thread = None
        
    def connect_and_start(self, config_file=None):
        """连接雷达并开始收集数据"""
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
        
        # 开始收集线程
        self.is_collecting = True
        self.collection_thread = threading.Thread(target=self._collection_loop, daemon=True)
        self.collection_thread.start()
        
        print("模型数据收集已启动")
        return True
    
    def stop_collection(self):
        """停止数据收集"""
        self.is_collecting = False
        
        if self.collection_thread:
            self.collection_thread.join(timeout=2.0)
        
        self.radar_reader.stop_reading()
        self.radar_reader.disconnect()
        
        print("数据收集已停止")
    
    def _extract_features(self, frame_data):
        """
        从帧数据中提取特征
        返回: [x, y, z, vx, vy, vz] 或 None
        """
        try:
            # 优先使用跟踪数据（3D People Tracking）
            if ('trackData' in frame_data and 
                'numDetectedTracks' in frame_data and 
                frame_data['numDetectedTracks'] > 0):
                
                tracks = frame_data['trackData'][:frame_data['numDetectedTracks']]
                
                # 选择最接近的目标（距离原点最近）
                best_track = None
                min_distance = float('inf')
                
                for track in tracks:
                    if len(track) >= 6:
                        x, y, z, vx, vy, vz = track[1:7]  # 跳过track ID
                        distance = np.sqrt(x**2 + y**2 + z**2)
                        
                        if distance < min_distance:
                            min_distance = distance
                            best_track = [x, y, z, vx, vy, vz]
                
                return best_track
            
            # 如果没有跟踪数据，使用点云数据
            elif ('pointCloud' in frame_data and 
                  'numDetectedPoints' in frame_data and 
                  frame_data['numDetectedPoints'] > 0):
                
                points = frame_data['pointCloud'][:frame_data['numDetectedPoints']]
                
                if len(points) > 0:
                    # 选择最接近的点
                    best_point = None
                    min_distance = float('inf')
                    
                    for point in points:
                        if len(point) >= 4:
                            x, y, z, doppler = point[:4]
                            distance = np.sqrt(x**2 + y**2 + z**2)
                            
                            if distance < min_distance:
                                min_distance = distance
                                # 将多普勒速度转换为速度分量（简化处理）
                                speed = doppler * 0.1  # 假设的比例因子
                                best_point = [x, y, z, speed, 0, 0]  # vx=doppler, vy=vz=0
                    
                    return best_point
            
            return None
            
        except Exception as e:
            print(f"特征提取失败: {e}")
            return None
    
    def _collection_loop(self):
        """数据收集主循环"""
        print("数据收集线程启动")
        
        while self.is_collecting:
            try:
                # 获取最新帧数据
                frame_data = self.radar_reader.get_latest_frame()
                
                if frame_data:
                    # 提取特征
                    features = self._extract_features(frame_data)
                    
                    if features is not None:
                        # 添加到帧缓冲区
                        self.frame_buffer.append(features)
                        
                        # 检查是否收集到足够的帧
                        if len(self.frame_buffer) == self.sequence_length:
                            # 创建序列
                            sequence = np.array(list(self.frame_buffer), dtype=np.float64)
                            
                            # 添加到序列缓冲区
                            self.sequence_buffer.append(sequence)
                            self.sequence_count += 1
                            
                            print(f"收集到序列 {self.sequence_count}: 形状 {sequence.shape}")
                            
                            # 定期保存
                            if len(self.sequence_buffer) >= self.save_interval:
                                self._save_sequences()
                
                time.sleep(0.01)  # 10ms延迟
                
            except Exception as e:
                print(f"数据收集错误: {e}")
                time.sleep(0.1)
    
    def _save_sequences(self):
        """保存收集到的序列"""
        if not self.sequence_buffer:
            return
        
        try:
            # 转换为numpy数组
            sequences = np.array(self.sequence_buffer, dtype=np.float64)
            
            # 生成文件名
            timestamp = int(time.time())
            filename = f"radar_data_{timestamp}.npy"
            
            # 保存数据
            np.save(filename, sequences)
            
            print(f"保存了 {len(self.sequence_buffer)} 个序列到 {filename}")
            print(f"数据形状: {sequences.shape}")
            
            # 清空缓冲区
            self.sequence_buffer = []
            self.saved_count += 1
            
        except Exception as e:
            print(f"保存数据失败: {e}")
    
    def get_statistics(self):
        """获取统计信息"""
        stats = self.radar_reader.get_statistics()
        stats.update({
            'sequence_count': self.sequence_count,
            'saved_count': self.saved_count,
            'frame_buffer_size': len(self.frame_buffer),
            'sequence_buffer_size': len(self.sequence_buffer),
            'is_collecting': self.is_collecting
        })
        return stats
    
    def save_current_data(self):
        """手动保存当前数据"""
        self._save_sequences()
    
    def get_latest_sequence(self):
        """获取最新的完整序列"""
        if len(self.frame_buffer) == self.sequence_length:
            return np.array(list(self.frame_buffer), dtype=np.float64)
        return None


def main():
    """主函数示例"""
    # 配置
    CLI_PORT = 'COM4'
    DATA_PORT = 'COM6'
    CONFIG_FILE = '3m.cfg'
    
    # 创建收集器
    collector = ModelDataCollector(
        cli_port=CLI_PORT,
        data_port=DATA_PORT,
        sequence_length=30,  # 30帧序列
        save_interval=10     # 每10个序列保存一次
    )
    
    try:
        # 连接并开始收集
        if not collector.connect_and_start(CONFIG_FILE):
            return
        
        print("开始收集模型数据，按 Ctrl+C 停止...")
        print("数据将自动保存为 .npy 文件")
        
        # 主循环
        while True:
            # 显示统计信息
            stats = collector.get_statistics()
            if stats['sequence_count'] % 5 == 0 and stats['sequence_count'] > 0:
                print(f"统计: 序列数={stats['sequence_count']}, "
                      f"已保存={stats['saved_count']}, "
                      f"帧率={stats['frame_rate']:.1f}Hz")
            
            # 检查是否有完整序列
            latest_seq = collector.get_latest_sequence()
            if latest_seq is not None:
                print(f"最新序列形状: {latest_seq.shape}")
                print(f"特征范围: X[{latest_seq[:,0].min():.2f}, {latest_seq[:,0].max():.2f}], "
                      f"Y[{latest_seq[:,1].min():.2f}, {latest_seq[:,1].max():.2f}], "
                      f"Z[{latest_seq[:,2].min():.2f}, {latest_seq[:,2].max():.2f}]")
            
            time.sleep(1)  # 1秒更新一次
            
    except KeyboardInterrupt:
        print("\n正在停止数据收集...")
    finally:
        # 保存剩余数据
        collector.save_current_data()
        collector.stop_collection()
        print("程序已退出")


if __name__ == "__main__":
    main() 