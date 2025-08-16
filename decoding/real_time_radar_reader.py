#!/usr/bin/env python3
"""
实时IWR6843雷达数据读取器
低延迟数据收集和解码
"""

import serial
import time
import struct
import numpy as np
import threading
import queue
import logging
from datetime import datetime
from collections import deque

# 导入解析模块
import sys
import os

# 添加当前目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

try:
    from parseFrame import parseStandardFrame
    from demo_defines import *
except ImportError as e:
    print(f"❌ 导入错误: {e}")
    print(f"📁 当前目录: {current_dir}")
    print(f"📁 可用文件: {os.listdir(current_dir)}")
    raise

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RealTimeRadarReader:
    def __init__(self, cli_port, data_port, baud_rate=921600, buffer_size=1000):
        """
        初始化实时雷达读取器
        Args:
            cli_port: CLI串口名称 (如 'COM3')
            data_port: 数据串口名称 (如 'COM4') 
            baud_rate: 数据串口波特率
            buffer_size: 数据缓冲区大小
        """
        self.cli_port = cli_port
        self.data_port = data_port
        self.baud_rate = baud_rate
        self.buffer_size = buffer_size
        
        # 串口连接
        self.cli_com = None
        self.data_com = None
        
        # 数据缓冲区
        self.data_queue = queue.Queue(maxsize=buffer_size)
        self.frame_buffer = deque(maxlen=10)  # 最近10帧数据
        self.feature_window = deque(maxlen=30)  # 用于存储30帧4维特征 [x, y, z, intensity]
        
        # 控制标志
        self.is_running = False
        self.is_connected = False
        
        # 统计信息
        self.frame_count = 0
        self.error_count = 0
        self.last_frame_time = 0
        self.frame_rate = 0
        
        # 魔法字
        self.UART_MAGIC_WORD = bytearray(b'\x02\x01\x04\x03\x06\x05\x08\x07')
        
        # 线程
        self.reader_thread = None
    
    def connect(self):
        """连接串口"""
        try:
            logger.info(f"正在连接CLI串口: {self.cli_port}")
            self.cli_com = serial.Serial(
                self.cli_port, 
                115200, 
                parity=serial.PARITY_NONE, 
                stopbits=serial.STOPBITS_ONE, 
                timeout=0.1
            )
            logger.info(f"CLI串口连接成功: {self.cli_port}")
            time.sleep(0.5)
            logger.info(f"正在连接数据串口: {self.data_port}")
            self.data_com = serial.Serial(
                self.data_port, 
                self.baud_rate, 
                parity=serial.PARITY_NONE, 
                stopbits=serial.STOPBITS_ONE, 
                timeout=0.1
            )
            logger.info(f"数据串口连接成功: {self.data_port}")
            self.data_com.reset_input_buffer()
            self.data_com.reset_output_buffer()
            self.is_connected = True
            logger.info(f"成功连接到串口: CLI={self.cli_port}, DATA={self.data_port}")
            return True
        except serial.SerialException as e:
            logger.error(f"串口连接失败: {e}")
            if self.cli_com and self.cli_com.is_open:
                self.cli_com.close()
            if self.data_com and self.data_com.is_open:
                self.data_com.close()
            self.cli_com = None
            self.data_com = None
            self.is_connected = False
            return False
        except Exception as e:
            logger.error(f"连接串口时发生未知错误: {e}")
            if self.cli_com and self.cli_com.is_open:
                self.cli_com.close()
            if self.data_com and self.data_com.is_open:
                self.data_com.close()
            self.cli_com = None
            self.data_com = None
            self.is_connected = False
            return False
    
    def disconnect(self):
        """断开串口连接"""
        self.is_running = False
        if self.cli_com and self.cli_com.is_open:
            self.cli_com.close()
        if self.data_com and self.data_com.is_open:
            self.data_com.close()
        self.is_connected = False
        logger.info("已断开串口连接")
    
    def send_config(self, config_lines):
        """发送配置命令"""
        if not self.is_connected:
            logger.error("串口未连接")
            return False
        try:
            for line in config_lines:
                if line.strip() and not line.startswith('%'):
                    if not line.endswith('\n'):
                        line += '\n'
                    self.cli_com.write(line.encode())
                    time.sleep(0.01)
                    ack = self.cli_com.readline()
                    if ack:
                        logger.debug(f"配置确认: {ack.decode().strip()}")
            logger.info("配置发送完成")
            return True
        except Exception as e:
            logger.error(f"发送配置失败: {e}")
            return False
    
    def find_magic_word(self):
        """查找魔法字，返回完整帧数据"""
        frame_data = bytearray()
        index = 0
        while True:
            magic_byte = self.data_com.read(1)
            if len(magic_byte) < 1:
                index = 0
                frame_data = bytearray()
                continue
            if magic_byte[0] == self.UART_MAGIC_WORD[index]:
                index += 1
                frame_data.append(magic_byte[0])
                if index == 8:
                    break
            else:
                if index == 0:
                    continue
                index = 0
                frame_data = bytearray()
        return frame_data
    
    def read_frame(self):
        """读取一帧数据"""
        try:
            frame_data = self.find_magic_word()
            version_bytes = self.data_com.read(4)
            frame_data.extend(version_bytes)
            length_bytes = self.data_com.read(4)
            frame_data.extend(length_bytes)
            frame_length = int.from_bytes(length_bytes, byteorder='little')
            remaining_length = frame_length - 16
            remaining_data = self.data_com.read(remaining_length)
            frame_data.extend(remaining_data)
            return frame_data
        except Exception as e:
            logger.error(f"读取帧数据失败: {e}")
            return None
    
    def parse_frame_data(self, frame_data):
        """解析帧数据"""
        try:
            parsed_data = parseStandardFrame(frame_data)
            # 隐藏详细数据输出
            # print("parseStandardFrame 返回:", parsed_data)
            parsed_data['timestamp'] = time.time()
            parsed_data['frame_id'] = self.frame_count
            return parsed_data
        except Exception as e:
            logger.error(f"解析帧数据失败: {e}")
            return None
    
    def data_reader_thread(self):
        """数据读取线程"""
        logger.info("数据读取线程启动")
        while self.is_running:
            try:
                frame_data = self.read_frame()
                if frame_data:
                    parsed_data = self.parse_frame_data(frame_data)
                    if parsed_data:
                        # 隐藏详细数据输出
                        # print("采集到并解析的数据:", parsed_data)
                        self.frame_count += 1
                        current_time = time.time()
                        if self.last_frame_time > 0:
                            self.frame_rate = 1.0 / (current_time - self.last_frame_time)
                        self.last_frame_time = current_time
                        self.process_coordinate_data(parsed_data)
                        try:
                            self.data_queue.put_nowait(parsed_data)
                            self.frame_buffer.append(parsed_data)
                            # 隐藏详细数据输出
                            # print("frame_buffer 当前长度:", len(self.frame_buffer))
                        except queue.Full:
                            try:
                                self.data_queue.get_nowait()
                                self.data_queue.put_nowait(parsed_data)
                            except:
                                pass
                    else:
                        self.error_count += 1
                else:
                    time.sleep(0.001)
            except Exception as e:
                logger.error(f"数据读取线程错误: {e}")
                self.error_count += 1
                time.sleep(0.01)
    
    def start_reading(self):
        """开始实时数据读取"""
        if not self.is_connected:
            logger.error("串口未连接")
            return False
        self.is_running = True
        self.reader_thread = threading.Thread(target=self.data_reader_thread, daemon=True)
        self.reader_thread.start()
        logger.info("开始实时数据读取")
        return True
    
    def stop_reading(self):
        """停止数据读取"""
        self.is_running = False
        if self.reader_thread:
            self.reader_thread.join(timeout=1.0)
        logger.info("停止数据读取")
    
    def get_latest_frame(self):
        """获取最新的一帧数据"""
        try:
            return self.data_queue.get_nowait()
        except queue.Empty:
            return None
    
    def get_frame_buffer(self):
        """获取帧缓冲区中的所有数据"""
        return list(self.frame_buffer)
    
    def get_latest_frame_peek(self):
        """只读不取出队列的最新一帧（或None）"""
        if self.frame_buffer:
            return self.frame_buffer[-1]
        return None
    
    def get_statistics(self):
        """获取统计信息"""
        return {
            'frame_count': self.frame_count,
            'error_count': self.error_count,
            'frame_rate': self.frame_rate,
            'queue_size': self.data_queue.qsize(),
            'buffer_size': len(self.frame_buffer),
            'is_connected': self.is_connected,
            'is_running': self.is_running
        }
    
    def process_coordinate_data(self, parsed_data):
        """处理坐标数据，提取4维特征[x, y, z, intensity] - 直接使用雷达SNR数据"""
        try:
            feature = None
            
            # 优先使用跟踪数据（trackData）
            if ('trackData' in parsed_data and 
                'numDetectedTracks' in parsed_data and 
                parsed_data['numDetectedTracks'] > 0):
                
                track = parsed_data['trackData'][0]
                if len(track) >= 4:
                    # 提取位置信息
                    x, y, z = track[1], track[2], track[3]
                    
                    # 从对应的点云数据中获取SNR作为强度
                    intensity = 255.0  # 默认强度（匹配训练数据）
                    
                    # 如果有点云数据，找到最接近的点并使用其SNR
                    if ('pointCloud' in parsed_data and 
                        'numDetectedPoints' in parsed_data and 
                        parsed_data['numDetectedPoints'] > 0):
                        
                        points = parsed_data['pointCloud'][:parsed_data['numDetectedPoints']]
                        if len(points) > 0 and points.shape[1] >= 5:  # 确保有SNR数据 (第5列)
                            # 找到距离跟踪点最近的点云点
                            distances = np.sqrt((points[:, 0] - x)**2 + 
                                              (points[:, 1] - y)**2 + 
                                              (points[:, 2] - z)**2)
                            nearest_idx = np.argmin(distances)
                            snr_value = points[nearest_idx, 4]  # 原始SNR值
                            # 将SNR缩放到训练时的强度范围 (255)
                            intensity = np.clip(snr_value / 50.0 * 255.0, 0, 255)
                            print(f"📡 使用最近点云SNR缩放为训练范围: {snr_value:.1f}dB -> {intensity:.1f}")
                    
                    feature = [x, y, z, intensity]
            
            # 如果没有跟踪数据，直接使用点云数据
            elif ('pointCloud' in parsed_data and 
                  'numDetectedPoints' in parsed_data and 
                  parsed_data['numDetectedPoints'] > 0):
                
                points = parsed_data['pointCloud'][:parsed_data['numDetectedPoints']]
                if len(points) > 0:
                    # 计算点云中心位置
                    center = np.mean(points[:, :3], axis=0)
                    
                    # 直接使用雷达提供的SNR作为强度
                    if points.shape[1] >= 5:  # 第5列是SNR
                        snr_values = points[:, 4]  # 原始SNR值
                        # 将SNR缩放到训练时的强度范围 (255)
                        intensity = np.mean(np.clip(snr_values / 50.0 * 255.0, 0, 255))
                        print(f"📡 使用点云平均SNR缩放为训练范围: 平均SNR {np.mean(snr_values):.1f}dB -> 强度{intensity:.1f} (共{len(points)}点)")
                    elif points.shape[1] >= 4:
                        # 如果只有基础4维数据，使用多普勒速度的绝对值估算强度
                        doppler_data = points[:, 3]
                        # 多普勒速度映射到255范围
                        intensity = np.mean(np.abs(doppler_data)) * 25.0  # 调整缩放因子
                        intensity = np.clip(intensity, 0, 255)  # 限制在训练范围
                        print(f"⚠️  使用多普勒估算强度(训练范围): {intensity:.1f} (基于多普勒)")
                    else:
                        intensity = 255.0  # 训练时的默认强度值
                        print(f"⚠️  使用训练默认强度: {intensity:.1f}")
                    
                    feature = [center[0], center[1], center[2], intensity]
            
            # 如果成功提取到特征，添加到特征窗口
            if feature is not None:
                # 确保特征是4维的
                if len(feature) != 4:
                    # 如果维度不对，进行填充或截断
                    if len(feature) < 4:
                        feature.extend([255.0] * (4 - len(feature)))  # 用训练默认强度填充
                    else:
                        feature = feature[:4]
                
                # 数据类型转换和验证
                feature = [float(x) for x in feature]
                
                # 强度值范围检查 (训练时使用255范围)
                if feature[3] < 0:
                    feature[3] = 0.0
                elif feature[3] > 255:
                    feature[3] = 255.0
                
                # 检查是否有无效值
                if not any(np.isnan(feature)) and not any(np.isinf(feature)):
                    self.feature_window.append(feature)
                    # 每10帧输出一次特征信息
                    if len(self.feature_window) % 10 == 0:
                        print(f"✅ 4D特征已添加: X={feature[0]:.2f}, Y={feature[1]:.2f}, Z={feature[2]:.2f}, 强度={feature[3]:.1f}dB")
                else:
                    logger.warning(f"检测到无效特征值: {feature}")
                    
        except Exception as e:
            logger.error(f"处理坐标数据时出错: {e}")
            import traceback
            traceback.print_exc()
    
    def get_latest_feature_sequence(self):
        """获取最近30帧的4维特征序列，shape=(30,4) - [x, y, z, intensity]"""
        if len(self.feature_window) == 30:
            return np.array(self.feature_window)
        return None


def main():
    """主函数示例 - 实时4维特征数据读取 [x, y, z, intensity]"""
    CLI_PORT = 'COM4'
    DATA_PORT = 'COM6'
    reader = RealTimeRadarReader(CLI_PORT, DATA_PORT)
    try:
        print("正在连接雷达...")
        if not reader.connect():
            print("连接失败，请检查串口配置")
            return
        print("正在发送配置...")
        try:
            import os
            config_path = os.path.join(os.path.dirname(__file__), '3m.cfg')
            with open(config_path, 'r') as f:
                config_lines = f.readlines()
            if reader.send_config(config_lines):
                print(f"✓ 配置 {config_path} 发送成功")
            else:
                print(f"⚠ 配置 {config_path} 发送失败，但连接正常")
        except FileNotFoundError:
            print(f"⚠ 未找到配置文件: {config_path}，使用默认配置")
        except Exception as e:
            print(f"⚠ 配置发送异常: {e}")
        print("开始读取数据...")
        reader.start_reading()
        print("=" * 60)
        print("实时4维特征数据读取已启动 [x, y, z, intensity]")
        print("按 Ctrl+C 停止...")
        print("=" * 60)
        while True:
            frame = reader.get_latest_frame()
            if frame:
                print(f"\n时间: {datetime.now().strftime('%H:%M:%S.%f')[:-3]}")
                # 打印目标坐标
                if 'trackData' in frame and 'numDetectedTracks' in frame and frame['numDetectedTracks'] > 0:
                    print(f"检测到 {frame['numDetectedTracks']} 个目标:")
                    for track in frame['trackData'][:frame['numDetectedTracks']]:
                        if len(track) >= 4:
                            x, y, z = track[1], track[2], track[3]
                            
                            # 尝试从点云数据获取对应的SNR强度
                            intensity_str = "N/A"
                            if ('pointCloud' in frame and 
                                'numDetectedPoints' in frame and 
                                frame['numDetectedPoints'] > 0):
                                
                                points = frame['pointCloud'][:frame['numDetectedPoints']]
                                if len(points) > 0 and points.shape[1] >= 5:
                                    # 找到最近的点云点的SNR
                                    distances = np.sqrt((points[:, 0] - x)**2 + 
                                                      (points[:, 1] - y)**2 + 
                                                      (points[:, 2] - z)**2)
                                    nearest_idx = np.argmin(distances)
                                    snr_intensity = points[nearest_idx, 4]
                                    intensity_str = f"{snr_intensity:.1f} dB (SNR)"
                            
                            print(f"  TrackID: {int(track[0])} 位置: X={x:.2f}, Y={y:.2f}, Z={z:.2f}")
                            print(f"          强度: {intensity_str}")
                elif 'pointCloud' in frame and 'numDetectedPoints' in frame and frame['numDetectedPoints'] > 0:
                    points = frame['pointCloud'][:frame['numDetectedPoints']]
                    points_np = np.array(points)
                    if len(points_np) > 0:
                        center = np.mean(points_np[:, :3], axis=0)
                        # 显示SNR强度信息
                        if points_np.shape[1] >= 5:
                            # 直接使用雷达提供的SNR
                            avg_snr = np.mean(points_np[:, 4])
                            max_snr = np.max(points_np[:, 4])
                            min_snr = np.min(points_np[:, 4])
                            print(f"点云中心: X={center[0]:.2f}, Y={center[1]:.2f}, Z={center[2]:.2f}")
                            print(f"          SNR强度: 平均={avg_snr:.1f}dB, 最大={max_snr:.1f}dB, 最小={min_snr:.1f}dB (共{len(points_np)}点)")
                        elif points_np.shape[1] >= 4:
                            # 基础4维数据，使用多普勒估算
                            doppler_data = points_np[:, 3]
                            estimated_intensity = np.mean(np.abs(doppler_data)) * 5.0
                            print(f"点云中心: X={center[0]:.2f}, Y={center[1]:.2f}, Z={center[2]:.2f}")
                            print(f"          估算强度: {estimated_intensity:.1f} (基于多普勒，共{len(points_np)}点)")
                        else:
                            print(f"点云中心: X={center[0]:.2f}, Y={center[1]:.2f}, Z={center[2]:.2f} (共{len(points_np)}点)")
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\n正在停止...")
    finally:
        reader.stop_reading()
        reader.disconnect()
        print("程序已退出")

if __name__ == "__main__":
    main() 