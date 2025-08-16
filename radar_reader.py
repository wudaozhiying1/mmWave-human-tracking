#!/usr/bin/env python3
"""
雷达数据读取器
从串口4和6读取IWR6843雷达数据
"""

import serial
import time
import struct
import threading
import json
import os
import sys
from datetime import datetime

# 添加decoding目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
decoding_dir = os.path.join(current_dir, 'decoding')
if decoding_dir not in sys.path:
    sys.path.insert(0, decoding_dir)

try:
    from parseFrame import parseStandardFrame
    from demo_defines import *
    from tlv_defines import *
except ImportError as e:
    print(f"❌ 导入错误: {e}")
    print(f"📁 当前目录: {current_dir}")
    print(f"📁 decoding目录: {decoding_dir}")
    raise

class RadarReader:
    def __init__(self, cli_com='COM4', data_com='COM6', baud_rate=921600):
        """
        初始化雷达读取器
        Args:
            cli_com: CLI串口名称 (COM4)
            data_com: 数据串口名称 (COM6)
            baud_rate: 波特率
        """
        self.cli_com = cli_com
        self.data_com = data_com
        self.baud_rate = baud_rate
        
        # 串口连接
        self.cli_serial = None
        self.data_serial = None
        
        # 控制标志
        self.is_running = False
        
        # 数据存储
        self.data4 = []
        self.raw_frames = []  # 存储原始帧数据
        
        # 魔法字
        self.MAGIC_WORD = b'\x02\x01\x04\x03\x06\x05\x08\x07'
        
        # 创建数据保存文件夹
        self.create_data_folder()
        
        print(f"🚀 雷达读取器初始化完成")
        print(f"📡 CLI串口: {cli_com}")
        print(f"📡 数据串口: {data_com}")
        print(f"⚡ 波特率: {baud_rate}")
        
    def create_data_folder(self):
        """创建数据保存文件夹"""
        try:
            # 创建主文件夹
            self.data_folder = "radar_data"
            if not os.path.exists(self.data_folder):
                os.makedirs(self.data_folder)
                print(f"📁 创建数据文件夹: {self.data_folder}")
            
            # 创建时间戳子文件夹
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.session_folder = os.path.join(self.data_folder, f"session_{timestamp}")
            os.makedirs(self.session_folder)
            print(f"📁 创建会话文件夹: {self.session_folder}")
            
            # 创建子文件夹
            self.pointcloud_folder = os.path.join(self.session_folder, "pointcloud")
            self.tracking_folder = os.path.join(self.session_folder, "tracking")
            self.raw_folder = os.path.join(self.session_folder, "raw")
            
            os.makedirs(self.pointcloud_folder)
            os.makedirs(self.tracking_folder)
            os.makedirs(self.raw_folder)
            
            print(f"📁 创建子文件夹:")
            print(f"   📊 点云数据: {self.pointcloud_folder}")
            print(f"   🎯 跟踪数据: {self.tracking_folder}")
            print(f"   📦 原始数据: {self.raw_folder}")
            
        except Exception as e:
            print(f"❌ 创建文件夹失败: {e}")
            # 使用当前目录作为备选
            self.session_folder = "."
            self.pointcloud_folder = "."
            self.tracking_folder = "."
            self.raw_folder = "."
        
    def connect_ports(self):
        """连接串口"""
        try:
            print(f"\n🔌 正在连接串口...")
            
            # 连接CLI串口
            print(f"   📡 连接CLI串口: {self.cli_com}")
            self.cli_serial = serial.Serial(
                self.cli_com, 
                115200,  # CLI串口通常使用115200波特率
                timeout=1
            )
            print(f"   ✅ CLI串口连接成功")
            
            # 连接数据串口
            print(f"   📡 连接数据串口: {self.data_com}")
            self.data_serial = serial.Serial(
                self.data_com, 
                self.baud_rate,  # 数据串口使用921600波特率
                timeout=1
            )
            print(f"   ✅ 数据串口连接成功")
            
            print(f"✅ 所有串口连接成功")
            return True
            
        except Exception as e:
            print(f"❌ 串口连接失败: {e}")
            return False
    
    def load_config_file(self, config_path):
        """加载配置文件"""
        try:
            with open(config_path, 'r') as f:
                return [line.strip() for line in f.readlines()]
        except Exception as e:
            print(f"❌ 加载配置文件失败: {e}")
            return []
    
    def send_config(self, config_lines):
        """发送配置命令到CLI串口 - 使用gui_parser.py的方法"""
        try:
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
                
                print(f"   📤 {line.strip()} -> {ack.decode().strip()}")
                
                # 读取第二行确认
                ack = self.cli_serial.readline()
                if ack:
                    print(f"   📤 确认: {ack.decode().strip()}")
            
            # 给缓冲区一些时间清理
            time.sleep(0.03)
            self.cli_serial.reset_input_buffer()
            
            print(f"✅ 配置文件发送完成")
            return True
            
        except Exception as e:
            print(f"❌ 发送配置文件失败: {e}")
            return False
    
    def disconnect_ports(self):
        """断开串口连接"""
        if self.cli_serial and self.cli_serial.is_open:
            self.cli_serial.close()
        if self.data_serial and self.data_serial.is_open:
            self.data_serial.close()
        print("🔌 串口已断开")
    
    def read_frame(self, serial_port):
        """读取一帧数据 - 使用gui_parser.py的方法"""
        try:
            # 查找魔法字
            index = 0
            magicByte = serial_port.read(1)
            frameData = bytearray(b'')
            
            while True:
                # 检查是否有数据
                if len(magicByte) < 1:
                    print("❌ 串口超时，没有数据")
                    return None
                
                # 找到匹配的字节
                if magicByte[0] == self.MAGIC_WORD[index]:
                    index += 1
                    frameData.append(magicByte[0])
                    if index == 8:  # 找到完整的魔法字
                        break
                    magicByte = serial_port.read(1)
                else:
                    # 重置索引
                    if index == 0:
                        magicByte = serial_port.read(1)
                    index = 0
                    frameData = bytearray(b'')
            
            # 读取版本号
            versionBytes = serial_port.read(4)
            frameData += bytearray(versionBytes)
            
            # 读取长度
            lengthBytes = serial_port.read(4)
            frameData += bytearray(lengthBytes)
            frameLength = int.from_bytes(lengthBytes, byteorder='little')
            
            # 减去已经读取的字节（魔法字、版本、长度）
            frameLength -= 16
            
            # 读取帧的其余部分
            frameData += bytearray(serial_port.read(frameLength))
            
            return frameData
            
        except Exception as e:
            print(f"❌ 读取帧数据失败: {e}")
            return None
    
    def parse_frame(self, frame_data):
        """解析帧数据"""
        try:
            parsed_data = parseStandardFrame(frame_data)
            return parsed_data
        except Exception as e:
            print(f"❌ 解析帧数据失败: {e}")
            return None
    
    def read_port_data(self, serial_port, port_name, data_list):
        """读取单个串口的数据"""
        print(f"📊 开始读取{port_name}数据...")
        
        frame_count = 0
        last_debug_time = time.time()
        
        while self.is_running:
            try:
                frame_data = self.read_frame(serial_port)
                if frame_data:
                    frame_count += 1
                    print(f"📦 读取到第 {frame_count} 帧数据，长度: {len(frame_data)} 字节")
                    
                    # 保存原始帧数据
                    raw_frame_info = {
                        'timestamp': time.time(),
                        'frame_number': frame_count,
                        'frame_length': len(frame_data),
                        'frame_data_hex': frame_data.hex()
                    }
                    self.raw_frames.append(raw_frame_info)
                    
                    # 解析数据
                    parsed_data = self.parse_frame(frame_data)
                    if parsed_data:
                        print(f"✅ 解析成功，包含 {len(parsed_data)} 个TLV")
                        
                        # 提取点云数据
                        if 'pointCloud' in parsed_data:
                            points = parsed_data['pointCloud']
                            num_points = parsed_data.get('numDetectedPoints', 0)
                            print(f"🎯 检测到 {num_points} 个点云点")
                            
                            if num_points > 0:
                                print(f"📊 点云数据示例:")
                                for i in range(min(3, num_points)):  # 只显示前3个点
                                    if i < len(points):
                                        point = points[i]
                                        if len(point) >= 3:
                                            print(f"   点{i+1}: x={point[0]:.2f}, y={point[1]:.2f}, z={point[2]:.2f}")
                            
                            for i in range(num_points):
                                if i < len(points):
                                    point = points[i]
                                    if len(point) >= 3:
                                        coord = {
                                            'timestamp': time.time(),
                                            'port': port_name,
                                            'x': float(point[0]),
                                            'y': float(point[1]),
                                            'z': float(point[2]),
                                            'doppler': float(point[3]) if len(point) > 3 else 0.0,
                                            'snr': float(point[4]) if len(point) > 4 else 0.0,
                                            'noise': float(point[5]) if len(point) > 5 else 0.0
                                        }
                                        data_list.append(coord)
                        else:
                            print(f"⚠️  未找到点云数据")
                            print(f"📋 可用数据键: {list(parsed_data.keys())}")
                        
                        # 提取跟踪数据
                        if 'trackData' in parsed_data:
                            tracks = parsed_data['trackData']
                            num_tracks = parsed_data.get('numDetectedTracks', 0)
                            print(f"🎯 检测到 {num_tracks} 个跟踪目标")
                            
                            for i in range(num_tracks):
                                if i < len(tracks):
                                    track = tracks[i]
                                    track_info = {
                                        'timestamp': time.time(),
                                        'port': port_name,
                                        'track_id': int(track[0]),
                                        'pos_x': float(track[1]),
                                        'pos_y': float(track[2]),
                                        'pos_z': float(track[3]),
                                        'vel_x': float(track[4]),
                                        'vel_y': float(track[5]),
                                        'vel_z': float(track[6])
                                    }
                                    print(f"🎯 跟踪目标 {track_info['track_id']}: 位置=({track_info['pos_x']:.2f}, {track_info['pos_y']:.2f}, {track_info['pos_z']:.2f})")
                    else:
                        print(f"❌ 帧数据解析失败")
                else:
                    # 每10秒打印一次调试信息
                    current_time = time.time()
                    if current_time - last_debug_time >= 10:
                        print(f"⏳ 等待数据中... 串口状态: {serial_port.in_waiting} 字节可用")
                        last_debug_time = current_time
                
                time.sleep(0.01)  # 短暂休眠
                
            except Exception as e:
                print(f"❌ {port_name} 读取异常: {e}")
                time.sleep(0.1)
        
        print(f"⏹️  {port_name} 读取线程已停止")
    
    def start_reading(self):
        """开始读取数据"""
        if not self.data_serial:
            print("❌ 数据串口未连接")
            return False
        
        self.is_running = True
        
        # 启动数据读取线程
        thread = threading.Thread(
            target=self.read_port_data,
            args=(self.data_serial, 'RADAR', self.data4)  # 使用data4存储所有数据
        )
        thread.daemon = True
        thread.start()
        
        print("✅ 数据读取已启动")
        return True
    
    def stop_reading(self):
        """停止读取数据"""
        self.is_running = False
        print("⏹️  正在停止数据读取...")
        time.sleep(2)  # 等待线程结束
    
    def save_data(self, filename_prefix="radar_data"):
        """保存数据到文件"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 分离点云数据和跟踪数据
        pointcloud_data = []
        tracking_data = []
        
        for item in self.data4:
            if 'track_id' in item:
                tracking_data.append(item)
            else:
                pointcloud_data.append(item)
        
        # 保存点云数据
        if pointcloud_data:
            filename = os.path.join(self.pointcloud_folder, f"pointcloud_{timestamp}.json")
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(pointcloud_data, f, indent=2, ensure_ascii=False)
            print(f"💾 点云数据已保存: {filename} ({len(pointcloud_data)} 个点)")
        
        # 保存跟踪数据
        if tracking_data:
            filename = os.path.join(self.tracking_folder, f"tracking_{timestamp}.json")
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(tracking_data, f, indent=2, ensure_ascii=False)
            print(f"💾 跟踪数据已保存: {filename} ({len(tracking_data)} 个目标)")
        
        # 保存完整数据
        if self.data4:
            filename = os.path.join(self.session_folder, f"complete_data_{timestamp}.json")
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(self.data4, f, indent=2, ensure_ascii=False)
            print(f"💾 完整数据已保存: {filename} ({len(self.data4)} 个数据点)")
        
        # 保存原始帧数据
        if self.raw_frames:
            filename = os.path.join(self.raw_folder, f"raw_frames_{timestamp}.json")
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(self.raw_frames, f, indent=2, ensure_ascii=False)
            print(f"💾 原始帧数据已保存: {filename} ({len(self.raw_frames)} 帧)")
    
    def print_statistics(self):
        """打印统计信息"""
        print("\n📊 === 数据统计 ===")
        
        # 分离点云数据和跟踪数据
        pointcloud_count = sum(1 for item in self.data4 if 'track_id' not in item)
        tracking_count = sum(1 for item in self.data4 if 'track_id' in item)
        
        print(f"📡 总数据点: {len(self.data4)}")
        print(f"📊 点云数据: {pointcloud_count} 个点")
        print(f"🎯 跟踪数据: {tracking_count} 个目标")
        print(f"📦 原始帧数: {len(self.raw_frames)} 帧")
        print("=" * 30)

def main():
    """主函数"""
    print("🚀 雷达数据读取器")
    print("=" * 50)
    
    # 创建读取器
    reader = RadarReader(
        cli_com='COM4',
        data_com='COM6',
        baud_rate=921600
    )
    
    try:
        # 连接串口
        if not reader.connect_ports():
            print("❌ 串口连接失败，程序退出")
            return
        
        # 加载并发送配置文件
        config_path = os.path.join(decoding_dir, '3m.cfg')
        config_lines = reader.load_config_file(config_path)
        if config_lines:
            if not reader.send_config(config_lines):
                print("❌ 配置文件发送失败")
                return
        else:
            print("❌ 配置文件加载失败")
            return
        
        # 等待雷达启动
        print("⏳ 等待雷达启动...")
        time.sleep(2)
        
        # 检查数据串口是否有数据
        print("🔍 检查数据串口状态...")
        if reader.data_serial.in_waiting > 0:
            print(f"📡 数据串口有 {reader.data_serial.in_waiting} 字节数据")
            # 读取前几个字节看看
            test_data = reader.data_serial.read(min(16, reader.data_serial.in_waiting))
            print(f"🔍 前16字节: {test_data.hex()}")
        else:
            print("📡 数据串口暂无数据")
        
        # 等待更长时间让雷达开始发送数据
        print("⏳ 等待雷达开始发送数据...")
        time.sleep(5)
        
        # 再次检查数据串口
        if reader.data_serial.in_waiting > 0:
            print(f"📡 5秒后数据串口有 {reader.data_serial.in_waiting} 字节数据")
        else:
            print("⚠️  5秒后数据串口仍无数据，可能雷达未正确启动")
        
        # 开始读取数据
        if not reader.start_reading():
            print("❌ 启动数据读取失败")
            return
        
        print("\n📊 数据读取中... (按 Ctrl+C 停止)")
        print("=" * 50)
        
        # 主循环
        start_time = time.time()
        last_stats_time = start_time
        
        while True:
            try:
                time.sleep(5)  # 每5秒检查一次
                current_time = time.time()
                
                # 每30秒打印一次统计信息
                if current_time - last_stats_time >= 30:
                    reader.print_statistics()
                    last_stats_time = current_time
                
                # 每60秒保存一次数据
                if current_time - start_time >= 60:
                    reader.save_data()
                    start_time = current_time
                    
            except KeyboardInterrupt:
                print("\n⏹️  收到停止信号...")
                break
            except Exception as e:
                print(f"❌ 主循环异常: {e}")
                break
    
    except Exception as e:
        print(f"❌ 程序异常: {e}")
    
    finally:
        # 清理资源
        reader.stop_reading()
        reader.save_data()  # 最终保存
        reader.disconnect_ports()
        print("✅ 程序已退出")

if __name__ == "__main__":
    main() 