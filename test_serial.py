#!/usr/bin/env python3
"""
简单的串口测试脚本
"""

import serial
import time
import serial.tools.list_ports

def test_serial_ports():
    """测试串口连接"""
    print("🔍 检测可用串口...")
    ports = serial.tools.list_ports.comports()
    for port in ports:
        print(f"   📡 {port.device}: {port.description}")
    
    # 测试CLI端口
    print("\n🔍 测试CLI端口 (COM4)...")
    try:
        cli_serial = serial.Serial('COM4', 115200, timeout=1)
        print("✅ CLI端口连接成功")
        
        # 发送测试命令
        cli_serial.write(b'sensorStop\n')
        time.sleep(0.1)
        response = cli_serial.readline().decode().strip()
        print(f"📤 sensorStop -> {response}")
        
        cli_serial.close()
    except Exception as e:
        print(f"❌ CLI端口连接失败: {e}")
    
    # 测试数据端口
    print("\n🔍 测试数据端口 (COM6)...")
    try:
        data_serial = serial.Serial('COM6', 115200, timeout=1)
        print("✅ 数据端口连接成功")
        
        # 监听数据
        print("📡 监听数据端口...")
        start_time = time.time()
        data_count = 0
        
        while time.time() - start_time < 10:  # 监听10秒
            if data_serial.in_waiting > 0:
                data = data_serial.read(data_serial.in_waiting)
                data_count += len(data)
                print(f"📊 接收到 {len(data)} 字节数据")
                print(f"📊 前16字节: {data[:16].hex()}")
            
            time.sleep(0.1)
        
        if data_count > 0:
            print(f"✅ 总共接收到 {data_count} 字节数据")
        else:
            print("⚠️ 没有接收到任何数据")
        
        data_serial.close()
        
    except Exception as e:
        print(f"❌ 数据端口连接失败: {e}")

if __name__ == "__main__":
    test_serial_ports() 