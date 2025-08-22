#!/usr/bin/env python3
"""
Simple Serial Port Test Script
"""

import serial
import time
import serial.tools.list_ports

def test_serial_ports():
    """Test serial port connections"""
    print("Detecting available serial ports...")
    ports = serial.tools.list_ports.comports()
    for port in ports:
        print(f"   {port.device}: {port.description}")
    
    # Test CLI port
    print("\nTesting CLI port (COM4)...")
    try:
        cli_serial = serial.Serial('COM4', 115200, timeout=1)
        print("CLI port connected successfully")
        
        # Send test command
        cli_serial.write(b'sensorStop\n')
        time.sleep(0.1)
        response = cli_serial.readline().decode().strip()
        print(f"sensorStop -> {response}")
        
        cli_serial.close()
    except Exception as e:
        print(f"CLI port connection failed: {e}")
    
    # Test data port
    print("\nTesting data port (COM6)...")
    try:
        data_serial = serial.Serial('COM6', 115200, timeout=1)
        print("Data port connected successfully")
        
        # Listen for data
        print("Listening to data port...")
        start_time = time.time()
        data_count = 0
        
        while time.time() - start_time < 10:  # Listen for 10 seconds
            if data_serial.in_waiting > 0:
                data = data_serial.read(data_serial.in_waiting)
                data_count += len(data)
                print(f"Received {len(data)} bytes of data")
                print(f"First 16 bytes: {data[:16].hex()}")
            
            time.sleep(0.1)
        
        if data_count > 0:
            print(f"Total received {data_count} bytes of data")
        else:
            print("No data received")
        
        data_serial.close()
        
    except Exception as e:
        print(f"Data port connection failed: {e}")

if __name__ == "__main__":
    test_serial_ports() 