import sys
import os
import time
from pathlib import Path

# Add parent directory to path to import the SIM928 module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.instruments.SIM928 import SIM928

def test_real_sim928(port_name, address=1):
    """
    Test the SIM928 instrument class with a real device.
    
    This test will:
    1. Initialize the connection
    2. Print instrument information
    3. Set the voltage to 0V
    4. Turn output on
    5. Ramp the voltage up and down
    6. Turn output off
    7. Close the connection
    
    Parameters:
    -----------
    port_name : str
        Serial port name (e.g., 'COM9' on Windows, '/dev/ttyUSB0' on Linux)
    address : int
        Module address in the SIM900 mainframe (usually 1-8)
    """
    print("=" * 60)
    print(f"SIM928 Real Hardware Test on port {port_name}, address {address}")
    print("=" * 60)
    
    try:
        # 1. Initialize the connection
        print("\n[1] Initializing SIM928...")
        sim928 = SIM928(port_name, address)
        result = sim928.initialize()
        print(f"    {result}")
        
        # 2. Print instrument information
        print("\n[2] Getting instrument info...")
        info = sim928.info(verbose=True)
        print(info)
        # print(f"    {info.replace(chr(10), chr(10) + '    ')}")
        
        # 3. Set the voltage to 0V
        print("\n[3] Setting voltage to 0V...")
        sim928.set_voltage(0.0)
        time.sleep(0.5)  # Give it time to set
        current_voltage = sim928.get_voltage()
        print(f"    Current voltage reading: {current_voltage:.6f} V")
        
        # 4. Turn output on
        print("\n[4] Turning output ON...")
        sim928.set_output(True)
        time.sleep(0.5)
        output_state = "ON" if sim928._check_output() else "OFF"
        print(f"    Output state: {output_state}")
        
        # 5. Ramp the voltage up and down
        print("\n[5] Ramping voltage up and down...")
        voltages = [0.1, 0.5, 1.0, 1.5, 2.0, 1.5, 1.0, 0.5, 0.1, 0.0]
        
        for target_v in voltages:
            print(f"    Setting voltage to {target_v:.1f}V...")
            sim928.set_voltage(target_v)
            time.sleep(0.5)  # Give it time to set
            
            # Read back voltage
            current_v = sim928.get_voltage()
            print(f"    Measured voltage: {current_v:.6f}V")
        
        # 6. Turn output off
        print("\n[6] Turning output OFF...")
        sim928.set_output(False)
        time.sleep(0.5)
        output_state = "ON" if sim928._check_output() else "OFF"
        print(f"    Output state: {output_state}")
        
        # 7. Close the connection
        print("\n[7] Closing connection...")
        sim928.close_connection()
        print("    Connection closed.")
        
        print("\nTest completed successfully!")
        return True
        
    except Exception as e:
        print(f"\n[ERROR] Test failed: {str(e)}")
        return False


if __name__ == "__main__":
    # Get port name from command line arguments or prompt user
    if len(sys.argv) > 1:
        port_name = sys.argv[1]
    else:
        # Detect OS and suggest default port format
        if os.name == 'nt':  # Windows
            default_port = 'COM9'
        else:  # Assume Linux/Mac
            default_port = '/dev/ttyUSB0'
            
        port_name = input(f"Enter serial port name [{default_port}]: ")
        if not port_name:
            port_name = default_port
    
    # Get module address (optional)
    if len(sys.argv) > 2:
        try:
            address = int(sys.argv[2])
        except ValueError:
            address = 1
    else:
        addr_input = input("Enter module address in SIM900 mainframe [1]: ")
        try:
            address = int(addr_input) if addr_input else 1
        except ValueError:
            address = 1
    
    print(f"Testing SIM928 on port {port_name}, address {address}")
    test_real_sim928(port_name, address)