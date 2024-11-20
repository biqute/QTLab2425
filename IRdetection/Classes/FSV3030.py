import socket
import sys
import re
import time

class FSV3030:
    """
    class to control the Rohde & Schwarz FSV3030 Spectrum Analyzer.
    """
    def __init__(self, ip_address, port=5025, set_defaults=True):
        """
        Initialize the connection to the device.
        
        :param ip_address: IP address of the device
        :param port: Port number
        """
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.ip_address = ip_address
        self.port = port
        self.sock.settimeout(3)
        
        # Connect to the server
        try:
            self.sock.connect((ip_address, port))
        except socket.error as e:
            print(f"Socket error: {e}")
            sys.exit(1)
            
        # Set default parameters if required
        if set_defaults:
            self.set_sweep_type('FFT')
            self.send_command('SENSe:ADJust:LEVel')
            
    def get_spectrum(self, center_freq, span, num_points='auto', sweep_count=1):
        """
        Get the power spectrum from the device.
        
        :param start_freq: Start frequency in Hz
        :param stop_freq: Stop frequency in Hz
        :param step: Frequency step in Hz
        :return: List of power spectrum data
        """
        # Check that there are no queued commands
        self.wait_opc(timeout=150)
        # Set up the frequency sweep
        self.send_command(f'FREQuency:CENTer {center_freq}') # Set center frequency
        self.send_command(f'FREQuency:SPAN {span}') # Set span
        # Set the frequency step if provided
        if num_points != 'auto' and isinstance(num_points, int):
            if num_points < 101:
                raise ValueError('Number of points should be at least 101')
            self.send_command(f'SENS:SWEep:POINts {num_points}')  # Set number of points
        # Set sweep parameters
        self.send_command(f'SWEep:COUNt {sweep_count}') # Set sweep count 
        self.send_command('DISPlay:TRACe1:MODE AVERage') # Set trace mode to average
        
        # Perform frequency sweep
        self.send_command('INITiate')
        self.wait_opc() # Wait for the operation to complete
        
        # Get Y-axis data
        trace_data = self.query('TRACe:DATA? TRACe1')
        trace_data = re.sub(r'E', 'e', trace_data).split(',')
        trace_len = len(trace_data)
        # Extract the data from the list
        y_data = [float(x) for x in trace_data if x != '']  # Remove empty
        
        # Reconstruct x data (frequency for each point) as it can not be directly read from the instrument
        freq_start = self.query('FREQuency:STARt?', return_type='float')
        freq_span = self.query('FREQuency:SPAN?', return_type='float')
        step_size = freq_span / (trace_len-1)
        x_data = [freq_start + i * step_size for i in range(trace_len)]
        
        return x_data, y_data
    
    # ----------------- Setters -----------------
    def set_resolution_bandwidth(self, rbw):
        """
        Set the resolution bandwidth.
        
        :param rbw: Resolution bandwidth in Hz
        """
        self.send_command(f'BAND:RES {rbw}')
        
    def set_sweep_time(self, sweep_time):
        """
        Set the sweep time.
        
        :param sweep_time: Sweep time in seconds or 'auto'
        """
        if sweep_time == 'auto':
            self.send_command('SWEep:TIME:AUTO ON')
        else:
            self.send_command(f'SWEep:TIME {sweep_time}')
            
    def set_dB_offset(self, offset):
        """
        Set the dB offset.
        
        :param offset: Offset value in dB
        """
        self.send_command(f'DISP:TRAC:Y:RLEV:OFFS {offset}dB')
    
    def set_sweep_type(self, sweep_type):
        """
        Set the sweep type.
        
        :param sweep_type: Sweep type (AUTO, FFT, Sweep)
        """
        if sweep_type == 'AUTO':
            self.send_command('SWEep:TYPE:AUTO ON')
        elif sweep_type == 'FFT':
            self.send_command('SWEep:TYPE FFT')
        elif sweep_type == 'Sweep':
            self.send_command('SWEep:TYPE SWEep')
        else:
            raise ValueError('Invalid sweep type, allowed values are: AUTO, FFT, Sweep')
    
    # ----------------- Analysis functions -----------------
    def get_peak(self, x_data, y_data):
        peak_idx = y_data.index(max(y_data))
        peak_freq = x_data[peak_idx]
        peak_power = y_data[peak_idx]
        
        return peak_freq, peak_power
      
    # ----------------- Interface functions ----------------- 
    def send_command(self, command):
        self.sock.send((command + "\r\n").encode('utf-8'))
    
    def _read_until_EOF(self, buffer_size=1024):
        data = b''
        while True:
            part = self.sock.recv(buffer_size)
            data += part
            if len(part) < buffer_size:
                break
        return data
    
    def read(self):
        response = self._read_until_EOF()
        return response.decode('utf-8')
    
    def query(self, command, timeout=0.5, return_type='str'):
        self.send_command(command)
        time.sleep(timeout)
        result = self.read()
        if return_type == 'float':
            return float(result)
        elif return_type == 'int':
            return int(result)
        elif return_type == 'bool':
            test = result == '1\n'
            return result == '1\n'
        elif return_type == 'str':
            return result
        else:
            raise ValueError('Invalid return type')
    
    def query_opc(self):
        return self.query('*OPC?', return_type='bool')
    
    def wait_opc(self, time_sleep=0.1, timeout=10):
        start_time = time.time()
        while not self.query_opc():
            if time.time() - start_time > timeout:
                raise TimeoutError('Timeout waiting for OPC')
            time.sleep(time_sleep)
    
    def reset(self):
        self.send_command('*RST')        
    
    def close_connection(self):
        self.sock.close()

if __name__ == "__main__":
    fsv = FSV3030('192.168.3.50')
    # Set resolution bandwidth
    fsv.set_resolution_bandwidth(500)
    fsv.set_sweep_time('auto')
    #fsv.set_dB_offset(50)
    x_data, y_data = fsv.get_spectrum(6e9, 5e6, num_points=1000, sweep_count=60)
    peak_freq, peak_power = fsv.get_peak(x_data, y_data)
    #plot 
    import matplotlib.pyplot as plt
    plt.plot(x_data, y_data)
    # Highlight the peak
    #plt.axvline(x=peak_freq, color='r', linestyle='--', linewidth=1)
    plt.plot(peak_freq, peak_power, 'rx')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power (dBm)')
    plt.show()

    fsv.reset()
    fsv.close_connection()