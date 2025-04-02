import socket
import sys
import re
import time
import json
import os
from src.abstract.Instrument import Instrument

class FSV3030(Instrument):
    """
    class to control the Rohde & Schwarz FSV3030 Spectrum Analyzer.
    """
    def __init__(self, ip_address, port=5025, set_defaults=True, name=None):
        """
        Initialize the connection to the device.
        
        :param ip_address: IP address of the device
        :param port: Port number
        """
        name = name if name else 'FSV3030_' + str(ip_address)
        super().__init__(name)

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.ip_address = ip_address
        self.port = port
        self.sock.settimeout(3)
        self.set_defaults = set_defaults
    

    def initialize(self):
        """
        Initialize the instrument by activating remote mode.
        """
        # Connect to the server
        try:
            self.sock.connect((self.ip_address, self.port))
        except socket.error as e:
            print(f"Socket error: {e}")
            sys.exit(1)
            
        # Set default parameters if required
        if self.set_defaults:
            self.set_sweep_type('FFT')
            

    def info(self, verbose=False):
        """
        Get information about the instrument.
        
        :param verbose: If True, display detailed information.
        """
        info = f"FSV3030 Spectrum Analyzer connected at {self.ip_address}:{self.port}"
        info += f"\nName: {self.name}"
        if verbose:
            info += "\nIDN string:"
            info += "\n" + self.query('*IDN?')
        return info


    def acq_get_spectrum(self, center_freq='auto', span='auto', num_points='auto', sweep_count='auto', trace_mode='auto') -> tuple[list[float], list[float]]:
        """
        Get the power spectrum from the device.
        
        **Note**: if all parameters are set to 'auto', the spectrum will be the one currently displayed on the device.
        
        :param center_freq: Center frequency in Hz
        :param span: Frequency span in Hz
        :param num_points: Number of points or 'auto'
        :param sweep_count: Number of sweeps
        :return: Tuple of (x_data, y_data)
        """
        # Check that there are no queued commands
        self.wait_opc(timeout=15)
        # Set up the frequency sweep
        if center_freq != 'auto':
            self.set_frequency_center(center_freq)
        else:
            center_freq = self.get_frequency_center()
    
        if span != 'auto':
            self.set_frequency_span(span)
        else:
            span = self.get_frequency_span()
        # Set the frequency step if provided
        if num_points != 'auto' and isinstance(num_points, int):
            if num_points < 101:
                raise ValueError('Number of points should be at least 101')
            self.set_sweep_points(num_points)
        # Set sweep parameters
        if sweep_count != 'auto':
            self.set_sweep_count(sweep_count)
        if trace_mode != 'auto':
            self.set_trace_mode(trace_mode)
        
        # Perform frequency sweep
        self.send_command('INITiate')
        self.wait_opc()  # Wait for the operation to complete
        
        # Get Y-axis data
        trace_data = self.query('TRACe:DATA? TRACe1')
        trace_data = re.sub(r'E', 'e', trace_data).split(',')
        trace_len = len(trace_data)
        
        # Extract the data from the list
        y_data = [float(x) for x in trace_data if x != '']  # Remove empty
        
        # Reconstruct x data (frequency for each point) as it cannot be directly read from the instrument
        step_size = span / (trace_len - 1)
        x_data = [(center_freq - (span/2)) + i * step_size for i in range(trace_len)]
        
        return x_data, y_data
        
    # ----------------- Analysis functions -----------------
    def acq_get_peak(self, x_data: list[float], y_data: list[float]) -> tuple:
        peak_idx = y_data.index(max(y_data))
        peak_freq = x_data[peak_idx]
        peak_power = y_data[peak_idx]
        
        return peak_freq, peak_power
    
    # ----------------- Setters -----------------
    def set_resolution_bandwidth(self, rbw: float) -> None:
        """
        Set the resolution bandwidth.
        
        :param rbw: Resolution bandwidth in Hz
        """
        self.send_command(f'BANDwidth:RESolution {rbw}')
        
    def set_video_bandwidth_ratio(self, ratio: float) -> None:
        """
        Set the video bandwidth ratio.
        
        :param ratio: Video bandwidth ratio
        """
        self.send_command('BANDwidth:VIDeo:AUTO ON')
        self.send_command(f'BANDwidth:VIDeo:RATio {ratio}')
        
    def set_sweep_time(self, sweep_time: float | str) -> None:
        """
        Set the sweep time.
        
        :param sweep_time: Sweep time in seconds or 'auto'
        """
        if sweep_time == 'auto':
            self.send_command('SWEep:TIME:AUTO ON')
        else:
            self.send_command(f'SWEep:TIME {sweep_time}')
    
    def set_sweep_type(self, sweep_type: str) -> None:
        """
        Set the sweep type.
    
        :param sweep_type: Sweep type (AUTO, FFT, Sweep)
        """
        commands = {
            'AUTO': 'SWEep:TYPE:AUTO ON',
            'FFT': 'SWEep:TYPE FFT',
            'Sweep': 'SWEep:TYPE Sweep'
        }
        try:
            self.send_command(commands[sweep_type])
        except KeyError:
            raise ValueError('Invalid sweep type, allowed values are: AUTO, FFT, Sweep')
    
    def set_trace_mode(self, mode: str) -> None:
        """
        Set the trace mode. I.e. select wheter the different sweeps (counts) are averaged, max hold or min hold.
        
        :param mode: Trace mode (MAXHold, AVERage, MINHold)
        """
        commands = {
            'WRITE': 'DISPlay:TRACe1:MODE MAXHold',
            'AVERAGE': 'DISPlay:TRACe1:MODE AVERage',
            'VIEW': 'DISPlay:TRACe1:MODE MINHold'
        }
        try:
            self.send_command(commands[mode.upper()])
        except KeyError:
            raise ValueError('Invalid trace mode, allowed values are: WRITE, AVERAGE, VIEW') 
    
    def set_sweep_count(self, count: int) -> None:
        """
        Set the sweep count.
        
        :param count: Number of sweeps
        """
        self.send_command(f'SWEep:Count {count}')
    
    def set_sweep_points(self, points: int) -> None:
        """
        Set the number of sweep points.
        
        :param points: Number of sweep points
        """
        self.send_command(f'SENSe:SWEep:POINts {points}')
        
    def set_dB_reference(self, reference: float) -> None:
        """
        Set the dB reference.
        
        :param reference: Reference value in dBm
        """
        self.send_command(f'DISPlay:TRACe:Y:RLEVel {reference}dBm')
    
    def set_dB_ref_position(self, position: float) -> None:
        """
        Set the dB reference position as screen percentage from the bottom.
        
        :param position: Reference position in percentage
        """
        if position < 0 or position > 100:
            raise ValueError('Position should be between 0 and 100')
        self.send_command(f'DISPlay:TRACe:Y:RPOSition {position}PCT')
            
    def set_dB_range(self, range: float) -> None:
        """
        Set the dB range.
        
        :param range: Range value in dB
        """
        self.send_command(f'DISPlay:TRACe:Y {range}dB')
    
    def set_frequency_center(self, freq: float) -> None:
        """
        Set the center frequency.
        
        :param freq: Center frequency in Hz
        """
        self.send_command(f'FREQuency:CENTer {freq}')
    
    def set_frequency_span(self, span: float) -> None:
        """
        Set the frequency span.
        
        :param span: Frequency span in Hz
        """
        self.send_command(f'FREQuency:SPAN {span}')
        
    # ----------------- Getters -----------------
    def get_resolution_bandwidth(self) -> float:
        return self.query('BANDwidth:RESolution?', return_type='float')
    
    def get_video_bandwidth_ratio(self) -> float:
        return self.query('BANDwidth:VIDeo:RATio?', return_type='float')
    
    def get_sweep_time(self):
        response = self.query('SWEep:TIME?', return_type='str')
        return response if response.lower() == 'auto on' else float(response)
    
    def get_sweep_type(self) -> str:
        response = self.query('SWEep:TYPE?', return_type='str')
        if 'AUTO' in response.upper():
            return 'AUTO'
        elif 'FFT' in response.upper():
            return 'FFT'
        elif 'SWEEP' in response.upper():
            return 'Sweep'
        else:
            return response
    
    def get_sweep_count(self) -> int:
        return self.query('SWEep:Count?', return_type='int')
    
    def get_sweep_points(self) -> int:
        return self.query('SENSe:SWEep:POINts?', return_type='int')
    
    def get_dB_reference(self) -> float:
        value = self.query('DISPlay:TRACe:Y:RLEVel?', return_type='str')
        return float(value.rstrip('dBm'))
    
    def get_dB_range(self) -> float:
        value = self.query('DISPlay:TRACe:Y?', return_type='str')
        return float(value.rstrip('dB'))
    
    def get_frequency_center(self) -> float:
        return self.query('FREQuency:CENTer?', return_type='float')
    
    def get_frequency_span(self) -> float:
        return self.query('FREQuency:SPAN?', return_type='float')
        
    # ----------------- Interface functions ----------------- 
    def send_command(self, command: str) -> None:
        """
        Send a command to the instrument.

        :param command: The command string to send.
        """
        self.sock.send((command + "\r\n").encode('utf-8'))
    
    def _read_until_EOF(self, buffer_size: int = 1024) -> bytes:
        """
        Read data from the socket until EOF.

        :param buffer_size: Size of each read buffer.
        :return: The received data as bytes.
        """
        data = b''
        while True:
            part = self.sock.recv(buffer_size)
            data += part
            if len(part) < buffer_size:
                break
        return data
    
    def read(self) -> str:
        """
        Read the response from the instrument.

        :return: The decoded response string.
        """
        response = self._read_until_EOF()
        return response.decode('utf-8')
    
    def query(self, command: str, timeout: float = 0.5, return_type: str = 'str') -> any:
        """
        Send a command and retrieve the response.

        :param command: The command string to send.
        :param timeout: Time to wait for the response.
        :param return_type: The expected return type ('float', 'int', 'bool', 'str').
        :return: The response converted to the specified type.
        """
        self.send_command(command)
        time.sleep(timeout)
        result = self.read().strip()
        return {
            'float': float,
            'int': int,
            'bool': lambda x: x == '1',
            'str': str
        }.get(return_type, lambda x: ValueError('Invalid return type'))(result)
    
    def query_opc(self) -> bool:
        """
        Query the *OPC? command to check if the operation is complete.

        :return: True if operation is complete, False otherwise.
        """
        return self.query('*OPC?', return_type='bool')
    
    def wait_opc(self, time_sleep: float = 0.1, timeout: float = 10) -> None:
        """
        Wait until the operation is complete or timeout is reached.

        :param time_sleep: Time to sleep between checks.
        :param timeout: Maximum time to wait for operation completion.
        :raises TimeoutError: If the operation does not complete within the timeout.
        """
        start_time = time.time()
        while not self.query_opc():
            if time.time() - start_time > timeout:
                raise TimeoutError('Timeout waiting for OPC')
            time.sleep(time_sleep)
            
    def set_auto(self, center_freq:float, config: str ='FSV3030_default_configs.json'):
        """
        set the instrument to default parameters read from json config file
        
        :param config: json file containing default parameters
        """
        # First reset the instrument
        self._reset()
        # Set center frequency
        self.set_frequency_center(center_freq)
        
        # Calculate path to config file relative to the current file
        current_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(current_dir, '..', 'cfgs', config)
        
        # Read the config file
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
            self.set_frequency_span(config_dict["frequency_span"])
            self.set_resolution_bandwidth(config_dict["resolution_bandwidth"])
            self.set_video_bandwidth_ratio(config_dict["video_bandwidth_ratio"])
            self.set_sweep_time(config_dict["sweep_time"])
            self.set_sweep_type(config_dict["sweep_type"])
            self.set_sweep_count(config_dict["sweep_count"])
            self.set_sweep_points(config_dict["sweep_points"])
            self.set_dB_reference(config_dict["dB_reference"])
            self.set_dB_range(config_dict["dB_range"])
            self.set_dB_ref_position(config_dict["dB_ref_position"])
            self.set_trace_mode(config_dict["trace_mode"])
        
        self.wait_opc()

    def _Instrument_activate(self):
        """
        For the FSV3030, no activation is required. This is a dummy function.
        """
        pass

    def reset(self):
        self.send_command('*RST') 
        self.wait_opc()  # Wait for the operation to complete
        print("FSV3030 reset to default state.")      
    
    def close_connection(self):
        self.sock.close()
        print("FSV3030 connection closed.")
        
    def shutdown(self): # To implement real shutdown in the future
        self.close_connection()
        print("FSV3030 shutdown.")
        
    def kill(self):
        self.reset()
        self.close_connection()
        print("FSV3030 kill.")
        
    def __del__(self):
        self.close_connection()