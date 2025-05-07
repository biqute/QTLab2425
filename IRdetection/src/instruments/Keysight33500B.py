import socket
import sys
import re
import time
import json
import os
from src.abstract.Instrument import Instrument

class Keysight33500B(Instrument):
    """
    Class to control the Keysight 33500B Series Arbitrary Waveform Generator.
    Supports configuration of various waveforms including DC, square, and pulse.
    """
    
    def __init__(self, ip_address, port=5025, name=None):
        """
        Initialize the connection to the device.
        
        :param ip_address: IP address of the device
        :param port: Port number (default is 5025 for SCPI over socket)
        :param name: Optional custom name for the instrument
        """
        name = name if name else 'Keysight33500B_' + str(ip_address)
        super().__init__(name)

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.ip_address = ip_address
        self.port = port
        self.sock.settimeout(3)
        
        # Default parameters
        self.channel = 1
        
    def initialize(self):
        """
        Initialize the instrument by connecting to it.
        """
        try:
            self.sock.connect((self.ip_address, self.port))
            # Check if connection is successful by querying the IDN
            idn = self.query("*IDN?")
            print(idn)
            if not idn or "33522B" not in idn: # The model name in the IDN string is actually "33522B"
                raise ConnectionError(f"Connected to device but it doesn't appear to be a 33500 Series: {idn}")
            
            # Reset the instrument to a known state
            self.send_command("*RST")
            self.wait_opc()
            
            # Turn off all channels by default for safety
            for ch in [1, 2]:
                self.send_command(f"OUTPUT{ch} OFF")
                
            return f"Keysight33500B initialized at {self.ip_address}:{self.port}, IDN: {idn}"
        except socket.error as e:
            raise ConnectionError(f"Failed to connect to Keysight33500B at {self.ip_address}:{self.port}: {e}")
    
    def info(self, verbose=False):
        """
        Get information about the instrument.
        
        :param verbose: If True, display detailed information.
        """
        info = f"Keysight33500B at {self.ip_address}:{self.port}"
        info += f"\nName: {self.name}"
        if verbose:
            info += "\nIDN: " + self.query("*IDN?")
            info += "\nActive channel: " + str(self.channel)
            # Get output state for each channel
            output_states = []
            for ch in [1, 2]:
                state = self.query(f"OUTPUT{ch}:STATE?")
                output_states.append(f"CH{ch}: {'ON' if int(state) else 'OFF'}")
            info += "\nOutput states: " + ", ".join(output_states)
        return info
    
    def _activate(self):
        """
        For the Keysight33500B, activation is turning on the output.
        """
        self.set_output(True)
    
    # ---------------------- Waveform Configuration Methods ----------------------
    
    def set_channel(self, channel: int):
        """
        Set the active channel for subsequent commands.
        
        :param channel: Channel number (1 or 2)
        """
        if channel not in [1, 2]:
            raise ValueError("Channel must be 1 or 2")
        self.channel = channel
        
    def set_waveform(self, waveform_type: str):
        """
        Set the waveform type for the current channel.
        
        :param waveform_type: Type of waveform (SINusoid, SQUare, RAMP, PULSe, NOISe, DC, etc.)
        """
        valid_types = ["SINusoid", "SQUare", "RAMP", "PULSe", "NOISe", "DC", "PRBS", "ARBitrary"]
        waveform_type = waveform_type.upper()
        
        # Accept shortened versions like SIN, SQU etc.
        for valid_type in valid_types:
            if waveform_type in valid_type.upper():
                waveform_type = valid_type
                break
        
        if waveform_type not in valid_types:
            raise ValueError(f"Invalid waveform type. Must be one of {valid_types}")
        
        self.send_command(f"SOURce{self.channel}:FUNCtion {waveform_type}")
        
    def set_frequency(self, frequency: float):
        """
        Set the frequency for the current channel's waveform.
        
        :param frequency: Frequency in Hz
        """
        self.send_command(f"SOURce{self.channel}:FREQuency {frequency}")
        
    def set_amplitude(self, amplitude: float, unit: str = "VPP"):
        """
        Set the amplitude for the current channel's waveform.
        
        :param amplitude: Amplitude value
        :param unit: Unit (VPP, VRMS, DBM)
        """
        valid_units = ["VPP", "VRMS", "DBM"]
        unit = unit.upper()
        
        if unit not in valid_units:
            raise ValueError(f"Invalid amplitude unit. Must be one of {valid_units}")
        
        self.send_command(f"SOURce{self.channel}:VOLTage {amplitude}")
        self.send_command(f"SOURce{self.channel}:VOLTage:UNIT {unit}")
        
    def set_offset(self, offset: float):
        """
        Set the DC offset for the current channel's waveform.
        
        :param offset: Offset value in volts
        """
        self.send_command(f"SOURce{self.channel}:VOLTage:OFFSet {offset}")
        
    def set_high_low_levels(self, high: float, low: float):
        """
        Set the high and low voltage levels for the current channel's waveform.
        
        :param high: High level in volts
        :param low: Low level in volts
        """
        self.send_command(f"SOURce{self.channel}:VOLTage:HIGH {high}")
        self.send_command(f"SOURce{self.channel}:VOLTage:LOW {low}")
        
    def set_output(self, state: bool):
        """
        Enable or disable the output for the current channel.
        
        :param state: True to enable, False to disable
        """
        state_str = "ON" if state else "OFF"
        self.send_command(f"OUTPut{self.channel}:STATe {state_str}")
        
    def set_output_load(self, load_ohms=50):
        """
        Set the expected output load for the current channel.
        
        :param load_ohms: Load impedance in ohms, or "INFinity" for high-Z
        """
        load_param = load_ohms if load_ohms != float('inf') else "INFinity"
        self.send_command(f"OUTPut{self.channel}:LOAD {load_param}")
    
    # ---------------------- DC Waveform Methods ----------------------
    
    def set_dc_voltage(self, voltage: float):
        """
        Configure the channel to output a DC voltage.
        
        :param voltage: DC voltage level in volts
        """
        self.set_waveform("DC")
        self.send_command(f"SOURce{self.channel}:VOLTage:OFFSet {voltage}")
        
    # ---------------------- Square Waveform Methods ----------------------
    
    def set_square_waveform(self, frequency: float, amplitude: float, offset: float = 0, duty_cycle: float = 50):
        """
        Configure a square wave on the current channel.
        
        :param frequency: Frequency in Hz
        :param amplitude: Amplitude in volts peak-to-peak
        :param offset: DC offset in volts
        :param duty_cycle: Duty cycle as percentage (0-100)
        """
        self.set_waveform("SQUare")
        self.set_frequency(frequency)
        self.set_amplitude(amplitude)
        self.set_offset(offset)
        self.set_square_duty_cycle(duty_cycle)
        
    def set_square_duty_cycle(self, duty_cycle: float):
        """
        Set the duty cycle for a square wave.
        
        :param duty_cycle: Duty cycle as percentage (0-100)
        """
        if not 0 <= duty_cycle <= 100:
            raise ValueError("Duty cycle must be between 0 and 100")
        
        self.send_command(f"SOURce{self.channel}:FUNCtion:SQUare:DCYCle {duty_cycle}")
        
    # ---------------------- Pulse Waveform Methods ----------------------
    
    def set_pulse_waveform(self, frequency: float, amplitude: float, offset: float = 0, 
                          width: float = None, duty_cycle: float = None, 
                          edge_time: float = None):
        """
        Configure a pulse wave on the current channel.
        
        :param frequency: Frequency in Hz
        :param amplitude: Amplitude in volts peak-to-peak
        :param offset: DC offset in volts
        :param width: Pulse width in seconds (if None, duty_cycle will be used)
        :param duty_cycle: Duty cycle as percentage (0-100) (used only if width is None)
        :param edge_time: Leading and trailing edge time in seconds (if specified)
        """
        self.set_waveform("PULSe")
        self.set_frequency(frequency)
        self.set_amplitude(amplitude)
        self.set_offset(offset)
        
        # Set either pulse width or duty cycle
        if width is not None:
            self.set_pulse_width(width)
        elif duty_cycle is not None:
            self.set_pulse_duty_cycle(duty_cycle)
            
        # Set edge time if specified
        if edge_time is not None:
            self.set_pulse_edge_time(edge_time)
            
    def set_pulse_width(self, width: float):
        """
        Set the pulse width for a pulse waveform.
        
        :param width: Pulse width in seconds
        """
        self.send_command(f"SOURce{self.channel}:FUNCtion:PULSe:WIDTh {width}")
        
    def set_pulse_duty_cycle(self, duty_cycle: float):
        """
        Set the duty cycle for a pulse waveform.
        
        :param duty_cycle: Duty cycle as percentage (0-100)
        """
        if not 0 <= duty_cycle <= 100:
            raise ValueError("Duty cycle must be between 0 and 100")
            
        self.send_command(f"SOURce{self.channel}:FUNCtion:PULSe:DCYCle {duty_cycle}")
        
    def set_pulse_edge_time(self, edge_time: float):
        """
        Set both leading and trailing edge times for a pulse waveform.
        
        :param edge_time: Edge time in seconds
        """
        self.send_command(f"SOURce{self.channel}:FUNCtion:PULSe:TRANsition:LEADing {edge_time}")
        self.send_command(f"SOURce{self.channel}:FUNCtion:PULSe:TRANsition:TRAiling {edge_time}")
        
    def set_pulse_leading_edge(self, edge_time: float):
        """
        Set the leading edge time for a pulse waveform.
        
        :param edge_time: Leading edge time in seconds
        """
        self.send_command(f"SOURce{self.channel}:FUNCtion:PULSe:TRANsition:LEADing {edge_time}")
        
    def set_pulse_trailing_edge(self, edge_time: float):
        """
        Set the trailing edge time for a pulse waveform.
        
        :param edge_time: Trailing edge time in seconds
        """
        self.send_command(f"SOURce{self.channel}:FUNCtion:PULSe:TRANsition:TRAiling {edge_time}")
    
    # ---------------------- Communication Methods ----------------------
    
    def send_command(self, command: str) -> None:
        """
        Send a command to the instrument.
        
        :param command: The command string to send
        """
        try:
            self.sock.send((command + "\n").encode('utf-8'))
        except socket.error as e:
            raise ConnectionError(f"Failed to send command to Keysight33500B: {e}")
    
    def read(self) -> str:
        """
        Read the response from the instrument.
        
        :return: The decoded response string
        """
        try:
            response = b''
            chunk = self.sock.recv(4096)
            response += chunk
            
            # Continue reading if there's more data
            while len(chunk) == 4096:
                chunk = self.sock.recv(4096)
                response += chunk
                
            return response.decode('utf-8').strip()
        except socket.error as e:
            raise ConnectionError(f"Failed to read from Keysight33500B: {e}")
    
    def query(self, command: str) -> str:
        """
        Send a command and retrieve the response.
        
        :param command: The command string to send
        :return: The response string
        """
        self.send_command(command)
        time.sleep(0.1)  # Small delay to allow the instrument to process
        return self.read()
    
    def query_opc(self) -> bool:
        """
        Query the *OPC? command to check if the operation is complete.
        
        :return: True if operation is complete, False otherwise
        """
        try:
            response = self.query("*OPC?")
            return response.strip() == "1"
        except Exception:
            return False
    
    def wait_opc(self, time_sleep: float = 0.1, timeout: float = 10) -> None:
        """
        Wait until the operation is complete or timeout is reached.
        
        :param time_sleep: Time to sleep between checks
        :param timeout: Maximum time to wait for operation completion
        :raises TimeoutError: If the operation does not complete within the timeout
        """
        start_time = time.time()
        while not self.query_opc():
            if time.time() - start_time > timeout:
                raise TimeoutError('Timeout waiting for operation to complete')
            time.sleep(time_sleep)
    
    # ---------------------- Standard Instrument Methods ----------------------
    
    def reset(self):
        """
        Reset the instrument to its default state.
        """
        self.send_command("*RST")
        self.wait_opc()
        print(f"{self.name} reset to default state.")
    
    def close_connection(self):
        """
        Close the connection to the instrument.
        """
        try:
            self.sock.close()
            print(f"{self.name} connection closed.")
        except Exception as e:
            print(f"Error closing connection to {self.name}: {e}")
    
    def shutdown(self):
        """
        Shut down the instrument by turning off outputs and closing the connection.
        """
        try:
            # Turn off all outputs
            for ch in [1, 2]:
                self.send_command(f"OUTPUT{ch} OFF")
            self.close_connection()
            print(f"{self.name} shutdown complete.")
        except Exception as e:
            print(f"Error during shutdown of {self.name}: {e}")
    
    def kill(self):
        """
        Emergency kill of the instrument connection.
        """
        try:
            self.reset()
            self.shutdown()
            print(f"{self.name} killed.")
        except Exception as e:
            print(f"Error during kill of {self.name}: {e}")
            # Force close the socket
            try:
                self.sock.close()
            except:
                pass
    
    def __del__(self):
        """
        Destructor to ensure connection is closed when object is deleted.
        """
        try:
            self.close_connection()
        except:
            pass