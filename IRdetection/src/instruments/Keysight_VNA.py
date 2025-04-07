import numpy as np
import pyvisa
import sys
from src.abstract.Instrument import Instrument

class EthernetDevice:
    """
    Pyvisa ethernet device abstraction

    Conventions:
        - All arrays are numpy arrays
    
    Units:
        - frequency [Hz]
        - time [ms]
        - amplitude TODO
        - tension [V]
    """

    _name = ""
    _ip = ""
    __timeout = 0
    __res = None
    debug = False
    debug_prefix = ""

    def __init__(self, ip_address_string):
        self.__res_manager = pyvisa.ResourceManager()
        self.__res = self.__res_manager.open_resource(f"tcpip0::{ip_address_string}::INSTR")

        self._ip = ip_address_string
        self._name = self.query_expect("*IDN?")

        self.timeout = 10e3

        if self.on_init:
            self.on_init(ip_address_string)
    
    def __del__(self):
        self.close()

    def close(self):
        if self.debug: print("[CLOSE]")
        self.__res_manager.close()
    
    def write(self, command):
        self.__res.write(command)

    def query(self, command):
        return self.__res.query(command)

    def write_expect(self, command, error_msg = None):
        """Send write command to device and check for operation complete"""
        if self.__res is None: raise Exception("No connection.")

        result = self.query(f"{command}; *OPC?")

        if self.debug: 
            print(f"{self.debug_prefix}[{command}] {result.strip()}")
            sys.stdout.flush()
        if '0' in result:
            if error_msg is None:
                raise Exception(f"Operation '{command}' could not complete.")
            else:
                raise Exception(error_msg)
            
    def query_expect(self, command, error_msg = None):
        """Send query command to device and check for operation complete. Returns the queried value if no error occurs."""
        if self.__res is None: raise Exception("No connection.")

        data = self.query(f"{command}")
        result = self.query("*OPC?")

        if self.debug: 
            print(f"[{command}] {result.strip()}")
            sys.stdout.flush()
        if '0' in result:
            if error_msg is None:
                raise Exception(f"Operation '{command}' could not complete.")
            else:
                raise Exception(error_msg)
        
        return data

    # TIMEOUT

    @property
    def timeout(self):
        return self.__timeout
    
    @timeout.setter
    def timeout(self, millis):
        """Set request response timeout (in milliseconds)"""
        if self.__res is None: raise Exception("No connection.")
        self.__res.timeout = millis
        self.__timeout = millis



class VNA(Instrument, EthernetDevice):
    """
    Vector Network Analyzer (VNA)

    The class has the following properties
    - min_freq 
    - max_freq 
    - point_count 
    - bandwidth 
    - avg_count 
    - power 

    The class has the following methods
    - read_frequency_data
    - read_data
    """

    __min_freq = 0
    __max_freq = 0
    __point_count = 0
    __bandwidth = 0
    __avg_count = 0
    __power = 0
    
    def __init__(self, ip_address_string, name="VNA"):
        Instrument.__init__(self, name)
        EthernetDevice.__init__(self, ip_address_string)
        self.initialize()
    
    def initialize(self):
        """Initialize the instrument."""
        self.write_expect("*CLS") # clear settings
        self.write_expect("INST:SEL 'NA'", "Failed to select NA mode.") # Newtwork Analyzer
        self.write_expect("SENS:AVER:MODE SWEEP") # Average mode set to sweep
        self.write_expect("DISP:WIND:TRAC1:Y:AUTO") # Turn on autoscaling on the y axis
        
        self.timeout = 10e3
        self.min_freq = 4e9
        self.max_freq = 6e9
        self.point_count = 400 
        self.bandwidth = 10e3 # Hz
        self.avg_count = 1
        self.power = -40 # dBm
    
    def info(self, verbose=False):
        """Get information about the instrument."""
        info = {
            "name": self.name,
            "ip": self._ip,
            "id": self._name,
            "min_freq": self.__min_freq,
            "max_freq": self.__max_freq,
            "point_count": self.__point_count,
            "bandwidth": self.__bandwidth,
            "avg_count": self.__avg_count,
            "power": self.__power,
        }
        
        if verbose:
            # Add more detailed information if verbose
            info["status"] = self.query("*STB?").strip()
            
        return info
    
    def _activate(self):
        """Activate the instrument. Put it in remote mode."""
        # VNAs typically don't need explicit remote mode activation
        pass
    
    def reset(self):
        """Reset the instrument."""
        self.write_expect("*RST")
        self.initialize()
    
    def close_connection(self):
        """Disconnect the instrument."""
        self.close()
    
    def shutdown(self):
        """Shut down the instrument."""
        # self.write_expect("SYST:PRES")  # Preset the system
        self.close_connection()
    
    def kill(self):
        """Kill the instrument."""
        # Force close connection without waiting
        self.close_connection()
        
    def on_init(self, ip_address_string):
        """Legacy initialization method, redirects to initialize"""
        # This method is kept for backward compatibility
        pass
    
    def set_freq_range(self, center_freq, span):
        """Set the frequency range based on center frequency and span."""
        self.min_freq = center_freq - span / 2
        self.max_freq = center_freq + span / 2
    # MIN_FREQ
    @property
    def min_freq(self):
        return self.__min_freq

    @min_freq.setter
    def min_freq(self, f):
        """Set minimum frequency in Hz"""
        self.write_expect(f"SENS:FREQ:START {f}")
        self.__min_freq = f

        if int(self.query("SENS:FREQ:START?")) != f: 
            raise Exception(f"Could not set 'min_freq' to {f}.")

    # MAX_FREQ

    @property
    def max_freq(self):
        return self.__max_freq

    @max_freq.setter
    def max_freq(self, f):
        """Set maximum frequency in Hz"""
        self.write_expect(f"SENS:FREQ:STOP {f}")
        self.__max_freq = f

        if int(self.query("SENS:FREQ:STOP?")) != f: 
            raise Exception(f"Could not set 'max_freq' to {f}.")
    
    # POINT_COUNT
    
    @property
    def point_count(self):
        return self.__point_count
    
    @point_count.setter
    def point_count(self, n):
        """Set the number of datapoints"""
        self.write_expect(f"SENS:SWE:POIN {n}")
        self.__point_count = n

        if int(self.query("SENS:SWE:POIN?")) != n: 
            raise Exception(f"Could not set 'point_count' to {n}.")

    # BANDWIDTH

    @property
    def bandwidth(self):
        return self.__bandwidth
    
    @bandwidth.setter
    def bandwidth(self, bw):
        """Set the bandwidth in Hz"""
        self.write_expect(f"SENS:BWID {bw}")
        self.__bandwidth = bw

        if int(self.query("SENS:BWID?")) != bw: 
            raise Exception(f"Could not set 'bandwidth' to {bw}.")

    # AVERAGE COUNT

    @property
    def avg_count(self):
        return self.__avg_count
    
    @avg_count.setter
    def avg_count(self, n):
        """Set the number of averages (1 = no averages)"""
        self.write_expect(f"AVER:COUN {n}")
        self.__avg_count = n

        if int(self.query("AVER:COUN?")) != n: 
            raise Exception(f"Could not set 'avg_count' to {n}.")
        
    # POWER

    @property
    def power(self):
        return self.__power

    @power.setter
    def power(self, value):
        """Set output power in dBm"""
        self.write_expect(f"SOUR:POW {value}")
        self.__power = value

        if float(self.query("SOUR:POW?").strip()) != value: 
            raise Exception(f"Could not set 'power' to {value}.")

    # FREQUENCY SPECTRUM

    def read_frequency_data(self):
        return np.array(list(map(float, self.query_expect("FREQ:DATA?", "Frequency data readout failed.").split(","))))
        

    # DATA ACQUISITION
    
    def read_data(self, Sij):
        """Sij is a string of value "S11", "S12", "S21" or "S22"."""
        self.write_expect("INIT:CONT 0")
        self.write_expect(f"CALC:PAR:DEF {Sij}") # Chose which data to read

        for _ in range(self.avg_count): self.write_expect("INIT:IMMediate") # Trigger now
        
        data = np.array(list(map(float, self.query_expect("CALC:DATA:SDATA?", "Data readout failed.").split(","))))
        data_real = data[0::2] # all even entries 0,2,4,6,8
        data_imag = data[1::2] # all odd entries 1,3,5,7,9

        self.write_expect("INIT:CONT 1")

        return {"real": data_real, "imag": data_imag}
    
    # Add acquisition method example
    def acq_s_parameters(self, param="S21"):
        """
        Acquire S-parameters from the VNA
        
        Args:
            param: S-parameter to measure (S11, S12, S21, S22)
        
        Returns:
            Dictionary with frequency, real and imaginary parts
        """
        freq_data = self.read_frequency_data()
        s_data = self.read_data(param)
        return {
            "frequency": freq_data,
            "real": s_data["real"],
            "imag": s_data["imag"]
        }

