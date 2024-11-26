import pyvisa
import numpy as np


class SMA:
    """
    SMA100B signal generator

    Conventions:
        - All arrays are numpy arrays
    
    Units:
        - frequency [Hz]
        - time [ms]
        - amplitude TODO
    """

    _name = ""
    _ip = ""
    __freq = 0
    __power_level = 0
    debug = True
    debug_prefix = ""

    def __init__(self, ip_address_string):
        res_manager = pyvisa.ResourceManager()
        self.__res = res_manager.open_resource(f"tcpip0::{ip_address_string}::INSTR")
        self._ip = ip_address_string

        self.write_expect("*RST") # clear settings
        self.write_expect("*CLS") # clear settings
        self._name = self.query_expect("*IDN?")
        self.write_expect("SOUR:FREQ:MODE CW") # set mode to Continous Wave (CW)

        self.freq = 5e9
        self.power_level = -20
    
    def write(self, command):
        self.__res.write(command)

    def query(self, command):
        return self.__res.query(command)

    def write_expect(self, command, error_msg = None):
        """Send write command to device and check for operation complete"""
        if self.__res is None: raise Exception("No connection.")

        result = self.query(f"{command}; *OPC?")

        if self.debug: print(f"{self.debug_prefix}[{command}] {result.strip()}")
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

        if self.debug: print(f"[{command}] {result.strip()}")
        if '0' in result:
            if error_msg is None:
                raise Exception(f"Operation '{command}' could not complete.")
            else:
                raise Exception(error_msg)
        
        return data
    
     
    # FREQUENCY

    @property
    def freq(self):
        return self.__freq

    @freq.setter
    def freq(self, f):
        """Set frequency in Hz"""
        self.write_expect(f"SOUR:FREQ:OFFSET 0 Hz")
        self.write_expect("SOUR:FREQ:MULTIPLIER 1.0")
        self.write_expect(f"SOUR:FREQ:CW {f} Hz") # set CW frequency
        self.__freq = f

        if int(self.query("SOUR:FREQ:CW?")) != f: 
            raise Exception(f"Could not set 'freq' to {f}.")
     
    # POWER LEVEL

    @property
    def power_level(self):
        return self.__power_level

    @power_level.setter
    def power_level(self, pow_lvl):
        """Set output power level in dBm"""
        self.write_expect(f"SOUR:POW:POW {pow_lvl} dBm") # set power level
        self.__power_level = pow_lvl

        if int(self.query("SOUR:POW:POW?")) != pow_lvl: 
            raise Exception(f"Could not set 'power_level' to {pow_lvl}.")
        
    # ON/OFF

    def turn_on(self):
        self.write_expect("OUTP ON")
    
    def turn_off(self):
        self.write_expect("OUTP OFF")
