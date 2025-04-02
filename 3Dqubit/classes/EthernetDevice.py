import pyvisa
import sys

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
    __res_manager = None
    debug = True
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
    
    def write_binary_values(self, command, data):
        """Send binary values to device"""
        if self.__res is None: raise Exception("No connection.")

        self.__res.write_binary_values(command, data, datatype='H', is_big_endian=False)

        if self.debug: 
            print(f"[{command}]")
            sys.stdout.flush()
        

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

