import pyvisa


class VNA:
    _name = ""
    _ip = ""
    _min_freq = 0
    _max_freq = 0
    _point_count = 0
    _timeout = 0

    def __init__(self, ip_address_string):
        res_manager = pyvisa.ResourceManager()
        self.__res = res_manager.open_resource(f"tcpip0::{ip_address_string}::INSTR")
        self._ip = ip_address_string

        self.__res.write("*CLS") # clear settings
        self.__res.write("*RST")
        self._name = self.__res.query("*IDN?")

        res = self.__res.query("INST:SEL 'NA'; *OPC?") # Newtwork Analyzer
        if res[0] != '1': raise Exception("Failed to select NA mode.")
        
        self.timeout = 10e3
        self.min_freq = 4e9
        self.max_freq = 6e9
        self.point_count = 400 

    # TIMEOUT

    @property
    def timeout(self):
        return self._timeout
    
    @timeout.setter
    def timeout(self, millis):
        """Set request response timeout (in milliseconds)"""
        self.__res.timeout = millis
        self._timeout = millis
    
    # MIN_FREQ

    @property
    def min_freq(self):
        return self._min_freq

    @min_freq.setter
    def min_freq(self, n):
        """Set minimum frequency in Hz"""
        self.__res.write("SENS:FREQ:START " + str(n))
        self._min_freq = n

    # MAX_FREQ

    @property
    def max_freq(self):
        return self._max_freq

    @max_freq.setter
    def max_freq(self, n):
        """Set maximum frequency in Hz"""
        self.__res.write("SENS:FREQ:STOP " + str(n))
        self._max_freq = n
    
    # POINT_COUNT
    
    @property
    def point_count(self):
        return self._point_count
    
    @point_count.setter
    def point_count(self, n):
        """Set the number of datapoints"""
        self.__res.write("SENS:SWE:POIN " + str(n))
        self._point_count = n

    # DATA ACQUISITION
    
    def read_data(self, Sij):
        self.__res.query(f"CALC:PAR:DEF {Sij}; *OPC?")
        self.__res.query("INIT:IMMediate;*OPC?")
        data = list(map(float, self.__res.query("CALC:DATA:SDATA?").split(",")))
        data_real = data[0::2] # all even entries 0,2,4,6,8
        data_imag = data[1::2] # all odd entries 1,3,5,7,9
        return {"real": data_real, "imag": data_imag}
    
    # (...)

