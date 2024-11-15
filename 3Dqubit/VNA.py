import pyvisa


class VNA:
    _name = ""
    _ip = ""
    __min_freq = 0
    __max_freq = 0
    __point_count = 0
    __timeout = 0
    __bandwidth = 0
    __avg_count = 0
    debug = True

    def __init__(self, ip_address_string):
        res_manager = pyvisa.ResourceManager()
        self.__res = res_manager.open_resource(f"tcpip0::{ip_address_string}::INSTR")
        self._ip = ip_address_string

        self.write_expect("*CLS") # clear settings
        self._name = self.query_expect("*IDN?")

        self.write_expect("INST:SEL 'NA'", "Failed to select NA mode.") # Newtwork Analyzer

        self.write_expect("SENS:AVER:MODE SWEEP") # Average mode set to sweep
        
        self.timeout = 10e3
        self.min_freq = 4e9
        self.max_freq = 6e9
        self.point_count = 400 
        self.bandwidth = 10e3
        self.avg_count = 1
    
    def write(self, command):
        self.__res.write(command)

    def query(self, command):
        return self.__res.query(command)

    def write_expect(self, command, error_msg = None):
        """Send write command to device and check for operation complete"""
        if self.__res is None: raise Exception("No connection.")

        result = self.query(f"{command}; *OPC?")

        if self.debug: print(f"[{command}] {result.strip()}")
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

    # TIMEOUT

    @property
    def timeout(self):
        return self.__timeout
    
    @timeout.setter
    def timeout(self, millis):
        """Set request response timeout (in milliseconds)"""
        self.__res.timeout = millis
        self.__timeout = millis
    
    # MIN_FREQ

    @property
    def min_freq(self):
        return self.__min_freq

    @min_freq.setter
    def min_freq(self, n):
        """Set minimum frequency in Hz"""
        self.write("SENS:FREQ:START " + str(n))
        self.__min_freq = n

    # MAX_FREQ

    @property
    def max_freq(self):
        return self.__max_freq

    @max_freq.setter
    def max_freq(self, n):
        """Set maximum frequency in Hz"""
        self.write("SENS:FREQ:STOP " + str(n))
        self.__max_freq = n
    
    # POINT_COUNT
    
    @property
    def point_count(self):
        return self.__point_count
    
    @point_count.setter
    def point_count(self, n):
        """Set the number of datapoints"""
        self.write("SENS:SWE:POIN " + str(n))
        self.__point_count = n

    # BANDWIDTH

    @property
    def bandwidth(self):
        return self.__bandwidth
    
    @bandwidth.setter
    def bandwidth(self, bw):
        """Set the bandwidth in Hz"""
        self.write("SENS:BWID " + str(bw))
        self.__bandwidth = bw

    # AVERAGE COUNT

    @property
    def avg_count(self):
        return self.__avg_count
    
    @avg_count.setter
    def avg_count(self, n):
        """Set the number of averages (1 = no averages)"""
        self.write("AVER:COUN " + str(n))
        self.__avg_count = n

    # DATA ACQUISITION
    
    def read_data(self, Sij):
        self.write_expect("INIT:CONT 0")
        self.write_expect(f"CALC:PAR:DEF {Sij}") # Chose which data to read

        for _ in range(self.avg_count): self.write_expect("INIT:IMMediate") # Trigger now
        
        data = list(map(float, self.query_expect("CALC:DATA:SDATA?", "Data readout failed.").split(",")))
        data_real = data[0::2] # all even entries 0,2,4,6,8
        data_imag = data[1::2] # all odd entries 1,3,5,7,9

        self.write_expect("INIT:CONT 1")

        return {"real": data_real, "imag": data_imag}
    
    # (...)

