import numpy as np
import pyvisa

class PSA :

    def __init__ (self, ip) :

        rm1 = pyvisa.ResourceManager()
        self.__PSA =  rm1.open_resource(f"tcpip0::{ip}::inst0::INSTR")
        print ("device connected")

    def off (self) :

        self.__PSA.clear()
        self.__PSA.close()

    def set_freq_limits (self, set_min, set_max) :

        self.__PSA.write(f'FREQ:STAR {set_min}')
        self.__PSA.write(f'FREQ:STOP {set_max}')

    def get_freq_limits (self) :

        get_min = self.__PSA.query("SENS:FREQ:START?")
        get_max = self.__PSA.query("SENS:FREQ:STOP?")

        return get_min, get_max

    def set_freq_span (self, set_center, set_span) :

        self.__PSA.write(f'FREQ:CENT {set_center}')   
        self.__PSA.write(f'FREQ:SPAN {set_span}')

    def get_freq_center (self) :

        get_center = self.__PSA.query("SENS:FREQ:CENT?")
        get_span = self.__PSA.write("SENS:FREQ:SPAN?")

        return get_center, get_span