import pyvisa

class SG :

    def __init__(self, ip) :
        
        rm = pyvisa.ResourceManager ()
        self.__SG = rm.open_resource (f"tcpip0::{ip}::inst0::INSTR")
        self.__SG.write("SOUR:FREQ:MODE CW") # set mode to Continous Wave (CW)


    def set_freq(self, f) :

        self.__SG.write(f"SOUR:FREQ:CW {f} Hz") # set CW frequency


    def set_power(self, pw) :
        
        self.__SG.write(f"SOUR:POW:POW {pw} dBm") # set power level

    
    def turn_on(self) :

        self.__SG.write("OUTP ON")


    def turn_off(self) :

        self.__SG.write("OUTP OFF")

    def quadra (self) :

        self.__SG.write("SOUR:PULM:STAT ON")  
        self.__SG.write("SOUR:PULM:SOUR INT")  
        self.__SG.write("SOUR:PULM:TRIG:MODE SING")
        self.__SG.write("SOUR:PULM:MODE DOUB")  
        self.__SG.write("SOUR:PULM:TTYP SMO")  
        self.__SG.write("SOUR:PULM:OUTP:VID:POL INV")
        self.__SG.write("SOUR:PULM:TRIG:IMM")
        #da finire

