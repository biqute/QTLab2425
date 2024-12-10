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

    def pulse_on(self, sour = "INT", ttyp = "FAST"):
        self.__SG.write("SOUR:PULM:STAT ON")  #turns on pulse
        self.__SG.write(f"SOUR:PULM:SOUR {sour}") #sets an internal pulse source 
        self.__SG.write(f"SOUR:PULM:TTYP {ttyp}") #sets the transition mode

    def pulse_off(self) :
        self.__SG.write("SOUR:PULM:STAT OFF")

    def pulse_set (self, delay, width, period, double_d = 0, double_w = 0, mode = "SING", trig = "SING") :

        mode = mode if mode in ['SING', 'DOUB'] else 'SING'

        self.__SG.write(f"SOUR:PULM:MODE {mode}") #single pulse setting
        
        if mode == "DOUB":
            self.__SG.write(f"SOUR:PULM:DOUB DEL {double_d}") #delay setting        
            self.__SG.write(f"SOUR:PULM:DOUB WIDT {double_w}") #width of the pulse
         
        trig = trig if trig in ['SING', 'AUTO'] else 'SING'

        self.__SG.write(f"SOUR:PULM:TRIG:MODE {trig}") #single trigger setting

        self.__SG.write(f"SOUR:PULM:PER {period}")
       
        self.__SG.write(f"SOUR:PULM:DEL {delay}") #delay setting 

        self.__SG.write(f"SOUR:PULM:WIDT {width}") #width of the pulse
    
    def pulse_trig(self) :
        self.__SG.write("SOUR:PULM:TRIG:IMM") #initiates a trigger if trigger mode is single


