import numpy as np
import pyvisa
import matplotlib.pyplot as plt
import time
# aggiungere metodo per trovare i picchi senza usare i marker
class VNA :

    def __init__(self, ip) :

        rm1 = pyvisa.ResourceManager()
        self.__VNA = rm1.open_resource(f"tcpip0::{ip}::inst0::INSTR") 
        print("device connected")

        self.__VNA.write("*CLS")
        VNA =self.__VNA.query("INST:SEL 'NA'; *OPC?")
        if VNA[0] != '1': raise Exception("Failed to select NA mode")

        self.__VNA.write("AVER:MODE POINT") # Average mode set to sweep

        self.__VNA.write("DISP:WIND:TRAC1:Y:AUTO") # Turn on autoscaling on the y axis

        self.__VNA.write("CALC:SMO 0")

    def off (self) :
        self.__VNA.clear()
        self.__VNA.close()

    def wait(self, wait_time):
        time.sleep(wait_time)  # Aspetta il tempo specificato
            

    def set_freq_limits (self, set_min, set_max) :

        set_min_G = set_min*1000000000
        set_max_G = set_max*1000000000
        self.__VNA.write(f'FREQ:STAR {set_min_G}')
        self.__VNA.write(f'FREQ:STOP {set_max_G}')

        return self.__VNA.query("*OPC?")
    
    
    def set_freq_span (self, set_center, set_span) :

        set_center_G = set_center*1000000000
        set_span_G = set_span*1000000000
        self.__VNA.write(f'FREQ:CENT {set_center_G}')   
        self.__VNA.write(f'FREQ:SPAN {set_span_G}')

        return self.__VNA.query("*OPC?")
    

    def set_power(self, set_power) :

        self.__VNA.write(f'SOUR:POW {set_power}')

        return self.__VNA.query("*OPC?")
    
    
    def set_ifband(self, ifband) : 

        self.__VNA.write(f'BWID {ifband}')

        return self.__VNA.query("*OPC?")
    

    def set_sweep_time(self, sweep_time) :

        self.__VNA.write(f'SWE:TIME  {sweep_time}')

        return self.__VNA.query("*OPC?")
    
    
    def set_sweep_points(self, sweep_points) :
        
        self.__VNA.write(f'SWE:POIN {sweep_points}')

        return self.__VNA.query("*OPC?")


    def set_n_means(self, n_means) :
        self.__VNA.write(f"SENS:AVER:COUN {n_means}")
        return self.__VNA.query("*OPC?")


    def get_freq_limits (self) :

        min_freq = self.__VNA.query("SENS:FREQ:START?")
        max_freq = self.__VNA.query("SENS:FREQ:STOP?")

        return float(min_freq), float(max_freq)


    def get_freq_center (self) :

        return float(self.__VNA.query("SENS:FREQ:CENT?")), float(self.__VNA.query("SENS:FREQ:SPAN?"))
    

    def get_sweep_points(self) :

        return int(self.__VNA.query("SENSE:SWE:POIN?"))


    def get_n_means(self) :

        return int(self.__VNA.query("SENS:AVER:COUN?"))
    

    def get_S_parameters(self, Sij) :
        self.__VNA.write(f"CALC:PAR:DEF {Sij}")
        
        data = np.array(list(map(float, self.__VNA.query("CALC:DATA:SDATA?").split(","))))
        real = np.array(data[0::2]) 
        imag = np.array(data[1::2])
        return real, imag
    
    def get_power(self) :
        real, imag =  self.get_S_parameters("S21")
        return np.array(real**2 + imag**2)
    
    def get_phase(self) :
        real, imag =  self.get_S_parameters("S21")
        phase = np.arctan2(imag, real)
        return phase
    
    def get_dbm (self) :
        Pow = self.get_power ()
        return np.array(10*np.log10 (Pow))
            
        
    
    def get_freq(self) :
        
        return list(map(float, self.__VNA.query("FREQ:DATA?").split(",")))

    def get_spectrum(self) :
        fig, ax = plt.subplots()
        freq = self.get_freq()
        Pow = self.get_dbm ()
        plt.title("spectrum")
        #ax.set_yscale("log")
        plt.xlabel("Frequency [Hz]")
        plt.ylabel("Amplitude [dBm]")
        ax.plot(freq, Pow)
        plt.show()

        return self.__VNA.query("*OPC?")

