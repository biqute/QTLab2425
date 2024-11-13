import numpy as np
import pyvisa
import matplotlib.pyplot as plt  # Import per la visualizzazione

class PSA :

    def __init__ (self, ip) :

        rm1 = pyvisa.ResourceManager()
        self.__PSA =  rm1.open_resource(f"tcpip0::{ip}::inst0::INSTR")
        print ("device connected.")

        self.__PSA.write("*CLS") # clear settings
        PSA = self.__PSA.query("INST:SEL 'SA'; *OPC?") # set spectrum analyzer
        if PSA[0] != '1': raise Exception("Failed to select SA mode.")

    def off (self) :

        self.__PSA.clear()
        self.__PSA.close()


    # SET PARAMETERS


    def set_freq_limits (self, set_min, set_max) :

        set_min_G = set_min*1000000000
        set_max_G = set_max*1000000000
        self.__PSA.write(f'FREQ:STAR {set_min_G}')
        self.__PSA.write(f'FREQ:STOP {set_max_G}')

        return self.__PSA.query("*OPC?")
    
    
    def set_freq_span (self, set_center, set_span) :

        set_center_G = set_center*1000000000
        set_span_G = set_span*1000000000
        self.__PSA.write(f'FREQ:CENT {set_center_G}')   
        self.__PSA.write(f'FREQ:SPAN {set_span_G}')

        return self.__PSA.query("*OPC?")
    

    def set_power(self, set_power) :

        self.__PSA.write(f'SOUR:POW {set_power}')

        return self.__PSA.query("*OPC?")
    
    

    def set_ifband(self, ifband) : 

        self.__PSA.write(f'BAND {ifband}')

        return self.__PSA.query("*OPC?")
    


    def set_sweep_time(self, sweep_time) :

        self.__PSA.write(f'SWE:TIME  {sweep_time}')

        return self.__PSA.query("*OPC?")
    

    
    def set_sweep_points(self, sweep_points) :
        
        self.__PSA.write(f'SWE:POIN {sweep_points}')

        return self.__PSA.query("*OPC?")

    def set_n_means(self, n_means) :
        self.__PSA.write(f"SENS:AVER:COUN {n_means}")
        return self.__PSA.query("*OPC?")

    # GET PARAMETERS


    def get_freq_limits (self) :

        min_freq = self.__PSA.query("SENS:FREQ:START?")
        max_freq = self.__PSA.query("SENS:FREQ:STOP?")

        return float(min_freq), float(max_freq)



    def get_freq_center (self) :

        return float(self.__PSA.query("SENS:FREQ:CENT?")), float(self.__PSA.query("SENS:FREQ:SPAN?"))
    


    def get_sweep_points(self) :

        return int(self.__PSA.query("SENSE:SWE:POIN?"))


    def get_n_means(self) :

        return int(self.__PSA.query("SENS:AVER:COUN?"))
    

    def get_trace(self) :
        
        return list(map(float, self.__PSA.query("TRACE:DATA?").split(",")))    
    

    
    def get_spectrum(self) :
        # Query the FieldFox response data
        amp = self.__PSA.query("TRACE:DATA?")

        # Convert amp data to a list of floats
        amp_Array = list(map(float, amp.split(",")))
        min , max = self.get_freq_limits()
        n_points = self.get_sweep_points()
        freq = np.linspace(min , max, n_points)
        import matplotlib.pyplot as plt
        plt.title("spectrum")
        plt.xlabel("Frequency [Hz]")
        plt.ylabel("Amplitude [dBm]")
        plt.plot(freq, amp_Array)
        plt.show()

        return self.__PSA.query("*OPC?")