import numpy as np
from EthernetDevice import EthernetDevice
import warnings

class AWG(EthernetDevice):
    """Arbitrary Waveform Generator (AWG)"""

    __output = False
    __freq = 0
    __amplitude = 0
    __modulation = False
    __waveform = None

    def on_init(self, ip_address_string):
        self.write_expect("*RST") # clear settings

        self.output = False # Whether the signal is outputted or not (ON/OFF)
        self.freq = 1000 # 1kHz
        self.amplitude = 4 # 4Vpp
        self.modulation = False # Whether the signal modulation is active or not (ON/OFF)

    def status(self):
        arr1 = self.query_expect(f"C1:BASIC_WAVE?").strip()[8:].split(",")
        arr2 = self.query_expect(f"C1:OUTP?").strip()[8:].split(",")
        arr3 = self.query_expect(f"C1:MODULATEWAVE?").strip()[8:].split(",")
        arr4 = self.query_expect(f"C1:ARWV?").strip()[8:].split(",")
        
        print(arr4)

        read_from_arr = lambda arr, field: arr[arr.index(field) + 1] # panics if the field is not found

        # Sanity checks
        if arr2[0] not in ["ON", "OFF"]: raise Exception(f"Output should be either ON or OFF, but '{arr2[0]}' was found") 
        if read_from_arr(arr3, "STATE") not in ["ON", "OFF"]: raise Exception(f"Modulation should be either ON or OFF, but '{arr3[1]}' was found") 

        return {
            "freq": float(read_from_arr(arr1, "FRQ")[:-2]),
            "amplitude": float(read_from_arr(arr1, "AMP")[:-1]),
            "output": arr2[0] == "ON",
            "modulation": read_from_arr(arr3, "STATE") == "ON",
            "waveform": read_from_arr(arr4, "NAME"),
        }
    
    def upload_waveform(self, name, func, interval, samples_per_second = None, samples = None):
        """
        Upload a waveform to the AWG
        
        Parameters
        - `name` (`string`): name of the waveform
        - `func` (`function`): function to be used to generate the waveform. It should take a single argument (time in seconds) and return a single value (amplitude).
        - `interval` (`tuple`): tuple of length two with the extremes of the interval of func to sample from (in seconds)
        - `samples_per_second` (`float`): number of samples per second (optional)
        - `samples` (`int`): number of samples (optional)
        """
        
        if len(interval) != 2: 
            raise Exception(f"interval should be a tuple of length 2, but a length of \"{len(interval)}\" was found") 
        if samples is not None and samples_per_second is not None:
            raise Exception(f"You should not specify both samples and samples_per_second, but neither is set to None")
        if " " in name or "," in name: 
            raise Exception(f"name should not contain spaces ' ' nor commas ',', but \"{name}\" was found") 
        
        if samples is None and samples_per_second is None:
            samples_per_second = 2.4e9
        
        duration = interval[1] - interval[0]
        if samples is None: samples = int(duration * samples_per_second)
        if samples_per_second is None: samples_per_second = samples / duration

        if samples_per_second > 2.4e9: 
            raise Exception(f"The maximum value of samples_per_second is 2.4e9, but {samples_per_second} was found")  
        if samples > 1e6: 
            raise Exception(f"The maximum number of samples is 1e6, but {samples} was found")
        
        array = np.zeros(samples, dtype=np.int16)
        N = 16
        cropped = 0
        for i in range(0,samples):
            f = func(interval[0] + i / samples_per_second)
            n = int(round((2**N)*f))
            if n > (2**N - 1): 
                n = 2**N - 1
                cropped += 1
            if n < -2**N: 
                n = -2**N
                cropped += 1
            array[i] = n
        
        if cropped > 0 and self.debug: 
            warnings.warn(f"The function 'func' was cropped to the range [-1, +1], for a total of {cropped} cropped samples.", stacklevel=2)
            
        self.write_binary_values_expect(f"C1:WVDT WVNM,{name},LENGTH,{samples*2},WAVEDATA,", array, datatype="h") # https://pyvisa.readthedocs.io/en/1.8/rvalues.html#writing-binary-values !!!

    # OUTPUT

    @property
    def output(self):
        return self.__output

    @output.setter
    def output(self, value):
        """Turn ON and OFF the output"""
        self.write_expect(f"C1:OUTP {"ON" if value else "OFF"},LOAD,HZ,PLRT,NOR")
        self.__output = value

        s = self.status()
        if self.output != s["output"]: 
            raise Exception(f"There's a mismatch between the value of 'output', which is {self.output}, and the corresponding value on the device, which is {s["output"]}.")
    
    
    # FREQ

    @property
    def freq(self):
        return self.__freq

    @freq.setter
    def freq(self, f):
        """Set frequency in Hz of the generated waveform"""
        self.write_expect(f"C1:BASIC_WAVE FRQ,{f}")
        self.__freq = f

        s = self.status()
        if self.freq != s["freq"]: 
            raise Exception(f"There's a mismatch between the value of 'freq', which is {self.freq}Hz, and the corresponding value on the device, which is {s["freq"]}Hz.")
    

    # AMPLITUDE

    @property
    def amplitude(self):
        return self.__amplitude

    @amplitude.setter
    def amplitude(self, amp):
        """Set amplitude in Vpp of the generated waveform"""
        self.write_expect(f"C1:BASIC_WAVE AMP,{amp}")
        self.__amplitude = amp

        s = self.status()
        if self.amplitude != s["amplitude"]: 
            raise Exception(f"There's a mismatch between the value of 'amplitude', which is {self.amplitude}Vpp, and the corresponding value on the device, which is {s["amplitude"]}Vpp.")
    
    # MODULATION

    @property
    def modulation(self):
        return self.__modulation

    @modulation.setter
    def modulation(self, value):
        """Turn ON and OFF the modulation"""
        # AM = amplitude modulation, MDSP = modulation wave shape, ARB = arbitrary
        if value:
            self.write(f"C1:MDWV STATE,ON")
            self.write(f"C1:MDWV AM")
            self.write(f"C1:MDWV AM,SRC,INT")
            self.write(f"C1:MDWV MDSP,ARB,INDEX,19")
            print(self.query(f"*OPC?"))
        else:
            self.write_expect(f"C1:MDWV STATE,OFF,AM,MDSP,ARB")
        self.__modulation = value

        s = self.status()
        if self.modulation != s["modulation"]: 
            raise Exception(f"There's a mismatch between the value of 'modulation', which is {self.modulation}, and the corresponding value on the device, which is {s["modulation"]}.")
    
    # WAVEFORM

    @property
    def waveform(self):
        return self.__waveform

    @waveform.setter
    def waveform(self, name):
        """Set waveform of the generated waveform"""
        
        # self.write_expect(f"C1:BASIC_WAVE WVTP,SINE,FRQ,1000,AMP,4")
        self.write(f"C1:BASIC_WAVE WVTP,ARB")
        # TODO: NEED A COMMAND TO SAY THAT THE WAVEFORM IS TAKEN FROM FILE
        print(self.query_expect(f"STL? USER"))
        self.write(f"C1:ARWV NAME,\"{name}\"")
        self.write(f"C1:SRATE MODE,DDS")
        
        # self.write(f"C1:SRATE MODE,TARB,VALUE,1e6")
        # self.write_expect(f"C1:ARWV INDEX,{value}")
    
        self.__waveform = name

        s = self.status()
        if self.waveform != s["waveform"]: 
            pass
            # raise Exception(f"There's a mismatch between the value of 'waveform', which is '{self.waveform}', and the corresponding value on the device, which is '{s["waveform"]}'.")
    