import numpy as np
from EthernetDevice import EthernetDevice

class AWG(EthernetDevice):
    """Arbitrary Waveform Generator (AWG)"""

    __output = False
    __freq = 0
    __amplitude = 0
    __modulation = False

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

        # Sanity checks
        if arr2[0] not in ["ON", "OFF"]: raise Exception(f"Output should be either ON or OFF, but '{arr2[0]}' was found") 
        if arr3[1] not in ["ON", "OFF"]: raise Exception(f"Modulation should be either ON or OFF, but '{arr3[1]}' was found") 

        print(arr3)

        return {
            "freq": int(arr1[3][:-2]),
            "amplitude": int(arr1[7][:-1]),
            "output": arr2[0] == "ON",
            "modulation": arr3[1] == "ON",
        }
    
    def upload_waveform(self, name, func, duration, samples_per_second = 2.4e9):
        points = int(duration * samples_per_second)

        if " " in name or "," in name: 
            raise Exception(f"name should not contain spaces ' ' nor commas ',', but \"{name}\" was found") 
        if samples_per_second > 2.4e9: 
            raise Exception(f"The maximum value of samples_per_second is 2.4e9, but {samples_per_second} was found")  
        if points > 1e6: 
            raise Exception(f"The maximum number of points is 1e6, but {points} was found")

        # dat = "Xpos,value"
        # for i in range(0,points):
        #     dat += f"\n{i},{func(i / samples_per_second)}" 
        # self.write_expect(f"C1:WVDT WVNM,{name},WAVEDATA,'{dat}'")

        # bin = ""
        # for i in range(0,points):
        #     f = func(i / samples_per_second)
        #     n = int((2**16)*f)
        #     if n < -32768: n = -32768
        #     if n > +32767: n = +32767
        #     string = hex(n)[2:]
        #     string = (4 - len(string))*"0" + string
        #     bin += string

        #print(bin)
        

        # test = [0x0080, 0x0070, 0x0060, 0x0050, 0x0060, 0x0070, 0x0080]
        
        array = np.zeros(points, dtype=np.int16)
        N = 7
        for i in range(0,points):
            f = func(i / samples_per_second)
            n = int(round((2**N)*f))
            if n > (2**N - 1): n = 2**N - 1
            array[i] = n
        self.write_binary_values(f"C1:WVDT WVNM,TEST,LENGTH,{points*2},WAVEDATA,", array) # https://pyvisa.readthedocs.io/en/1.8/rvalues.html#writing-binary-values !!!

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
    