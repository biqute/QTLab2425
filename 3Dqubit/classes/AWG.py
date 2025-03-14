import numpy as np
from EthernetDevice import EthernetDevice

class AWG(EthernetDevice):
    """Arbitrary Waveform Generator (AWG)"""

    __freq = 0
    __amplitude = 0
    __output = False

    def on_init(self, ip_address_string):
        self.write_expect("*RST") # clear settings

        self.freq = 1000 # 1kHz
        self.amplitude = 4 # 4Vpp
        self.output = False # Whether the signal is outputted or not (ON/OFF)

    def status(self):
        arr1 = self.query_expect(f"C1:BASIC_WAVE?").strip()[8:].split(",")
        arr2 = self.query_expect(f"C1:OUTP?").strip()[8:].split(",")
        return {
            "freq": int(arr1[3][:-2]),
            "amplitude": int(arr1[7][:-1]),
            "output": arr2[0] == "ON",
        }

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
    