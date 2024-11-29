import pyvisa
from EthernetDevice import EthernetDevice


class SMA(EthernetDevice):
    """SMA100B signal generator"""

    __freq = 0
    __power_level = 0

    def on_init(self, ip_address_string):
        self.write_expect("*RST") # clear settings
        self.write_expect("*CLS") # clear settings
        self.write_expect("SOUR:FREQ:MODE CW") # set mode to Continous Wave (CW)

        self.freq = 5e9
        self.power_level = -20
     
    # FREQUENCY

    @property
    def freq(self):
        return self.__freq

    @freq.setter
    def freq(self, f):
        """Set frequency in Hz"""
        self.write_expect(f"SOUR:FREQ:OFFSET 0 Hz")
        self.write_expect("SOUR:FREQ:MULTIPLIER 1.0")
        self.write_expect(f"SOUR:FREQ:CW {f} Hz") # set CW frequency
        self.__freq = f

        if int(self.query("SOUR:FREQ:CW?")) != f: 
            raise Exception(f"Could not set 'freq' to {f}.")
     
    # POWER LEVEL

    @property
    def power_level(self):
        return self.__power_level

    @power_level.setter
    def power_level(self, pow_lvl):
        """Set output power level in dBm"""
        self.write_expect(f"SOUR:POW:POW {pow_lvl} dBm") # set power level
        self.__power_level = pow_lvl

        if int(self.query("SOUR:POW:POW?")) != pow_lvl: 
            raise Exception(f"Could not set 'power_level' to {pow_lvl}.")
        
    # ON/OFF

    def turn_on(self):
        self.write_expect("OUTP ON")
    
    def turn_off(self):
        self.write_expect("OUTP OFF")
