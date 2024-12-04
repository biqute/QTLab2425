import pyvisa as visa
import time
import numpy as np


class VNA:
    def __init__(self, resource_name='GPIB0::16::INSTR'):
        # SET INSTRUMENT LIMITS
        self.MAX_FREQUENCY = 6e9
        self.MIN_FREQUENCY = 3e4
        
        self.rm = visa.ResourceManager()
        self.vna = self.rm.open_resource(resource_name)
        print(self.vna.query('*IDN?'))
        
    # -------------------- DATA --------------------
    def get_matrix_element(self, element='S11'):
        data = self.get_rawdata(element)
        print(data)
        data = data.split(',')
        data = [float(i) for i in data]
        return data
    
    # ------------------ SETTERS ------------------
    def set_start_frequency(self, start_freq, unit='GHZ'):
        self.validate_frequency(start_freq, unit)        
        self.vna.write(f'STAR {start_freq}{unit}')
    
    def set_stop_frequency(self, stop_freq, unit='GHZ'):
        self.validate_frequency(stop_freq, unit)
        self.vna.write(f'STOP {stop_freq}{unit}')
    
    def set_center_frequency(self, center_freq, unit='GHZ', auto_span=True):
        self.validate_frequency(center_freq, unit)
        span = self.get_frequency_span(unit)
        if auto_span:
            try:
                self.validate_frequency(center_freq - span/2, unit)
                self.validate_frequency(center_freq + span/2, unit)
            except:
                span_target = 1 # GHz
                max_allowed_span = min(abs(center_freq - self.MIN_FREQUENCY*1e-9), abs(self.MAX_FREQUENCY*1e-9 - center_freq))
                span = min(span_target, max_allowed_span)
                self.set_frequency_span(span) # GHz
        else:
            self.validate_frequency(center_freq - span/2, unit)
            self.validate_frequency(center_freq + span/2, unit)
        self.vna.write(f'CENT {center_freq}{unit}')
        
    def set_frequency_span(self, freq_span, unit='GHZ'):
        center = self.get_center_frequency(unit)
        self.validate_frequency(center - freq_span/2, unit)
        self.validate_frequency(center + freq_span/2, unit)
        self.vna.write(f'SPAN {freq_span}{unit}')
     
    # ------------------ GETTERS ------------------
    def get_start_frequency(self, unit='HZ'):
        start_freq = self.convert_frequency_unit(self.query('STAR', float), 'HZ', unit)
        return start_freq
        
    def get_stop_frequency(self, unit='HZ'):
        stop_freq = self.convert_frequency_unit(self.query('STOP', float), 'HZ', unit)
        return stop_freq
    
    def get_center_frequency(self, unit='HZ'):
        center_freq = self.convert_frequency_unit(self.query('CENT', float), 'HZ', unit)
        return center_freq
    
    def get_frequency_span(self, unit='HZ'):
        frequency_span = self.convert_frequency_unit(self.query('SPAN', float), 'HZ', unit)
        return frequency_span
    
    def get_rawdata(self, element='S11'):
        index = {'S11': 1, 'S21': 2, 'S12': 3, 'S22': 4}
        test = self.query('OUTPDATA')
        return test
    
    # ------------------ UTILITY ------------------
    def query(self, command, return_type=str):
        response = self.vna.query(command+'?')
        try:
            return return_type(response)
        except ValueError:
            raise ValueError(f'Error converting response to {return_type}. Plain response: {response}')
        
    def convert_frequency_unit(self, value, origin, target):
        units = {'HZ': 1, 'KHZ': 1e3, 'MHZ': 1e6, 'GHZ': 1e9}
        return value * units[origin] / units[target]
    
    def validate_frequency(self, freq, unit):
        if freq < 0:
            raise ValueError('Frequency must be greater than 0')
        if self.convert_frequency_unit(freq, unit, 'HZ') > self.MAX_FREQUENCY or self.convert_frequency_unit(freq, unit, 'HZ') < self.MIN_FREQUENCY:
            raise ValueError(f'Frequency must be between {self.MIN_FREQUENCY} and {self.MAX_FREQUENCY}')
        
    def query_raw(self, command):
        self.vna.write(command)
        return self.vna.read_raw()
    
    def clear_all(self):
        self.vna.write('*RST')
    

# Example usage
vna = VNA()

print('Testing center frequency and frequency span')
vna.set_start_frequency(1)
vna.set_stop_frequency(3)
#vna.set_frequency_span(2)
vna.set_center_frequency(4)
print(vna.get_matrix_element())