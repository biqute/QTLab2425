import pyvisa as visa
import time
import numpy as np
import struct
import matplotlib.pyplot as plt

class VNA:
    def __init__(self, resource_name='GPIB0::16::INSTR'):
        # SET INSTRUMENT LIMITS
        self.MAX_FREQUENCY = 6e9
        self.MIN_FREQUENCY = 3e4
        
        self.rm = visa.ResourceManager()
        self.vna = self.rm.open_resource(resource_name)
        self.set_array_output_dtype('float32')
        print(self.vna.query('*IDN?'))
        
    # -------------------- DATA --------------------
    def get_sweep_data(self):
        """
        Get sweep data from VNA. The data is returned as a numpy array of complex numbers.
        """
        raw_data = self.query_raw('OUTPFORM;')
        return self.decode_raw(raw_data)
    
    # ------------------ SETTERS ------------------
    def set_start_frequency(self, start_freq, unit='GHZ'):
        self.validate_frequency(start_freq, unit)        
        self.write(f'STAR {start_freq}{unit}')
    
    def set_stop_frequency(self, stop_freq, unit='GHZ'):
        self.validate_frequency(stop_freq, unit)
        self.write(f'STOP {stop_freq}{unit}')
    
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
        self.write(f'CENT {center_freq}{unit}')
        
    def set_frequency_span(self, freq_span, unit='GHZ'):
        center = self.get_center_frequency(unit)
        self.validate_frequency(center - freq_span/2, unit)
        self.validate_frequency(center + freq_span/2, unit)
        self.write(f'SPAN {freq_span}{unit}')
     
    def set_array_output_dtype(self, format='float64'):
        """
        Sets output mode for array data. Available formats are:
            - 'float32': 32-bit floating point (with 4 byte header)
            - 'float64': 64-bit floating point (with 4 byte header)
            - 'ASCII': ASCII (with no header)
            
        Parameters:
        
        :param format: str -> Output dtype for array data
        """
        formats = {'float32': 2, 'float64': 3, 'ASCII': 4}
        if format not in formats:
            raise ValueError(f'Invalid format. Available formats: {formats.keys()}')
        self.write(f'FORM{formats[format]}')
        self.array_format = formats[format]
        
    def set_array_output_format(self, format='polar'):
        """
        Sets output format for array data. Available formats are:
            - 'polar': Polar format (magnitude and phase)
            - 'log magnitude': Logarithmic magnitude
            - 'phase': Phase data
            - 'delay': Delay data
            - 'smith chart': Smith Chart representation
            - 'linear magnitude': Linear magnitude
            - 'standing wave ratio': Standing Wave Ratio (SWR)
            - 'real': Real part
            - 'imaginary': Imaginary part
        
        Parameters:
        
        :param format: str -> Output format for array data
        """
        formats = {
            'polar': 'POLA',
            'log magnitude': 'LOGM',
            'phase': 'PHAS',
            'delay': 'DELA',
            'smith chart': 'SMIC',
            'linear magnitude': 'LINM',
            'standing wave ratio': 'SWR',
            'real': 'REAL',
            'imaginary': 'IMAG'
            }
        
        if format not in formats:
            raise ValueError(f'Invalid format. Available formats: {formats.keys()}')
        
        self.write(f'{formats[format]}')

     
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
    
    def get_array_format(self):
        # Ask to the VNA for the current format
        response = self.query_raw('FORM?')
        
        formats = {2: 'float64', 3: 'float32', 4: 'ASCII'}
        return formats[self.array_format]
        
    # ------------------ UTILITY ------------------
    def write(self, command):
        self.vna.write(command)
    
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
        self.write(command)
        return self.vna.read_raw()
    
    def decode_raw(self, raw_data):
        """
        Decode raw output from VNA. The output is a byte array encoded in the instrument's current format.
        
        Parameters:
        :param raw_data: bytes -> Raw data from VNA
        
        :return: tuple of numpy arrays
        """
        data = None
        # Read float32
        if self.array_format == 2:
            data = np.frombuffer(raw_data[4:], dtype='>f4') # 4 bytes per float. '>' is for big-endian, 'f' is for float, '4' is for 4 bytes
        # Read float64
        elif self.array_format == 3:
            data = np.frombuffer(raw_data[4:], dtype='>f8')     
        # Read ASCII
        elif self.array_format == 4:
            decoded_data = raw_data.decode('utf-8').strip()
            import re
            # Use regex to extract all numbers
            numbers = re.findall(r'[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?', decoded_data)
            data = np.array(numbers, dtype=np.float32)
            
        col1 = data[0::2]
        col2 = data[1::2]
        return col1, col2
    
    def read_data_lol(self):
        self.write('FORM3;')
        self.write('POIN?;')
        num_points = int(float((self.vna.read('\n'))))
        
        self.write('OUTPDATA')
        num_bytes = 16*int(num_points)+4
        raw_bytes = self.vna.read_bytes(num_bytes)

        #print(raw_bytes)

        trimmed_bytes = raw_bytes[4:]
        tipo='>'+str(2*num_points)+'d'
        x = struct.unpack(tipo, trimmed_bytes)
        #x.byteswap()
        
        #del x[1::2]
        return list(x)
        
    
    def clear_all(self):
        self.write('*RST')
    

# Example usage
vna = VNA()

print('Testing center frequency and frequency span')
vna.set_start_frequency(2)
vna.set_stop_frequency(6)
#vna.set_frequency_span(2)
vna.set_array_output_dtype('float32')
vna.set_array_output_format('log magnitude')
#print(vna.get_array_format())
r, _ = vna.get_sweep_data()
# Plot polar data with r=0 in the center
ax = plt.subplot()
x_axis = np.linspace(0, 100, len(r))
ax.plot(x_axis, r)
plt.show()


# TODO:
# - Add channel selection
# - Add Matrix element selection
# - Add channel coupling switch (ON/OFF)
# - Add frequency getter
# - Add high level function to get i,q,freq
# - Add high level function to get magnitude and frequency