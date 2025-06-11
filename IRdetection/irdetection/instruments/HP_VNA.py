import pyvisa as visa
import time
import numpy as np
import struct
from irdetection.abstract.Instrument import Instrument

class VNA(Instrument):
    def __init__(self, resource_name='GPIB0::16::INSTR', name=None):
        name = name if name is not None else "VNA_" + str(resource_name)
        super().__init__(name)
        
        self.resource_name = resource_name
        
        # SET INSTRUMENT LIMITS
        self.MAX_FREQUENCY = 6e9
        self.MIN_FREQUENCY = 3e4
        
        self.rm = visa.ResourceManager()
        
    def initialize(self):
        self.vna = self.rm.open_resource(self.resource_name)
        self.vna.timeout = 5000  # Set timeout to 25 seconds
        self.set_array_output_dtype('float32')
        
    # -------------------- DATA --------------------
    def acq_get_sweep_data(self):
        """
        Get sweep data from VNA. The data is returned as a numpy array of complex numbers.
        """
        raw_data = self.query_raw('OUTPFORM;')
        return self.decode_raw(raw_data)

    def acq_read_data(self, matrix_element=None, dtype='float64'):
        """
        Read data from VNA. The data is returned as a numpy array of complex numbers.
        """
        if matrix_element is not None:
            self.set_matrix_element(matrix_element)
        self.set_array_output_dtype(dtype)
        self.set_array_output_format('real')
        real, _ = self.get_sweep_data()
        self.set_array_output_format('imaginary')
        imaginary, _ = self.get_sweep_data()

        return real+1j*imaginary
    
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

    def set_channels_coupling(self, coupling='ON'):
        """
        Set channels coupling. Available options are:
            - 'ON': Coupling is enabled
            - 'OFF': Coupling is disabled
        """
        if coupling not in ['ON', 'OFF']:
            raise ValueError('Invalid coupling option. Available options: ON, OFF')
        self.write(f'COUC {coupling}')
    
    def set_channel(self, channel=1):
        """
        Set active channel. Available options are: 1, 2, 3, 4
        """
        if channel not in [1, 2, 3, 4]:
            raise ValueError('Invalid channel. Available options: 1, 2, 3, 4')
        self.write(f'CHAN{channel}')
    
    def set_matrix_element(self, element='S11'):
        """
        Set matrix element. Available options are: S11, S21 S12, S22
        """
        if element not in ['S11', 'S21', 'S12', 'S22']:
            raise ValueError('Invalid element. Available options: S11, S21, S12, S22')
        self.write(f'{element};')
    
    def set_sweep_points(self, npoints=201):
        self.write(f'POIN {npoints}')
        
    def set_IFbandwidth(self, bandwidth):
        """
        Set IF bandwidth. The bandwidth is in Hz.
        Possible values are: 10, 30, 100, 300, 1000, 3000, 3700, 6000
        """
        if bandwidth not in [10, 30, 100, 300, 1000, 3000, 3700, 6000]:
            raise ValueError('Invalid bandwidth. Possible values are: 10, 30, 100, 300, 1000, 3000, 3700, 6000')
        if bandwidth == 10:
            self.vna.timeout = 25000  # Set timeout to 25 seconds
        else:
            self.vna.timeout = 5000  # Set timeout to 5 seconds
        self.write(f'IFBW {bandwidth}')
     
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
        
        formats = {2: 'float32', 3: 'float64', 4: 'ASCII'}
        return formats[self.array_format]
    
    def get_channels_coupling(self):
        return self.query('COUC') # ON or OFF
    
    def get_channel(self):
        self.write('OUTPCHAN')
        result = self.vna.read()
        return int(result)
    
    def get_matrix_element(self):
        active_matrix_elements = [self.query(m_el, int) for m_el in ['S11', 'S21', 'S12', 'S22']]
        m_el_dict = {0: 'S11', 1: 'S21', 2: 'S12', 3: 'S22'}
        # get index of active element
        active_index = active_matrix_elements.index(1)
        return m_el_dict[active_index]

    def get_frequency_points(self):
        start_freq = self.get_start_frequency()
        stop_freq = self.get_stop_frequency()
        npoints = self.get_sweep_points()
        return np.linspace(start_freq, stop_freq, npoints)
    
    def get_sweep_points(self):
        return self.query('POIN', int)
    
    def get_IFbandwidth(self):
        return self.query('IFBW', int)
    
    # ------------------ UTILITY ------------------
    def write(self, command):
        self.vna.write(command)
    
    def query(self, command, return_type=str):
        response = self.vna.query(command+'?')
        # if ressponse has just one '\n' remove it
        if response[-1] == '\n':
            response = response[:-1]
        try:
            if return_type == int:
                return return_type(float(response))
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
            data = np.frombuffer(raw_data[4:], dtype='>f8') # 8 bytes per float
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
    
    def complex_to_dB(self, data):
        return 20*np.log10(np.absolute(data))
    
    
# ---- Instrument methods ----

    def info(self, verbose=False):
        """
        Get information about the instrument.
        """
        message = f"Device name: {self.name}"
        message += f"\nResource name: {self.vna.resource_name}"
        if verbose:
            message += f"\nMax frequency: {self.MAX_FREQUENCY}"
            message += f"\nMin frequency: {self.MIN_FREQUENCY}"
            message += f"\nTimeout: {self.vna.timeout}"
            # Instrument IDN response
            message += f"\nIDN: {self.vna.query('*IDN?')}"
        return message
    
    def _Instrument_activate(self): # Dummy method
        pass
    
    def reset(self):
        """
        Reset the instrument.
        """
        self.write('*RST')
        
    def close_connection(self):
        """
        Close the connection to the instrument.
        """
        self.vna.close()
        
    def shutdown(self):
        """ 
        Shut down the instrument.
        """
        self.close_connection()
        
    def kill(self):
        """
        Kill the instrument.
        """
        self.reset()
        self.close_connection()
        
    def __del__(self):
        self.close_connection()