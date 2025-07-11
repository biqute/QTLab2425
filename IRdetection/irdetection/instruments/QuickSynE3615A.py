import serial
import time
from irdetection.abstract.Instrument import Instrument

class QuickSyn(Instrument):
    
    def __init__(self, port_name, name=None):
        name = name if name is not None else "QuickSyn_" + str(port_name)
        super().__init__(name)
        self.port_name = port_name
      
    def initialize(self):
        self.ser = serial.Serial(self.port_name)  # open serial port
        # Clear the input buffer to ensure there are no pending commands
        self.ser.flushInput()
        #check
        if not self.ser.is_open:
            raise ValueError('Connection failed. Check the port name and the device.')
        # Set the output state to ON
        self._Instrument_activate()
    
    def set_frequency(self, frequency, order="GHz"):
        """
        Set frequency.
        
        :param frequency: Frequency value in order specified
        :param order: Frequency order (GHz, MHz, KHz, mlHz)
        """
        if order not in ["GHz", "MHz", "KHz", "mlHz"]:
            raise ValueError('Order should be one of: GHz, MHz, KHz, mlHz.')   
        
        self.__write(f'FREQ {float(frequency)}{order}')
        # Set output state to ON
        time.sleep(0.5)
        
    
    def set_output_state(self, state):
        """
        Set output flow control.
        
        :param state: 'ON' or 'OFF'
        """
        if state not in ['ON', 'OFF']:
            raise ValueError('State should be one of: ON, OFF.')
        
        self.__write(f'OUTP:STAT {state}')
    
    def get_frequency(self, order="GHz"):
        """
        Get frequency for the specified order (GHz, MHz, KHz, Hz, mlHz).
        
        :return: Frequency value in GHz
        """
        
        
        if order not in ["GHz", "MHz", "KHz", "Hz", "mlHz"]:
            raise ValueError('Order should be one of: GHz, MHz, KHz, Hz, mlHz.')   
        
        self.__write('FREQ?')
        frequency = self.ser.readline().decode('utf-8').strip() # default order is mlHz
        
        
        if order == "GHz":
            frequency = float(frequency) / 1e12
        elif order == "MHz":
            frequency = float(frequency) / 1e9
        elif order == "kHz":
            frequency = float(frequency) / 1e6
        elif order == "Hz":
            frequency = float(frequency) / 1e3
        else:
            frequency = float(frequency)
                
        return frequency
        
    def __write(self, command_str):
        command_str += '\r\n'
        encoded_command = command_str.encode(encoding='utf-8')
        self.ser.write(encoded_command)
        
    def close_connection(self):
        # #turn off the output
        # self.__write('OUTP:STAT OFF')
        # Close the serial port
        self.ser.close()
        # Check if the port is closed
        if self.ser.is_open:
            raise ValueError('Connection failed to close. Check the port name and the device.')
    
    def shutdown(self):
        self.__write('OUTP:STAT OFF')
        self.close_connection()
    
    def kill(self):
        self.__write("*RST")
        time.sleep(0.5)
        # Reset to factory settings
        self.__write("*RCL 0")
        self.shutdown()
        print("Device has been killed :)")
        
    def _Instrument__activate(self):
        self.__write('OUTP:STAT ON')
        
    def info(self, verbose=False):
        message = "Device: " + self.name + "\n"
        message += "Port: " + self.ser.name + "\n"
        if verbose:
            message += "Serial connection: " + str(self.ser.is_open) + "\n"
            # add freqency settings
            message += "Frequency: " + str(self.get_frequency()) + " GHz\n"
        return message
    
    def reset(self):
        # Default is 10 MHz
        self.set_frequency(10, 'MHz')
            
    
    def __del__(self):
        self.close_connection()