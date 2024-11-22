import serial
import time

class QuickSyn(serial.Serial):
    
    def __init__(self, name):
        self.ser = serial.Serial(name)  # open serial port
        # Clear the input buffer to ensure there are no pending commands
        self.ser.flushInput()
        #check
        if not self.ser.is_open:
            raise ValueError('Connection failed. Check the port name and the device.')
    
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
        self.__write('OUTP:STAT ON')
    
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
        
    def close_conncetion(self):
        #turn off the output
        self.__write('OUTP:STAT OFF')
        # Close the serial port
        self.ser.close()
        # Check if the port is closed
        if self.ser.is_open:
            raise ValueError('Connection failed to close. Check the port name and the device.')