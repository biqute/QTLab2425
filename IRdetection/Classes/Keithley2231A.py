import serial
import time

allowed =[1,2,3]
default_current = 0.1

class Keithley2231A(serial.Serial):
        
    def __init__(self, name):
                
        self.ser = serial.Serial(name)  # open serial port

        # Clear the input buffer to ensure there are no pending commands
        self.ser.flushInput()
        # Command to set the device to remote mode
        self.__write('SYSTem:REMote')
        time.sleep(0.5)
        # Enable the output
        self.__write('OUTPut:STATe ON')    #use this command before setting the values of the output channels
        #check
        if not self.ser.is_open:
            raise ValueError('Connection failed. Check the port name and the device.')
    
    
    def check_open(self):
        if not self.ser.is_open:
            raise ValueError('Connection failed. Check the port name and the device.')

    def set_voltage(self, voltage, current=None, ch=1):
        """
        Set voltage to the specified channel and current limit.
        
        :param voltage: Voltage value in volts
        :param current: Current limit in amperes
        :param ch: Channel number (1, 2 or 3)
        
        :return: None
        """
        
        if int(ch) not in allowed:
            raise ValueError('Channel should be one of: 1, 2 or 3.')
                    
        ch_str = 'CH' + str(int(ch))
        
        # Set voltage and current
        actual_current = current if current is not None else self.get_current_limit(ch)
        command = f'APPLy {ch_str},{voltage},{actual_current}'
        self.__write(command)
        
    def set_current_limit(self, current, ch=1):
        """
        Set current limit to the specified channel.
        
        :param current: Current limit in amperes
        :param ch: Channel number (1, 2 or 3)
        
        :return: None
        """
            
        if int(ch) not in allowed:
            raise ValueError('Channel should be one of: 1, 2 or 3.')
        
        ch_str = 'CH' + str(int(ch))
        
        # Set voltage and current
        actual_voltage = self.get_voltage(ch)
        command = f'APPLy {ch_str},{actual_voltage},{current}'
        self.__write(command)

    
    def get_channel_values(self, ch=1):
        """
        Get the values of the specified channel.
        
        :param ch: Channel number (1, 2 or 3)
        
        :return: values of the channel (string)
        """
            
        if int(ch) not in allowed:
            raise ValueError('Channel should be one of: 1, 2 or 3.')
        
        ch_str = 'CH' + str(int(ch))
        
        # Query the voltage value
        self.__write(f'APPLy? {ch_str}')
        time.sleep(0.5)
        response = self.ser.read_all()
        response = response.decode('utf-8')
        return response
        
        
    def get_voltage(self, ch=1):
        """
        Get the voltage value of the specified channel.
        
        :param ch: Channel number (1, 2 or 3)
        
        :return: Voltage value in volts
        """
        
        if int(ch) not in allowed:
            raise ValueError('Channel should be one of: 1, 2 or 3.')
        
        response = self.get_channel_values(ch)
        voltage = response.split(',')[0]
        return float(voltage)
    
    def get_current_limit(self, ch=1):
        """
        Get the current limit value of the specified channel.
        
        :param ch: Channel number (1, 2 or 3)
        
        :return: Current limit value in amperes
        """
        
        if int(ch) not in allowed:
            raise ValueError('Channel should be one of: 1, 2 or 3.')
        
        response = self.get_channel_values(ch)
        current = response.split(',')[1][1:]
        return float(current)
    
    def __write(self, command_str):
        command_str += '\r\n'
        encoded_command = command_str.encode(encoding='utf-8')
        self.ser.write(encoded_command)
       
    def __activate(self):
        self.__write('SYSTem:REMote')
        self.__write('OUTPut:STATe ON')
    
    def reset(self):
        """
        Reset the device to its default settings.
        """
        self.__write('*RST')
        time.sleep(0.5)
        self.__activate()
        self.ser.flushInput()



    def close_conncetion(self):
        # Close the serial port
        self.ser.close()
        # Check if the port is closed
        if self.ser.is_open:
            raise ValueError('Connection failed to close. Check the port name and the device.')