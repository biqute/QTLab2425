import serial
import time

class QuickSyn(serial.Serial):
    
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
        
    def __write(self, command_str):
        command_str += '\r\n'
        encoded_command = command_str.encode(encoding='utf-8')
        self.ser.write(encoded_command)