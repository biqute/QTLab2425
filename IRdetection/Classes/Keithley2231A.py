import serial
import time

class Keithley2231a(serial.Serial):
        
    def __init__(self, name):
                
        self.ser = serial.Serial(name)  # open serial port

        # Clear the input buffer to ensure there are no pending commands
        self.ser.flushInput()
        # Command to set the device to remote mode
        self.ser.write(b'SYSTem:REMote\r\n')
        time.sleep(2)
        # Enable the output
        self.ser.write(b'OUTPut:STATe ON\r\n')    #use this command before setting the values of the output channels
        #check
        print('Is the connection open? ', self.ser.is_open)
        time.sleep(2)
        
    def is_open(self):
        print('Is the connection open? ', self.ser.is_open)

    def set_vi_1(self, voltage, current):
        # Set the voltage and current of the output channel 1
        
        # Set voltage and current
        setting_original  = str('APPLy CH1,', voltage, ',', current, '\r\n')      #the problem is here                      #It's not possible to write settings on the same line for all the channels
        setting_encoded = setting_original.encode(encoding='utf-8')
        self.ser.write(setting_encoded)



   # ser.write(b'APPLy CH2,1.59,0.2\r\n')
   # ser.write(b'APPLy CH3,1.4,0.4\r\n')
   # time.sleep(1)  # Allow time for the device to process the command

    # Optionally, verify the settings by querying the device
    #ser.write(b'APPLy? CH1\r\n')
    #ser.write(b'APPLy? CH2\r\n')
  #  ser.write(b'APPLy? CH3\r\n')
  #  time.sleep(1)  # Allow time for the device to respond

    # Read the response
   # response = ser.read_all()
   # print(response.decode('utf-8'))

    def close(self):
        # Close the serial port
        self.ser.close()
        #check
        print('Is the connection closed? ', not self.ser.is_open)