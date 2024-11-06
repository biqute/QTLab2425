import serial
import time

ser = serial.Serial('COM5')  # open serial port

print(ser.is_open)

# Clear the input buffer to ensure there are no pending commands
ser.flushInput()


# Command to set the device to remote mode
ser.write(b'SYSTem:REMote\r\n')
time.sleep(2)

# Enable the output
ser.write(b'OUTPut:STATe ON\r\n')    #use this command before setting the values of the output channels


# Set voltage and current
ser.write(b'APPLy CH1,1.5,0.5\r\n')                            #It's not possible to write settings on the same line for all the channels
ser.write(b'APPLy CH2,1.59,0.2\r\n')
ser.write(b'APPLy CH3,1.4,0.4\r\n')
time.sleep(1)  # Allow time for the device to process the command

# Optionally, verify the settings by querying the device
ser.write(b'APPLy? CH1\r\n')
ser.write(b'APPLy? CH2\r\n')
ser.write(b'APPLy? CH3\r\n')
time.sleep(1)  # Allow time for the device to respond

# Read the response
response = ser.read_all()
print(response.decode('utf-8'))

# Close the serial port
ser.close()