import serial
import time

ser = serial.Serial('COM5')  # open serial port

print(ser.is_open)

# Clear the input buffer to ensure there are no pending commands
ser.flushInput()


# Command to set the device to remote mode
ser.write(b'*RST\r\n')
time.sleep(2)

# Enable the output
ser.write(b'OUTPut:STATe ON\r\n')