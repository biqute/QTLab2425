import serial
import time

# Connect to COM3 port
ser = serial.Serial('COM3', 9600, timeout=1)
time.sleep(2)  # Wait for the connection to establish

# Ask for device name
ser.write(b'*IDN?\r\n')
time.sleep(1)  # Wait for the response

# Read the response
device_name = ser.readline().decode('utf-8').strip()
print(f'Device name: {device_name}')


ser.write(b'FREQ 4.81359GHz\r\n')
time.sleep(1)
ser.write(b'FREQ?\r\n')



# Read the response
frequency = ser.readline().decode('utf-8').strip()
print(f'Frequency: {frequency}')

ser.write(b'OUTP:STAT ON\r\n')
# Close the serial connection
ser.close()