import socket
import sys
import time

# Instantiate TCP/IP socket on this machine
computer_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
fsv_ip = '192.168.3.50'
port = 139

# Set a timeout for the socket
computer_sock.settimeout(3)  # Timeout after 10 seconds if no response

# Connect to the server
try:
    computer_sock.connect((fsv_ip, port))
except socket.error as e:
    print(f"Socket error: {e}")
    sys.exit(1)

# Send a command: *IDN?
command = '*IDN?\r\n'
computer_sock.sendall(command.encode('utf-8'))
print(f"Command sent: {command}")
time.sleep(1)

# Receive the response
try:
    data = computer_sock.recv(1024)  # This will now timeout after 10 seconds
    print(f"Response: {data.decode('utf-8')}")
except socket.timeout:
    print("Timed out waiting for a response from the server")
finally:
    computer_sock.close()  # Make sure to close the socket
