import socket
from matplotlib import pyplot as plt
import re
from scipy.signal import find_peaks

def query_power_spectrum_analyzer(host, port, start_freq, stop_freq, step):
    try:
        # Create a socket object
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        
        # Connect to the instrument
        s.connect((host, port))
        
        # Set start frequency
        s.sendall(f"FREQ:START {start_freq}\n".encode())
        
        # Set stop frequency
        s.sendall(f"FREQ:STOP {stop_freq}\n".encode())
        
        # Perform frequency sweep
        s.sendall(f"FREQ:STEP {step}\n".encode())
        s.sendall("INIT:CONT OFF\n".encode())  # Turn off continuous sweep
        s.sendall("INIT; *WAI\n".encode())     # Initiate and wait for completion
        
        # Query the power spectrum
        s.sendall("TRAC? TRACE1\n".encode())
        
        # Receive the response
        response = s.recv(65536)
        # Put data in a list
        data = re.sub(r'E', 'e', response.decode()).split(',')
        # Extract the data from the list
        data = [float(x) for x in data if x != '']  # Remove empty
        
        return data

    except socket.error as e:
        print(f"Socket error: {e}")
    finally:
        # Close the socket connection
        s.close()
        
def find_peak(x, y):
    # find the peak as max y value
    peak_idx = y.index(max(y))
    return x[peak_idx], y[peak_idx]

# 6.1e9 - 5.9e9 = 2e8
# Example usage
start_freq = 5.9e9
stop_freq = 6.1e9
step = 1e5     # leave this at 1e5
data = query_power_spectrum_analyzer('192.168.3.50', 5025, start_freq, stop_freq, step)
print(f'Number of data points: {len(data)}')
x = [start_freq + i * step for i in range(len(data))]

# Peak
peak_x, peak_y = find_peak(x, data)

# Plot the data
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(x, data)

# Plot the peak
ax.plot(peak_x, peak_y, 'ro')
ax.set_xlabel('Frequency')
ax.set_title('Power Spectrum')
ax.set_ylabel('Power (dBm)')
plt.show()