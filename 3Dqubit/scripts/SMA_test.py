import sys; sys.path.append("../classes")
from SMA import SMA

f = 5e9
df = 0.5e9

# SMA100B
mySMA = SMA("192.168.40.15")
mySMA.freq = f + df # Hz
mySMA.power_level = -10 # dBm
mySMA.turn_on()


