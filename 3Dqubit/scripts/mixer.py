import sys; sys.path.append("../classes")
from SMA import SMA
from LO import LO


myLO = LO("COM4")
mySMA = SMA("192.168.40.15")

myLO.debug_prefix = "LO"
mySMA.debug_prefix = "SMA"

f = 4e9
df = 0.050e9

myLO.freq = f
mySMA.freq = f + df
mySMA.power_level = -10
myLO.turn_on()
mySMA.turn_on()

