import serial
import time
from Keithley2231A import Keithley2231A as KL

gen = KL("COM5")
gen.reset()
#time.sleep(5)
gen.set_voltage(3.0)
time.sleep(1)
gen.set_current_limit(0.08)

gen.close_conncetion()