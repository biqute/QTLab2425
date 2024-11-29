import matplotlib.pyplot as plt
import sys; sys.path.append("../classes")
from Oscilloscope import Oscilloscope

myOsc = Oscilloscope("192.168.40.20")
myOsc.timeout = 30e3
print(myOsc._name)
myOsc.write_expect("DATA:SOURCE CH1")
myOsc.query_expect("DATA:SOURCE?")

#myOsc.write_expect("WFMpre:NR_Pt 1000")

# temp = myOsc.query_expect("WFMO?").split(";")
# info = {
#     "encoding": temp[2],
# }
# print(temp)

myOsc.write_expect("WFMO:ENC ascii")
datay = list(map(int, myOsc.query_expect("CURVE?").split(",")[0:200]))
datax = range(0, len(datay))

plt.plot(datax, datay)
plt.title("Channel 1")
plt.xlabel("t [units]")
plt.ylabel("Voltage [units]")
plt.grid()
plt.show()
