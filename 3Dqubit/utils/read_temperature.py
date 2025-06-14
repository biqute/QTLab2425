import sys; sys.path.append("../classes")
from EthernetDevice import EthernetDevice

def read_temperature(address = "212.189.204.22:33576", chamber = 8):
    dev = EthernetDevice(address)
    dev.timeout = 10e3 # ms
    temp = dev.query_expect(f"READ:DEV:T{chamber}:TEMP:SIG:TEMP", f"Could not read temperature of chamber {chamber} from device {dev._name}.")
    return temp