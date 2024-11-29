import pyvisa

class OSC :

    def __init__(self, ip) :

        rm1 = pyvisa.ResourceManager()
        self.__OSC =  rm1.open_resource(f"tcpip0::{ip}::inst0::INSTR")
        print ("device connected.")