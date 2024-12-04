import pyvisa

class OSC :

    def __init__(self, ip) :

        rm1 = pyvisa.ResourceManager()
        self.__OSC =  rm1.open_resource(f"tcpip0::{ip}::inst0::INSTR")
        print ("device connected.")

    def acquire_data(self, channel) :

        self.__OSC.write(f"DATA:SOURCE {channel}")
        self.__OSC.write("HEAD 0")
        return list(map(float,self.__OSC.query("CURVE?").split(",")))