import pyvisa

rm_2 = pyvisa.ResourceManager()
#rm_2.list_resources()
resource_2 = rm_2.open_resource("tcpip0::192.168.40.10::INSTR")
#resource_2 = rm_2.open_resource("tcpip0::192.168.40.15::inst0::INSTR")
print(resource_2.query("*IDN?"))

#rm_1 = pyvisa.ResourceManager()
#rm_1.list_resources()
#resource_1 = rm_1.open_resource('tcpip0::169.254.69.24::inst0::INSTR')
#print(resource_1.query("*IDN?"))

