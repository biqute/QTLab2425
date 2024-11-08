import pyvisa


class PSA:
    _name = ""

    def __init__(self, resource_string):
        res_manager = pyvisa.ResourceManager()
        self.__res = res_manager.open_resource(resource_string)

        self.__res.write("*CLS") # clear settings
        self._name = self.__res.query("*IDN?")
        res = self.__res.query("INST:SEL 'SA'; *OPC?")
        
        if res != '1': raise Exception("Failed to select SA mode.")

    def set_point_count(self, n):
        self.__res.write("SENS:SWE:POIN " + str(n))
        return self.__res.query("*OPC?")

    def set_min_freq(self, n):
        self.__res.write("SENS:FREQ:START " + str(n))
        return self.__res.query("*OPC?")

    def set_max_freq(self, n):
        self.__res.write("SENS:FREQ:STOP " + str(n))
        return self.__res.query("*OPC?")   
