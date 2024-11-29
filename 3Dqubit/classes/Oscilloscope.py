from EthernetDevice import EthernetDevice


class Oscilloscope(EthernetDevice):
    """Oscilloscope (DPO3014)"""
    
    def on_init(self, ip_address_string):
        self.write_expect("WFMO:ENC ascii")


    # DATA ACQUISITION
    
    def read_data(self, CH):
        """CH is a string of value "CH1", "CH2", "CH3" or "CH4"."""
        self.write_expect(f"DATA:SOURCE {CH}") # select channel
        self.write_expect("HEAD 0") # turn off headers
        return list(map(float, self.query_expect("CURVE?").split(",")))