import numpy as np
import pyvisa
import time
import serial

class LO (serial.Serial) :

    def __init__(self, device_address) :

        self.device = serial.Serial(device_address, baudrate=115200, timeout=1.5)


    def write(self, msg_string) :
        
        self.device.write(bytes(msg_string))
        
        return True
    
    def read(self): #shorter read syntax for get function
        
        return self.device.readline().decode() #removes byte encoding syntax
    
    def ask(self,msg_string): #general get function
        
        self.write(msg_string) #takes an already encoded string
        
        return self.read()
    
    #Set parameters
    
    def set_frequency(self,freq):  # default units in GHz
        
        cmd_string = 'FREQ ' + str(freq) + 'GHz\r'
        
        self.write(str.encode(cmd_string))    # this converts a string to bytes
        
        return "Frequency set to "+str(freq)+" GHz."
    
    def set_ref_source(self, source):
        #as above: ref source can only be EXT (external) or INT internal
        if (not (source == "EXT" or source == "INT")):
            return "Invalid entry. Ref source must be set to EXT or IN"
        else:
            cmd_string = str.encode('ROSC:SOUR  ' + (source))
            self.write(cmd_string) #send command to set ref source
            return "Ref source set to "+source
        
    def set_output(self, set):
        #turn RF output ON or OFF (TURN DEVICE ON OR OFF)
        if (not (set == "ON" or set == "OFF")):
            return "Invalid entry. Ref out may only be set to ON or OFF"
        else:
            if set == "ON":
                self.write(b'0F01')
            elif set == "OFF":
                self.write(b'0F00')
            return "RF set to "+set
    
    
    #Get parameters

    def get_frequency(self):
        
        
        f = self.ask(b'FREQ?\r') 
        
        return float(f)
    
    
    def get_temp(self):

        return (self.ask(b'DIAG:MEAS? 21\r')).strip() + ' °C' 
    

    def get_ref_source(self): #device needs a ref source for frequency calibration
         #returns "EXT" for external ref source (REF IN pin)
         #returns "INT" for internal ref source (built in clock)
        return 'Ref source: '+self.ask(b'ROSC:SOUR?\r')
    
    def get_output(self):
        #get RF status
        #returns 1 (ON) or 0 (OFF)
        return 'RF '+ (self.ask(b'0x02'))