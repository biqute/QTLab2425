import serial
import time

class LO(serial.Serial):
    """
    Local Oscillator (LO) (or Frequency Synthesizer)

    Conventions:
        - All arrays are numpy arrays
    
    Units:
        - frequency [Hz]
        - time [ms]
        - amplitude TODO
    """

    __freq = 0.0

    def __init__(self, name):
        self.__ser = serial.Serial(name)  # open serial port
        self.__ser.flushInput() # Clear the input buffer to ensure there are no pending commands

        if not self.__ser.is_open: raise Exception("Connection failed.")

        self.frequency = 5e9 # 5GHz

    def write(self, unterminated_command):
        command_utf8 = (unterminated_command + "\r\n").encode(encoding="utf-8")
        self.__ser.write(command_utf8)

    def query(self, unterminated_command):        
        self.write(unterminated_command)
        string = self.__ser.readline().decode("utf-8").strip()
        return string
    
    def turn_on(self):
        self.write("OUTP:STAT ON")
    
    def close(self):
        self.write("OUTP:STAT OFF") # turn off the output
        self.__ser.close() # close the serial port

        if self.ser.is_open: raise Exception("Connection failed to close.")

    # FREQUENCY

    @property
    def frequency(self):
        return self.__freq

    @frequency.setter
    def frequency(self, f):
        """Set synthetized frequency in Hz"""
        f_millis = f * 1000 # f in mHz
        self.write(f"FREQ {f_millis}mlHz")
        time.sleep(0.005)
        # self.write('OUTP:STAT ON')
        self.__freq = f

        if int(self.query("FREQ?")) != f_millis: 
            raise Exception(f"Could not set 'freq' to {f}.")