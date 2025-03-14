import serial
import time
from src.abstract.Instrument import Instrument

ALLOWED_CHANNELS = [1, 2, 3]

class Keithley2231A(Instrument):
    """
    Class for interfacing with the Keithley 2231A device.
    """

    def __init__(self, port_name, name=None):
        """
        Initialize the connection to the device.

        :param port_name: Serial port name
        """
        name = name if name is not None else "Keithley2231A_" + str(port_name)
        super().__init__(name)

        self.port_name = port_name
       

    def initialize(self):
        """
        Initialize the instrument by activating remote mode.
        """
        self.ser = serial.Serial(self.port_name)  # Open serial port
        # Clear the input buffer to ensure there are no pending commands
        self.ser.flushInput()
        # Activate the device using the abstract interface method
        self._Instrument__activate()
        if not self.ser.is_open:
            raise ValueError('Connection failed. Check the port name and the device.')
        
        return f"Keithley2231A initialized on port {self.port_name}"

    def info(self, verbose=False):
        """
        Return information about the instrument.
        
        :param verbose: If True, include detailed serial connection parameters.
        """
        info_str = f"Keithley2231A on port {self.port_name}"
        if verbose:
            info_str += f"\nSerial settings: {self.ser}"
        return info_str

    def _Instrument__activate(self):
        # Set the device to remote mode and enable the output
        self.__write('SYSTem:REMote')
        self.__write('OUTPut:STATe ON')  # Use this command before setting output channel values

    def set_voltage(self, voltage, current=None, channel=1):
        """
        Set voltage and current limit for the specified channel.

        :param voltage: Voltage value in volts
        :param current: Current limit in amperes
        :param channel: Channel number (1, 2, or 3)
        """
        self.__validate_channel(channel)
        actual_current = current if current is not None else self.get_current_limit(channel)
        self.__write(f'APPLy CH{channel},{voltage},{actual_current}')

    def set_current_limit(self, current, channel=1):
        """
        Set current limit for the specified channel.

        :param current: Current limit in amperes
        :param channel: Channel number (1, 2, or 3)
        """
        self.__validate_channel(channel)
        actual_voltage = self.get_voltage(channel)
        self.__write(f'APPLy CH{channel},{actual_voltage},{current}')

    def get_channel_values(self, channel=1):
        """
        Get the values of the specified channel.

        :param channel: Channel number (1, 2, or 3)
        :return: String containing the voltage and current limit
        """
        self.__validate_channel(channel)
        self.__write(f'APPLy? CH{channel}')
        time.sleep(0.5)
        response = self.ser.read_all()
        response = response.decode('utf-8')
        return response

    def get_voltage(self, channel=1):
        """
        Get the voltage value of the specified channel.

        :param channel: Channel number (1, 2 or 3)
        :return: Voltage value in volts
        """
        response = self.get_channel_values(channel)
        voltage = response.split(',')[0]
        return float(voltage)

    def get_current_limit(self, channel=1):
        """
        Get the current limit value of the specified channel.

        :param channel: Channel number (1, 2 or 3)
        :return: Current limit value in amperes
        """
        response = self.get_channel_values(channel)
        current = response.split(',')[1][1:]
        return float(current)

    def __write(self, command_str):
        command_str += '\r\n'
        encoded_command = command_str.encode(encoding='utf-8')
        self.ser.write(encoded_command)

    def reset(self):
        """
        Reset the device to its default settings.
        """
        self.__write('*RST')
        time.sleep(0.5)
        self._Instrument__activate()
        self.ser.flushInput()

    def close_connection(self):
        """
        Close the serial connection to the device.
        """
        self.ser.close()
        if self.ser.is_open:
            raise ValueError('Connection failed to close. Check the port name and the device.')
        
    def shutdown(self):
        """
        Shut down the device.
        """
        self.close_connection()
        
    def kill(self):
        """
        Kill the device.
        """
        self.reset()
        self.close_connection()

    def __validate_channel(self, channel):
        """
        Validate if the channel number is allowed.

        :param channel: Channel number to validate
        :raises ValueError: If the channel is not valid
        """
        if int(channel) not in ALLOWED_CHANNELS:
            raise ValueError(f'Channel should be one of: {ALLOWED_CHANNELS}.')
        
    def __del__(self):
        # Toggle remote control off and close the connection
        self.__write('SYSTem:LOCal')
        self.close_connection()
        print(f"Connection to {self.name} closed.")