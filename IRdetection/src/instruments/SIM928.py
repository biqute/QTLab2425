import time
import serial
from src.abstract.Instrument import Instrument


class SIM928(Instrument):
    """
    Class for interfacing with the Stanford Research Systems SIM928 DC Voltage Source.
    The SIM928 is an isolated voltage source that provides ultra-clean DC voltage.
    
    Documentation: SIM928m.pdf
    """

    def __init__(self, port_name, address=1, name=None):
        """
        Initialize the SIM928 voltage source.

        Parameters:
        -----------
        port_name : str
            Serial port name (COM port on Windows, /dev/tty* on Linux)
        address : int
            Module address in the SIM900 mainframe (1-8)
        name : str, optional
            Custom name for this instrument instance
        """
        name = name if name is not None else f"SIM928_{str(port_name)}_addr{address}"
        super().__init__(name)
        
        self.port_name = port_name
        self.address = address
        self.ser = None
        self.voltage_range = (-20.0, 20.0)  # Voltage range in volts

    def initialize(self):
        """Initialize the connection to the SIM928."""
        try:
            self.ser = serial.Serial(
                port=self.port_name,
                baudrate=9600,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                timeout=1
            )
            
            if not self.ser.is_open:
                self.ser.open()
                
            # Clear any pending commands
            self.ser.reset_input_buffer()
            self.ser.reset_output_buffer()
            
            # Activate the device
            self._activate()
            
            return f"SIM928 initialized on port {self.port_name}, address {self.address}"
        except Exception as e:
            raise ConnectionError(f"Failed to initialize SIM928: {str(e)}")

    def info(self, verbose=False):
        """
        Get information about the instrument.
        
        Parameters:
        -----------
        verbose : bool, optional
            If True, return additional information
            
        Returns:
        --------
        str
            Information about the instrument
        """
        info_str = f"SIM928 DC Voltage Source on port {self.port_name}, address {self.address}"
        
        if verbose and self.ser and self.ser.is_open:
            # Get voltage
            voltage = self.get_voltage()
            # Get output state
            output_state = "ON" if self._check_output() else "OFF"
            
            info_str += f"\nCurrent voltage: {voltage} V"
            info_str += f"\nOutput state: {output_state}"
            info_str += f"\nVoltage range: {self.voltage_range[0]} to {self.voltage_range[1]} V"
            info_str += f"\nSerial settings: {self.ser}"
            
        return info_str

    def _activate(self):
        """Activate the instrument by setting it to remote mode."""
        # Clear communication by sending a termination character
        self._write("")
        
        # Select the module in the SIM900 mainframe
        self._write(f'CONN {self.address}, "esc"', expect_response=False)
        
        # Small delay to ensure command is processed
        time.sleep(0.1)

    def _write(self, command, expect_response=False):
        """
        Write a command to the instrument.
        
        Parameters:
        -----------
        command : str
            Command to send
        expect_response : bool, optional
            Whether to expect and return a response
            
        Returns:
        --------
        str or None
            Response from the instrument if expect_response is True
        """
        if not self.ser or not self.ser.is_open:
            raise ConnectionError("Serial connection is not open")
        
        # Add termination character if not already present
        if command and not command.endswith("\n"):
            command += "\n"
        
        self.ser.write(command.encode('ascii'))
        
        if expect_response:
            time.sleep(0.1)  # Give the instrument time to respond
            response = self.ser.read_all().decode('ascii').strip()
            return response
        
        return None

    def set_voltage(self, voltage):
        """
        Set the output voltage.
        
        Parameters:
        -----------
        voltage : float
            Voltage value in volts (range: -20.0 to 20.0)
            
        Raises:
        -------
        ValueError
            If the voltage is outside the allowed range
        """
        if voltage < self.voltage_range[0] or voltage > self.voltage_range[1]:
            raise ValueError(f"Voltage must be between {self.voltage_range[0]} and {self.voltage_range[1]} V")
        
        self._write(f"VOLT {voltage:.6f}")

    def get_voltage(self):
        """
        Get the currently set voltage.
        
        Returns:
        --------
        float
            Current voltage in volts
        """
        response = self._write("VOLT?", expect_response=True)
        if response:
            try:
                return float(response)
            except ValueError:
                raise ValueError(f"Invalid response from instrument: {response}")
        else:
            raise ConnectionError("No response from instrument")

    def set_output(self, state):
        """
        Turn the output on or off.
        
        Parameters:
        -----------
        state : bool
            True to turn output on, False to turn it off
        """
        cmd = "OPON" if state else "OPOF"
        self._write(cmd)

    def _check_output(self):
        """
        Check if the output is enabled.
        
        Returns:
        --------
        bool
            True if output is on, False if off
        """
        response = self._write("EXON?", expect_response=True)
        if response:
            try:
                return int(response) == 1
            except ValueError:
                raise ValueError(f"Invalid response from instrument: {response}")
        else:
            # Default to assuming output is off if no response
            return False

    def reset(self):
        """Reset the instrument to default settings."""
        # Set voltage to 0V
        self.set_voltage(0.0)
        # Turn off output
        self.set_output(False)

    def close_connection(self):
        """Close the connection to the instrument."""
        if self.ser and self.ser.is_open:
            # Exit connection to the specific module and return to mainframe control
            self._write("END")
            self._write("esc")
            self.ser.close()

    def shutdown(self):
        """Safely shut down the instrument."""
        try:
            # Set voltage to 0 and turn output off before closing
            self.set_voltage(0.0)
            self.set_output(False)
            self.close_connection()
        except Exception as e:
            print(f"Error during shutdown: {str(e)}")

    def kill(self):
        """
        Force close the connection to the instrument.
        This is a more aggressive shutdown that does not attempt to reset settings.
        """
        if self.ser and self.ser.is_open:
            self.ser.close()

    def __del__(self):
        """Destructor to ensure the connection is closed when the object is deleted."""
        try:
            self.shutdown()
        except:
            # Ignore errors during deletion
            pass