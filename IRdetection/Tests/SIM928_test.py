import sys
import os
import unittest
from unittest.mock import patch, MagicMock, call
import serial

# Add parent directory to path to import the SIM928 module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.instruments.SIM928 import SIM928

class TestSIM928(unittest.TestCase):
    """Test cases for the SIM928 voltage source instrument class."""

    @patch('serial.Serial')
    def setUp(self, mock_serial):
        """Set up a SIM928 instance with mocked serial connection for testing."""
        self.mock_serial = mock_serial.return_value
        self.mock_serial.is_open = True
        self.mock_serial.read_all.return_value = b'0.0\r\n'  # Default response for voltage queries
        
        # Create the SIM928 instance with mock serial
        self.sim928 = SIM928('COM9', address=1, name='TestSIM928')
        self.sim928.initialize()

    def tearDown(self):
        """Clean up after tests."""
        if hasattr(self, 'sim928'):
            self.sim928 = None

    def test_initialization(self):
        """Test that the SIM928 initializes with the correct parameters."""
        self.assertEqual(self.sim928.name, 'TestSIM928')
        self.assertEqual(self.sim928.port_name, 'COM9')
        self.assertEqual(self.sim928.address, 1)
        self.assertEqual(self.sim928.voltage_range, (-20.0, 20.0))
        
        # Check that serial was initialized properly
        self.mock_serial.reset_input_buffer.assert_called_once()
        self.mock_serial.reset_output_buffer.assert_called_once()
        
        # Check that the instrument was activated (CONN command sent)
        self.mock_serial.write.assert_called_with(b"CONN 1\n")

    def test_set_voltage(self):
        """Test setting the voltage."""
        self.sim928.set_voltage(5.0)
        self.mock_serial.write.assert_called_with(b'VOLT 5.000000\n')
        
        # Test with negative voltage
        self.sim928.set_voltage(-10.0)
        self.mock_serial.write.assert_called_with(b'VOLT -10.000000\n')
        
        # Test with voltage at boundary
        self.sim928.set_voltage(20.0)
        self.mock_serial.write.assert_called_with(b'VOLT 20.000000\n')
        
    def test_voltage_bounds(self):
        """Test that voltage bounds are enforced."""
        # Test voltage too high
        with self.assertRaises(ValueError):
            self.sim928.set_voltage(30.0)
            
        # Test voltage too low
        with self.assertRaises(ValueError):
            self.sim928.set_voltage(-25.0)

    def test_get_voltage(self):
        """Test getting the current voltage."""
        self.mock_serial.read_all.return_value = b'5.43\r\n'
        voltage = self.sim928.get_voltage()
        self.assertEqual(voltage, 5.43)
        self.mock_serial.write.assert_called_with(b'VOLT?\n')

    def test_set_output(self):
        """Test turning output on and off."""
        # Test turning output on
        self.sim928.set_output(True)
        self.mock_serial.write.assert_called_with(b'OPON\n')
        
        # Test turning output off
        self.sim928.set_output(False)
        self.mock_serial.write.assert_called_with(b'OPOF\n')

    def test_check_output(self):
        """Test checking if output is enabled."""
        # Test output is on
        self.mock_serial.read_all.return_value = b'1\r\n'
        output_on = self.sim928._check_output()
        self.assertTrue(output_on)
        self.mock_serial.write.assert_called_with(b'EXON?\n')
        
        # Test output is off
        self.mock_serial.read_all.return_value = b'0\r\n'
        output_off = self.sim928._check_output()
        self.assertFalse(output_off)
        self.mock_serial.write.assert_called_with(b'EXON?\n')

    def test_reset(self):
        """Test resetting the instrument."""
        self.sim928.reset()
        
        # Check that voltage is set to 0 and output is turned off
        expected_calls = [
            call(b'VOLT 0.000000\n'),
            call(b'OPOF\n')
        ]
        self.mock_serial.write.assert_has_calls(expected_calls, any_order=False)

    def test_close_connection(self):
        """Test closing the connection."""
        self.sim928.close_connection()
        
        # Check that END command is sent and serial connection is closed
        self.mock_serial.write.assert_called_with(b'END\n')
        self.mock_serial.close.assert_called_once()

    def test_shutdown(self):
        """Test safely shutting down the instrument."""
        self.sim928.shutdown()
        
        # Verify proper shutdown sequence: set voltage to 0, turn off output, close connection
        expected_calls = [
            call(b'VOLT 0.000000\n'),
            call(b'OPOF\n'),
            call(b'END\n')
        ]
        self.mock_serial.write.assert_has_calls(expected_calls, any_order=False)
        self.mock_serial.close.assert_called_once()

    def test_info_basic(self):
        """Test the basic info method."""
        info = self.sim928.info(verbose=False)
        self.assertIn('SIM928 DC Voltage Source', info)
        self.assertIn('COM9', info)
        self.assertIn('address 1', info)

    @patch.object(SIM928, 'get_voltage')
    @patch.object(SIM928, '_check_output')
    def test_info_verbose(self, mock_check_output, mock_get_voltage):
        """Test the verbose info method."""
        # Set up mock return values
        mock_get_voltage.return_value = 5.0
        mock_check_output.return_value = True
        
        info = self.sim928.info(verbose=True)
        
        # Check that verbose info includes voltage and output state
        self.assertIn('Current voltage: 5.0 V', info)
        self.assertIn('Output state: ON', info)
        self.assertIn('Voltage range', info)
        
        # Verify methods were called
        mock_get_voltage.assert_called_once()
        mock_check_output.assert_called_once()

    def test_write_with_response(self):
        """Test sending a command and getting a response."""
        self.mock_serial.read_all.return_value = b'TEST_RESPONSE\r\n'
        response = self.sim928._write('TEST_COMMAND', expect_response=True)
        
        self.assertEqual(response, 'TEST_RESPONSE')
        self.mock_serial.write.assert_called_with(b'TEST_COMMAND\n')

    def test_write_no_response(self):
        """Test sending a command without expecting a response."""
        response = self.sim928._write('TEST_COMMAND', expect_response=False)
        
        self.assertIsNone(response)
        self.mock_serial.write.assert_called_with(b'TEST_COMMAND\n')

    @patch('serial.Serial', side_effect=Exception("Connection error"))
    def test_initialize_error(self, mock_serial):
        """Test handling of initialization errors."""
        sim = SIM928('COM9')
        with self.assertRaises(ConnectionError):
            sim.initialize()

if __name__ == '__main__':
    unittest.main()