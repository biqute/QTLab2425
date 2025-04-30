import unittest
import time
import sys
import os

# Add the parent directory to the path to import modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.instruments.Keysight33500B import Keysight33500B

class TestKeysight33500B(unittest.TestCase):
    """Test class for the Keysight33500B arbitrary waveform generator."""

    # Change this to match your instrument's IP address
    IP_ADDRESS = "192.168.3.7"  # Replace with your instrument's IP address
    
    def setUp(self):
        """Set up the test fixture - create an instance of the instrument."""
        try:
            self.awg = Keysight33500B(ip_address=self.IP_ADDRESS)
            self.awg.initialize()
            # Default to channel 1 for all tests
            self.awg.set_channel(1)
            # Make sure output is off at start of test
            self.awg.set_output(False)
            print(f"Successfully connected to: {self.awg.query('*IDN?')}")
        except Exception as e:
            self.skipTest(f"Could not connect to Keysight33500B: {str(e)}")
    
    def tearDown(self):
        """Tear down the test fixture - clean up and close connection."""
        if hasattr(self, 'awg'):
            # Make sure output is off
            self.awg.set_output(False)
            self.awg.reset()
            self.awg.close_connection()
    
    def test_square_wave_basic(self):
        """Test basic square wave generation with default parameters."""
        self.awg.set_waveform("SQUare")
        self.awg.set_frequency(1000)  # 1 kHz
        self.awg.set_amplitude(1.0)   # 1 Vpp (within ±2V limit)
        self.awg.set_offset(0.0)      # No DC offset
        self.awg.set_square_duty_cycle(50)  # 50% duty cycle
        
        # Enable output
        self.awg.set_output(True)
        
        # Wait a moment to allow the waveform to stabilize
        time.sleep(1)
        
        # Query parameters to verify they were set correctly
        freq = float(self.awg.query(f"SOURce{self.awg.channel}:FREQuency?"))
        ampl = float(self.awg.query(f"SOURce{self.awg.channel}:VOLTage?"))
        offset = float(self.awg.query(f"SOURce{self.awg.channel}:VOLTage:OFFSet?"))
        duty = float(self.awg.query(f"SOURce{self.awg.channel}:FUNCtion:SQUare:DCYCle?"))
        
        # Verify parameters
        self.assertAlmostEqual(freq, 1000, delta=0.1)
        self.assertAlmostEqual(ampl, 1.0, delta=0.01)
        self.assertAlmostEqual(offset, 0.0, delta=0.01)
        self.assertAlmostEqual(duty, 50.0, delta=0.1)
        
        # Turn off output
        self.awg.set_output(False)
    
    def test_square_wave_frequency_sweep(self):
        """Test square wave generation with different frequencies."""
        frequencies = [100, 1000, 10000, 100000, 1000000]  # 100 Hz to 1 MHz
        
        self.awg.set_waveform("SQUare")
        self.awg.set_amplitude(1.5)   # 1.5 Vpp (within ±2V limit)
        self.awg.set_offset(0.0)      # No DC offset
        self.awg.set_square_duty_cycle(50)  # 50% duty cycle
        
        for freq in frequencies:
            self.awg.set_frequency(freq)
            self.awg.set_output(True)
            
            # Wait a moment to allow the waveform to stabilize
            time.sleep(0.5)
            
            # Query frequency to verify it was set correctly
            actual_freq = float(self.awg.query(f"SOURce{self.awg.channel}:FREQuency?"))
            
            # Verify frequency (allow 0.1% tolerance)
            self.assertAlmostEqual(actual_freq, freq, delta=freq*0.001)
            
            # Turn off output
            self.awg.set_output(False)
    
    def test_square_wave_duty_cycle(self):
        """Test square wave generation with different duty cycles."""
        duty_cycles = [20, 50, 80]  # 20%, 50%, 80%
        
        self.awg.set_waveform("SQUare")
        self.awg.set_frequency(5000)   # 5 kHz
        self.awg.set_amplitude(1.8)    # 1.8 Vpp (within ±2V limit)
        self.awg.set_offset(0.0)       # No DC offset
        
        for duty in duty_cycles:
            self.awg.set_square_duty_cycle(duty)
            self.awg.set_output(True)
            
            # Wait a moment to allow the waveform to stabilize
            time.sleep(0.5)
            
            # Query duty cycle to verify it was set correctly
            actual_duty = float(self.awg.query(f"SOURce{self.awg.channel}:FUNCtion:SQUare:DCYCle?"))
            
            # Verify duty cycle (allow 0.1% tolerance)
            self.assertAlmostEqual(actual_duty, duty, delta=0.1)
            
            # Turn off output
            self.awg.set_output(False)
    
    def test_square_wave_amplitude(self):
        """Test square wave generation with different amplitudes (within ±2V limit)."""
        amplitudes = [0.5, 1.0, 1.5, 2.0]  # 0.5 to 2.0 Vpp (within ±2V limit)
        
        self.awg.set_waveform("SQUare")
        self.awg.set_frequency(10000)  # 10 kHz
        self.awg.set_offset(0.0)       # No DC offset
        self.awg.set_square_duty_cycle(50)  # 50% duty cycle
        
        for ampl in amplitudes:
            self.awg.set_amplitude(ampl)
            self.awg.set_output(True)
            
            # Wait a moment to allow the waveform to stabilize
            time.sleep(0.5)
            
            # Query amplitude to verify it was set correctly
            actual_ampl = float(self.awg.query(f"SOURce{self.awg.channel}:VOLTage?"))
            
            # Verify amplitude (allow 1% tolerance)
            self.assertAlmostEqual(actual_ampl, ampl, delta=ampl*0.01)
            
            # Turn off output
            self.awg.set_output(False)
    
    def test_square_wave_combined_params(self):
        """Test square wave generation with various combinations of parameters."""
        # Test combinations of frequency, amplitude, and duty cycle
        test_params = [
            (1000, 1.0, 25),    # 1 kHz, 1.0 Vpp, 25% duty cycle
            (5000, 1.5, 50),    # 5 kHz, 1.5 Vpp, 50% duty cycle
            (10000, 2.0, 75),   # 10 kHz, 2.0 Vpp, 75% duty cycle
        ]
        
        self.awg.set_waveform("SQUare")
        self.awg.set_offset(0.0)  # No DC offset
        
        for freq, ampl, duty in test_params:
            self.awg.set_frequency(freq)
            self.awg.set_amplitude(ampl)
            self.awg.set_square_duty_cycle(duty)
            self.awg.set_output(True)
            
            # Wait a moment to allow the waveform to stabilize
            time.sleep(0.5)
            
            # Query parameters to verify they were set correctly
            actual_freq = float(self.awg.query(f"SOURce{self.awg.channel}:FREQuency?"))
            actual_ampl = float(self.awg.query(f"SOURce{self.awg.channel}:VOLTage?"))
            actual_duty = float(self.awg.query(f"SOURce{self.awg.channel}:FUNCtion:SQUare:DCYCle?"))
            
            # Verify parameters
            self.assertAlmostEqual(actual_freq, freq, delta=freq*0.001)
            self.assertAlmostEqual(actual_ampl, ampl, delta=ampl*0.01)
            self.assertAlmostEqual(actual_duty, duty, delta=0.1)
            
            # Turn off output
            self.awg.set_output(False)

if __name__ == '__main__':
    # Run the tests
    unittest.main()