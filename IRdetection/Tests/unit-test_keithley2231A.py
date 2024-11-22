import unittest
from Keithley2231A import Keithley2231A

class TestKeithley2231A(unittest.TestCase):
    def test_set_voltage(self):
        keithley = Keithley2231A("COM5")
        keithley.reset()
        keithley.set_voltage(1.5)
        self.assertEqual(keithley.get_voltage(), 1.5)
        keithley.close_conncetion()

    def test_limit_current(self):
        keithley = Keithley2231A("COM5")
        keithley.reset()
        keithley.set_current_limit(0.05)
        self.assertEqual(keithley.get_current_limit(), 0.05)
        keithley.close_conncetion()
        
    def test_set_voltage_after_current(self):
        keithley = Keithley2231A("COM5")
        keithley.reset()
        keithley.set_current_limit(0.05)
        keithley.set_voltage(1.5)
        self.assertEqual(keithley.get_voltage(), 1.5)
        self.assertEqual(keithley.get_current_limit(), 0.05)
        keithley.close_conncetion()
        
    def test_all_channels(self):
        keithley = Keithley2231A("COM5")
        keithley.reset()
        # Test channel 1
        keithley.set_voltage(1.5, ch=1)
        keithley.set_current_limit(0.05, ch=1)
        self.assertEqual(keithley.get_voltage(ch=1), 1.5)
        self.assertEqual(keithley.get_current_limit(ch=1), 0.05)
        # Test channel 2 
        keithley.set_voltage(2.5, ch=2)
        keithley.set_current_limit(0.1, ch=2)
        self.assertEqual(keithley.get_voltage(ch=2), 2.5)
        self.assertEqual(keithley.get_current_limit(ch=2), 0.1)
        # Test channel 3
        keithley.set_voltage(3.5, ch=3)
        keithley.set_current_limit(0.15, ch=3)
        self.assertEqual(keithley.get_voltage(ch=3), 3.5)
        self.assertEqual(keithley.get_current_limit(ch=3), 0.15)

        keithley.close_conncetion()
        
    # Main test function to run tests sequentially
    def monolithic_test(self):
        print('Running test_set_voltage')
        self.test_set_voltage()
        print('Running test_limit_current')
        self.test_limit_current()
        print('Running test_set_voltage_after_current')
        self.test_set_voltage_after_current()
        print('Running test_all_channels')
        self.test_all_channels()

if __name__ == '__main__':
    suite = unittest.TestSuite()
    suite.addTest(TestKeithley2231A('monolithic_test'))
    
    runner = unittest.TextTestRunner()
    runner.run(suite)
    