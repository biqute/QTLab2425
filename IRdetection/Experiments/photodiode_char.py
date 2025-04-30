import numpy as np
import h5py
import os
import time
from matplotlib import pyplot as plt

from src.instruments.Keysight_VNA import VNA
from src.instruments.Keysight33500B import Keysight33500B
from src.instruments.PicoScope import PicoScope
from src.experiment.Experiment import Experiment
from src.experiment.Callback import Callback, MakePeakGraphCallback
from tqdm import tqdm

"""
Photodiode characterization experiment.

Objectives:
1. Find the voltage at which the diode saturates. Set the signal to DC and do binary search for the saturation voltage.
2. Set square wave and find the best configuration of frequancy and duty cycle that allows to have the narrowest burst.
"""

class PhotodiodeChar(Experiment):
    def __init__(self, AWG_address, Pico_port, name="PhotodiodeChar", experiment_dir="Experiments"):
        super().__init__(name, experiment_dir)
        self.AWG_address = AWG_address
        self.Pico_port = Pico_port
        
        # Initialize instruments
        self.awg = Keysight33500B(ip_address=self.AWG_address, name="AWG")
        self.ps = PicoScope(port=self.Pico_port, name="PicoScope")
        
        self.add_instrument(self.awg)
        self.add_instrument(self.ps)
        
        # Set up the experiment parameters
        self.start_offset = -0.5  # V
        self.stop_offset = -1.1  # V
        
    def get_mean_voltage(self, channel="A"):
        """
        Read the DC voltage from the PicoScope and return the mean value (in V).
        """
        self.ps.set_channel(channel, enabled=True, coupling="DC", range=5, offset=0)
        data = self.ps.acq_block(sample_rate=1e6, post_trigger_samples=1000)
        mean_voltage = np.mean(data[channel]) / 1e3 # Convert to volts
        return mean_voltage
    
    def is_saturation(self, mean_voltage, max_voltage, tollerance=0.1):
        """
        Voltage saturation criteria.
        """
        return abs(mean_voltage - max_voltage) < tollerance # Go to lowr voltages 
        
        
    def routine(self, **kwargs):
        # 1 - Find the saturation voltage of the photodiode ------------------------
        # Set the AWG to DC mode
        self.awg.set_waveform("DC")
        self.awg.set_offset(self.stop_offset)  # Start from the minimum offset
        max_voltage = self.get_mean_voltage()
        self.voltage_history = []
        self.voltage_history.append(max_voltage)
        # Binary search for the saturation voltage
        
        self.logger.info("Starting binary search for saturation voltage")
        
        # Initialize binary search parameters
        low = self.start_offset
        high = self.stop_offset
        saturation_voltage = None
        offset_history = []
        
        # Binary search loop
        while abs(high - low) > 0.01:  # 10 mV precision
            mid = (low + high) / 2
            self.logger.info(f"Testing offset voltage: {mid:.3f} V")
            
            # Set the AWG to the mid voltage
            self.awg.set_offset(mid)
            time.sleep(0.1)  # Allow time for the voltage to stabilize
            
            # Measure the voltage
            mean_voltage = self.get_mean_voltage()
            
            # Record data
            offset_history.append(mid)
            self.voltage_history.append(mean_voltage)
            
            # Check if we've reached saturation
            if self.is_saturation(mean_voltage, max_voltage):
                self.logger.info(f"Saturation detected at {mid:.3f} V")
                saturation_voltage = mid
                high = mid  # Search for lower voltages
            else:
                self.logger.info(f"No saturation at {mid:.3f} V")
                low = mid  # Search for higher voltages
        
        # Final result
        if saturation_voltage is not None:
            self.logger.info(f"Photodiode saturation voltage found: {saturation_voltage:.3f} V")
        else:
            saturation_voltage = (low + high) / 2
            self.logger.info(f"Best approximation of saturation voltage: {saturation_voltage:.3f} V")
        
        # Save results
        results = {
            "saturation_voltage": saturation_voltage,
            "offset_history": offset_history,
            "voltage_history": self.voltage_history
        }
        self.save_data(results)
        
        # Plot results
        self._plot_saturation_results(offset_history, self.voltage_history, saturation_voltage)
        
        return saturation_voltage
    
    def _plot_saturation_results(self, offset_history, voltage_history, saturation_voltage):
        """
        Plot the results of the saturation voltage search.
        """
        plt.figure(figsize=(10, 6))
        
        # Plot all tested points
        plt.plot(offset_history, voltage_history[1:], 'o-', label='Measured Voltage')
        
        # Mark the saturation voltage
        plt.axvline(x=saturation_voltage, color='r', linestyle='--', 
                    label=f'Saturation Voltage: {saturation_voltage:.3f} V')
        
        # Reference max voltage
        plt.axhline(y=voltage_history[0], color='g', linestyle='-', 
                    label=f'Max Voltage: {voltage_history[0]:.3f} V')
        
        plt.xlabel('Offset Voltage (V)')
        plt.ylabel('Measured Voltage (V)')
        plt.title('Photodiode Saturation Voltage Search')
        plt.grid(True)
        plt.legend()
        
        # Save the plot
        plot_path = os.path.join(self.experiment_dir, f"{self.name}_saturation_curve.png")
        plt.savefig(plot_path)
        plt.close()
        self.logger.info(f"Saved saturation curve plot to {plot_path}")
