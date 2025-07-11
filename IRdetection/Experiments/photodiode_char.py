import numpy as np
import h5py
import os
import time
from matplotlib import pyplot as plt

from irdetection.instruments.Keysight_VNA import VNA
from irdetection.instruments.Keysight33500B import Keysight33500B
from irdetection.instruments.PicoScope import PicoScope
from irdetection.experiment.Experiment import Experiment
from irdetection.experiment.Callback import Callback, MakePeakGraphCallback, RestartRunCallback
from tqdm import tqdm

"""
Photodiode characterization experiment.

Objectives:
1. Find the voltage at which the diode saturates. Set the signal to DC and do binary search for the saturation voltage.
2. Set square wave and find the best configuration of frequancy and duty cycle that allows to have the narrowest burst.
"""

class PhotodiodeChar(Experiment):
    def __init__(self, AWG_address, name="PhotodiodeChar", experiment_dir="Experiments"):
        super().__init__(name, experiment_dir)
        self.AWG_address = AWG_address
        
        # Initialize instruments
        self.awg = Keysight33500B(ip_address=self.AWG_address, name="AWG")
        self.ps = PicoScope(resolution="12", name="PicoScope")
        
        self.add_instrument(self.awg)
        self.add_instrument(self.ps)
        
        # Set up the experiment parameters
        self.start_offset = -0.6  # V
        self.stop_offset = -0.8  # V
        self.sensitivity = 0.0005  # V (1 mV)
        
    def get_mean_voltage(self, channel="A"):
        """
        Read the DC voltage from the PicoScope and return the mean value (in V).
        """
        self.ps.set_channel(channel, enabled=True, coupling="DC", range='5V', offset=0.0)
        data = self.ps.acq_block(sample_rate=1e6, post_trigger_samples=1000)
        mean_voltage = np.mean(data[channel]) / 1e3 # Convert to volts
        return mean_voltage
    
    def get_sigmoid(self):
        """
        Get a sweep of points to see the sigmoid curve of the photodiode.
        """
        self.awg.set_waveform("DC")
        self.awg.set_offset(self.start_offset)
        self.awg.set_output(True)
        time.sleep(0.5)  # Allow time for the voltage to stabilize
        linspace = np.linspace(self.start_offset, self.stop_offset, 100)

        voltages = []
        for offset in tqdm(linspace, desc="Measuring sigmoid curve", ncols=100):
            self.awg.set_offset(offset)
            time.sleep(0.5)
            voltages.append(self.get_mean_voltage())
        
        return linspace, voltages
    
    def save_data(self, data, filename=None):
        """
        Save the data to an HDF5 file.
        """
        if filename is None:
            filename = os.path.join(self.get_run_folder(), f"{self.name}.h5")
        
        with h5py.File(filename, 'w') as f:
            for key, value in data.items():
                f.create_dataset(key, data=value)
        
        self.logger.log_info(f"Data saved to {filename}")
        
    def load_data(self, filename):
        """
        Load the data from an HDF5 file.
        """
        with h5py.File(filename, 'r') as f:
            data = {key: f[key][()] for key in f.keys()}
        return data
        
    def routine(self, **kwargs):
        # 1 - Find the saturation voltage of the photodiode ------------------------
        # Set the AWG to DC mode
        self.awg.set_waveform("DC")
        self.awg.set_offset(self.stop_offset)  # Start from the minimum offset
        # Activate awg output
        self.awg.set_output(True)
        time.sleep(0.5)  # Allow time for the voltage to stabilize
        high = self.get_mean_voltage()
        self.logger.log_info(f"High voltage: {high:.3f} V")
        self.awg.set_offset(self.start_offset)
        time.sleep(0.5)
        low = self.get_mean_voltage()
        self.logger.log_info(f"Low voltage: {low:.3f} V")
        
        self.voltage_history = []
        offset_history = []
        self.voltage_history.append(low)
        offset_history.append(self.start_offset)
        self.voltage_history.append(high)
        offset_history.append(self.stop_offset)
        # Binary search for the saturation voltage
        
        self.logger.log_info("Starting binary search for saturation voltage")
        
        # Initialize binary search parameters
        low_offset = self.start_offset
        high_offset = self.stop_offset
        saturation_voltage = None
        
        
        # Binary search loop
        while not np.isclose(low, high, atol=self.sensitivity): # tollerance of 0.1 mV (awg sensitivity)
            mid_offset = (low_offset + high_offset) / 2
            #print(f"Testing offset: {mid_offset} V")
            
            # Set the AWG to the mid voltage
            self.awg.set_offset(mid_offset)
            time.sleep(0.5)  # Allow time for the voltage to stabilize
            
            # Measure the voltage
            mid = self.get_mean_voltage()
            
            # Record data
            offset_history.append(mid_offset)
            self.voltage_history.append(mid)
            
            if abs(mid-low) > self.sensitivity:
                low = mid
                low_offset = mid_offset
            else:
                high = mid
                high_offset = mid_offset
                
            print(f'high: {high} V, low: {low} V, mid: {mid} V, mid_offset: {mid_offset} V')
            # print(f"high offset: {high_offset:.3f} V, low offset: {low_offset:.3f} V, mid offset: {mid_offset:.3f} V")
                
        # Save the last measured voltage
        saturation_voltage = mid
        saturation_offset = mid_offset
        
        self.logger.log_info(f"Saturation voltage: {saturation_voltage:.3f} V")
        self.logger.log_info(f"Saturation offset: {saturation_offset:.3f} V")
        
        # Save results
        results = {
            "saturation_voltage": saturation_voltage,
            "saturation_offset": saturation_offset,
            "offset_history": offset_history,
            "voltage_history": self.voltage_history
        }
        self.save_data(results)
        
        # Plot results
        self._plot_saturation_results(offset_history, self.voltage_history)
        
        # Plot the sigmoid curve
        linspace, voltages = self.get_sigmoid()
        plt.figure(figsize=(10, 6))
        plt.scatter(linspace, voltages, label='Sigmoid Curve')
        plt.axvline(x=saturation_offset, color='r', linestyle='--', label='Saturation Offset')
        plt.xlabel('Offset Voltage (V)')
        plt.ylabel('Measured Voltage (V)')
        plt.title('Photodiode Sigmoid Curve')
        plt.grid(True)
        plt.legend()
        plt.savefig(os.path.join(self.get_run_folder(), f"{self.name}_sigmoid_curve.png"))
        plt.close()
        
        return saturation_voltage
    
    def _plot_saturation_results(self, offset_history, voltage_history):
        """
        Plot the results of the saturation voltage search.
        """
        plt.figure(figsize=(10, 6))
        
        # Plot all tested points
        plt.scatter(offset_history, voltage_history, marker='o', label='Measured Voltage')
        plt.plot(offset_history, voltage_history, '-', alpha=0.5, color='gray')  # Add line connecting points
        
        plt.xlabel('Offset Voltage (V)')
        plt.ylabel('Measured Voltage (V)')
        plt.title('Photodiode Saturation Voltage Search')
        plt.grid(True)
        plt.legend()
        
        # Save the plot
        plot_path = os.path.join(self.get_run_folder(), f"{self.name}_saturation_curve.png")
        plt.savefig(plot_path)
        plt.close()
        self.logger.log_info(f"Saved saturation curve plot to {plot_path}")


if __name__ == "__main__":
    # Define the experiment
    experiment = PhotodiodeChar(AWG_address="192.168.3.7")
    
    # Add callbacks
    experiment.add_callback(RestartRunCallback(experiment))
    
    # Run the experiment
    experiment.run(override_last_run=False)
