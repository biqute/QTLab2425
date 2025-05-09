import numpy as np
import h5py
import os
import time
from scipy import integrate

from src.instruments.Keysight33500B import Keysight33500B
from src.instruments.PicoScope import PicoScope
from src.experiment.Experiment import Experiment
from src.experiment.Callback import Callback, RestartRunCallback


"""
Photodiode area characterization experiment.
"""

class PhotodiodeArea(Experiment):
    def __init__(self, AWG_address, name="PhotodiodeArea", experiment_dir="Experiments"):
        super().__init__(name, experiment_dir)
        self.AWG_address = AWG_address
        
        # Initialize instruments
        self.awg = Keysight33500B(ip_address=self.AWG_address, name="AWG")
        self.ps = PicoScope(resolution="12", name="PicoScope")
        
        self.add_instrument(self.awg)
        self.add_instrument(self.ps)
        
        # Set up the experiment parameters
        self.frequency = 6e3  # Hz
        self.amplitude = 0.712  # V
        self.offset = -0.356  # V
        self.duty_cycle = 99 # %

    def save_raw_data(self, data, offset, amplitude, frequency, duty_cycle, filename=None):
        """
        Save the data to an HDF5 file with experiment metadata.
        """
        if filename is None:
            filename = os.path.join(self.get_run_folder(), "raw_data.h5")
        voltage = data['A']
        time = data['time']

        data_to_save = np.array([time, voltage])
        with h5py.File(filename, 'w') as f:
            dataset_name = f"freq-{frequency}_duty-{duty_cycle}"
            dataset = f.create_dataset(dataset_name, data=data_to_save)
            
            # Save metadata
            dataset.attrs['offset'] = offset
            dataset.attrs['amplitude'] = amplitude
            dataset.attrs['frequency'] = frequency
            dataset.attrs['duty_cycle'] = duty_cycle
        
        self.logger.log_info(f"New raw data saved to {filename}")
        
    def Integrate_Diode(self, y, time):
        sum = (np.sum(y)/len(y)) * time
        return sum

    def compute_post_trigger_samples(self, sample_rate, duty_cycle, offset=0.1):
        """
        Compute the number of post-trigger samples based on the sample rate and duty cycle.
        """
        # Calculate the period of the waveform in seconds
        period = 1 / self.frequency
        # Calculate the time for the post-trigger samples in seconds
        post_trigger_time = period * (1 - (duty_cycle / 100))
        post_trigger_time += post_trigger_time * offset  # Add the offset time
        # Calculate the number of samples for the post-trigger time
        post_trigger_samples = int(post_trigger_time * sample_rate)
        return post_trigger_samples
    
        
    def routine(self, **kwargs):
        
        # 1 - Set AWG ------------------------------------------------
        self.awg.set_waveform("SQUare")
        self.awg.set_square_waveform(self.frequency, self.amplitude, self.offset, self.duty_cycle)
        self.awg.set_output(True)
        time.sleep(0.5)  # Allow time for the voltage to stabilize
        
        # 2 - Acquire Diode response with PicoScope ------------------
        
        sample_rate = 500e6  # Hz
        pre_trigger_samples = 50
        post_trigger_samples = self.compute_post_trigger_samples(sample_rate, self.duty_cycle)
        threshold = -900  # mV
        
        self.ps.set_channel("A", enabled=True, coupling="DC", range='5V', offset=0.0)
        self.ps.set_trigger("A", threshold=threshold, direction="FALLING", delay=0)
        
        data = self.ps.acq_block(sample_rate, post_trigger_samples=post_trigger_samples, pre_trigger_samples=pre_trigger_samples, downsampling_mode='none')

        self.save_raw_data(data, self.offset, self.amplitude, self.frequency, self.duty_cycle) 
        
        Integral = self.Integrate_Diode(data['A'], (post_trigger_samples+pre_trigger_samples)/sample_rate)
        
        Integral_scipy = integrate.simpson(data['A'], dx=1/sample_rate)
        self.logger.log_info(f"Integral calculated: {Integral} Vs")
        self.logger.log_info(f"Integral calculated with scipy: {Integral_scipy} Vs")

if __name__ == "__main__":
    # Define the experiment
    experiment = PhotodiodeArea(AWG_address="192.168.3.7")
    
    # Add callbacks
    experiment.add_callback(RestartRunCallback(experiment))
    
    # Run the experiment
    experiment.run(override_last_run=True)
    