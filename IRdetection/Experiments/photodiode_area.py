import numpy as np
import h5py
import os
import time
from scipy import integrate

from src.instruments.Keysight33500B import Keysight33500B
from src.instruments.PicoScope import PicoScope
from src.experiment.Experiment import Experiment
from src.experiment.Callback import Callback, RestartRunCallback
from tqdm import tqdm


"""
Photodiode area characterization experiment.
"""

class PhotodiodeArea(Experiment):
    def __init__(self, AWG_address, name="PhotodiodeArea", experiment_dir="Experiments"):
        super().__init__(name, experiment_dir)
        self.AWG_address = AWG_address
        
        # Initialize instruments
        self.awg = Keysight33500B(ip_address=self.AWG_address, name="AWG")
        self.ps = PicoScope(resolution="8", name="PicoScope")
        
        self.add_instrument(self.awg)
        self.add_instrument(self.ps)
        
        # Set up the experiment parameters
        self.amplitude = 0.712  # V
        self.offset = -0.356  # V

    def save_raw_data(self, data, offset, amplitude, frequency, duty_cycle, filename=None, tail=False):
        """
        Save the data to an HDF5 file with experiment metadata.
        """
        if filename is None:
            filename = os.path.join(self.get_run_folder(), "raw_data.h5")
        voltage = data['A']
        time = data['time']

        data_to_save = np.array([time, voltage])
        with h5py.File(filename, 'a') as f:
            dataset_name = f"freq-{frequency}_duty-{duty_cycle}" #if tail == False else f"tail_freq-{frequency}_duty-{duty_cycle}"
            dataset = f.create_dataset(dataset_name, data=data_to_save)
            
            # Save metadata
            dataset.attrs['offset'] = offset
            dataset.attrs['amplitude'] = amplitude
            dataset.attrs['frequency'] = frequency
            dataset.attrs['duty_cycle'] = duty_cycle
        
        self.logger.log_info(f"New raw data saved to {filename}")
        
    def save_integral_data(self, integral, derivative_integral, frequency, duty_cycle, filename=None):
        """
        Save the integration results to integals.h5 file.
        """
        if filename is None:
            filename = os.path.join(self.get_run_folder(), "integrals.h5")
        with h5py.File(filename, 'a') as f:
            dataset_name = f"freq-{frequency}_duty-{duty_cycle}"
            dataset = f.create_dataset(dataset_name, data=[integral, derivative_integral])
            
            # Save metadata
            dataset.attrs['frequency'] = frequency
            dataset.attrs['duty_cycle'] = duty_cycle
            
        self.logger.log_info(f"New integral data saved to {filename}")  
    
    def Integrate_Diode(self, y, time):
        sum = (np.sum(y)/len(y)) * time
        return sum

    def compute_post_trigger_samples(self, frequency, sample_rate, duty_cycle, offset):
        """
        Compute the number of post-trigger samples based on the sample rate and duty cycle.
        """
        # Calculate the period of the waveform in seconds
        period = 1 / frequency
        # Calculate the time for the post-trigger samples in seconds
        post_trigger_time = period * (1 - duty_cycle)
        post_trigger_time += post_trigger_time * offset  # Add the offset time
        # Calculate the number of samples for the post-trigger time
        post_trigger_samples = int(post_trigger_time * sample_rate)
        return post_trigger_samples
    
    def acquire_data(self, frequency, duty_cycle):
        
        # 1 - Set AWG ------------------------------------------------
        self.awg.set_waveform("SQUare")
        self.awg.set_square_waveform(frequency, self.amplitude, self.offset, duty_cycle*100)
        self.awg.set_output(True)
        time.sleep(0.5)  # Allow time for the voltage to stabilize
        
        threshold_1 = -900  # mV
        self.ps.set_channel("A", enabled=True, coupling="DC", range='5V', offset=0.0)
        self.ps.set_trigger("A", threshold=threshold_1, direction="FALLING", delay=0)

        # 2 - Acquire Diode response with PicoScope ------------------
        sample_rate = 1e6 * frequency  # Hz
        post_trigger_samples = self.compute_post_trigger_samples(frequency, sample_rate, duty_cycle, offset=2.5)
        pre_trigger_samples = int(0.05*post_trigger_samples)
        
        data = self.ps.acq_block(sample_rate, post_trigger_samples=post_trigger_samples, pre_trigger_samples=pre_trigger_samples, downsampling_mode='none')
        self.awg.set_output(False)
        
        self.save_raw_data(data, self.offset, self.amplitude, frequency, duty_cycle) 
        
        integral = self.Integrate_Diode(data['A'], (post_trigger_samples+pre_trigger_samples)/sample_rate)
        derivative = np.abs(np.gradient(data['A']))
        derivative_integral = self.Integrate_Diode(derivative, (post_trigger_samples+pre_trigger_samples)/sample_rate)
        
        self.logger.log_info(f"Integral calculated: {integral} Vs")
        self.logger.log_info(f"Integral calculated on the derivative: {derivative_integral} V")
        
        time.sleep(0.5)  # Allow time for the voltage to stabilize
        tail_correction = self.acquire_tail_integral(frequency)
        corrected_integral = integral - tail_correction
        self.logger.log_info(f"Corrected integral: {corrected_integral} Vs")
        
        self.save_integral_data(corrected_integral, derivative_integral, frequency, duty_cycle)
        
    def acquire_tail_integral(self, frequency):
        
        # 1 - Set AWG ------------------------------------------------
        self.awg.set_waveform("SQUare")
        self.awg.set_square_waveform(frequency, self.amplitude, self.offset, 99)
        self.awg.set_output(True)
        time.sleep(0.5)  # Allow time for the voltage to stabilize
        threshold_2 = -1400  # mV
        self.ps.set_trigger("A", threshold=threshold_2, direction="RISING", delay=0)
        
        # 2 - Acquire Diode response with PicoScope ------------------
        sample_rate = 1e6 * frequency  # Hz
        post_trigger_samples = self.compute_post_trigger_samples(frequency, sample_rate, 0.99, offset=0.5)
        
        data = self.ps.acq_block(sample_rate, post_trigger_samples=post_trigger_samples, downsampling_mode='none')

        #self.save_raw_data(data, self.offset, self.amplitude, frequency, 0.99, tail=True) 
        
        self.awg.set_output(False)
        
        return self.Integrate_Diode(data['A'], (post_trigger_samples)/sample_rate)
        
    def max_dutycycle(self, frequency):
        dc = 1 - (frequency*16e-9)
        return dc if dc < 0.9999 else 0.9999
        
    def routine(self, **kwargs):
        # constructr the sweep grid
        frequencies = np.logspace(1, 3, 20)
        max_dcs = [self.max_dutycycle(f) for f in frequencies]
        min_dc = 0.98
        max_dc = np.max(max_dcs)
        duty_cycles = np.round(np.linspace(min_dc, max_dc, 10).T, 4)

        # Calculate total number of valid combinations for progress bar
        total_iterations = sum(1 for f in frequencies for dc in duty_cycles if dc <= self.max_dutycycle(f))
        
        # Create progress bar
        with tqdm(total=total_iterations, desc="Running experiment") as pbar:
            for f in frequencies:
                for dc in duty_cycles:
                    if dc <= self.max_dutycycle(f):
                        # Acquire data for each frequency and duty cycle
                        self.logger.log_info(f"Acquiring data for frequency: {f} Hz, duty cycle: {dc}")
                        self.acquire_data(f, dc)
                        time.sleep(0.5)
                        pbar.update(1)
        

        
        
        
                        


if __name__ == "__main__":
    # Define the experiment
    experiment = PhotodiodeArea(AWG_address="192.168.3.7")
    
    # Add callbacks
    experiment.add_callback(RestartRunCallback(experiment))
    
    # Run the experiment
    experiment.run(override_last_run=False)
    