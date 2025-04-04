from matplotlib import pyplot as plt
import numpy as np
import h5py
import os

from src.instruments.Keysight_VNA import VNA
from src.experiment.Experiment import Experiment
from src.experiment.Callback import Callback
from src.experiment.Logger import Logger

class MakePeakGraph(Callback):
    def make_peak_graph(self, experiment, peak_name, power, peak_data):
        """Callback to plot the peak data."""
        plt.figure(figsize=(10, 6))
        plt.plot(peak_data['frequency'], np.abs(peak_data['real'] + 1j * peak_data['imag']), label='Magnitude')
        plt.title(f"Peak Data: {peak_name} at {power} dBm")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("S21 (dB)")
        plt.legend()
        plt.grid()
        # Save the figure
        fig_dir = f"{experiment.experiment_dir}/run-{experiment.config['run_id']}/figures/{peak_name}"
        os.makedirs(fig_dir, exist_ok=True)
        fig_path = f"{fig_dir}/{peak_name}_{power}_dBm.png"
        plt.savefig(fig_path)
        plt.close()

class ResonatorsExperiment(Experiment):
    """A simple test experiment that demonstrates the basic functionality."""
    
    def __init__(self, name="ResonatorsExperiment", experiment_dir="Experiments"):
        super().__init__(name, experiment_dir=experiment_dir)
        self.vna = VNA(ip_address_string="192.168.40.10", name="VNA")
        self.add_instrument(self.vna)
    
    def routine(self, **kwargs):

        self.instruments["VNA"].point_count = 1000 # 1000
        # self.instruments["VNA"].bandwidth = 1000 # Hz # 1000
        self.instruments["VNA"].avg_count = 20 # 20
        self.instruments["VNA"].timeout = 600e3 # ms

        vna_config = {
            "point_count": self.instruments["VNA"].point_count,
            "bandwidth (Hz)": self.instruments["VNA"].bandwidth,
            "avg_count": self.instruments["VNA"].avg_count,
            "attenuation (dBm)": -20,
            "cryostat_temperature (mK)": 44.6538,
        }
        
        # Define peaks info (name and center frequency) and output power settings
        peak_info = [
            ("peak_1", 4.396e9),
            ("peak_2", 5.257e9),
            ("peak_3", 5.812e9),
            ("peak_4", 6.236e9)
        ]
        freq_span = 0.010e9 # 15 MHz

        max_power_output = -5 # dBm
        min_power_output = -45 # dBm
        power_step = 5 # dBm

        # Loop through power output and each peak
        for power in range(min_power_output, max_power_output + power_step, power_step):
            self.instruments["VNA"].power = power
            if power <= -35:
                self.instruments["VNA"].bandwidth = 100 # Hz # 1000
            else:
                self.instruments["VNA"].bandwidth = 1000
            if power >= -15:
                self.instruments["VNA"].point_count = 1600
               
            vna_config["power (dBm)"] = self.instruments["VNA"].power
            for peak_name, center_freq in peak_info:
                # Set frequency range using center frequency and span
                self.instruments["VNA"].set_freq_range(center_freq, freq_span)
                
                # Acquire S parameters
                peak_data = self.instruments["VNA"].acq_s_parameters(param="S21")
                
                # Update VNA config with current frequency settings
                vna_config["min_freq (Hz)"] = self.instruments["VNA"].min_freq
                vna_config["max_freq (Hz)"] = self.instruments["VNA"].max_freq
                
                # Save data and create graph
                pk_name = f"{peak_name}_{power}_dBm"
                self.save_peak(peak_data, peak_name, power, vna_config)
                self.trigger('make_peak_graph', peak_name=peak_name, power=power, peak_data=peak_data)
                self.logger.log_info(f"{pk_name} acquired")

    def save_peak(self, peak_data, peak_name, power, vna_config):
        """Save the peak data to an HDF5 file."""
        data_dir = f"{self.experiment_dir}/run-{self.config['run_id']}/data/"
        os.makedirs(data_dir, exist_ok=True)
        data_path = f"{data_dir}/peaks_data.h5"
        
        # Prepare data array
        data_array = np.array([peak_data['frequency'], peak_data['real'], peak_data['imag']])
        with h5py.File(data_path, "a") as f:
            # Create a directory named after the peak name in the h5 file
            if peak_name not in f:
                peak_group = f.create_group(peak_name)
            else:
                peak_group = f[peak_name]
            # Create a dataset within the peak group
            dataset = peak_group.create_dataset(f"{peak_name}_{power}_dBm", data=data_array, compression="gzip")
            
            # Set column names
            dataset.attrs['column_names'] = ['frequency', 'real', 'imag']
            
            # Attributes for metadata
            for key, value in vna_config.items():
                dataset.attrs[key] = value


if __name__ == "__main__":
    # Create and run the test experiment
    experiment = ResonatorsExperiment()
    # Add callbacks
    callback = MakePeakGraph()
    experiment.add_callback(callback)
    try:
        # Run the experiment
        experiment.run(override_last_run=False)
    except Exception as e:
        print(f"An error occurred: {e}")

