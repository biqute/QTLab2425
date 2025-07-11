from matplotlib import pyplot as plt
import numpy as np
import h5py
import os

from irdetection.instruments.Keysight_VNA import VNA
from irdetection.experiment.Experiment import Experiment
from irdetection.experiment.Callback import Callback

class MakePeakGraph(Callback):
    def make_peak_graph(self, experiment, peak_name, bias_current, peak_data):
        """Callback to plot the peak data."""
        plt.figure(figsize=(10, 6))
        plt.plot(peak_data['frequency'], np.abs(peak_data['real'] + 1j * peak_data['imag']), label='Magnitude')
        plt.title(f"Peak Data with bias current of {bias_current} $\mu$A")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("S21 (dB)")
        plt.legend()
        plt.grid()
        # Save the figure
        fig_dir = f"{experiment.experiment_dir}/run-{experiment.config['run_id']}/figures"
        os.makedirs(fig_dir, exist_ok=True)
        fig_path = f"{fig_dir}/{peak_name}.png"
        plt.savefig(fig_path)
        plt.close()

class WaitForUserInput(Callback):
    def on_wait_input(self, experiment):
        """
        Callback to wait for user input.
        """
        new_swipe_conf = input("Please enter the bias voltage and frequency center (and span) in MHz: ")
        while not self._check_input(experiment, new_swipe_conf):
            new_swipe_conf = input("Please enter the bias voltage and frequency center (and span) in MHz: ")


    def _check_input(self, experiment, user_input):
        """
        Check if user input is correct. User input should be one of the following:
        - "<bias_voltage> <frequency_center> <frequency_span>"
        - "<bias_voltage> <frequency_center>"
        - "stop" (stop the experiment)
        It also check the range validity of the parameters.
        """
        if user_input == "stop":
            experiment.go_on = False
            return True
        try:
            if len(user_input.split()) == 2:
                bias_voltage, frequency_center = map(float, user_input.split())
                # check that bias voltage is within the range [0, 8]V
                if not (0 <= bias_voltage <= 8):
                    raise ValueError("Bias voltage must be between 0 and 8.")
                experiment.bias_voltage = bias_voltage
                experiment.center_frequency = frequency_center * 1e6
            elif len(user_input.split()) == 3:
                bias_voltage, frequency_center, frequency_span = map(float, user_input.split())
                # check that bias voltage is within the range [0, 8]V
                if not (0 <= bias_voltage <= 8):
                    raise ValueError("Bias voltage must be between 0 and 8.")
                experiment.bias_voltage = bias_voltage 
                experiment.center_frequency = frequency_center * 1e6
                # Check that frequency span is within the range [0, 80]MHz
                if not (0 <= frequency_span <= 80):
                    raise ValueError("Frequency span must be between 0 and 80 MHz.")
                experiment.frequency_span = frequency_span * 1e6 # convert to Hz
            return True
        except ValueError:
            print("Invalid input. Please enter in the format: <bias_voltage> <frequency_center> <frequency_span> \n Frequency center and span in MHz, bias voltage in V.")
            return False
        except Exception as e:
            print(f"Error: {e}")
            return False

class TRswipeExperiment(Experiment):
    def __init__(self, name="TRswipe", experiment_dir="Experiments"):
        super().__init__(name, experiment_dir=experiment_dir)
        self.vna = VNA(ip_address_string="192.168.40.10", name="VNA")
        self.add_instrument(self.vna)

        # Parameters
        self.frequency_span = 10e6 # 10 MHz
        self.center_frequency = None
        self.bias_voltage = None
        self.go_on = True
        self.resistance = 2e6 # Ohm

    def routine(self, **kwargs):
        """
        We manually change the bias current, we find the resonance frequency and we acquire the data.
        """

        # Configure VNA settings
        self.instruments["VNA"].point_count = 1000
        self.instruments["VNA"].bandwidth = 1000 # Hz
        self.instruments["VNA"].avg_count = 5
        self.instruments["VNA"].timeout = 600e3 # ms
        self.instruments["VNA"].power = -30 # dBm

        sweep_config = {
            "vna_point_count": self.instruments["VNA"].point_count,
            "vna_bandwidth (Hz)": self.instruments["VNA"].bandwidth,
            "vna_avg_count": self.instruments["VNA"].avg_count,
            "vna_power (dBm)": self.instruments["VNA"].power,
            "attenuation (dBm)": -20,
            "cryostat_temperature (K)": 1.4,
        }

        while self.go_on:
            # restore avg count to 5 for better resolution
            self.instruments["VNA"].avg_count = 5
            self.trigger('on_wait_input') # Wait for user input to set bias voltage and frequency center
            # Update configuration with current settings
            self.bias_current = (self.bias_voltage * 1e6) / self.resistance # Bias current in uA
            print(f"Bias current: {self.bias_current} uA")
            sweep_config["bias_current (uA)"] = self.bias_current
        
            # Sweep
            self.instruments["VNA"].set_freq_range(self.center_frequency, self.frequency_span)
            # Acquire S parameters
            peak_data = self.instruments["VNA"].acq_s_parameters(param="S21")
            # Update VNA config with current frequency settings
            sweep_config["vna_min_freq (Hz)"] = self.instruments["VNA"].min_freq
            sweep_config["vna_max_freq (Hz)"] = self.instruments["VNA"].max_freq
            # Save data and create graph
            pk_name = f"peak_bc_{self.bias_current}_uA"
            self.save_peak(peak_data, pk_name, sweep_config)
            self.trigger('make_peak_graph', peak_name=pk_name, bias_current=self.bias_current, peak_data=peak_data)
            self.logger.log_info(f"{pk_name} acquired. Bias current: {self.bias_current} uA, Frequency center: {self.center_frequency} Hz")
            # Set average count to 1 for easier manual search
            self.instruments["VNA"].avg_count = 1
        
        self.logger.log_info("Sweeps terminated.")

    
    
    def save_peak(self, peak_data, peak_name, sweep_config):
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
            dataset = peak_group.create_dataset(f"{peak_name}_{self.bias_current}_uA", data=data_array, compression="gzip")
            
            # Set column names
            dataset.attrs['column_names'] = ['frequency', 'real', 'imag']
            
            # Attributes for metadata
            for key, value in sweep_config.items():
                dataset.attrs[key] = value

            

if __name__ == "__main__":
    # Create and run the test experiment
    experiment = TRswipeExperiment()
    experiment.add_callback(MakePeakGraph())
    experiment.add_callback(WaitForUserInput())
    
    # Start the experiment
    experiment.run(override_last_run=False)