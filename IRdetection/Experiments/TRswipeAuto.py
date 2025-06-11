import numpy as np
import h5py
import os
import time
from matplotlib import pyplot as plt

from irdetection.instruments.Keysight_VNA import VNA
from irdetection.instruments.SIM928 import SIM928
from irdetection.experiment.Experiment import Experiment
from irdetection.experiment.Callback import Callback, MakePeakGraphCallback
from tqdm import tqdm

class TRswipeAutoExperiment(Experiment):
    """
    Tunable Resonator Automated Sweep Experiment
    
    This experiment automatically sweeps through a range of bias voltages using a SIM928 voltage source
    and acquires S21 I and Q response data from a tunable resonator using a VNA.
    All data is saved to an HDF5 file with improved organization.
    
    The center frequency for each sweep point is calculated based on the bias current according to:
    center_freq = -0.5*f0*((I^2/a^2) + (I^4/b^4))+f0
    where f0 is the resonance frequency at zero bias, I is the bias current,
    and a, b are model parameters.
    """
    
    def __init__(self, name="TRswipeAuto", experiment_dir="Experiments",
                 port_name="COM3", sim_address=1):
        super().__init__(name, experiment_dir=experiment_dir)
        
        # Initialize instruments
        self.vna = VNA(ip_address_string="192.168.40.10", name="VNA")
        self.voltage_source = SIM928(port_name=port_name, address=sim_address, name="SIM928")
        
        # Add instruments to the experiment
        self.add_instrument(self.vna)
        self.add_instrument(self.voltage_source)
        
        # Default parameters
        self.frequency_span = 10e6  # 10 MHz
        self.f0 = 7e9  # Base resonance frequency at zero bias (Hz)
        self.resistance = 1.997e3  # 1.997 kOhm series resistance
        self.param_a = 1e-3  # Model parameter 'a' (A)
        self.param_b = 1e-3  # Model parameter 'b' (A)
        self.voltage_min = 0.0  # V
        self.voltage_max = 5.0  # V
        self.voltage_step = 0.1  # V
        self.settling_time = 1.0  # seconds to wait after changing bias voltage

    def set_sweep_parameters(self, f0=None, span_hz=None, 
                            v_min=None, v_max=None, v_step=None,
                            settling_time=None, param_a=None, param_b=None):
        """
        Set the parameters for the voltage sweep and frequency scan.
        
        Parameters:
        -----------
        f0 : float, optional
            Base resonance frequency at zero bias (Hz)
        span_hz : float, optional
            Frequency span in Hz
        v_min : float, optional
            Minimum bias voltage in V
        v_max : float, optional
            Maximum bias voltage in V
        v_step : float, optional
            Voltage step size in V
        settling_time : float, optional
            Time to wait after setting voltage (seconds)
        param_a : float, optional
            Parameter 'a' in the frequency model (A)
        param_b : float, optional
            Parameter 'b' in the frequency model (A)
        """
        if f0 is not None:
            self.f0 = f0
        if span_hz is not None:
            self.frequency_span = span_hz
        if v_min is not None:
            self.voltage_min = v_min
        if v_max is not None:
            self.voltage_max = v_max
        if v_step is not None:
            self.voltage_step = v_step
        if settling_time is not None:
            self.settling_time = settling_time
        if param_a is not None:
            self.param_a = param_a
        if param_b is not None:
            self.param_b = param_b
            
        self.logger.log_info(f"Sweep parameters set: f0={self.f0/1e9} GHz, "
                           f"Span={self.frequency_span/1e6} MHz, "
                           f"Voltage range: {self.voltage_min}V to {self.voltage_max}V "
                           f"in {self.voltage_step}V steps, "
                           f"Model parameters: a={self.param_a}, b={self.param_b}")

    def calculate_center_frequency(self, bias_current):
        """
        Calculate the center frequency for a given bias current using the model:
        center_freq = -0.5*f0*((I^2/a^2) + (I^4/b^4))+f0
        
        Parameters:
        -----------
        bias_current : float
            Bias current in Amperes
            
        Returns:
        --------
        float
            Center frequency in Hz
        """
        term1 = (bias_current**2) / (self.param_a**2)
        term2 = (bias_current**4) / (self.param_b**4)
        center_freq = -0.5 * self.f0 * (term1 + term2) + self.f0
        # Convert to scientific notation and round to 5 decimal places
        center_freq = "{:.5e}".format(center_freq)
        center_freq = float(center_freq)  # Convert back to float
        return center_freq

    def routine(self, **kwargs):
        """
        Main experiment routine that:
        1. Initializes the voltage source
        2. Configures the VNA
        3. Sweeps through bias voltages
        4. Adjusts center frequency based on bias current
        5. Acquires and saves S21 data for each voltage point
        """
        # Extract parameters from kwargs if provided
        f0 = kwargs.get('f0', self.f0)
        span_hz = kwargs.get('span_hz', self.frequency_span)
        v_min = kwargs.get('v_min', self.voltage_min)
        v_max = kwargs.get('v_max', self.voltage_max)
        v_step = kwargs.get('v_step', self.voltage_step)
        settling_time = kwargs.get('settling_time', self.settling_time)
        param_a = kwargs.get('param_a', self.param_a)
        param_b = kwargs.get('param_b', self.param_b)
        
        # Update sweep parameters
        self.set_sweep_parameters(f0, span_hz, v_min, v_max, v_step, settling_time, param_a, param_b)
        
        # Configure VNA base settings
        self.vna.point_count = 1000
        self.vna.bandwidth = 1000  # Hz
        self.vna.avg_count = 5
        self.vna.timeout = 600e3  # ms
        self.vna.power = -30  # dBm
        
        # Common metadata for all measurements
        sweep_config = {
            "f0 (Hz)": self.f0,
            "vna_point_count": self.vna.point_count,
            "vna_bandwidth (Hz)": self.vna.bandwidth,
            "vna_avg_count": self.vna.avg_count,
            "vna_power (dBm)": self.vna.power,
            "frequency_span (Hz)": self.frequency_span,
            "attenuation (dBm)": -20,
            "cryostat_temperature (K)": 1.4,  # Approximate value
            "resistance (Ohm)": self.resistance,
            "voltage_min (V)": self.voltage_min,
            "voltage_max (V)": self.voltage_max,
            "voltage_step (V)": self.voltage_step,
            "param_a": self.param_a,
            "param_b": self.param_b,
        }
        
        # Initialize data storage
        data_dir = f"{self.experiment_dir}/run-{self.config['run_id']}/data/"
        os.makedirs(data_dir, exist_ok=True)
        h5_path = f"{data_dir}/tr_sweep_data.h5"
        
        # Initialize voltage source
        self.voltage_source.set_voltage(0.0)  # Start at 0V
        self.voltage_source.set_output(True)  # Turn on output
        
        try:
            # Calculate voltage points for the sweep
            voltage_points = np.arange(self.voltage_min, self.voltage_max + self.voltage_step/2, self.voltage_step)
            
            # Create HDF5 file and store metadata
            with h5py.File(h5_path, 'w') as h5file:
                # Store experiment configuration as attributes in the root group
                for key, value in sweep_config.items():
                    h5file.attrs[key] = value
                
                # Create a group for all sweeps
                sweeps_group = h5file.create_group("voltage_sweeps")
                
                # Loop through each voltage point
                # Create a progress bar for the voltage sweep
                
                for voltage in tqdm(voltage_points, desc="Voltage sweep progress", bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]"):
                    # Set the bias voltage
                    self.voltage_source.set_voltage(voltage)
                    self.logger.log_info(f"Set bias voltage to {voltage:.3f} V")
                    
                    # Wait for system to settle
                    time.sleep(self.settling_time)
                    
                    # Calculate bias current
                    bias_current = voltage / self.resistance  # A
                    bias_current_mA = bias_current * 1e3  # mA (for display and logging)
                    
                    # Calculate center frequency based on the bias current
                    center_frequency = self.calculate_center_frequency(bias_current_mA)
                    self.logger.log_info(f"Calculated center frequency: {center_frequency/1e9:.6f} GHz for bias current: {bias_current_mA:.3f} mA")
                    
                    # Update VNA frequency range
                    self.vna.set_freq_range(center_frequency, self.frequency_span)
                    time.sleep(15)  # Allow time for VNA to settle
                    
                    # Acquire S parameters
                    peak_data = self.vna.acq_s_parameters(param="S21")
                    
                    # Create dataset name
                    dataset_name = f"bias_{voltage:.3f}V"
                    
                    # Save data in HDF5 file
                    dataset = sweeps_group.create_dataset(
                        dataset_name,
                        data=np.array([
                            peak_data['frequency'],
                            peak_data['real'],
                            peak_data['imag']
                        ]),
                        compression="gzip"
                    )
                    
                    # Set dataset attributes
                    dataset.attrs['bias_voltage'] = voltage
                    dataset.attrs['bias_current_mA'] = bias_current_mA
                    dataset.attrs['center_frequency'] = center_frequency
                    dataset.attrs['column_names'] = ['frequency', 'real', 'imag']
                    
                    # Create and save graph
                    peak_name = f"peak_bv_{voltage:.3f}V"
                    self.trigger('make_peak_graph', peak_name=peak_name, bias_value=f"{voltage:.3f}V ({bias_current_mA:.3f} mA)", peak_data=peak_data)
                    
                    self.logger.log_info(f"Acquired data at bias voltage: {voltage:.3f}V (current: {bias_current_mA:.3f} mA)")
            
            self.logger.log_info(f"Voltage sweep completed successfully. Data saved to {h5_path}")
                
        finally:
            # Ensure voltage source is properly shut down
            self.voltage_source.set_voltage(0.0)
            self.voltage_source.set_output(False)
            self.voltage_source.shutdown()
            self.logger.log_info("Voltage source reset and shutdown properly")

if __name__ == "__main__":
    # Create the experiment
    experiment = TRswipeAutoExperiment()
    
    # Add callback for generating peak graphs
    experiment.add_callback(MakePeakGraphCallback())
    
    # Define experiment parameters
    # Values can also be passed as kwargs to the run method
    # params_peak1 = {
    #     'f0': 6234427256,       
    #     'span_hz': 7.5e6,        # 50 MHz span
    #     'v_min': 0.0,           # 0V
    #     'v_max': 8.0,           # 5V
    #     'v_step': 0.2,          # 0.2V steps
    #     'settling_time': 1.5,   # 1 second settling time
    #     'param_a': 12.6,      # 200 mA (parameter 'a' in the model)
    #     'param_b': 9.28,      # 500 mA (parameter 'b' in the model)
    # }
    
    # # Start the experiment with specified parameters
    # experiment.run(override_last_run=False, **params_peak1)

    # Second run with different parameters for peak2
    params_peak2 = {
        'f0': 5810574575,      
        'span_hz': 7.5e6,        # 50 MHz span
        'v_min': 0.0,           # 0V
        'v_max': 4.0,           # 5V
        'v_step': 0.1,          # 0.2V steps
        'settling_time': 1.5,   # 1 second settling time
        'param_a': 5.97,      # 200 mA (parameter 'a' in the model)
        'param_b': 4.5,      # 500 mA (parameter 'b' in the model)
    }

    # Start the second run with specified parameters
    experiment.run(override_last_run=False, **params_peak2)