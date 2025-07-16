import h5py
import numpy as np
import matplotlib.pyplot as plt
import os
import threading
import time


from irdetection.experiment.Experiment import Experiment
from irdetection.experiment.Callback import RestartRunCallback, GraphCallback
from irdetection.instruments.Keysight33500B import Keysight33500B
from irdetection.instruments.PicoScope import PicoScope
from irdetection.io.h5 import H5Manager as H5


class PulseAcquisition(Experiment):
    def __init__(self, AWG_address, name="PulseAcquisition", experiment_dir="Experiments"):
        super().__init__(name, experiment_dir)
        self.AWG_address = AWG_address
        
        # Initialize instruments
        self.awg = Keysight33500B(ip_address=self.AWG_address, name="AWG")
        self.ps = PicoScope(resolution="8", name="PicoScope")
        
        self.add_instrument(self.awg)
        self.add_instrument(self.ps)
        
        # Set up the experiment parameters
        self.amplitude = -0.712  # V
        self.offset = -0.356  # V
        self.duty_cycle = 99.91
        self.frequency = 8333.333  # Hz
        
    def _acquire_data(self) -> dict:
        """Acquire data from the PicoScope."""
        data = self.ps.acq_block(
            sample_rate=self.frequency * 1.2e5,
            post_trigger_samples=500,
            pre_trigger_samples=200,
            downsampling_mode='none'
        )
        return data

    def _acquire_with_trigger(self) -> dict:
        """Handle acquisition timing with threading."""
        # Container to store the result from the thread
        acquisition_result = {}
        acquisition_error = {}
        
        def acquisition_thread():
            try:
                data = self._acquire_data()
                acquisition_result['data'] = data
            except Exception as e:
                acquisition_error['error'] = e
        
        # Start acquisition in separate thread
        acq_thread = threading.Thread(target=acquisition_thread)
        acq_thread.start()
        
        # Small delay to ensure acquisition has started
        time.sleep(0.1)
        
        # Trigger the impulse
        self.awg.send_software_trigger()
        
        # Wait for acquisition to complete
        acq_thread.join()
        
        # Check for errors
        if 'error' in acquisition_error:
            raise acquisition_error['error']
        
        return acquisition_result['data']

    def routine(self):
        # Set up the AWG burst
        self.awg.set_software_triggered_burst_square(
                frequency=self.frequency,
                amplitude=self.amplitude,
                duty_cycle=self.duty_cycle,
                offset=self.offset
           )
        # Configure acquisition settings
        threshold_1 = -1700  # mV
        self.ps.set_channel("B", enabled=True, coupling="DC", range='5V', offset=0.0)
        self.ps.set_trigger("B", threshold=threshold_1, direction="BELOW", delay=0)
        
        # Start acquisition, trigger, and wait for data
        data = self._acquire_with_trigger()
        
        # Save the data
        manager = H5(os.path.join(self.get_run_folder(), "data.h5"))
        manager.add_dataset("data", np.array([data["B"], data["time"]]))
        manager.add_metadata(".", {"amplitude": self.amplitude, 
                                  "offset": self.offset,
                                  "duty_cycle": self.duty_cycle,
                                  "frequency": self.frequency,
                                  "trigger_threshold": threshold_1})
        
        self.trigger('XYGraph', x_data=data['time']*1e9, y_data=data['B'], title="Pulse Acquisition", xlabel="Time (ns)", ylabel="Voltage (mV)")
        
        
        
if __name__ == "__main__":
    # Create an instance of the PulseAcquisition experiment
    experiment = PulseAcquisition(AWG_address="192.168.3.7")
    experiment.add_callback(RestartRunCallback())
    experiment.add_callback(GraphCallback())
    experiment.run(override_last_run=False)



# P = 1.8e-3 W    t = 30 ns    lambda = 1550 nm
# hbar = 1.055e-34 J.s
# c = 3e8 m/s
# Nphot = P*t / (hbar*c/lambda)

P = 1.8e-3  # W
t = 30e-9  # s
lambda_val = 1550e-9  # m
h = 6.626e-34  # J.s
c = 2.98e8  # m/s

Nphot = (P * t) / (h * c / lambda_val)
print(f"Estimated number of photons = {Nphot}")

