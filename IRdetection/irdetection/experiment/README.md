# Experiment Package (IRdetection/irdetection/experiment)

## Introduction
The `experiment` package provides a flexible and extensible framework for running scientific experiments with instrument control, data management, logging, and callback support. It is designed to facilitate reproducible and automated laboratory workflows, allowing users to define experiments as Python classes that manage instruments, execute routines, and handle experiment lifecycle events. The package abstracts common experiment operations, such as initialization, data acquisition, error handling, and shutdown, making it easier to implement complex measurement protocols and integrate with laboratory hardware.

The core of the package is the `Experiment` abstract base class, which defines the structure and behavior of an experiment. Users can create custom experiment classes by subclassing `Experiment` and implementing the `routine` method. The package also provides callback mechanisms for extending experiment functionality, such as automated data plotting, run management, and custom event handling.

## Usage
### Basic Structure
To use the experiment package, create a new experiment class that inherits from `Experiment`. Implement the `routine` method to define the main logic of your experiment. Instruments can be added using `add_instrument`, and callbacks can be registered with `add_callback`.

#### Example: Pulse Acquisition Experiment
```python
from irdetection.experiment.Experiment import Experiment
from irdetection.experiment.Callback import RestartRunCallback, GraphCallback
from irdetection.instruments.Keysight33500B import Keysight33500B
from irdetection.instruments.PicoScope import PicoScope
from irdetection.io.h5 import H5Manager as H5
import numpy as np
import os

class PulseAcquisition(Experiment):
    def __init__(self, AWG_address, name="PulseAcquisition", experiment_dir="Experiments"):
        super().__init__(name, experiment_dir)
        self.AWG_address = AWG_address
        self.awg = Keysight33500B(ip_address=self.AWG_address, name="AWG")
        self.ps = PicoScope(resolution="8", name="PicoScope")
        self.add_instrument(self.awg)
        self.add_instrument(self.ps)
        # ... set experiment parameters ...

    def routine(self):
        # ... experiment logic ...
        data = self.ps.acq_block(...)
        manager = H5(os.path.join(self.get_run_folder(), "data.h5"))
        manager.add_dataset("data", np.array([data["B"], data["time"]]))
        self.trigger('XYGraph', x_data=data['time'], y_data=data['B'], title="Pulse Acquisition", xlabel="Time (ns)", ylabel="Voltage (mV)")

if __name__ == "__main__":
    experiment = PulseAcquisition(AWG_address="192.168.3.7")
    experiment.add_callback(RestartRunCallback())
    experiment.add_callback(GraphCallback())
    experiment.run(override_last_run=False)
```

#### Example: Automated Resonator Sweep
```python
from irdetection.experiment.Experiment import Experiment
from irdetection.experiment.Callback import MakePeakGraphCallback
from irdetection.instruments.Keysight_VNA import VNA
from irdetection.instruments.SIM928 import SIM928

class TRswipeAutoExperiment(Experiment):
    def __init__(self, ...):
        super().__init__(...)
        self.vna = VNA(...)
        self.voltage_source = SIM928(...)
        self.add_instrument(self.vna)
        self.add_instrument(self.voltage_source)
        # ... set parameters ...

    def routine(self, **kwargs):
        # ... sweep logic ...
        self.set_sweep_parameters(...)
        # ... acquire and save data ...

if __name__ == "__main__":
    experiment = TRswipeAutoExperiment()
    experiment.add_callback(MakePeakGraphCallback())
    params = { ... }
    experiment.run(override_last_run=False, **params)
```

### Key Features
- **Instrument Management:** Add and control laboratory instruments using `add_instrument`.
- **Callback System:** Register callbacks for custom actions (e.g., plotting, run management) using `add_callback`.
- **Data Management:** Use built-in methods to save data and metadata for each experiment run.
- **Logging:** Automatic logging of experiment events, errors, and system messages.
- **Run Management:** Each experiment run is stored in a dedicated folder with configuration and data files.
- **Error Handling:** Robust exception handling for experiment and instrument errors.

### Extending the Framework
- Subclass `Experiment` and implement the `routine` method for custom experiments.
- Create custom callbacks by subclassing `Callback` and overriding relevant methods.
- Integrate new instruments by implementing the required interface and adding them to your experiment.

## References
- See `Pulseacquisition.py` and `TRswipeAuto.py` in the `Experiments` folder for full example implementations.
- See `Callback.py` for available callback classes and usage.
- See `Logger.py` and `Exceptions.py` for logging and error handling details.

## Folder Structure
- `Experiment.py`: Abstract base class for experiments
- `Callback.py`: Callback classes for experiment events
- `Logger.py`: Logging and configuration management
- `Exceptions.py`: Custom exception classes

## License
This package is distributed under the terms of the LICENSE file in the repository.
