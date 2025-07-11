import sys
import time
import os
from typing import Any

# Add the src directory to the path to import local modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from irdetection.experiment.Experiment import Experiment
from irdetection.experiment.Callback import Callback
from irdetection.abstract.Instrument import Instrument

# Create a simple mock instrument for testing
class MockInstrument(Instrument):
    def __init__(self, name: str):
        super().__init__(name)
        self.is_initialized = False
        self.is_active = False
        
    def initialize(self):
        print(f"Initializing instrument {self.name}")
        self.is_initialized = True
        
    def info(self, verbose=False):
        status = "initialized" if self.is_initialized else "not initialized"
        return f"MockInstrument '{self.name}': {status}"
        
    def _activate(self):
        print(f"Activating instrument {self.name}")
        self.is_active = True
        
    def reset(self):
        print(f"Resetting instrument {self.name}")
        
    def close_connection(self):
        print(f"Closing connection to instrument {self.name}")
        
    def shutdown(self):
        print(f"Shutting down instrument {self.name}")
        self.is_active = False
        
    def kill(self):
        print(f"Emergency killing instrument {self.name}")
        self.is_active = False
        self.is_initialized = False

class TestCallback(Callback):
    """A test callback to demonstrate the callback mechanism."""
    
    def on_experiment_start(self, experiment):
        print(f"[Callback] Experiment '{experiment.__class__.__name__}' starting...")
        
    def on_experiment_end(self, experiment):
        print(f"[Callback] Experiment '{experiment.__class__.__name__}' completed successfully!")
        
    def on_exception(self, experiment, exception):
        print(f"[Callback] Experiment '{experiment.__class__.__name__}' encountered an error: {exception}")

    def on_3rd_iteration(self, experiment):
        print(f"[Callback] Experiment '{experiment.__class__.__name__}' reached the 3rd iteration!")

class SimpleTestExperiment(Experiment):
    """A simple test experiment that demonstrates the basic functionality."""
    
    def __init__(self, name="SimpleTest", experiment_dir="Tests/test_experiment"):
        super().__init__(name, experiment_dir=experiment_dir)
        self.iteration_count = 5
        
        # Add a mock instrument
        mock_inst = MockInstrument("test_device")
        self.add_instrument(mock_inst)
        
        
    def on_experiment_start(self, **kwargs: Any) -> None:
        self.logger.log_info("Initializing simple test experiment...")
        self.logger.log_info(f"Will run {self.iteration_count} iterations")
        print(f"Starting experiment with {self.iteration_count} iterations")
        
    def routine(self, **kwargs: Any) -> dict:
        self.logger.log_system("Starting experiment routine...")
        results = {"iterations": []}
        
        for i in range(self.iteration_count):
            iteration_data = {"iteration": i+1, "timestamp": time.time()}
            print(f"Running iteration {i+1}/{self.iteration_count}")
            self.logger.log_info(f"Running iteration {i+1}/{self.iteration_count}")
            
            if i == 2:
                self.trigger("on_3rd_iteration")
                
            # Simulate some work
            time.sleep(0.5)
            
            # Store iteration result
            results["iterations"].append(iteration_data)
            
        print("Routine completed!")
        self.logger.log_system("Experiment routine completed")
        return results
        
    def on_shutdown(self, **kwargs: Any) -> None:
        self.logger.log_info("Cleaning up experiment resources...")
        print("Cleaning up experiment resources...")
        

if __name__ == "__main__":
    # Create and run the test experiment
    experiment = SimpleTestExperiment()
    
    # Add a callback
    callback = TestCallback()
    experiment.add_callback(callback)
    
    try:
        # Run the experiment
        results = experiment.run()
        print(f"Experiment completed with {len(results['iterations'])} iterations")
    except Exception as e:
        print(f"Experiment failed: {e}")
