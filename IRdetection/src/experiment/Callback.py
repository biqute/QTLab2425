import os
import shutil

class Callback:
    def on_experiment_start(self, experiment):
        pass

    def on_experiment_end(self, experiment):
        pass

    def on_exception(self, experiment, exception):
        pass


class RestartRunCallback(Callback):
    """Callback to restart a run if run override is set to True."""
    def __init__(self, clean_run_folder=True, use_custom_override=False):
        """
        Initialize the RestartRunCallback.


        Parameters:
        -----------
        clean_run_folder : bool
            If True, cleans the run folder before restarting the run.
        use_custom_override : bool
            If True, uses a custom override for restarting the run. that must be defined in the child class in the custom_run_override method.
        """
        self.clean_run_folder = clean_run_folder
        self.use_custom_override = use_custom_override

        # Check that if the custom override is set to True the custom_run_override method is defined in the child class.
        if self.use_custom_override and not hasattr(self, 'custom_run_override'):
            raise ValueError("If use_custom_override is set to True, the custom_run_override method must be defined in the child class.")
    
    def on_run_override(self, experiment, run_id):
        """
        Restart the run if the run override is set to True.
        
        Parameters:
        -----------
        experiment : Experiment
            The experiment instance
        run_id : int
            The ID of the run to be restarted
        """
        if self.clean_run_folder:
            run_folder = f"{experiment.experiment_dir}/run-{run_id}"
            if os.path.exists(run_folder):
                shutil.rmtree(run_folder)
                print(f"Cleaned run folder: {run_folder}")
            else:
                print(f"Run folder does not exist: {run_folder}")
        else:
            print(f"Run folder clean skipped: {experiment.experiment_dir}/run-{run_id}")

        if self.use_custom_override:
            # Call the custom run override method defined in the child class.
            try:
                self.custom_run_override(experiment, run_id)
            except Exception as e:
                print(f"Error in custom run override: {e}")

class MakePeakGraphCallback(Callback):
    """Callback to plot and display peak data."""
    
    def make_peak_graph(self, experiment, peak_name, bias_value, peak_data):
        """
        Generate and save a plot of the peak data.
        
        Parameters:
        -----------
        experiment : Experiment
            The experiment instance
        peak_name : str
            Name for the peak/measurement
        bias_value : float
            Current bias value (voltage or current)
        peak_data : dict
            Dictionary containing frequency, real, and imag data
        """
        import matplotlib.pyplot as plt
        import numpy as np
        import os
        
        plt.figure(figsize=(10, 6))
        plt.plot(peak_data['frequency'], np.abs(peak_data['real'] + 1j * peak_data['imag']), label='Magnitude')
        plt.title(f"Peak Data with bias value of {bias_value}")
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