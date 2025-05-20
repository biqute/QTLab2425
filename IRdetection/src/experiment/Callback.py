import os
import shutil

class Callback:
    """
    Base class for experiment callbacks.
    
    This class defines the interface for callbacks that can be registered
    with an experiment to hook into various lifecycle events.
    
    Methods
    -------
    on_experiment_start(experiment)
        Called when an experiment starts
    on_experiment_end(experiment)
        Called when an experiment ends successfully
    on_exception(experiment, exception)
        Called when an experiment encounters an exception
    """
    def on_experiment_start(self, experiment):
        """
        Called when an experiment starts.
        
        Parameters
        ----------
        experiment : Experiment
            The experiment that is starting
        """
        pass

    def on_experiment_end(self, experiment):
        """
        Called when an experiment ends successfully.
        
        Parameters
        ----------
        experiment : Experiment
            The experiment that is ending
        """
        pass

    def on_exception(self, experiment, exception):
        """
        Called when an experiment encounters an exception.
        
        Parameters
        ----------
        experiment : Experiment
            The experiment that encountered an exception
        exception : Exception
            The exception that was raised
        """
        pass

    def on_run_override(self, experiment, run_id):
        """
        Called when a run is being restarted with override.
        
        Parameters
        ----------
        experiment : Experiment
            The experiment instance
        run_id : int
            The ID of the run to be restarted
        """
        pass


class RestartRunCallback(Callback):
    """
    Callback to restart a run when run override is set to True.
    
    Optionally cleans the run folder before restarting.
    
    Parameters
    ----------
    clean_run_folder : bool, default=True
        If True, cleans the run folder before restarting the run
    use_custom_override : bool, default=False
        If True, calls custom_run_override which must be defined in a subclass
    
    Raises
    ------
    ValueError
        If use_custom_override is True but custom_run_override method is not defined
        
    Methods
    -------
    on_run_override(experiment, run_id)
        Called when a run is being restarted with override
    """
    def __init__(self, clean_run_folder=True, use_custom_override=False):
        self.clean_run_folder = clean_run_folder
        self.use_custom_override = use_custom_override

        # Check that if the custom override is set to True the custom_run_override method is defined in the child class.
        if self.use_custom_override and not hasattr(self, 'custom_run_override'):
            raise ValueError("If use_custom_override is set to True, the custom_run_override method must be defined in the child class.")
    
    def on_run_override(self, experiment, run_id):
        """
        Restart the run if the run override is set to True.
        
        Parameters
        ----------
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
    """
    Callback to plot and display peak data from resonance measurements.
    
    Methods
    -------
    make_peak_graph(experiment, peak_name, bias_value, peak_data)
        Generate and save a plot of peak data
    """
    
    def make_peak_graph(self, experiment, peak_name, bias_value, peak_data):
        """
        Generate and save a plot of the peak data.
        
        Parameters
        ----------
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