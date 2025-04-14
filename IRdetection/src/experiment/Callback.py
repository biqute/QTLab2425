class Callback:
    def on_experiment_start(self, experiment):
        pass

    def on_experiment_end(self, experiment):
        pass

    def on_exception(self, experiment, exception):
        pass


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