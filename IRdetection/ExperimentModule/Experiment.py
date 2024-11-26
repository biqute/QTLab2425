from Instruments import Instrument
import json
import time

class Experiment:
    """
    ## Experiment class

    This class is used to define an experiment. An experiment is a collection of instruments and a run procedure.
    The class allows to save a detailed log of the experiment runs.
    """

    def __init__(self, name: str, experiment_folder: str):
        """
        Initialize the experiment.

        :param name: Name of the experiment
        :param experiment_folder: Root folder for the experiment. All runs logs, data and results will be saved here.
        """
        self.name = name
        self.experiment_folder = experiment_folder
        self.instruments = {}
        self.runs = []

    def add_instrument(self, instrument: Instrument) -> None:
        """
        Add an instrument to the experiment.

        :param instrument: Instrument object
        """
        self.instruments[instrument.name] = instrument

    def remove_instrument(self, instrument_name: str) -> None:
        """
        Remove an instrument from the experiment.

        :param instrument_name: Name of the instrument
        """
        if instrument_name in self.instruments:
            del self.instruments[instrument_name]
        else:
            print(f"Instrument {instrument_name} is not in the experiment.")
            pass


    def run(self, run_name: str, run_procedure: callable) -> bool:
        """
        Run the experiment.

        :param run_name: Name of the run
        :param run_procedure: Procedure to run

        :return: True if the run was successful, False otherwise
        """
        run_log = {
            "run_name": run_name,
            "run_procedure": run_procedure.__name__,
            "start_time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "end_time": None,
            "result": None, # What the run_procedure returns
            "instuments_logs": {} # All kinds of logs from the instruments. Dict with instrument name as key
        }

        result = run_procedure()

        run_log["end_time"] = time.strftime("%Y-%m-%d %H:%M:%S")
        run_log["results"] = result
        self.runs.append(run_log)
    pass