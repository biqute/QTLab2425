from abc import ABC, abstractmethod
import traceback
import os
from typing import Dict, List, Optional, Any, TypeVar, Union
import sys

from irdetection.experiment.Logger import Logger, ExperimentConfig
from irdetection.abstract.Instrument import Instrument
from irdetection.experiment.Callback import Callback
from irdetection.experiment.Exceptions import (
    ExperimentError, InstrumentError, CallbackError,
    ExperimentRuntimeError, ExperimentSetupError, ConfigurationError
)


T = TypeVar('T', bound=Instrument)  # For instrument type hints

class Experiment(ABC):
    """
    Abstract base class for experiment implementations.
    
    This class provides a framework for running scientific experiments with
    instrument control, logging, callbacks, and error handling.
    
    Parameters
    ----------
    name : str
        Name of the experiment
    experiment_dir : str, optional
        Directory to store experiment data
    predefined_startup : bool, default=True
        Whether to run predefined startup routine
    predefined_shutdown : bool, default=True
        Whether to run predefined shutdown routine
        
    Methods
    -------
    add_instrument(instrument)
        Add an instrument to the experiment
    add_callback(callback)
        Add a callback to the experiment
    routine(**kwargs)
        Main experiment routine (to be implemented by subclass)
    run(override_last_run=False, **kwargs)
        Run the experiment
    on_experiment_start(**kwargs)
        Hook executed before the routine (can be overridden)
    on_shutdown(**kwargs)
        Hook executed after the routine (can be overridden)
    get_instrument(name, required=False)
        Get an instrument by name
    get_run_folder()
        Get the folder for the current run
    get_run_id()
        Get the ID for the current run
    trigger(hook_name, **kwargs)
        Trigger a callback hook
    """
    def __init__(self, name: str, experiment_dir: Optional[str] = None, 
                predefined_startup: bool = True, predefined_shutdown: bool = True) -> None:
        self.name = name
        self.experiment_dir = f"{experiment_dir}/{name}" if experiment_dir else f"./{name}"
        
        try:
            os.makedirs(self.experiment_dir, exist_ok=True)
        except OSError as e:
            raise ExperimentSetupError(f"Failed to create experiment directory: {e}")
        
        self.instruments: Dict[str, Instrument] = {}
        self.callbacks: List[Callback] = []
        self.status: str = "initialized"
        
        # Initialize or load the configuration
        try:
            self.config = ExperimentConfig(f"{self.experiment_dir}/{self.name}")
            self.config['parameters']['predefined_startup'] = predefined_startup
            self.config['parameters']['predefined_shutdown'] = predefined_shutdown
        except Exception as e:
            raise ConfigurationError(f"Failed to initialize configuration: {e}")

    def add_instrument(self, instrument: Instrument) -> None:
        """
        Add an instrument to the experiment.
        
        Parameters
        ----------
        instrument : Instrument
            The instrument to add
            
        Raises
        ------
        TypeError
            If the provided object is not an Instrument
        InstrumentError
            If an instrument with the same name already exists
        """
        if not isinstance(instrument, Instrument):
            raise TypeError(f"Expected Instrument object, got {type(instrument)}")
        
        if instrument.name in self.instruments:
            raise InstrumentError(f"Instrument with name '{instrument.name}' already exists")
        
        self.instruments[instrument.name] = instrument

    def add_callback(self, callback: Callback) -> None:
        """
        Add a callback to the experiment.
        
        Parameters
        ----------
        callback : Callback
            The callback object to add
            
        Raises
        ------
        TypeError
            If the provided object is not a Callback
        """
        if not isinstance(callback, Callback):
            raise TypeError(f"Expected Callback object, got {type(callback)}")
        
        self.callbacks.append(callback)

    # RUNNING THE EXPERIMENT -----------------------------------------------------
    def _predefined_startup(self) -> None:
        """
        Predefined startup routine, runs before on_experiment_start.
        
        This method can be disabled by setting the predefined_startup parameter to False.
        Initializes all registered instruments.
        
        Raises
        ------
        InstrumentError
            If any instrument fails to initialize or activate
        """
        self.logger.log_system(f'Experiment `{self.name}` started. Run ID {self.config["run_id"]}.')
        
        for instrument_name, instrument in self.instruments.items():
            try:
                instrument.initialize()
                self.logger.log_info(f'Instrument `{instrument.name}` initialized and activated.')
            except Exception as e:
                self.logger.log_error(f'Failed to initialize instrument {instrument_name}: {e}')
                raise InstrumentError(f"Failed to initialize instrument '{instrument_name}': {e}") from e

    def on_experiment_start(self, **kwargs: Any) -> None:
        """
        Initialize the experiment. Runs before the routine.
        
        Override this method to add custom initialization.
        
        Parameters
        ----------
        **kwargs : dict
            Additional arguments to customize initialization
        """
        pass

    @abstractmethod
    def routine(self, **kwargs: Any) -> Any:
        """
        Define the experiment routine.
        
        This is the main experiment implementation that must be overridden
        by concrete experiment classes.
        
        Parameters
        ----------
        **kwargs : dict
            Additional arguments for the experiment routine
            
        Returns
        -------
        Any
            The result of the experiment
        """
        pass

    def on_shutdown(self, **kwargs: Any) -> None:
        """
        Shutdown the experiment. Runs after the routine.
        
        Override this method to add custom shutdown procedures.
        
        Parameters
        ----------
        **kwargs : dict
            Additional arguments for shutdown customization
        """
        pass
    
    def _predefined_shutdown(self) -> None:
        """
        Predefined shutdown routine, runs after on_shutdown.
        
        This method can be disabled by setting the predefined_shutdown parameter to False.
        Shuts down all registered instruments.
        """
        shutdown_errors = []
        
        for instrument_name, instrument in self.instruments.items():
            try:
                instrument.shutdown()
                self.logger.log_info(f'Instrument `{instrument.name}` shut down.')
            except Exception as e:
                error_msg = f"Failed to shut down instrument '{instrument_name}': {e}"
                self.logger.log_error(error_msg)
                shutdown_errors.append((instrument_name, str(e)))
        
        self.logger.log_system(f'Experiment `{self.name}` completed. Run ID {self.config["run_id"]}.')
        
        if shutdown_errors:
            error_details = "; ".join([f"{name}: {err}" for name, err in shutdown_errors])
            self.logger.log_error(f"Errors during shutdown: {error_details}")

    def run(self, override_last_run: bool = False, **kwargs: Any) -> None:
        """
        Run the experiment.
        
        Parameters
        ----------
        override_last_run : bool, default=False
            Whether to override the last run ID
        **kwargs : dict
            Additional arguments to pass to the experiment
            
        Returns
        -------
        Any
            The result returned by the routine method
            
        Raises
        ------
        ExperimentRuntimeError
            If the experiment encounters an error during execution
        """
        self.status = "starting"
        
        try:
            if not override_last_run:
                self.config['run_id'] += 1
                self.config.save_config()
            else:
                # Run override callback
                self.trigger("on_run_override", run_id=self.config['run_id'])
            
            # Initialize the logger
            try:
                run_dir = os.path.join(self.experiment_dir, f'run-{self.config["run_id"]}')
                os.makedirs(run_dir, exist_ok=True)
                self.logger = Logger(self.experiment_dir+f'/run-{self.config["run_id"]}/run-{self.config["run_id"]}')
            except Exception as e:
                raise ExperimentSetupError(f"Failed to initialize logger: {e}")
            
            self.trigger("on_experiment_start")
            self.status = "initializing"
            
            if self.config['parameters']['predefined_startup']:
                self._predefined_startup()
    
            self.on_experiment_start(**kwargs)
            self.status = "running"
            
            result = self.routine(**kwargs)
            self.status = "shutting_down"
            
            self.on_shutdown(**kwargs)

            if self.config['parameters']['predefined_shutdown']:
                print("Predefined shutdown")
                self._predefined_shutdown()

            self.trigger("on_experiment_end")
            self.status = "completed"
            
            return result

        except Exception as e:
            self.status = "failed"
            self.logger.log_error(f"Experiment failed with error: {e}")
            
            self.trigger("on_exception", exception=e)
            
            # Perform emergency shutdown if needed
            try:
                if self.config['parameters'].get('emergency_shutdown_on_failure', True):
                    self._emergency_shutdown()
            except Exception as shutdown_error:
                self.logger.log_error(f"Emergency shutdown also failed: {shutdown_error}")
            
            # Re-raise with more context
            raise ExperimentRuntimeError(f"Experiment '{self.name}' failed: {e}") from e

    def _emergency_shutdown(self) -> None:
        """
        Perform emergency shutdown in case of critical errors.
        
        This is a more aggressive shutdown that ensures resources are released.
        Uses instrument.kill() instead of instrument.shutdown() for guaranteed cleanup.
        """
        self.logger.log_system("Performing emergency shutdown")
        for instrument_name, instrument in self.instruments.items():
            try:
                instrument.kill()  # Use kill instead of shutdown for emergency
                self.logger.log_info(f"Emergency shutdown of {instrument_name}")
            except Exception as e:
                self.logger.log_error(f"Failed to kill {instrument_name}: {e}")

    # CALLBACKS ------------------------------------------------------------------
    def trigger(self, hook_name: str, **kwargs: Any) -> None:
        """
        Trigger a callback hook.
        
        Parameters
        ----------
        hook_name : str
            Name of the hook to trigger
        **kwargs : dict
            Arguments to pass to the callback
        """
        for callback in self.callbacks:
            method = getattr(callback, hook_name, None)
            if callable(method):
                try:
                    method(self, **kwargs)
                except Exception as e:
                    error_msg = f"Callback `{hook_name}` raised an exception: {e}"
                    self.logger.log_error(error_msg)
                    # Don't stop execution for callback errors
                    traceback.print_exc()

    # INTERNAL UTILITIES ---------------------------------------------------------------
    def _get_instrument(self, name: str) -> Optional[Instrument]:
        """
        Get an instrument by its name (internal method).
        
        Parameters
        ----------
        name : str
            Name of the instrument
            
        Returns
        -------
        Optional[Instrument]
            The instrument if found, None otherwise
        """
        instrument = self.instruments.get(name, None)
        if instrument is None:
            self.logger.log_info(f"Instrument '{name}' not found")
        return instrument
    
    def get_instrument(self, name: str, required: bool = False) -> Optional[Instrument]:
        """
        Get an instrument by its name.
        
        Parameters
        ----------
        name : str
            Name of the instrument
        required : bool, default=False
            Whether the instrument is required
            
        Returns
        -------
        Optional[Instrument]
            The instrument if found, None otherwise
            
        Raises
        ------
        InstrumentError
            If the instrument is required but not found
        """
        instrument = self._get_instrument(name)
        if instrument is None and required:
            raise InstrumentError(f"Required instrument '{name}' not found")
        return instrument
    
    def get_current_run_dir(self) -> str:
        """
        Get the directory for the current run.
        
        Returns
        -------
        str
            The directory path for the current run
        """
        return os.path.join(self.experiment_dir, f'run-{self.config["run_id"]}')
    
    # PUBLIC API UTILITIES -------------------------------------------------
    
    def get_run_folder(self) -> str:
        """
        Get the folder for the current run.
        
        Returns
        -------
        str
            The folder path for the current run
        """
        return os.path.join(self.experiment_dir, f'run-{self.config["run_id"]}')
    
    def get_run_id(self) -> int:
        """
        Get the run ID for the current experiment.
        
        Returns
        -------
        int
            The run ID
        """
        return self.config['run_id']
