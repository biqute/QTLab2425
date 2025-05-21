class ExperimentError(Exception):
    """
    Base exception for all experiment-related errors.
    
    This serves as the parent class for all specialized experiment
    exceptions to enable uniform error handling.
    """
    pass

class InstrumentError(ExperimentError):
    """
    Raised when there's an error related to instruments.
    
    Examples include connection failures, initialization errors,
    and communication problems with laboratory equipment.
    """
    pass

class CallbackError(ExperimentError):
    """
    Raised when there's an error in callbacks.
    
    Occurs when a registered callback encounters an issue during execution.
    """
    pass

class ConfigurationError(ExperimentError):
    """
    Raised when there's an error in the experiment configuration.
    
    Examples include missing or invalid configuration parameters.
    """
    pass

class ExperimentRuntimeError(ExperimentError):
    """
    Raised during the execution of an experiment.
    
    Occurs when the experiment encounters an unexpected issue during routine execution.
    """
    pass

class ExperimentSetupError(ExperimentError):
    """
    Raised during the setup phase of an experiment.
    
    Examples include directory creation failures or resource allocation problems.
    """
    pass