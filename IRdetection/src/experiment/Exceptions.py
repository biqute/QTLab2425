class ExperimentError(Exception):
    """Base exception for all experiment-related errors"""
    pass

class InstrumentError(ExperimentError):
    """Raised when there's an error related to instruments"""
    pass

class CallbackError(ExperimentError):
    """Raised when there's an error in callbacks"""
    pass

class ConfigurationError(ExperimentError):
    """Raised when there's an error in the experiment configuration"""
    pass

class ExperimentRuntimeError(ExperimentError):
    """Raised during the execution of an experiment"""
    pass

class ExperimentSetupError(ExperimentError):
    """Raised during the setup phase of an experiment"""
    pass