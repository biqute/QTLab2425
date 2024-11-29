from abc import ABC, abstractmethod

class BaseInstrument(ABC):
    """
    Abstract base class for all instruments.
    Defines the standard interface for all instruments.
    """

    @abstractmethod
    def connect(self) -> None: # or maybe better to return bool?
        """
        Establish a connection to the instrument.
        """
        pass

    @abstractmethod
    def disconnect(self) -> None:
        """
        Disconnect the instrument safely.
        """
        pass

    @abstractmethod
    def _reset(self) -> None:
        """
        Reset the instrument to its default state.
        """
        pass

    @abstractmethod
    def get_configs(self) -> dict:
        """
        Get the current configuration of the instrument.
        Configurations are instrument-specific.
        There is not a set_config method because each instrument handles configurations differently.
        The main purpose of this method is to provide a way to save the current configuration of the instrument and log it.
        """
        pass
