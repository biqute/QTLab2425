from abc import ABC, abstractmethod

class Instrument(ABC):
    @abstractmethod
    def initialize(self):
        pass

    @abstractmethod
    def info(self, verbose=False):
        """
        Get information about the instrument.
        """
        pass

    @abstractmethod
    def __activate(self):
        """
        Activate the instrument. Put the in remote mode.
        """
        pass

    @abstractmethod
    def reset(self):
        """
        Reset the instrument.
        """
        pass