import datetime
import os
import yaml

class Logger:
    """
    Logger class for experiment activity logging.
    
    Provides methods to log different types of messages to a text file.
    
    Parameters
    ----------
    log_file : str
        Base path for the log file (without extension)
        
    Methods
    -------
    log(message)
        Log a raw message without timestamp or category
    log_error(error_message)
        Log an error message with timestamp
    log_info(info_message)
        Log an informational message with timestamp
    log_system(system_message)
        Log a system message with timestamp
    """
    def __init__(self, log_file):
        self.log_file = log_file + "_log.txt"

    def log(self, message):
        """
        Log a raw message without timestamp or category.
        
        Parameters
        ----------
        message : str
            Message to log
        """
        with open(self.log_file, 'a') as f:
            f.write(message + '\n')

    def log_error(self, error_message):
        """
        Log an error message with timestamp.
        
        Parameters
        ----------
        error_message : str
            Error message to log
        """
        with open(self.log_file, 'a') as f:
            time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(f'({time})[ERROR]: {error_message}\n')

    def log_info(self, info_message):
        """
        Log an informational message with timestamp.
        
        Parameters
        ----------
        info_message : str
            Information message to log
        """
        with open(self.log_file, 'a') as f:
            time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(f'({time})[INFO]: {info_message}\n')

    def log_system(self, system_message):
        """
        Log a system message with timestamp.
        
        Parameters
        ----------
        system_message : str
            System message to log
        """
        with open(self.log_file, 'a') as f:
            time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(f'({time})[SYSTEM]: {system_message}\n')


class ExperimentConfig:
    """
    Configuration manager for experiments.
    
    Handles loading, saving, and accessing experiment configuration from YAML files.
    
    Parameters
    ----------
    config_file : str
        Base path for the config file (without extension)
        
    Methods
    -------
    load_config()
        Load configuration from file
    save_config()
        Save configuration to file
    __getitem__(key), __setitem__(key, value)
        Dictionary-like access to configuration values
    """
    def __init__(self, config_file):
        self.config_file = config_file + ".yaml"
        self.config = {}
        if os.path.exists(self.config_file):
            self.load_config()
        else:
            self._create_default_config()

    def load_config(self):
        """
        Load configuration from file.
        
        Raises
        ------
        FileNotFoundError
            If the configuration file is not found
        """
        if os.path.exists(self.config_file):
            with open(self.config_file, 'r') as f:
                self.config = yaml.safe_load(f)
        else:
            raise FileNotFoundError(f"Configuration file {self.config_file} not found.")
        
    def save_config(self):
        """
        Save configuration to file.
        """
        with open(self.config_file, 'w') as f:
            yaml.dump(self.config, f)

    def _create_default_config(self):
        """
        Create default configuration file.
        """
        self.config = {
            'experiment_name': 'default_experiment',
            'run_id': 0,
            'parameters': {}
        }
        self.save_config()

    # Make ExperimentConfig subscriptable
    def __getitem__(self, key):
        return self.config.get(key, None)
    def __setitem__(self, key, value):
        self.config[key] = value
        self.save_config()
    def __delitem__(self, key):
        if key in self.config:
            del self.config[key]
            self.save_config()
        else:
            raise KeyError(f"Key {key} not found in configuration.")
    def __contains__(self, key):
        return key in self.config
    def __iter__(self):
        return iter(self.config.items())
    def __len__(self):
        return len(self.config)
    def __str__(self):
        return str(self.config)