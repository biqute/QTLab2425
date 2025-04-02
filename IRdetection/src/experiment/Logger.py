import datetime
import os
import yaml

class Logger:
    def __init__(self, log_file):
        self.log_file = log_file + "_log.txt"

    def log(self, message):
        with open(self.log_file, 'a') as f:
            f.write(message + '\n')

    def log_error(self, error_message):
        with open(self.log_file, 'a') as f:
            time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(f'({time})[ERROR]: {error_message}\n')

    def log_info(self, info_message):
        with open(self.log_file, 'a') as f:
            time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(f'({time})[INFO]: {info_message}\n')

    def log_system(self, system_message):
        with open(self.log_file, 'a') as f:
            time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(f'({time})[SYSTEM]: {system_message}\n')


class ExperimentConfig:
    def __init__(self, config_file):
        self.config_file = config_file + ".yaml"
        self.config = {}
        if os.path.exists(self.config_file):
            self.load_config()
        else:
            self._create_default_config()

    def load_config(self):
        if os.path.exists(self.config_file):
            with open(self.config_file, 'r') as f:
                self.config = yaml.safe_load(f)
        else:
            raise FileNotFoundError(f"Configuration file {self.config_file} not found.")
        
    def save_config(self):
        with open(self.config_file, 'w') as f:
            yaml.dump(self.config, f)

    def _create_default_config(self):
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