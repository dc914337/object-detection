import yaml
from termcolor import colored
from argparse import Namespace


class YamlConfig:
    def __init__(self, config_path):
        with open(config_path, 'r') as stream:
            self._config_dict = yaml.safe_load(stream)
            self._called_params = set()

    def __getattr__(self, name):
        try:
            val = self._config_dict[name]
            if isinstance(val, dict):
                return YamlConfigDict(val, name, self._called_params)
            self._called_params.add(name)
            return val
        except KeyError:
            raise AttributeError(f"'YamlConfig' object has no attribute '{name}'")

    def __setattr__(self, name, value):
        if name.startswith('_'):
            super().__setattr__(name, value)
        else:
            self._config_dict[name] = value

    def unused_params(self):
        def unused_helper(d, prefix=''):
            unused = []
            for k, v in d.items():
                if isinstance(v, dict):
                    unused.extend(unused_helper(v, prefix + k + '.'))
                elif k not in self._called_params:
                    unused.append(prefix + k)
            return unused
        unused = unused_helper(self._config_dict)
        print(colored("Unused Parameters:", "yellow"))
        print(yaml.dump({param: self._config_dict[param] for param in unused}, default_flow_style=False))
        return unused

    def print_config(self):
        print(yaml.dump(self._config_dict, default_flow_style=False))

    def dict(self):
        return self._config_dict


class YamlConfigDict(YamlConfig):
    def __init__(self, config_dict, prefix, called_params):
        self._config_dict = config_dict
        self._prefix = prefix
        self._called_params = called_params

    def __getattr__(self, name):
        try:
            val = self._config_dict[name]
            if isinstance(val, dict):
                return YamlConfigDict(val, self._prefix + '.' + name, self._called_params)
            self._called_params.add(self._prefix + '.' + name)
            return val
        except KeyError:
            raise AttributeError(f"'YamlConfigDict' object has no attribute '{name}'")

    def __setattr__(self, name, value):
        if name.startswith('_'):
            super().__setattr__(name, value)
        else:
            self._config_dict[name] = value




