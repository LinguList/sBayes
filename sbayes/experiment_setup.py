#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" Setup of the Experiment"""
import json
import logging
import os
import warnings
try:
    import importlib.resources as pkg_resources     # PYTHON >= 3.7
except ImportError:
    import importlib_resources as pkg_resources     # PYTHON < 3.7

from sbayes.util import set_experiment_name
from sbayes import config

REQUIRED = '<REQUIRED>'
DEFAULT_CONFIG = json.loads(pkg_resources.read_text(config, 'default_config.json'))
DEFAULT_CONFIG_SIMULATION = json.loads(pkg_resources.read_text(config, 'default_config_simulation.json'))


class Experiment:
    def __init__(self, experiment_name="default", config_file=None, log=False):

        # Naming and shaming
        if experiment_name == "default":
            self.experiment_name = set_experiment_name()
        else:
            self.experiment_name = experiment_name

        self.config_file = None
        self.config = {}
        self.base_directory = None
        self.path_results = None

        if config_file is not None:
            self.load_config(config_file)

        if log:
            self.log_experiment()

    def load_config(self, config_file, custom_settings=None):

        # Get parameters from config_file
        self.base_directory, self.config_file = self.decompose_config_path(config_file)

        # Read the user specified config file
        with open(self.config_file, 'r') as f:
            self.config = json.load(f)

        # Load defaults
        set_defaults(self.config, DEFAULT_CONFIG)
        if 'simulation' in self.config:
            self.config['data'].pop("FEATURES")
            self.config['data'].pop("FEATURE_STATES")
            set_defaults(self.config['simulation'], DEFAULT_CONFIG_SIMULATION)

        if custom_settings is not None:
            update_recursive(self.config, custom_settings)

        # Verify config
        self.verify_config()

        # Set results path
        self.path_results = '{path}/{experiment}/'.format(
            path=self.config['results']['RESULTS_PATH'],
            experiment=self.experiment_name
        )

        # Compile relative paths, to be relative to config file

        self.path_results = self.fix_relative_path(self.path_results)

        if not os.path.exists(self.path_results):
            os.makedirs(self.path_results)

    @staticmethod
    def decompose_config_path(config_path):
        config_path = config_path.strip()
        if os.path.isabs(config_path):
            abs_config_path = config_path
        else:
            abs_config_path = os.path.abspath(config_path)

        base_directory = os.path.dirname(abs_config_path)

        return base_directory, abs_config_path.replace("\\", "/")

    def fix_relative_path(self, path):
        """Make sure that the provided path is either absolute or relative to
        the config file directory.

        Args:
            path (str): The original path (absolute or relative).

        Returns:
            str: The fixed path.
         """
        path = path.strip()
        if os.path.isabs(path):
            return path
        else:
            return os.path.join(self.base_directory, path).replace("\\", "/")

    def is_simulation(self):
        return 'simulation' in self.config

    def verify_priors(self, priors_cfg: dict, inheritance: bool):
        # Define which priors are required
        required_priors = ['geo', 'weights', 'universal', 'contact']
        if inheritance:
            required_priors.append('inheritance')
        else:
            if 'inheritance' in priors_cfg:

                warnings.warn("Inheritance is not considered in the model. PRIOR for inheritance"
                              "defined in " + self.config_file + " will not be used.")
                priors_cfg['inheritance'] = None

        # Check presence and validity of each required prior
        for key in required_priors:
            if key not in priors_cfg:
                NameError(f"Prior \'{key}\' is not defined in {self.config_file}.")

            prior = priors_cfg[key]
            if 'type' not in prior:
                raise NameError(f"type for prior \'{key}\' is not defined in {self.config_file}.")
            if prior['type'] == 'counts':
                if 'file_type' not in prior:
                    raise NameError(f"counts file for prior \'{key}\' is not defined in {self.config_file}.")
                if 'scale_counts' not in prior:
                    prior['scale_counts'] = None

                if key == 'universal':
                    if 'file' not in prior:
                        raise NameError(f"counts file for prior \'{key}\' is not "
                                        f"defined in {self.config_file}.")
                    prior['file'] = self.fix_relative_path(prior['file'])
                elif key == 'inheritance':
                    if 'files' not in prior:
                        raise NameError(f"counts files for prior \'{key}\' is not defined in {self.config_file}.")
                    for fam in prior['files']:
                        prior['files'][fam] = self.fix_relative_path(prior['files'][fam])

    def verify_config(self):
        for k, v in iter_items_recursive(self.config):
            if v == REQUIRED:
                raise NameError(f'{k} is not defined {self.config_file}')

        # Are priors complete and consistent?
        self.verify_priors(self.config['model']['PRIOR'],
                           inheritance=self.config['model']['INHERITANCE'])

        # SIMULATION
        if self.is_simulation():
            self.config['simulation']['SITES'] = self.fix_relative_path(self.config['simulation']['SITES'])
            if type(self.config['simulation']['AREA']) is list:
                self.config['simulation']['AREA'] = tuple(self.config['simulation']['AREA'])

            ### NN: replaced by default_config.json
            # if 'SITES' not in self.config['simulation']:
            #     raise NameError("SITES is not defined in " + self.config_file)
            # else:
            #     self.config['simulation']['SITES'] = self.fix_relative_path(self.config['simulation']['SITES'])
            # # Does the simulation part of the config file provide all required simulation parameters?
            # # Simulate inheritance?
            # if 'INHERITANCE' not in self.config['simulation']:
            #     raise NameError("INHERITANCE is not defined in " + self.config_file)
            # Strength of the contact signal
            # if 'E_CONTACT' not in self.config['simulation']:
            #     raise NameError("E_CONTACT is not defined in " + self.config_file)
            # if 'I_CONTACT' not in self.config['simulation']:
            #     raise NameError("I_CONTACT is not defined in " + self.config_file)
            # Area for which contact is simulated
            # if 'AREA' not in self.config['simulation']:
            #     raise NameError("AREA is not defined in " + self.config_file)
            # if type(self.config['simulation']['AREA']) is list:
            #     self.config['simulation']['AREA'] = tuple(self.config['simulation']['AREA'])
            #
            # # Which optional parameters are provided in the config file?
            # # Number of simulated features and states
            # if 'N_FEATURES' not in self.config['simulation']:
            #     self.config['simulation']['N_FEATURES'] = 35
            # if 'P_N_STATES' not in self.config['simulation']:
            #     self.config['simulation']['P_N_STATES'] = {"2": 0.4, "3": 0.3, "4": 0.3}
            #
            # # Strength of universal pressure
            # if 'I_UNIVERSAL' not in self.config['simulation']:
            #     self.config['simulation']['I_UNIVERSAL'] = 1.0
            # if 'E_UNIVERSAL' not in self.config['simulation']:
            # #     self.config['simulation']['E_UNIVERSAL'] = 1.0
            #
            # # Use only a subset of the data for simulation?
            # if 'SUBSET' not in self.config['simulation']:
            #     self.config['simulation']['SUBSET'] = False
            #
            # # Strength of inheritance
            # if self.config['simulation']['INHERITANCE']:
            #     if 'I_INHERITANCE' not in self.config['simulation']:
            #         self.config['simulation']['I_INHERITANCE'] = 0.2
            #     if 'E_INHERITANCE' not in self.config['simulation']:
            #         self.config['simulation']['E_INHERITANCE'] = 2
            # else:
            #     self.config['simulation']['I_INHERITANCE'] = None
            #     self.config['simulation']['E_INHERITANCE'] = None

        ### NN: Replaced by default_config.json
        # # Model
        # # Does the config file define a model?
        # if 'model' not in self.config:
        #     raise NameError("Information about the model was not found in"
        #                     + self.config_file + ". Include model as a key.")
        # # Number of areas
        # if 'N_AREAS' not in self.config['model']:
        #     raise NameError("N_AREAS is not defined in " + self.config_file)
        # # Consider inheritance as a confounder?
        # if 'INHERITANCE' not in self.config['model']:
        #     raise NameError("INHERITANCE is not defined in " + self.config_file)
        # # Priors
        # if 'PRIOR' not in self.config['model']:
        #     raise NameError("PRIOR is not defined in " + self.config_file)
        # if 'geo' not in self.config['model']['PRIOR']:
        #     raise NameError("geo PRIOR is not defined in " + self.config_file)
        # if 'weights' not in self.config['model']['PRIOR']:
        #     raise NameError("PRIOR for weights is not defined in " + self.config_file)
        # if 'contact' not in self.config['model']['PRIOR']:
        #     raise NameError("PRIOR for contact is not defined in " + self.config_file)
        # if 'universal' not in self.config['model']['PRIOR']:
        #     raise NameError("PRIOR for universal pressure is not defined in " + self.config_file)
        # if 'type' not in self.config['model']['PRIOR']['universal']:
        #     raise NameError("type for universal prior is not defined in " + self.config_file)
        # if self.config['model']['PRIOR']['universal']['type'] == 'counts':
        #     self.config['model']['PRIOR']['universal']['file'] = \
        #         self.fix_relative_path(self.config['model']['PRIOR']['universal']['file'])
        # if self.config['model']['PRIOR']['inheritance']:
        #     for fam in self.config['model']['PRIOR']['inheritance']:
        #         self.config['model']['PRIOR']['inheritance'][fam] = \
        #             self.fix_relative_path(self.config['model']['PRIOR']['inheritance'][fam])
        # if 'scale_counts' not in self.config['model']['PRIOR']:
        #     self.config['model']['PRIOR']['scale_counts'] = None
        # if self.config['model']['INHERITANCE']:
        #     if 'inheritance' not in self.config['model']['PRIOR']:
        #         raise NameError("PRIOR for inheritance (families) is not defined in " + self.config_file)
        # else:
        #     if 'inheritance' in self.config['model']['PRIOR']:
        #         warnings.warn("Inheritance is not considered in the model. PRIOR for inheritance"
        #                       "defined in " + self.config_file + "will not be used.")
        #         self.config['model']['PRIOR']['inheritance'] = None

        if 'NEIGHBOR_DIST' not in self.config['model']:
            self.config['model']['NEIGHBOR_DIST'] = "euclidean"
        if 'LAMBDA_GEO_PRIOR' not in self.config['model']:
            self.config['model']['LAMBDA_GEO_PRIOR'] = "auto_tune"
        if 'SAMPLE_FROM_PRIOR' not in self.config['model']:
            self.config['model']['SAMPLE_FROM_PRIOR'] = False

        # Minimum, maximum size of areas
        if 'MIN_M' not in self.config['model']:
            self.config['model']['MIN_M'] = 3
        if 'MAX_M' not in self.config['model']:
            self.config['model']['MAX_M'] = 50

        # MCMC
        # Is there an mcmc part in the config file?
        if 'mcmc' not in self.config:
            raise NameError("Information about the MCMC setup was not found in"
                            + self.config_file + ". Include mcmc as a key.")

        ### NN: Moved to default_config.json
        # # Which optional parameters are provided in the config file?
        # # Number of steps
        # if 'N_STEPS' not in self.config['mcmc']:
        #     self.config['mcmc']['N_STEPS'] = 30000
        # # Number of samples
        # if 'N_SAMPLES' not in self.config['mcmc']:
        #     self.config['mcmc']['N_SAMPLES'] = 1000
        # # Number of runs
        # if 'N_RUNS' not in self.config['mcmc']:
        #     self.config['mcmc']['N_RUNS'] = 1
        # if 'P_GROW_CONNECTED' not in self.config['mcmc']:
        #     self.config['mcmc']['P_GROW_CONNECTED'] = 0.85
        # if 'M_INITIAL' not in self.config['mcmc']:
        #     self.config['mcmc']['M_INITIAL'] = 5

        # todo: activate for MC3
        # Number of parallel Markov chains
        # if 'N_CHAINS' not in self.config['mcmc']:
        #     self.config['mcmc']['N_CHAINS'] = 5
        # # Steps between two attempted chain swaps
        # if 'SWAP_PERIOD' not in self.config['mcmc']:
        #     self.config['mcmc']['SWAP_PERIOD'] = 1000
        # # Number of attempted chain swaps
        # if 'N_SWAPS' not in self.config['mcmc']:
        #     self.config['mcmc']['N_SWAPS'] = 3
        if 'MC3' not in self.config['mcmc']:
            self.config['mcmc']['N_CHAINS'] = 1
        else:
            # todo: activate for MC3
            pass

        # Tracer does not like unevenly spaced samples
        spacing = self.config['mcmc']['N_STEPS'] % self.config['mcmc']['N_SAMPLES']

        if spacing != 0.:
            raise ValueError("Non-consistent spacing between samples. Set N_STEPS to be a multiple of N_SAMPLES. ")

        ### NN: Moved to default_config.json
        # # Precision of the proposal distribution
        # # PROPOSAL_PRECISION is in config --> check for consistency
        # if 'PROPOSAL_PRECISION' in self.config['mcmc']:
        #     if 'weights' not in self.config['mcmc']['PROPOSAL_PRECISION']:
        #         self.config['mcmc']['PROPOSAL_PRECISION']['weights'] = 30
        #     if 'universal' not in self.config['mcmc']['PROPOSAL_PRECISION']:
        #         self.config['mcmc']['PROPOSAL_PRECISION']['universal'] = 30
        #     if 'contact' not in self.config['mcmc']['PROPOSAL_PRECISION']:
        #         self.config['mcmc']['PROPOSAL_PRECISION']['contact'] = 30
        #     if self.config['model']['INHERITANCE']:
        #         if 'inheritance' not in self.config['mcmc']['PROPOSAL_PRECISION']:
        #             self.config['mcmc']['PROPOSAL_PRECISION']['inheritance'] = 30
        #     else:
        #         self.config['mcmc']['PROPOSAL_PRECISION']['inheritance'] = None
        #
        # # PROPOSAL_PRECISION is not in config --> use default values
        # else:
        #     if not self.config['model']['INHERITANCE']:
        #         self.config['mcmc']['PROPOSAL_PRECISION'] = {"weights": 15,
        #                                                      "universal": 40,
        #                                                      "contact": 20,
        #                                                      "inheritance": None}
        #     else:
        #         self.config['model']['PROPOSAL_PRECISION'] = {"weights": 15,
        #                                                       "universal": 40,
        #                                                       "contact": 20,
        #                                                       "inheritance": 20}

        ### NN: Moved to default_config.json
        # # Steps per operator
        # # STEPS is in config --> check for consistency
        # steps_complete = True
        # if 'STEPS' in self.config['mcmc']:
        #     if 'area' not in self.config['mcmc']['STEPS']:
        #         warnings.warn("STEPS for area are not defined in the config file, default STEPS will be used instead.")
        #         steps_complete = False
        #
        #     if 'weights' not in self.config['mcmc']['STEPS']:
        #         warnings.warn("STEPS for weights are not defined in the config file, "
        #                       "default STEPS will be used instead.")
        #         steps_complete = False
        #
        #     if 'universal' not in self.config['mcmc']['STEPS']:
        #         warnings.warn("STEPS for universal are not defined in the config file, "
        #                       "default STEPS will be used instead.")
        #         steps_complete = False
        #
        #     if 'universal' not in self.config['mcmc']['STEPS']:
        #         warnings.warn("STEPS for contact are not defined in the config file, "
        #                       "default STEPS will be used instead.")
        #         steps_complete = False
        #
        #     if self.config['model']['INHERITANCE']:
        #         if 'inheritance' not in self.config['mcmc']['STEPS']:
        #             warnings.warn("Inheritance is modelled in the MCMC, but STEPS for inheritance are not defined "
        #                           "in the config file, default STEPS will be used instead.")
        #             steps_complete = False
        #
        #     else:
        #         if 'inheritance' not in self.config['mcmc']['STEPS']:
        #             self.config['mcmc']['STEPS']['inheritance'] = 0.0
        #         elif self.config['mcmc']['STEPS']['inheritance'] > 0.0:
        #             warnings.warn("Inheritance is not modelled in the MCMC, but STEPS for inheritance are defined"
        #                           "in the config file, default STEPS will be used instead.")
        #             steps_complete = False
        #
        # ### NN: Moved to default_congig.json
        # # # STEPS is not in config --> use default
        # # if 'STEPS' not in self.config['mcmc'] or not steps_complete:
        # #     if self.config['model']['INHERITANCE']:
        # #         self.config['mcmc']['STEPS'] = {"area": 0.05,
        # #                                         "weights": 0.4,
        # #                                         "universal": 0.05,
        # #                                         "contact": 0.4,
        # #                                         "inheritance": 0.1}
        # #     else:
        # #         self.config['mcmc']['STEPS'] = {"area": 0.05,
        # #                                          "weights": 0.45,
        # #                                          "universal": 0.05,
        # #                                          "contact": 0.45,
        # #                                          "inheritance": 0.00}

        # Do not use inheritance steps if inheritance is disabled
        if not self.config['model']['INHERITANCE']:
            if self.config['mcmc']['STEPS'].get('inheritance', 0) == 0:
                logging.warning('STEPS for inheritance was set to 0, because ´inheritance´ is disabled.')
            self.config['mcmc']['STEPS']['inheritance'] = 0.0

        # Normalize weights
        weights_sum = sum(self.config['mcmc']['STEPS'].values())
        for operator, weight in self.config['mcmc']['STEPS'].items():
            self.config['mcmc']['STEPS'][operator] = weight / weights_sum

        if 'results' in self.config:
            if 'RESULTS_PATH' not in self.config['results']:
                self.config['results']['RESULTS_PATH'] = "results"
            if 'FILE_INFO' not in self.config['results']:
                self.config['results']['FILE_INFO'] = "n"

        else:
            self.config['results'] = {}
            self.config['results']['RESULTS_PATH'] = "results"
            self.config['results']['FILE_INFO'] = "n"

        # Data
        if 'data' not in self.config:
            raise NameError("Provide file paths to data.")
        else:
            if 'simulated' not in self.config['data']:
                self.config['data']['simulated'] = False

            if not self.config['data']['simulated']:
                if not self.config['data']['FEATURES']:
                    raise NameError("FEATURES is empty. Provide file paths to features file (e.g. features.csv)")
                else:
                    self.config['data']['FEATURES'] = self.fix_relative_path(self.config['data']['FEATURES'])
                if not self.config['data']['FEATURE_STATES']:
                    raise NameError("FEATURE_STATES is empty. Provide file paths to feature_states file (e.g. feature_states.csv)")
                else:
                    self.config['data']['FEATURE_STATES'] = self.fix_relative_path(self.config['data']['FEATURE_STATES'])

    def log_experiment(self):
        log_path = self.path_results + 'experiment.log'
        logging.basicConfig(format='%(message)s', filename=log_path, level=logging.DEBUG)
        logging.getLogger().addHandler(logging.StreamHandler())
        logging.info("Experiment: %s", self.experiment_name)
        logging.info("File location for results: %s", self.path_results)


def set_defaults(cfg: dict, default_cfg: dict):
    """Iterate through a recursive config dictionary and set all fields that are not
    present in cfg to the default values from default_cfg.

    == Usage ===
    >>> set_defaults(cfg={0:0, 1:{1:0}, 2:{2:1}},
    ...              default_cfg={1:{1:1}, 2:{1:1, 2:2}})
    {0: 0, 1: {1: 0}, 2: {2: 1, 1: 1}}
    >>> set_defaults(cfg={0:0, 1:1, 2:2},
    ...              default_cfg={1:{1:1}, 2:{1:1, 2:2}})
    {0: 0, 1: 1, 2: 2}
    """
    for key in default_cfg:
        if key not in cfg:
            # Field ´key´ is not defined in cfg -> use default
            cfg[key] = default_cfg[key]

        else:
            # Field ´key´ is defined in cfg
            # -> update recursively if the field is a dictionary
            if isinstance(default_cfg[key], dict) and isinstance(cfg[key], dict):
                set_defaults(cfg[key], default_cfg[key])

    return cfg


def update_recursive(cfg: dict, new_cfg: dict):
    """Iterate through a recursive config dictionary and update cfg in all fields that are specified in new_cfg.

    == Usage ===
    >>> update_recursive(cfg={0:0, 1:{1:0}, 2:{2:1}},
    ...                  new_cfg={1:{1:1}, 2:{1:1, 2:2}})
    {0: 0, 1: {1: 1}, 2: {2: 2, 1: 1}}
    >>> update_recursive(cfg={0:0, 1:1, 2:2},
    ...                  new_cfg={1:{1:1}, 2:{1:1, 2:2}})
    {0: 0, 1: {1: 1}, 2: {1: 1, 2: 2}}
    """
    for key in new_cfg:
        if (key in cfg) and isinstance(new_cfg[key], dict) and isinstance(cfg[key], dict):
            # Both dictionaries have another layer -> update recursively
            update_recursive(cfg[key], new_cfg[key])
        else:
            cfg[key] = new_cfg[key]

    return cfg


def iter_items_recursive(cfg: dict):
    """Recursively iterate through all key-value pairs in ´cfg´ and sub-dictionaries.

    Args:
        cfg (dict): Config dictionary, potentially containing sub-dictionaries.

    Yields:
        tuple: key-value pairs of the bottom level dictionaries

    == Usage ===
    >>> list(iter_items_recursive({0: 0, 1: {1: 0}, 2: {2: 1, 1: 1}}))
    [(0, 0), (1, 0), (2, 1), (1, 1)]
    """
    for key, value in cfg.items():
        if isinstance(value, dict):
            yield from iter_items_recursive(value)
        else:
            yield key, value


if __name__ == '__main__':
    import doctest
    doctest.testmod()
