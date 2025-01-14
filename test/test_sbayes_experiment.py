#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import unittest
from pathlib import Path

from sbayes.experiment_setup import Experiment
from sbayes.simulation import Simulation
from sbayes.mcmc_setup import MCMC


class TestExperiment(unittest.TestCase):

    """
    Test cases covering the general pipeline of sbayes experiments on simple and short runs.
    This should include data loading, configuration, mcmc setup and running the actual analysis.
    """

    CUSTOM_SETTINGS = {
        'simulation': {
            '_FEATURES': 5,
            'i_contact': 3,
            'e_contact': 0.5,
            'STRENGTH': 0,
            'area': 3,
            'correlation_threshold': 0.8
        },
        'mcmc': {
            'n_steps': 40,
            'n_samples': 20,
            'warm_up': {
                'n_warm_up_steps': 5,
                'n_warm_up_chains': 2
            }
        },
    }

    @staticmethod
    def test_sim_exp1():
        """Test whether simulation experiment 1 is running without errors."""
        custom_settings = TestExperiment.CUSTOM_SETTINGS
        TestExperiment.run_experiment(path=Path('experiments/simulation/sim_exp1/'),
                                      custom_settings=custom_settings)

        print('Experiment 1 passed\n')

    @staticmethod
    def test_sim_exp2():
        """Test whether simulation experiment 2 is running without errors."""
        custom_settings = {
            'model': {'inheritance': True},
            **TestExperiment.CUSTOM_SETTINGS
        }
        TestExperiment.run_experiment(path=Path('experiments/simulation/sim_exp2/'),
                                      custom_settings=custom_settings)

        print('Experiment 2 passed\n')

    @staticmethod
    def test_sim_exp3():
        """Test whether simulation experiment 3 is running without errors."""
        custom_settings = {
            'model': {'n_areas': 2},
            **TestExperiment.CUSTOM_SETTINGS
        }
        TestExperiment.run_experiment(path=Path('experiments/simulation/sim_exp3/'),
                                      custom_settings=custom_settings)

        print('Experiment 3 passed\n')

    @staticmethod
    def run_experiment(path: Path, custom_settings: dict):
        # 1. Initialize the experiment
        exp = Experiment()
        exp.load_config(config_file=path / 'config.json',
                        custom_settings=custom_settings)

        if exp.config['model']['n_areas'] == "TBD":
            exp.config['model']['n_areas'] = 2

        # 2. Simulate contact areas
        sim = Simulation(experiment=exp)
        sim.run_simulation()

        # 3. Define MCMC
        mc = MCMC(data=sim, experiment=exp)

        # 4. Warm-up sampler and sample from posterior
        mc.warm_up()
        mc.sample()

        # 5. Evaluate ground truth
        mc.eval_ground_truth()

        # 6. Log sampling statistics and save samples to file
        mc.save_samples(run=0)


if __name__ == '__main__':
    unittest.main()
