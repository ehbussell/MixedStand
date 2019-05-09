"""Test BOCOP optimisation implementation of mixed stand approximate model."""

import unittest
import numpy as np
import matplotlib.pyplot as plt
import pytest
from mixed_stand_model import mixed_stand_approx as ms_approx
from mixed_stand_model import parameters
from mixed_stand_model import visualisation
from mixed_stand_model import utils

@pytest.mark.slow
class TestOptimiser(unittest.TestCase):
    """Test accuracy of optimisation solutions."""

    def setUp(self):

        self.params = parameters.CORRECTED_PARAMS

        state_init = parameters.COBB_INIT_FIG4A
        state_init[1] = state_init[0]/400
        state_init[0] *= 399/400
        state_init[13] = state_init[12]/400
        state_init[12] *= 399/400

        # Now initialise space weights and recruitment rates
        self.params, state_init = utils.initialise_params(self.params, init_state=state_init)

        setup = {
            'state_init': state_init,
            'times': np.linspace(0, 50.0, 101)
        }

        beta = 0.5 * np.array([0.88, 0.81, 0.65, 0.48, 5.88, 0.0, 7.37])

        self.params['inf_tanoak_tanoak'] = beta[0:4]
        self.params['inf_bay_to_tanoak'] = beta[4]
        self.params['inf_tanoak_to_bay'] = beta[5]
        self.params['inf_bay_to_bay'] = beta[6]
        self.params['payoff_factor'] = 1.0
        self.params['treat_eff'] = 0.75
        self.params['max_budget'] = 100

        self.approx_model = ms_approx.MixedStandApprox(setup, self.params, beta)

    def test_optimiser_no_control(self):
        """Test BOCOP optimisation under no control."""

        approx_model_run, *_ = self.approx_model.run_policy(None)

        approx_model_run = approx_model_run[0:15]

        bocop_run, _, exit_text = self.approx_model.optimise(verbose=True)
        bocop_run = bocop_run[0:15]

        self.assertTrue(exit_text == "Optimal Solution Found." or
                        exit_text == "Solved To Acceptable Level.")

        times = self.approx_model.setup['times']

        fig = plt.figure()
        ax = fig.add_subplot(111)
        visualisation.plot_hosts(times, approx_model_run, ax=ax, proportions=False)

        small_tanoak = np.sum(bocop_run[0:6, :], axis=0)
        large_tanoak = np.sum(bocop_run[6:12, :], axis=0)
        bay = np.sum(bocop_run[12:14, :], axis=0)
        redwood = bocop_run[14, :]

        cmap = plt.get_cmap("tab20c")
        colours = [cmap(2.5*0.05), cmap(0.5*0.05), cmap(8.5*0.05), cmap(4.5*0.05)]

        ax.plot(times, small_tanoak, 'x', color=colours[0], label="Small tanoak")
        ax.plot(times, large_tanoak, 'x', color=colours[1], label="Large tanoak")
        ax.plot(times, bay, 'x', color=colours[2], label="Bay")
        ax.plot(times, redwood, 'x', color=colours[3], label="Redwood")
        ax.legend()
        plt.show()

        arg_max = np.unravel_index(np.argmax(abs(approx_model_run - bocop_run)), bocop_run.shape)
        print((abs(approx_model_run - bocop_run))[arg_max[0], arg_max[1]], *arg_max)
        self.assertTrue(np.allclose(approx_model_run, bocop_run, atol=1e-2))

    def test_optimiser_control(self):
        """Test BOCOP optimisation with control."""

        self.approx_model.params['primary_inf'] = 0.0000
        self.approx_model.params['rogue_rate'] = 0.5
        self.approx_model.params['thin_rate'] = 0.5
        self.approx_model.params['protect_rate'] = 0.5
        self.approx_model.params['rogue_cost'] = 20
        self.approx_model.params['thin_cost'] = 20
        self.approx_model.params['protect_cost'] = 1

        bocop_run, bocop_control_t, exit_text = self.approx_model.optimise(
            verbose=True)
        bocop_run = bocop_run[0:15]

        self.assertTrue(exit_text == "Optimal Solution Found." or
                        exit_text == "Solved To Acceptable Level.")

        approx_model_run, *_ = self.approx_model.run_policy(
            bocop_control_t, n_fixed_steps=1000)
        times = self.approx_model.setup['times']
        approx_model_run = approx_model_run[0:15]

        print(approx_model_run.shape, bocop_run.shape)

        fig = plt.figure()
        ax = fig.add_subplot(111)
        visualisation.plot_hosts(times, approx_model_run, ax=ax, proportions=False)

        small_tanoak = np.sum(bocop_run[0:6, :], axis=0)
        large_tanoak = np.sum(bocop_run[6:12, :], axis=0)
        bay = np.sum(bocop_run[12:14, :], axis=0)
        redwood = bocop_run[14, :]

        cmap = plt.get_cmap("tab20c")
        colours = [cmap(2.5*0.05), cmap(0.5*0.05), cmap(8.5*0.05), cmap(4.5*0.05)]

        ax.plot(times, small_tanoak, 'x', color=colours[0], label="Small tanoak")
        ax.plot(times, large_tanoak, 'x', color=colours[1], label="Large tanoak")
        ax.plot(times, bay, 'x', color=colours[2], label="Bay")
        ax.plot(times, redwood, 'x', color=colours[3], label="Redwood")
        ax.legend()
        plt.show()

        arg_max = np.unravel_index(np.argmax(abs(approx_model_run - bocop_run)), bocop_run.shape)
        print((approx_model_run - bocop_run)[arg_max[0], arg_max[1]], arg_max)
        self.assertTrue(np.allclose(approx_model_run, bocop_run, atol=1e-2))
