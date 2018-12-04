"""Test BOCOP optimisation implementation of mixed stand approximate model."""

import unittest
from nose.plugins.attrib import attr
import numpy as np
import matplotlib.pyplot as plt
import mixed_stand_approx as ms_approx
import parameters

@attr('slow')
class TestOptimiser(unittest.TestCase):
    """Test accuracy of optimisation solutions."""

    def setUp(self):

        self.params = parameters.CORRECTED_PARAMS

        S11, S12, S13, S14, S2, S3 = parameters.COBB_INIT_FIG4A
        state_init = np.array([399*S11/400, S11/400, 0.0, S12, 0.0, 0.0, S13, 0.0, 0.0,
                               S14, 0.0, 0.0, 399*S2/400, S2/400, S3])

        setup = {
            'state_init': state_init,
            'times': np.linspace(0, 50.0, 101)
        }

        beta = np.array([1.67, 0.692*1.67, 0.551*1.67, 0.411*1.67, 1.608, 0.0, 4.443])

        self.params['inf_tanoak_tanoak'] = beta[0:4]
        self.params['inf_bay_to_tanoak'] = beta[4]
        self.params['inf_tanoak_to_bay'] = beta[5]
        self.params['inf_bay_to_bay'] = beta[6]
        self.params['payoff_factor'] = 1.0
        self.params['treat_eff'] = 0.75

        self.approx_model = ms_approx.MixedStandApprox(setup, self.params, beta)

    def test_optimiser_no_control(self):
        """Test BOCOP optimisation under no control."""

        approx_model_run, approx_obj = self.approx_model.run_policy(None)

        bocop_state_t, _, exit_text = self.approx_model.optimise(verbose=True)

        self.assertTrue(exit_text == "Optimal Solution Found." or
                        exit_text == "Solved To Acceptable Level.")

        bocop_run = np.array([bocop_state_t(t)[:-1] for t in self.approx_model.setup['times']]).T

        fig = plt.figure()
        ax = fig.add_subplot(111)
        self.approx_model.plot(ax=ax)

        host_totals = np.sum(bocop_run, axis=0)
        times = self.approx_model.setup['times']

        small_tanoak_sus = np.sum(bocop_run[[0, 3], :], axis=0) / host_totals
        large_tanoak_sus = np.sum(bocop_run[[6, 9], :], axis=0) / host_totals
        small_tanoak_inf = np.sum(bocop_run[[1, 4], :], axis=0) / host_totals
        large_tanoak_inf = np.sum(bocop_run[[7, 10], :], axis=0) / host_totals

        bay_sus = bocop_run[12, :] / host_totals
        bay_inf = bocop_run[13, :] / host_totals
        redwood = bocop_run[14, :] / host_totals

        ax.plot(times, small_tanoak_sus + small_tanoak_inf, 'kx', label="Small tanoak")
        ax.plot(times, large_tanoak_sus + large_tanoak_inf, 'kx', label="Large tanoak")
        ax.plot(times, bay_sus+bay_inf, 'bx', label="Bay")
        ax.plot(times, bay_inf, 'bx', label="Bay I")
        ax.plot(times, redwood, 'rx', label="Redwood")
        ax.plot(times, small_tanoak_inf, 'rx', label="Small tanoak I", alpha=0.3)
        ax.plot(times, large_tanoak_inf, 'rx', label="Large tanoak I", alpha=0.3)
        plt.show()

        arg_max = np.unravel_index(np.argmax(abs(approx_model_run - bocop_run)), bocop_run.shape)
        print((abs(approx_model_run - bocop_run))[arg_max[0], arg_max[1]], *arg_max)
        self.assertTrue(np.allclose(approx_model_run, bocop_run, atol=1e-2))

    def test_optimiser_control(self):
        """Test BOCOP optimisation with control."""

        self.approx_model.params['primary_inf'] = 0.0000
        self.approx_model.params['control_rate'] = 0.5

        bocop_state_t, bocop_control_t, exit_text = self.approx_model.optimise(
            verbose=True)

        self.assertTrue(exit_text == "Optimal Solution Found." or
                        exit_text == "Solved To Acceptable Level.")

        bocop_run = np.array([bocop_state_t(t)[:-1] for t in self.approx_model.setup['times']]).T

        approx_model_run, apprx_obj = self.approx_model.run_policy(
            bocop_control_t, n_fixed_steps=1000)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        self.approx_model.plot(ax=ax)

        host_totals = np.sum(bocop_run, axis=0)
        times = self.approx_model.setup['times']

        small_tanoak_sus = np.sum(bocop_run[[0, 3], :], axis=0) / host_totals
        large_tanoak_sus = np.sum(bocop_run[[6, 9], :], axis=0) / host_totals
        small_tanoak_inf = np.sum(bocop_run[[1, 4], :], axis=0) / host_totals
        large_tanoak_inf = np.sum(bocop_run[[7, 10], :], axis=0) / host_totals

        bay_sus = bocop_run[12, :] / host_totals
        bay_inf = bocop_run[13, :] / host_totals
        redwood = bocop_run[14, :] / host_totals

        ax.plot(times, small_tanoak_sus + small_tanoak_inf, 'kx', label="Small tanoak")
        ax.plot(times, large_tanoak_sus + large_tanoak_inf, 'kx', label="Large tanoak")
        ax.plot(times, bay_sus+bay_inf, 'bx', label="Bay")
        ax.plot(times, bay_inf, 'bx', label="Bay I")
        ax.plot(times, redwood, 'rx', label="Redwood")
        ax.plot(times, small_tanoak_inf, 'rx', label="Small tanoak I", alpha=0.3)
        ax.plot(times, large_tanoak_inf, 'rx', label="Large tanoak I", alpha=0.3)
        plt.show()

        arg_max = np.unravel_index(np.argmax(abs(approx_model_run - bocop_run)), bocop_run.shape)
        print((approx_model_run - bocop_run)[arg_max[0], arg_max[1]], arg_max)
        self.assertTrue(np.allclose(approx_model_run, bocop_run, atol=1e-2))
