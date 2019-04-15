"""Test functionality of mixed stand simulator."""

import unittest
import numpy as np
from mixed_stand_model import utils

class TestObjCalculation(unittest.TestCase):
    """Test calculation of system objective."""

    def test_payoff(self):
        """Test calculation of payoff term"""

        # First test with length 15 array (single cell or approximate model)
        final_time = 100 * np.random.rand()
        state = np.random.rand(15)
        params = {
            'payoff_factor': np.random.rand(),
            'discount_rate': np.random.rand() * 0.05
        }

        payoff = utils.objective_payoff(final_time, state, params)

        expected_payoff = - (
            params['payoff_factor'] * np.exp(-params['discount_rate'] * final_time) *
            (state[6] + state[8] + state[9] + state[11]))

        self.assertEqual(payoff, expected_payoff)

        # Now test with multiple cells
        ncells = 100
        state = np.random.rand(15 * ncells)

        payoff = utils.objective_payoff(final_time, state, params)

        expected_payoff = - (
            params['payoff_factor'] * np.exp(-params['discount_rate'] * final_time) *
            np.sum([np.sum(state[x::15] for x in [6, 8, 9, 11])]))

        self.assertAlmostEqual(payoff, expected_payoff)

    def test_integrand(self):
        """Test calculation of integrand term."""

        # First test with length 15 array (single cell or approximate model)
        time = 100 * np.random.rand()
        state = np.random.rand(15)
        control = np.random.rand(9)
        params = {
            'div_cost': np.random.rand(),
            'cull_cost': np.random.rand(),
            'protect_cost': np.random.rand(),
            'control_rate': np.random.rand(),
            'discount_rate': np.random.rand(),
        }

        integrand = utils.objective_integrand(time, state, control, params)

        state_props = np.array([np.sum(state[0:6]) / np.sum(state),
                                np.sum(state[6:12]) / np.sum(state),
                                np.sum(state[12:14]) / np.sum(state),
                                np.sum(state[14]) / np.sum(state)])

        expected_integrand = np.exp(-params['discount_rate']*time) * (
            params['div_cost'] * np.sum(state_props * np.log(state_props)) +
            params['cull_cost'] * params['control_rate'] * (
                control[0] * (state[1] + state[4]) +
                control[1] * (state[7] + state[10]) +
                control[2] * (state[13]) +
                control[3] * np.sum(state[0:6]) +
                control[4] * np.sum(state[6:12]) +
                control[5] * np.sum(state[12:14]) +
                control[6] * state[14]) +
            params['protect_cost'] * params['control_rate'] * (
                control[7] * (state[0] + state[3]) +
                control[8] * (state[6] + state[9]))
            )

        self.assertEqual(integrand, expected_integrand)

        # Now test with multiple cells
        ncells = 100
        state = np.random.rand(15 * ncells)

        integrand = utils.objective_integrand(time, state, control, params)

        state = np.sum(np.reshape(state, (ncells, 15)), axis=0)

        state_props = np.array([np.sum(state[0:6]) / np.sum(state),
                                np.sum(state[6:12]) / np.sum(state),
                                np.sum(state[12:14]) / np.sum(state),
                                np.sum(state[14]) / np.sum(state)])

        expected_integrand = np.exp(-params['discount_rate']*time) * (
            params['div_cost'] * np.sum(state_props * np.log(state_props)) +
            params['cull_cost'] * params['control_rate'] * (
                control[0] * (state[1] + state[4]) +
                control[1] * (state[7] + state[10]) +
                control[2] * (state[13]) +
                control[3] * np.sum(state[0:6]) +
                control[4] * np.sum(state[6:12]) +
                control[5] * np.sum(state[12:14]) +
                control[6] * state[14]) +
            params['protect_cost'] * params['control_rate'] * (
                control[7] * (state[0] + state[3]) +
                control[8] * (state[6] + state[9]))
            )

        self.assertEqual(integrand, expected_integrand)
