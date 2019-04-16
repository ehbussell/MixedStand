"""Test functionality of mixed stand simulator."""

import unittest
import copy
import numpy as np
from mixed_stand_model import mixed_stand_simulator as ms_sim
from mixed_stand_model import parameters
from tests.utils import get_sis_params, sis_analytic, ZERO_PARAMS

# TODO add tests for vaccine decay
# TODO add tests for control

class TestNonSpatialDynamics(unittest.TestCase):
    """Test accuracy of simulator in non-spatial conditions."""

    def test_sis_accuracy(self):
        """Test non-spatial SIS setup matches analytic solution."""

        beta = 0.05
        mu = 1.0
        I0 = 0.1
        N = 100

        params = get_sis_params(beta, mu)

        state_init = np.zeros(15)
        state_init[0] = N - I0
        state_init[1] = I0

        setup = {
            'state_init': state_init,
            'landscape_dims': (1, 1),
            'times': np.linspace(0, 10.0, 101)
        }

        simulator = ms_sim.MixedStandSimulator(setup, params)
        simulator_results, *_ = simulator.run_policy()

        analytic_results = sis_analytic(setup['times'], beta, mu, I0, N)

        print(analytic_results)
        print(simulator_results[1, :])
        print(np.isclose(simulator_results[1, :], analytic_results, atol=1e-5))
        self.assertTrue(np.allclose(simulator_results[1, :], analytic_results, atol=1e-5))

class TestNonSpatialRates(unittest.TestCase):
    """Test calculations of state derivatives when system is non-spatial."""

    def setUp(self):
        S11, S12, S13, S14, S2, S3 = parameters.COBB_INIT_FIG4A
        state_init = np.tile([
            S11, 0.0, 0.0, S12, 0.0, 0.0, S13, 0.0, 0.0, S14, 0.0, 0.0, S2, 0.0, S3], 1)
        init_inf_cells = [0]
        init_inf_factor = 1.0
        for cell_pos in init_inf_cells:
            for i in [0, 4]:
                state_init[cell_pos*15+3*i+1] = init_inf_factor * state_init[cell_pos*15+3*i]
                state_init[cell_pos*15+3*i] *= (1.0 - init_inf_factor)

        setup = {
            'state_init': state_init,
            'landscape_dims': (1, 1),
            'times': np.linspace(0, 100, 101)
        }

        self.model = ms_sim.MixedStandSimulator(setup, ZERO_PARAMS)

    def test_recruit(self):
        """Test recruitment rates."""

        params = copy.deepcopy(ZERO_PARAMS)
        params['recruit_tanoak'] = np.random.rand(4)
        params['recruit_bay'] = np.random.rand()
        params['recruit_redwood'] = np.random.rand()

        self.model.params = params
        self.model._initialise()

        # Tanoak recruitment from single age class
        for i in range(4):
            state = np.array([0]*3*i + [0.01, 0.02, 0.03] + [0]*(9-3*i) + [0.04, 0.05, 0.06])
            deriv = self.model.state_deriv(0.0, np.append(state, 0))[:-1]
            test_deriv = np.array([
                params['recruit_tanoak'][i] * np.sum(state[3*i:3*(i+1)]) *
                (1.0 - np.sum(state))] + [0]*11)
            print("Tanoak recruit from {0}:".format(i+1), deriv[0:12], test_deriv)
            self.assertTrue(np.allclose(deriv[0:12], test_deriv))

        # Tanoak recruitment from all age classes
        state = np.random.rand(15) / 15
        deriv = self.model.state_deriv(0.0, np.append(state, 0))[:-1]
        test_deriv = np.zeros(12)
        for i in range(4):
            test_deriv[0] += (params['recruit_tanoak'][i] * np.sum(state[3*i:3*(i+1)]) *
                              (1.0 - np.sum(state)))
        print("Tanoak recruit from all:", deriv[0:12], test_deriv)
        self.assertTrue(np.allclose(deriv[0:12], test_deriv))

        # Bay recruitment
        test_deriv = np.array([
            params['recruit_bay'] * (state[12] + state[13]) * (1.0 - np.sum(state)), 0.0])
        print("Bay recruitment:", deriv[12:14], test_deriv)
        self.assertTrue(np.allclose(deriv[12:14], test_deriv))

        # Redwood recruitment
        test_deriv = params['recruit_redwood'] * state[14] * (1.0 - np.sum(state))
        print("Redwood recruitment:", deriv[14], test_deriv)
        self.assertTrue(np.allclose(deriv[14], test_deriv))

    def test_mortality(self):
        """Test mortality rates."""

        params = copy.deepcopy(ZERO_PARAMS)
        params['nat_mort_tanoak'] = np.random.rand(4)
        params['nat_mort_bay'] = np.random.rand()
        params['nat_mort_redwood'] = np.random.rand()
        params['inf_mort_tanoak'] = np.random.rand(4)

        self.model.params = params
        self.model._initialise()

        total_mort = np.append(
            np.repeat(params['nat_mort_tanoak'], 3),
            np.append(np.repeat(params['nat_mort_bay'], 2), [params['nat_mort_redwood']]))
        total_mort[1] += params['inf_mort_tanoak'][0]
        total_mort[4] += params['inf_mort_tanoak'][1]
        total_mort[7] += params['inf_mort_tanoak'][2]
        total_mort[10] += params['inf_mort_tanoak'][3]

        state = np.random.rand(15)
        expected_deriv = -total_mort * state
        deriv = self.model.state_deriv(0.0, np.append(state, 0))[:-1]
        print("Mortality rates:", deriv, expected_deriv)
        self.assertTrue(np.allclose(deriv, expected_deriv))

    def test_resprout(self):
        """Test resprouting rates of tanoak."""

        params = copy.deepcopy(ZERO_PARAMS)
        params['inf_mort_tanoak'] = np.random.rand(4)
        params['resprout_tanoak'] = np.random.rand()

        self.model.params = params
        self.model._initialise()

        total_mort = np.append(
            np.repeat(params['nat_mort_tanoak'], 3),
            np.append(np.repeat(params['nat_mort_bay'], 2), [params['nat_mort_redwood']]))
        total_mort[1] += params['inf_mort_tanoak'][0]
        total_mort[4] += params['inf_mort_tanoak'][1]
        total_mort[7] += params['inf_mort_tanoak'][2]
        total_mort[10] += params['inf_mort_tanoak'][3]

        state = np.random.rand(15)
        expected_deriv = -total_mort * state

        for i in range(4):
            expected_deriv[0] += (
                params['inf_mort_tanoak'][i] * params['resprout_tanoak'] * state[3*i+1])

        deriv = self.model.state_deriv(0.0, np.append(state, 0))[:-1]
        print("Resprout rates:", deriv, expected_deriv)
        self.assertTrue(np.allclose(deriv, expected_deriv))

    def test_age_transitions(self):
        """Test rates of age class transitions in tanoak."""

        params = copy.deepcopy(ZERO_PARAMS)
        params['trans_tanaok'] = np.random.rand(3)

        self.model.params = params
        self.model._initialise()

        state = np.random.rand(15)
        expected_deriv = np.zeros_like(state)

        for i in range(4):
            if i < 3:
                expected_deriv[3*i] -= (params['trans_tanoak'][i] * state[3*i])
                expected_deriv[3*i+1] -= (params['trans_tanoak'][i] * state[3*i+1])
                expected_deriv[3*i+2] -= (params['trans_tanoak'][i] * state[3*i+2])
            if i > 0:
                expected_deriv[3*i] += (params['trans_tanoak'][i-1] * state[3*(i - 1)])
                expected_deriv[3*i+1] += (params['trans_tanoak'][i-1] * state[3*(i - 1)+1])
                expected_deriv[3*i+2] += (params['trans_tanoak'][i-1] * state[3*(i - 1)+2])

        deriv = self.model.state_deriv(0.0, np.append(state, 0))[:-1]
        print("Age class transition rates:", deriv, expected_deriv)
        self.assertTrue(np.allclose(deriv, expected_deriv))

    def test_recovery(self):
        """Test rates of recovery."""

        params = copy.deepcopy(ZERO_PARAMS)
        params['recov_tanoak'] = np.random.rand()
        params['recov_bay'] = np.random.rand()

        self.model.params = params
        self.model._initialise()

        state = np.random.rand(15)
        expected_deriv = np.zeros_like(state)

        for i in range(4):
            expected_deriv[3*i+1] -= params['recov_tanoak'] * state[3*i+1]
            expected_deriv[3*i] += params['recov_tanoak'] * state[3*i+1]

        expected_deriv[13] -= params['recov_bay'] * state[13]
        expected_deriv[12] += params['recov_bay'] * state[13]

        deriv = self.model.state_deriv(0.0, np.append(state, 0))[:-1]
        print("Recovery rates:", deriv, expected_deriv)
        self.assertTrue(np.allclose(deriv, expected_deriv))

    def test_infection(self):
        """Test rates of infection."""

        params = copy.deepcopy(ZERO_PARAMS)
        params['inf_tanoak_tanoak'] = np.random.rand(4)
        params['inf_bay_to_bay'] = np.random.rand()
        params['inf_bay_to_tanoak'] = np.random.rand()
        params['inf_tanoak_to_bay'] = np.random.rand()
        params['primary_inf'] = np.random.rand()

        self.model.params = params
        self.model._initialise()

        state = np.random.rand(15)
        expected_deriv = np.zeros_like(state)

        for age in range(4):
            inf_rate = (
                params.get("primary_inf", 0.0) * state[3*age] +
                state[3*age] * (
                    params['inf_tanoak_tanoak'][age] * np.sum(state[1:12:3]) +
                    params['inf_bay_to_tanoak'] * state[13]))
            expected_deriv[3*age] -= inf_rate
            expected_deriv[3*age+1] += inf_rate
        inf_rate = (
            params.get("primary_inf", 0.0) * state[12] +
            state[12] * (
                params['inf_tanoak_to_bay'] * np.sum(state[1:12:3]) +
                params['inf_bay_to_bay'] * state[13]))
        expected_deriv[12] -= inf_rate
        expected_deriv[13] += inf_rate

        deriv = self.model.state_deriv(0.0, np.append(state, 0))[:-1]
        print("Infection rates:", deriv, expected_deriv)
        self.assertTrue(np.allclose(deriv, expected_deriv))

    def test_all_rates(self):
        """Test all rates together."""

        params = copy.deepcopy(ZERO_PARAMS)

        params['inf_tanoak_tanoak'] = np.random.rand(4)
        params['inf_bay_to_bay'] = np.random.rand()
        params['inf_bay_to_tanoak'] = np.random.rand()
        params['inf_tanoak_to_bay'] = np.random.rand()
        params['primary_inf'] = np.random.rand()

        params['nat_mort_tanoak'] = np.random.rand(4)
        params['nat_mort_bay'] = np.random.rand()
        params['nat_mort_redwood'] = np.random.rand()
        params['inf_mort_tanoak'] = np.random.rand(4)

        params['trans_tanaok'] = np.random.rand(3)
        params['recov_tanoak'] = np.random.rand()
        params['recov_bay'] = np.random.rand()

        params['recruit_tanoak'] = np.random.rand(4)
        params['recruit_bay'] = np.random.rand()
        params['recruit_redwood'] = np.random.rand()

        params['resprout_tanoak'] = np.random.rand()

        self.model.params = params
        self.model._initialise()

        state = np.random.rand(15) / 15
        expected_deriv = np.zeros_like(state)

        # Recruitment
        empty_space = 1 - np.sum(state)
        for i in range(4):
            expected_deriv[0] += (
                params['recruit_tanoak'][i] * np.sum(state[3*i:3*(i+1)]) * empty_space)
        expected_deriv[12] += params['recruit_bay'] * np.sum(state[12:14]) * empty_space
        expected_deriv[14] += params['recruit_redwood'] * state[14] * empty_space

        # Mortality
        total_mort = np.append(
            np.repeat(params['nat_mort_tanoak'], 3),
            np.append(np.repeat(params['nat_mort_bay'], 2), [params['nat_mort_redwood']]))
        for i in range(4):
            total_mort[3*i+1] += params['inf_mort_tanoak'][i]
        expected_deriv -= total_mort * state

        # Resprouting
        for i in range(4):
            expected_deriv[0] += (
                params['inf_mort_tanoak'][i] * params['resprout_tanoak'] * state[3*i+1])

        # Age transitions
        for i in range(4):
            if i < 3:
                expected_deriv[3*i] -= (params['trans_tanoak'][i] * state[3*i])
                expected_deriv[3*i+1] -= (params['trans_tanoak'][i] * state[3*i+1])
                expected_deriv[3*i+2] -= (params['trans_tanoak'][i] * state[3*i+2])
            if i > 0:
                expected_deriv[3*i] += (params['trans_tanoak'][i-1] * state[3*(i - 1)])
                expected_deriv[3*i+1] += (params['trans_tanoak'][i-1] * state[3*(i - 1)+1])
                expected_deriv[3*i+2] += (params['trans_tanoak'][i-1] * state[3*(i - 1)+2])

        # Recovery
        for i in range(4):
            expected_deriv[3*i+1] -= params['recov_tanoak'] * state[3*i+1]
            expected_deriv[3*i] += params['recov_tanoak'] * state[3*i+1]
        expected_deriv[13] -= params['recov_bay'] * state[13]
        expected_deriv[12] += params['recov_bay'] * state[13]

        # Infection
        for age in range(4):
            inf_rate = (
                params.get("primary_inf", 0.0) * state[3*age] +
                state[3*age] * (
                    params['inf_tanoak_tanoak'][age] * np.sum(state[1:12:3]) +
                    params['inf_bay_to_tanoak'] * state[13]))
            expected_deriv[3*age] -= inf_rate
            expected_deriv[3*age+1] += inf_rate
        inf_rate = (
            params.get("primary_inf", 0.0) * state[12] +
            state[12] * (
                params['inf_tanoak_to_bay'] * np.sum(state[1:12:3]) +
                params['inf_bay_to_bay'] * state[13]))
        expected_deriv[12] -= inf_rate
        expected_deriv[13] += inf_rate

        deriv = self.model.state_deriv(0.0, np.append(state, 0))[:-1]
        print("Infection rates:", deriv, expected_deriv)
        self.assertTrue(np.allclose(deriv, expected_deriv))

class TestControlRates(unittest.TestCase):
    """Test calculations of state derivatives when system is under control."""

    def setUp(self):
        S11, S12, S13, S14, S2, S3 = parameters.COBB_INIT_FIG4A
        state_init = np.tile([
            S11, 0.0, 0.0, S12, 0.0, 0.0, S13, 0.0, 0.0, S14, 0.0, 0.0, S2, 0.0, S3], 25)

        setup = {
            'state_init': state_init,
            'landscape_dims': (5, 5),
            'times': np.linspace(0, 100, 101)
        }

        self.model = ms_sim.MixedStandSimulator(setup, ZERO_PARAMS)
        self.ncells = 25

    def test_rogue(self):
        """Test roguing rates are correct."""

        params = copy.deepcopy(ZERO_PARAMS)
        params['rogue_rate'] = np.random.rand()

        self.model.params = params
        self.model._initialise()

        state = np.random.rand(15 * self.ncells)

        control = np.zeros(9)
        control[:3] = np.random.rand(3)

        def control_func(time):
            return control

        expected_deriv = np.zeros_like(state)
        for i in range(self.ncells):
            expected_deriv[15*i+1] = - control[0] * state[15*i+1] * params['rogue_rate']
            expected_deriv[15*i+4] = - control[0] * state[15*i+4] * params['rogue_rate']
            expected_deriv[15*i+7] = - control[1] * state[15*i+7] * params['rogue_rate']
            expected_deriv[15*i+10] = - control[1] * state[15*i+10] * params['rogue_rate']
            expected_deriv[15*i+13] = - control[2] * state[15*i+13] * params['rogue_rate']

        deriv = self.model.state_deriv(0.0, np.append(state, 0), control_func=control_func)[:-1]
        print("Rogue rates:", deriv, expected_deriv)
        self.assertTrue(np.allclose(deriv, expected_deriv))

    def test_thin(self):
        """Test thinning rates are correct."""

        params = copy.deepcopy(ZERO_PARAMS)
        params['thin_rate'] = np.random.rand()

        self.model.params = params
        self.model._initialise()

        state = np.random.rand(15 * self.ncells)

        control = np.zeros(9)
        control[3:7] = np.random.rand(4)

        def control_func(time):
            return control

        expected_deriv = np.zeros_like(state)
        for i in range(self.ncells):
            for j in range(6):
                expected_deriv[15*i+j] = - control[3] * state[15*i+j] * params['thin_rate']
            for j in range(6):
                expected_deriv[15*i+j+6] = - control[4] * state[15*i+j+6] * params['thin_rate']
            for j in range(2):
                expected_deriv[15*i+j+12] = - control[5] * state[15*i+j+12] * params['thin_rate']
            expected_deriv[15*i+14] = - control[6] * state[15*i+14] * params['thin_rate']

        deriv = self.model.state_deriv(0.0, np.append(state, 0), control_func=control_func)[:-1]
        print("Thinning rates:", deriv, expected_deriv)
        self.assertTrue(np.allclose(deriv, expected_deriv))

    def test_protect(self):
        """Test protection rates are correct."""

        params = copy.deepcopy(ZERO_PARAMS)
        params['protect_rate'] = np.random.rand()

        self.model.params = params
        self.model._initialise()

        state = np.random.rand(15 * self.ncells)

        control = np.zeros(9)
        control[7:] = np.random.rand(2)

        def control_func(time):
            return control

        expected_deriv = np.zeros_like(state)
        for i in range(self.ncells):
            for j in range(2):
                expected_deriv[15*i+3*j] = - control[7] * state[15*i+3*j] * params['protect_rate']
                expected_deriv[15*i+3*j+2] = control[7] * state[15*i+3*j] * params['protect_rate']
            for j in range(2):
                expected_deriv[15*i+3*j+6] = - control[8] * state[15*i+3*j+6] * params['protect_rate']
                expected_deriv[15*i+3*j+6+2] = control[8] * state[15*i+3*j+6] * params['protect_rate']

        deriv = self.model.state_deriv(0.0, np.append(state, 0), control_func=control_func)[:-1]
        print("Thinning rates:", deriv, expected_deriv)
        self.assertTrue(np.allclose(deriv, expected_deriv))

class TestSpatialInfection(unittest.TestCase):
    """Test calculations of infection rates when system is spatial."""

    def setUp(self):
        S11, S12, S13, S14, S2, S3 = parameters.COBB_INIT_FIG4A
        state_init = np.tile([
            S11, 0.0, 0.0, S12, 0.0, 0.0, S13, 0.0, 0.0, S14, 0.0, 0.0, S2, 0.0, S3], 25)
        init_inf_cells = [12]
        init_inf_factor = 0.5
        for cell_pos in init_inf_cells:
            for i in [0, 4]:
                state_init[cell_pos*15+3*i+1] = init_inf_factor * state_init[cell_pos*15+3*i]
                state_init[cell_pos*15+3*i] *= (1.0 - init_inf_factor)

        setup = {
            'state_init': state_init,
            'landscape_dims': (5, 5),
            'times': np.linspace(0, 100, 101)
        }

        self.model = ms_sim.MixedStandSimulator(setup, ZERO_PARAMS)

        self.ncells = 25
        self.inf_cell = 12

    def test_within_cell(self):
        """Test within cell infection rates."""

        params = copy.deepcopy(ZERO_PARAMS)

        params['inf_tanoak_tanoak'] = np.random.rand(4)
        params['inf_bay_to_bay'] = np.random.rand()
        params['inf_bay_to_tanoak'] = np.random.rand()
        params['inf_tanoak_to_bay'] = np.random.rand()

        params['spore_within'] = 0.5
        params['spore_between'] = 0.0
        params['num_nn'] = 4

        self.model.params = params
        self.model._initialise()

        state = np.random.rand(15*self.ncells)
        for i in range(self.ncells):
            if i != self.inf_cell:
                state[15*i+1:15*i+14:3] = 0
        expected_deriv = np.zeros_like(state)

        start_idx = 15*self.inf_cell

        for age in range(4):
            inf_rate = (
                state[start_idx+3*age] * params['spore_within'] * (
                    params['inf_tanoak_tanoak'][age] * np.sum(state[start_idx+1:start_idx+12:3]) +
                    params['inf_bay_to_tanoak'] * state[start_idx+13]))
            expected_deriv[start_idx+3*age] -= inf_rate
            expected_deriv[start_idx+3*age+1] += inf_rate
        inf_rate = (
            state[start_idx+12] * params['spore_within'] * (
                params['inf_tanoak_to_bay'] * np.sum(state[start_idx+1:start_idx+12:3]) +
                params['inf_bay_to_bay'] * state[start_idx+13]))
        expected_deriv[start_idx+12] -= inf_rate
        expected_deriv[start_idx+13] += inf_rate

        deriv = self.model.state_deriv(0.0, np.append(state, 0))[:-1]
        print("Infection rates:", deriv, expected_deriv)
        self.assertTrue(np.allclose(deriv, expected_deriv))

    def test_between_cell(self):
        """Test between cell infection rates."""

        params = copy.deepcopy(ZERO_PARAMS)

        params['inf_tanoak_tanoak'] = np.random.rand(4)
        params['inf_bay_to_bay'] = np.random.rand()
        params['inf_bay_to_tanoak'] = np.random.rand()
        params['inf_tanoak_to_bay'] = np.random.rand()

        params['spore_within'] = 0.0
        params['spore_between'] = 0.5 / 4
        params['num_nn'] = 4

        self.model.params = params
        self.model._initialise()

        state = np.random.rand(15*self.ncells)
        for i in range(self.ncells):
            if i != self.inf_cell:
                state[15*i+1:15*i+14:3] = 0
        expected_deriv = np.zeros_like(state)

        start_idx = 15*self.inf_cell

        adjacent_cells = [self.inf_cell-5, self.inf_cell-1, self.inf_cell+1, self.inf_cell+5]

        for cell in adjacent_cells:
            for age in range(4):
                inf_rate = (
                    state[15*cell+3*age] * params['spore_between'] * (
                        params['inf_tanoak_tanoak'][age] * np.sum(state[start_idx+1:start_idx+12:3])
                        + params['inf_bay_to_tanoak'] * state[start_idx+13]))
                expected_deriv[15*cell+3*age] -= inf_rate
                expected_deriv[15*cell+3*age+1] += inf_rate
            inf_rate = (
                state[15*cell+12] * params['spore_between'] * (
                    params['inf_tanoak_to_bay'] * np.sum(state[start_idx+1:start_idx+12:3]) +
                    params['inf_bay_to_bay'] * state[start_idx+13]))
            expected_deriv[15*cell+12] -= inf_rate
            expected_deriv[15*cell+13] += inf_rate

        deriv = self.model.state_deriv(0.0, np.append(state, 0))[:-1]
        print("Infection rates:", deriv, expected_deriv)
        self.assertTrue(np.allclose(deriv, expected_deriv))
