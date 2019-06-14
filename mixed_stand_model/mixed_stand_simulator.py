"""
Model for tanoak-bay laurel-redwood mixed stand SOD from Cobb 2012.

Species are labelled as:
    1) Tanoak
    2) Bay Laurel
    3) Redwood
"""

import copy
import logging
import pickle
import numpy as np
from scipy import integrate

from mixed_stand_model import utils


class MixedStandSimulator:
    """
    Class to implement a 3 species model of SOD within a forest stand.

    Initialisation requires the following setup dictionary of values:
        'state_init':       Can be either length 15 array giving [S11, I11, V11 ... S2 I2 S3] which
                            will be uniformly applied across landscape. Or length 15 * ncells array
                            giving this vector for each cell location.
        'landscape_dims':   (nx, ny).
        'times':            Times to solve for.

    and parameter object.
    """

    def __init__(self, setup, params):
        required_keys = ['state_init', 'landscape_dims', 'times']

        # Make sure required keys present
        try:
            self.setup = {k: setup[k] for k in required_keys}
        except KeyError as err:
            logging.exception("Missing required key!")
            raise err

        # Log if unnecessary keys present
        for key in setup:
            if key not in required_keys:
                logging.info("Unused setup parameter: %s", key)

        self.params = copy.deepcopy(params)

        self._linear_matrix = None
        self._inf_matrix = None
        self._indices = None
        self._space_weights = None
        self._recruit_rates = None
        self.ncells = np.prod(self.setup['landscape_dims'])

        # Object to store run results
        self.run = {
            'state': np.array([[]]),
            'control': np.array([[]]),
            'objective': None
        }

    @classmethod
    def load_run_class(cls, filename):
        """Load class from run file."""

        with open(filename, "rb") as infile:
            load_obj = pickle.load(infile)

        instance = cls(load_obj['setup'], load_obj['params'])
        instance.run = load_obj['run']

        return instance

    def save_run(self, filename):
        """Save run_data, control and run parameters to file."""

        dump_obj = {
            'run': self.run,
            'setup': self.setup,
            'params': self.params
        }

        with open(filename, "wb") as outfile:
            pickle.dump(dump_obj, outfile)

        logging.debug("Saved run data to %s", filename)

    def load_run(self, filename):
        """Load run_data, control and run parameters."""

        with open(filename, "rb") as infile:
            load_obj = pickle.load(infile)

        self.run = load_obj['run']
        self.setup = load_obj['setup']
        self.params = load_obj['params']

        logging.debug("Loaded run data from %s", filename)

    def _initialise(self):
        """Initialise ready for simulation - set up initial conditions, matrices and rates."""

        logging.debug("Starting initialisation")

        if len(self.setup['state_init']) == 15:
            # Uniform initialisation across landscape
            state_init = np.tile(self.setup['state_init'], self.ncells)
        else:
            # User specified landscape initialisation
            # Check dimensions consistent
            if len(self.setup['state_init']) / 15 != self.ncells:
                logging.error("Incorrect lenth of state initialisation array!")
                raise ValueError("Incorrect length of state initialisation array!")
            state_init = self.setup['state_init']

        # Construct infection and transition matrices
        self._construct_matrices()

        # For ease later arrays of indices to select particular host classes
        self._indices = {
            'inf_s_idx': np.array(
                [15*loc+np.arange(14, step=3) for loc in range(self.ncells)]).flatten(),
            'tan_s_idx': np.array(
                [15*loc+np.arange(12, step=3) for loc in range(self.ncells)]).flatten(),
            'tan_inf_force_idx': np.array(
                [5*loc+np.arange(4) for loc in range(self.ncells)]).flatten(),
            'recruit_idx': np.array([15*loc+np.array([0, 12, 14])
                                     for loc in range(self.ncells)]).flatten(),
        }

        # Create array of space weights for each state (len 15*ncells)
        self._space_weights = np.tile(np.append(
            np.repeat(self.params['space_tanoak'], 3),
            [self.params['space_bay'], self.params['space_bay'], self.params['space_redwood']]),
                                      self.ncells)

        # Create array of recruitment rates for each state (len 15*ncells)
        self._recruit_rates = np.tile(np.append(
            np.repeat(self.params['recruit_tanoak'], 3),
            [self.params['recruit_bay'], self.params['recruit_bay'],
             self.params['recruit_redwood']]), self.ncells)

        # Return the initial state
        return state_init

    def _get_state_idx(self, species, age_class, location):
        """Get index for particular host in state array."""

        if species == utils.Species.TANOAK:
            index = 3*age_class
        elif species == utils.Species.BAY:
            index = 12
        elif species == utils.Species.REDWOOD:
            index = 14
        else:
            raise ValueError("Unrecognised species!")

        index += 15 * location
        return index

    def _construct_matrices(self):
        """Construct matrices for calculation of equations RHS (linear parts and infection)."""

        # Linear matrix A - terms linear in the state, i.e. dX = A*X for each location
        A = np.zeros((15, 15))

        # Resprouting
        i = self._get_state_idx(utils.Species.TANOAK, 0, 0)
        for age_class in range(4):
            j = self._get_state_idx(utils.Species.TANOAK, age_class, 0) + 1
            A[i, j] += self.params['resprout_tanoak'] * self.params['inf_mort_tanoak'][age_class]

        # Mortality and recovery (tanoak)
        for age_class in range(4):
            i = self._get_state_idx(utils.Species.TANOAK, age_class, 0)
            j = self._get_state_idx(utils.Species.TANOAK, age_class, 0) + 1
            k = self._get_state_idx(utils.Species.TANOAK, age_class, 0) + 2
            A[i, i] += -self.params['nat_mort_tanoak'][age_class]
            A[j, j] += (-self.params['nat_mort_tanoak'][age_class]
                        -self.params['inf_mort_tanoak'][age_class]
                        -self.params['recov_tanoak'])
            A[i, j] += self.params['recov_tanoak']
            A[k, k] += -self.params['nat_mort_tanoak'][age_class]

        # Mortality and recovery (bay)
        i = self._get_state_idx(utils.Species.BAY, 0, 0)
        j = self._get_state_idx(utils.Species.BAY, 0, 0) + 1
        A[i, i] += -self.params['nat_mort_bay']
        A[j, j] += (-self.params['nat_mort_bay'] - self.params['recov_bay'])
        A[i, j] += self.params['recov_bay']

        # Mortality and recovery (redwood)
        i = self._get_state_idx(utils.Species.REDWOOD, 0, 0)
        A[i, i] += -self.params['nat_mort_redwood']

        # Age transitions
        for age_class in range(3):
            i = self._get_state_idx(utils.Species.TANOAK, age_class, 0)
            j = self._get_state_idx(utils.Species.TANOAK, age_class+1, 0)
            A[i, i] += -self.params['trans_tanoak'][age_class]
            A[j, i] += self.params['trans_tanoak'][age_class]
            A[i+1, i+1] += -self.params['trans_tanoak'][age_class]
            A[j+1, i+1] += self.params['trans_tanoak'][age_class]
            A[i+2, i+2] += -self.params['trans_tanoak'][age_class]
            A[j+2, i+2] += self.params['trans_tanoak'][age_class]

        # Primary infection
        for age_class in range(4):
            i = self._get_state_idx(utils.Species.TANOAK, age_class, 0)
            A[i, i] -= self.params.get('primary_inf', 0.0)
            A[i+1, i] += self.params.get('primary_inf', 0.0)
            A[i+2, i+2] -= (self.params.get('primary_inf', 0.0) *
                            self.params.get('treat_eff', 0.0))
            A[i+1, i+2] += (self.params.get('primary_inf', 0.0) *
                            self.params.get('treat_eff', 0.0))
        i = self._get_state_idx(utils.Species.BAY, 0, 0)
        A[i, i] -= self.params.get('primary_inf', 0.0)
        A[i+1, i] += self.params.get('primary_inf', 0.0)

        self._linear_matrix = A

        logging.debug("Completed linear matrix intialisation")

        # Infection matrix B. This is multiplied by infectious hosts to get change in susceptible
        # hosts. i.e. dSus = B * Inf
        B = np.zeros((5*self.ncells, 5*self.ncells))

        if self.params.get("kernel_type", "nn") == "nn":

            for location in range(self.ncells):
                # Find adjacent cells
                loc_coords = np.unravel_index(location, self.setup['landscape_dims'])
                adjacent_coords = []

                # Can vary number of nearest neighbours - either 4 or 8
                if self.params['num_nn'] == 4:
                    for row_change in [-1, 1]:
                        for col_change in [0]:
                            if not (row_change == 0 and col_change == 0):
                                new_coord = np.add(loc_coords, (row_change, col_change))
                                # Check inside domain:
                                check = (
                                    (new_coord[0] >= 0 and
                                    new_coord[0] < self.setup['landscape_dims'][0])
                                    and
                                    (new_coord[1] >= 0 and
                                    new_coord[1] < self.setup['landscape_dims'][1]))
                                if check:
                                    adjacent_coords.append(new_coord)
                    for row_change in [0]:
                        for col_change in [-1, 1]:
                            if not (row_change == 0 and col_change == 0):
                                new_coord = np.add(loc_coords, (row_change, col_change))
                                # Check inside domain:
                                check = (
                                    (new_coord[0] >= 0 and
                                    new_coord[0] < self.setup['landscape_dims'][0])
                                    and
                                    (new_coord[1] >= 0 and
                                    new_coord[1] < self.setup['landscape_dims'][1]))
                                if check:
                                    adjacent_coords.append(new_coord)
                elif self.params['num_nn'] == 8:
                    for row_change in [-1, 0, 1]:
                        for col_change in [-1, 0, 1]:
                            if not (row_change == 0 and col_change == 0):
                                new_coord = np.add(loc_coords, (row_change, col_change))
                                # Check inside domain:
                                check = (
                                    (new_coord[0] >= 0 and
                                    new_coord[0] < self.setup['landscape_dims'][0])
                                    and
                                    (new_coord[1] >= 0 and
                                    new_coord[1] < self.setup['landscape_dims'][1]))
                                if check:
                                    adjacent_coords.append(new_coord)
                else:
                    raise ValueError("Number of nearest neighbours must be 4 or 8!")

                if adjacent_coords:
                    adjacent_locs = np.ravel_multi_index(
                        np.array(adjacent_coords).T, self.setup['landscape_dims'])
                else:
                    adjacent_locs = []

                # Infection of tanoak:
                for age_class in range(4):
                    for age_class2 in range(4):
                        B[5*location+age_class, 5*location+age_class2] += (
                            self.params['spore_within'] * self.params['inf_tanoak_tanoak'][age_class])
                    B[5*location+age_class, 5*location+4] += (
                        self.params['spore_within'] * self.params['inf_bay_to_tanoak'])

                    for loc2 in adjacent_locs:
                        for age_class2 in range(4):
                            B[5*location+age_class, 5*loc2+age_class2] += (
                                self.params['spore_between'] *
                                self.params['inf_tanoak_tanoak'][age_class])
                        B[5*location+age_class, 5*loc2+4] += (
                            self.params['spore_between'] * self.params['inf_bay_to_tanoak'])

                # Infection of bay:
                for age_class2 in range(4):
                    B[5*location+4, 5*location+age_class2] += (
                        self.params['spore_within'] * self.params['inf_tanoak_to_bay'])
                B[5*location+4, 5*location+4] += (
                    self.params['spore_within'] * self.params['inf_bay_to_bay'])

                for loc2 in adjacent_locs:
                    for age_class2 in range(4):
                        B[5*location+4, 5*loc2+age_class2] += (
                            self.params['spore_between'] * self.params['inf_tanoak_to_bay'])
                    B[5*location+4, 5*loc2+4] += (
                        self.params['spore_between'] * self.params['inf_bay_to_bay'])

        elif self.params.get("kernel_type", "nn") == "exp":
            kernel_range = self.params['kernel_range']
            # First find normalisation for kernel
            kernel = np.zeros((2*kernel_range+1, 2*kernel_range+1))
            for i in range(2*kernel_range+1):
                for j in range(2*kernel_range+1):
                    dist = np.sqrt(np.power(i-kernel_range, 2) + np.power(j-kernel_range, 2))
                    if dist <= kernel_range:
                        kernel[i, j] = np.exp(-dist / self.params['kernel_scale'])
            kernel[kernel_range, kernel_range] = 0.0

            norm_factor = (1.0 - self.params['spore_within']) / np.sum(kernel)

            for loc1 in range(self.ncells):
                loc1_coords = np.unravel_index(loc1, self.setup['landscape_dims'])

                # Within cell infection
                # Infection of tanoak:
                for age_class in range(4):
                    for age_class2 in range(4):
                        B[5*loc1+age_class, 5*loc1+age_class2] += (
                            self.params['spore_within'] * self.params['inf_tanoak_tanoak'][age_class])
                    B[5*loc1+age_class, 5*loc1+4] += (
                        self.params['spore_within'] * self.params['inf_bay_to_tanoak'])

                # Infection of bay:
                for age_class2 in range(4):
                    B[5*loc1+4, 5*loc1+age_class2] += (
                        self.params['spore_within'] * self.params['inf_tanoak_to_bay'])
                B[5*loc1+4, 5*loc1+4] += (
                    self.params['spore_within'] * self.params['inf_bay_to_bay'])

                # Between cell dynamics
                for loc2 in range(self.ncells):
                    loc2_coords = np.unravel_index(loc2, self.setup['landscape_dims'])
                    dist = np.sqrt(
                        np.power(loc1_coords[0]-loc2_coords[0], 2) +
                        np.power(loc1_coords[1]-loc2_coords[1], 2))
                    if (dist == 0.0) or (dist > kernel_range):
                        continue

                    kernel_val = norm_factor * np.exp(-dist / self.params['kernel_scale'])

                    # Infection of tanoak:
                    for age_class in range(4):
                        for age_class2 in range(4):
                            B[5*loc1+age_class, 5*loc2+age_class2] += (
                                kernel_val * self.params['inf_tanoak_tanoak'][age_class])
                        B[5*loc1+age_class, 5*loc2+4] += (
                            kernel_val * self.params['inf_bay_to_tanoak'])

                    # Infection of bay:
                    for age_class2 in range(4):
                        B[5*loc1+4, 5*loc2+age_class2] += (
                            kernel_val * self.params['inf_tanoak_to_bay'])
                    B[5*loc1+4, 5*loc2+4] += (kernel_val * self.params['inf_bay_to_bay'])

        else:
            raise ValueError("Unrecognised kernel type!")

        self._inf_matrix = B

        logging.debug("Completed infection matrix intialisation")

    def _get_recruit(self, state):
        """Return recruitment rates for given full state."""

        empty_space = self._get_space(state)

        # Reduceat sums columns 0:12, 12:14, and 14: to give recruitment rates for each species
        recruit_rates = np.add.reduceat(
            np.reshape(self._recruit_rates * state, (self.ncells, 15)),
            [0, 12, 14], axis=1).flatten()

        return recruit_rates * np.repeat(empty_space, 3)

    def _get_space(self, state):
        """Return empty space for given full state."""

        space_occupied = np.sum(np.reshape(self._space_weights * state, (self.ncells, 15)), axis=1)
        empty_space = np.maximum(0, 1.0 - space_occupied)

        return empty_space

    def state_deriv(self, time, state, control_func=None):
        """Return state derivative for 3 species model.

        control_function:   Function of time returning proportion of control rate allocated to each
                            control method: rogue small tan
                                            rogue large tan
                                            rogue bay
                                            thin small tan
                                            thin large tan
                                            thin bay
                                            thin red
                                            protect small tan
                                            protect large tan.
        """

        # Get state without integrated objective value
        state = state[:-1]

        d_state = np.zeros_like(state)

        # Linear part of dX
        for loc in range(self.ncells):
            d_state[15*loc:15*(loc+1)] += self._linear_matrix @ state[15*loc:15*(loc+1)]

        # Recruitment rates
        recruit = self._get_recruit(state)

        # TODO this matrix multiplication slowest calculation step currently - try looping over
        # locations as with linear matrix?
        # Calculate force of infection:
        inf_force = np.matmul(self._inf_matrix, state[1+self._indices['inf_s_idx']])

        d_state[self._indices['recruit_idx']] += recruit
        d_state[self._indices['inf_s_idx']] -= inf_force * state[self._indices['inf_s_idx']]
        d_state[1 + self._indices['inf_s_idx']] += inf_force * state[self._indices['inf_s_idx']]
        d_state[self._indices['tan_s_idx'] + 2] -= (
            inf_force[self._indices['tan_inf_force_idx']] * state[self._indices['tan_s_idx'] + 2] *
            self.params.get('treat_eff', 0.0))
        d_state[1 + self._indices['tan_s_idx']] += (
            inf_force[self._indices['tan_inf_force_idx']] * state[self._indices['tan_s_idx'] + 2] *
            self.params.get('treat_eff', 0.0))

        # Vaccine decay
        d_state[self._indices['tan_s_idx'] + 2] -= (
            self.params.get("vaccine_decay", 0.0) * state[self._indices['tan_s_idx'] + 2])
        d_state[self._indices['tan_s_idx']] += (
            self.params.get("vaccine_decay", 0.0) * state[self._indices['tan_s_idx'] + 2])

        if control_func is not None:
            control = control_func(time)
            averaged_state = np.sum(np.reshape(state, (self.ncells, 15)), axis=0) / self.ncells

            expense = utils.control_expenditure(control, self.params, averaged_state)

            if expense > self.params['max_budget']:
                control *= self.params['max_budget'] / expense

            control[0:3] *= self.params.get('rogue_rate', 0.0)
            control[3:7] *= self.params.get('thin_rate', 0.0)
            control[7:] *= self.params.get('protect_rate', 0.0)

            roguing = np.tile(
                np.array([control[0], control[0], control[1], control[1], control[2]]), self.ncells)

            protection = np.tile(
                np.array([control[7], control[7], control[8], control[8]]), self.ncells)

            # Roguing
            d_state[1 + self._indices['inf_s_idx']] -= roguing * state[1+self._indices['inf_s_idx']]

            # Thinning
            for i in range(6):
                d_state[i::15] -= control[3] * state[i::15]
                d_state[i+6::15] -= control[4] * state[i+6::15]
            d_state[12::15] -= control[5] * state[12::15]
            d_state[13::15] -= control[5] * state[13::15]
            d_state[14::15] -= control[6] * state[14::15]

            # Phosphonite protectant
            d_state[self._indices['tan_s_idx']] -= protection * state[self._indices['tan_s_idx']]
            d_state[self._indices['tan_s_idx']+2] += protection * state[self._indices['tan_s_idx']]

        else:
            control = np.zeros(9)

        d_obj = utils.objective_integrand(time, state, control, self.params)

        logging.debug("Calculated state derivative at time %f", time)

        return np.append(d_state, [d_obj])

    def run_policy(self, control_policy=None, n_fixed_steps=None, obj_start=None):
        """Run forward simulation using a given control policy.

        Function control_policy(t, X) returns list of budget allocations for each control

        n_fixed_steps:  If not None then an explicit RK4 scheme is used with this number of
                        internal steps between time points as default.
        """

        state_init = self._initialise()
        logging.info("Initialisation complete")

        if obj_start is None:
            obj_start = 0.0

        ode = integrate.ode(self.state_deriv)
        ode.set_integrator('vode', nsteps=1000, atol=1e-10, rtol=1e-8)
        ode.set_initial_value(np.append(state_init, [obj_start]), self.setup['times'][0])
        ode.set_f_params(control_policy)

        logging.info("Starting ODE run")

        ts = [self.setup['times'][0]]
        state = np.zeros((len(state_init), len(self.setup['times'])))
        state[:, 0] = state_init
        obj = [obj_start]

        # Loop over times and advance ODE system
        for i, time in enumerate(self.setup['times'][1:]):
            if n_fixed_steps is not None:
                # Use fixed steps
                t_old_int = ts[-1]
                state_old_int = np.append(state[i], obj[-1])
                for t_int in np.linspace(ts[-1], time, n_fixed_steps+2)[1:]:
                    h = t_int - t_old_int
                    k1 = h * self.state_deriv(t_int, state_old_int, control_policy)
                    k2 = h * self.state_deriv(t_int + 0.5 * h, state_old_int + 0.5 * k1,
                                              control_policy)
                    k3 = h * self.state_deriv(t_int + 0.5 * h, state_old_int + 0.5 * k2,
                                              control_policy)
                    k4 = h * self.state_deriv(t_int + h, state_old_int + k3, control_policy)

                    state_old_int = np.clip(state_old_int + (k1 + k2 + k2 + k3 + k3 + k4) / 6,
                                            0.0, None)
                    t_old_int = t_int

                state[:, i+1] = state_old_int[:-1]
                obj.append(state_old_int[-1])
                ts.append(time)

            else:
                # use adaptive step solver
                if ode.successful():
                    ode.integrate(time)
                    ts.append(ode.t)
                    state[:, i+1] = ode.y[:-1]
                    obj.append(ode.y[-1])
                else:
                    logging.error("ODE solver error!")
                    raise RuntimeError("ODE solver error!")

        logging.info("ODE run completed")

        self.run['state'] = state
        if control_policy is None:
            self.run['control'] = None
        else:
            self.run['control'] = np.array([control_policy(t) for t in self.setup['times']]).T
        self.run['objective'] = utils.objective_payoff(ts[-1], state[:, -1], self.params) + obj[-1]

        return state, self.run['objective'], obj
