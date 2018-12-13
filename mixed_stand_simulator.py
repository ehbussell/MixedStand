"""
Model for tanoak-bay laurel-redwood mixed stand SOD from Cobb 2012.

Species are labelled as:
    1) Tanoak
    2) Bay Laurel
    3) Redwood
"""

import argparse
from enum import IntEnum
import copy
import warnings
import pickle
import pdb
import numpy as np
from scipy.interpolate import interp1d
from scipy import integrate
import matplotlib.pyplot as plt
from matplotlib import animation
import visualisation

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    args = parser.parse_args()


class Species(IntEnum):
    """Host species."""
    TANOAK = 0
    BAY = 1
    REDWOOD = 2


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

        for key in required_keys:
            if key not in setup:
                raise KeyError("Setup Parameter {0} not found!".format(key))

        self.setup = {k: setup[k] for k in required_keys}

        for key in setup:
            if key not in required_keys:
                warnings.warn("Unused setup parameter: {0}".format(key))

        self.params = copy.deepcopy(params)
        self._linear_matrix = None
        self._inf_matrix = None
        self._indices = None
        self._space_weights = None
        self.ncells = np.prod(self.setup['landscape_dims'])
        self.run = {
            'state': None,
            'control': None,
            'objective': None
        }

    def print_msg(self, msg):
        """Print message from class with class identifier."""
        identifier = "[" + self.__class__.__name__ + "]"
        print("{0:<20}{1}".format(identifier, msg))

    def save_run(self, filename):
        """Save run_data, control and run parameters to file."""

        dump_obj = {
            'run': self.run,
            'setup': self.setup,
            'params': self.params
        }

        with open(filename, "wb") as outfile:
            pickle.dump(dump_obj, outfile)

    def load_run(self, filename):
        """Load run_data, control and run parameters."""

        with open(filename, "rb") as infile:
            load_obj = pickle.load(infile)

        self.run = load_obj['run']
        self.setup = load_obj['setup']
        self.params = load_obj['params']

    def _initialise(self):
        """Initialise ready for simulation - set up initial conditions, matrices and rates."""

        if len(self.setup['state_init']) == 15:
            # Uniform initialisation across landscape
            state_init = np.tile(self.setup['state_init'], self.ncells)
        else:
            # User specified landscape initialisation
            # Check dimensions consistent
            if len(self.setup['state_init']) / 15 != self.ncells:
                raise ValueError("Incorrect length of state initialisation array!")
            state_init = self.setup['state_init']

        # Initialise rates s.t. if initial state was disease free, it is in dynamic equilibrium
        # Follows calculation in Cobb (2012)
        avg_state_init = np.sum(np.reshape(state_init, (self.ncells, 15)), axis=0) / self.ncells
        avg_df_state_init = np.array(
            [np.sum(avg_state_init[3*i:3*i+3]) for i in range(4)] +
            [avg_state_init[12] + avg_state_init[13]] + [avg_state_init[14]])

        # Space weights:
        if np.all(avg_df_state_init[:4] > 0):
            if np.all(np.isnan(self.params['space_tanoak'])):
                self.params['space_tanoak'] = 0.25 * np.sum(
                    avg_df_state_init[:4]) / avg_df_state_init[:4]
        else:
            self.params['space_tanoak'] = np.repeat(0.0, 4)

        # Recruitment rates:
        # Any recruitment rates that are nan in parameters are chosen to give dynamic equilibrium
        space_at_start = (1.0 - np.sum(self.params['space_tanoak'] * avg_df_state_init[:4]) -
                          self.params['space_bay'] * avg_df_state_init[4] -
                          self.params['space_redwood'] * avg_df_state_init[5])

        if np.isnan(self.params['recruit_bay']):
            self.params['recruit_bay'] = self.params['nat_mort_bay'] / space_at_start

        if np.isnan(self.params['recruit_redwood']):
            self.params['recruit_redwood'] = self.params['nat_mort_redwood'] / space_at_start

        if np.isnan(self.params['recruit_tanoak'][0]):
            A2 = self.params['trans_tanoak'][0] / (
                self.params['trans_tanoak'][1] + self.params['nat_mort_tanoak'][1])
            A3 = A2 * self.params['trans_tanoak'][1] / (
                self.params['trans_tanoak'][2] + self.params['nat_mort_tanoak'][2])
            A4 = A3 * self.params['trans_tanoak'][2] / self.params['nat_mort_tanoak'][3]

            self.params['recruit_tanoak'][0] = (
                (self.params['trans_tanoak'][0] + self.params['nat_mort_tanoak'][0]) /
                space_at_start - np.sum(self.params['recruit_tanoak'][1:] * np.array([A2, A3, A4])))

        # Construct infection and transition matrices
        self._construct_matrices()

        self._indices = {
            'all_s_idx': np.array([15*loc+np.array([0, 3, 6, 9, 12, 14])
                                   for loc in range(self.ncells)]).flatten(),
            'inf_s_idx': np.array(
                [15*loc+np.arange(14, step=3) for loc in range(self.ncells)]).flatten(),
            'tan_s_idx': np.array(
                [15*loc+np.arange(12, step=3) for loc in range(self.ncells)]).flatten(),
            'tan_inf_force_idx': np.array(
                [5*loc+np.arange(4) for loc in range(self.ncells)]).flatten(),
            'recruit_idx': np.array([15*loc+np.array([0, 12, 14])
                                     for loc in range(self.ncells)]).flatten(),
        }

        self._space_weights = np.tile(np.append(
            np.repeat(self.params['space_tanoak'], 3),
            [self.params['space_bay'],self.params['space_bay'], self.params['space_redwood']]),
            self.ncells)
        
        self._recruit_rates = np.tile(np.append(
            np.repeat(self.params['recruit_tanoak'], 3),
            [self.params['recruit_bay'],self.params['recruit_bay'], self.params['recruit_redwood']]),
            self.ncells)

        return state_init

    def _get_state_idx(self, species, age_class, location):
        if species == Species.TANOAK:
            index = 3*age_class
        elif species == Species.BAY:
            index = 12
        elif species == Species.REDWOOD:
            index = 14
        else:
            raise ValueError("Unrecognised species!")

        index += 15 * location
        return index

    def _construct_matrices(self):
        """Construct matrices for calculation of equations RHS (linear parts and infection)."""

        A = np.zeros((15, 15))

        # Resprouting
        i = self._get_state_idx(Species.TANOAK, 0, 0)
        for age_class in range(4):
            j = self._get_state_idx(Species.TANOAK, age_class, 0) + 1
            A[i, j] += self.params['resprout_tanoak']*self.params['inf_mort_tanoak'][age_class]

        # Mortality and recovery (tanoak)
        for age_class in range(4):
            i = self._get_state_idx(Species.TANOAK, age_class, 0)
            j = self._get_state_idx(Species.TANOAK, age_class, 0) + 1
            k = self._get_state_idx(Species.TANOAK, age_class, 0) + 2
            A[i, i] += -self.params['nat_mort_tanoak'][age_class]
            A[j, j] += (-self.params['nat_mort_tanoak'][age_class]
                        -self.params['inf_mort_tanoak'][age_class]
                        -self.params['recov_tanoak'])
            A[i, j] += self.params['recov_tanoak']
            A[k, k] += -self.params['nat_mort_tanoak'][age_class]

        # Mortality and recovery (bay)
        i = self._get_state_idx(Species.BAY, 0, 0)
        j = self._get_state_idx(Species.BAY, 0, 0) + 1
        A[i, i] += -self.params['nat_mort_bay']
        A[j, j] += (-self.params['nat_mort_bay'] - self.params['recov_bay'])
        A[i, j] += self.params['recov_bay']

        # Mortality and recovery (redwood)
        i = self._get_state_idx(Species.REDWOOD, 0, 0)
        A[i, i] += -self.params['nat_mort_redwood']

        # Age transitions
        for age_class in range(3):
            i = self._get_state_idx(Species.TANOAK, age_class, 0)
            j = self._get_state_idx(Species.TANOAK, age_class+1, 0)
            A[i, i] += -self.params['trans_tanoak'][age_class]
            A[j, i] += self.params['trans_tanoak'][age_class]
            A[i+1, i+1] += -self.params['trans_tanoak'][age_class]
            A[j+1, i+1] += self.params['trans_tanoak'][age_class]
            A[i+2, i+2] += -self.params['trans_tanoak'][age_class]
            A[j+2, i+2] += self.params['trans_tanoak'][age_class]

        # Primary infection
        for age_class in range(4):
            i = self._get_state_idx(Species.TANOAK, age_class, 0)
            A[i, i] -= self.params.get('primary_inf', 0.0)
            A[i+1, i] += self.params.get('primary_inf', 0.0)
            A[i+2, i+2] -= (self.params.get('primary_inf', 0.0) *
                            self.params.get('treat_eff', 0.0))
            A[i+1, i+2] += (self.params.get('primary_inf', 0.0) *
                            self.params.get('treat_eff', 0.0))
        i = self._get_state_idx(Species.BAY, 0, 0)
        A[i, i] -= self.params.get('primary_inf', 0.0)
        A[i+1, i] += self.params.get('primary_inf', 0.0)

        self._linear_matrix = A

        B = np.zeros((5*self.ncells, 5*self.ncells))

        for location in range(self.ncells):
            # Find adjacent cells
            loc_coords = np.unravel_index(location, self.setup['landscape_dims'])
            adjacent_coords = []

            if self.params['num_nn'] == 4:
                for row_change in [-1, 1]:
                    for col_change in [0]:
                        if not (row_change == 0 and col_change == 0):
                            new_coord = np.add(loc_coords, (row_change, col_change))
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

        self._inf_matrix = B

    def _get_recruit(self, state):
        """Return recruitment rates for given full state."""

        empty_space = self._get_space(state)

        recruit_rates = np.add.reduceat(
            np.reshape(self._recruit_rates * state, (self.ncells, 15)), [0,12,14], axis=1).flatten()

        return recruit_rates * np.repeat(empty_space, 3)

    def _get_space(self, state):
        """Return space for given full state."""
        
        space_occupied = np.sum(np.reshape(self._space_weights * state, (self.ncells, 15)), axis=1)
        empty_space = np.maximum(0, 1.0 - space_occupied)

        return empty_space


    def state_deriv(self, time, state, control_func=None):
        """Return state derivative for 3 species model.

        cull_function:      Function of time returning proportion of cull rate allocated to each
                            state class.
        treat_function:     Function of time returning proportion of treat rate allocated to
                            vaccinate each tanoak age class.
        """

        d_state = np.zeros_like(state)

        for loc in range(self.ncells):
            d_state[15*loc:15*(loc+1)] += self._linear_matrix @ state[15*loc:15*(loc+1)]

        recruit = self._get_recruit(state)

        # TODO this matrix multiplication slowest calculation step currently - change to loop over
        # locations as with linear matrix.
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
            control = control_func(time) * self.params.get('control_rate', 0.0)

            # bay_tot = np.sum(state[12::15]) + np.sum(state[13::15])
            # if bay_tot > 0.0:
            #     control[3] = control[3] / bay_tot
            # else:
            #     control[3] = 0.0

            # control[4] = np.minimum(control[4], 1000000000*(np.sum(state[14::15]) > 0))

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

        return d_state

    def run_policy(self, control_policy=None, n_fixed_steps=None):
        """Run forward simulation using a given control policy.

        Function control_policy(t, X) returns list of budget allocations for
            each region - [budget_S1, budget_I1, budget_S2, budget_I2, budget_S3]

        n_fixed_steps:  If not None then an explicit RK4 scheme is used with this number of
                        internal steps between time points as default.
        """

        state_init = self._initialise()

        ode = integrate.ode(self.state_deriv)
        ode.set_integrator('lsoda', nsteps=1000, atol=1e-10, rtol=1e-8)
        ode.set_initial_value(state_init, self.setup['times'][0])
        ode.set_f_params(control_policy)

        ts = [self.setup['times'][0]]
        xs = [state_init]

        for time in self.setup['times'][1:]:
            if n_fixed_steps is not None:
                t_old_int = ts[-1]
                state_old_int = xs[-1]
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

                xs.append(state_old_int)
                ts.append(t_int)

            else:
                if ode.successful():
                    ode.integrate(time)
                    ts.append(ode.t)
                    xs.append(ode.y)
                else:
                    pdb.set_trace()
                    raise RuntimeError("ODE solver error!")

        state = np.vstack(xs).T

        self.run['state'] = state
        self.run['control'] = np.array([control_policy(t) for t in self.setup['times']]).T

        return state

class MixedStandAnimator:
    """Plotting object for MixedStandSimulator results."""

    def __init__(self, simulator):
        self.simulator = simulator

    @staticmethod
    def _default_plot_func(state):
        """Default plot proportion of hosts infected."""

        total_inf = np.sum(state[1::3])
        total = np.sum(state)
        return total_inf / total

    def make_animation(self, plot_function=None, video_length=10, save_file=None, **kwargs):
        """Plot spatial animation of diseased proportion over time.

        plot_function:  If specified this function takes current state of a single cell and returns
                        the desired attribute to plot on the map. By default plots proportion of all
                        hosts that are infected.
        kwargs:         Keyword arguments passed to pcolormesh
        """

        if self.simulator.run['state'] is None:
            raise RuntimeError("No run has been simulated!")

        if plot_function is None:
            plot_function = self._default_plot_func

        run_data = interp1d(self.simulator.setup['times'], self.simulator.run['state'])
        fps = 30
        nframes = fps * video_length
        times = np.linspace(
            self.simulator.setup['times'][0], self.simulator.setup['times'][-1], nframes)

        # Setup plotting data
        dataset = np.zeros((nframes, *self.simulator.setup['landscape_dims']))
        for i, time in enumerate(times):
            dataset[i] = np.apply_along_axis(
                plot_function, 1, run_data(time).reshape((self.simulator.ncells, 15))).reshape(
                    self.simulator.setup['landscape_dims'])

        fig = plt.figure()
        ax = fig.add_subplot(111)
        vmin = kwargs.pop('vmin', 0)
        vmax = kwargs.pop('vmax', 1)

        im = ax.pcolormesh(dataset[0, :, :], vmin=vmin, vmax=vmax, **kwargs)
        fig.colorbar(im, ax=ax)
        fig.tight_layout()

        time_template = 'time = {0:.1f}'
        time_text = ax.text(0.05, 0.055, time_template.format(times[0]), transform=ax.transAxes,
                            bbox={'facecolor':'w', 'alpha':0.5, 'pad':5})
        time_text.set_animated(True)

        def update(frame_number):
            im.set_array(dataset[frame_number].ravel())
            time_text.set_text(time_template.format(times[frame_number]))

            return im, time_text

        im_ani = animation.FuncAnimation(fig, update, interval=1000*video_length/nframes,
                                         frames=nframes, blit=True, repeat=True)

        if save_file is not None:
            Writer = animation.writers['ffmpeg']
            writer = Writer(fps=fps, metadata=dict(artist='Me'), bitrate=1800, codec="h264")
            im_ani.save(save_file+'.mp4', writer=writer, dpi=300)

        return im_ani

    def plot_hosts(self, ax=None, proportions=True, combine_ages=True, **kwargs):
        """Plot simulator host numbers as a function of time."""

        if self.simulator.run['state'] is None:
            raise RuntimeError("No run has been simulated!")

        if ax is None:
            fig = plt.figure(111)
            ax = fig.add_subplot(111)

        ncells = np.product(self.simulator.setup['landscape_dims'])

        return visualisation.plot_hosts(
            self.simulator.setup['times'],
            np.sum(np.reshape(self.simulator.run['state'], (ncells, 15, -1)), axis=0) / ncells,
            ax=ax, combine_ages=combine_ages, **kwargs)

    def plot_dpcs(self, ax=None, proportions=True, combine_ages=True, **kwargs):
        """Plot simulator disease progress curves as a function of time."""

        if self.simulator.run['state'] is None:
            raise RuntimeError("No run has been simulated!")

        if ax is None:
            fig = plt.figure(111)
            ax = fig.add_subplot(111)

        ncells = np.product(self.simulator.setup['landscape_dims'])

        return visualisation.plot_dpcs(
            self.simulator.setup['times'],
            np.sum(np.reshape(self.simulator.run['state'], (ncells, 15, -1)), axis=0) / ncells,
            ax=ax, combine_ages=combine_ages, **kwargs)
