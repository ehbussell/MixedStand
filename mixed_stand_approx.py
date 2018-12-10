"""
Approximate model for tanoak-bay laurel-redwood mixed stand SOD .

Species are labelled as:
    1) Tanoak
    2) Bay Laurel
    3) Redwood
"""

import argparse
from enum import IntEnum
import copy
import pdb
import subprocess
import itertools
import os
import warnings
from scipy.interpolate import interp1d
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib as mpl
from scipy import integrate
from scipy.optimize import minimize
import bocop_utils
import mixed_stand_simulator as ms_sim
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

class MixedStandApprox:
    """
    Class to implement an approximate 3 species model of SOD within a forest stand.

    Initialisation requires the following setup dictionary of values:
        'state_init':       Length 15 array giving [S11, I11, V11 ... S2 I2 S3].
        'times':            Times to solve for.

    and parameter object. Some parameters will be taken from Cobb model, others require fitting.
    """

    # TODO add initialisation function

    def __init__(self, setup, params, fit):
        required_keys = ['state_init', 'times']

        for key in required_keys:
            if key not in setup:
                raise KeyError("Setup Parameter {0} not found!".format(key))

        self.setup = {'times': setup['times']}

        for key in setup:
            if key not in required_keys:
                warnings.warn("Unused setup parameter: {0}".format(key))

        self.set_state_init(setup['state_init'])

        self.params = copy.deepcopy(params)

        # If necessary initialise space weights and recruitment rates to give dynamic equilibrium
        df_state_init = np.array(
            [np.sum(self.setup['state_init'][3*i:3*i+3]) for i in range(4)] +
            [self.setup['state_init'][12] + self.setup['state_init'][13]] +
            [self.setup['state_init'][14]])

        # Space weights:
        if np.all(df_state_init[:4] > 0):
            if np.all(np.isnan(self.params['space_tanoak'])):
                self.params['space_tanoak'] = 0.25 * np.sum(
                    df_state_init[:4]) / df_state_init[:4]
        else:
            self.params['space_tanoak'] = np.repeat(0.0, 4)

        # Recruitment rates
        # Any recruitment rates that are nan in parameters are chosen to give dynamic equilibrium
        space_at_start = (1.0 - np.sum(self.params['space_tanoak'] * df_state_init[:4]) -
                          self.params['space_bay'] * df_state_init[4] -
                          self.params['space_redwood'] * df_state_init[5])

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

        if isinstance(fit, dict):
            self.beta = np.zeros(7)
            self.beta[0] = fit['beta_1,1']
            self.beta[1] = fit['beta_1,2']
            self.beta[2] = fit['beta_1,3']
            self.beta[3] = fit['beta_1,4']
            self.beta[4] = fit['beta_12']
            self.beta[5] = fit['beta_21']
            self.beta[6] = fit['beta_2']
        elif len(fit) == 7:
            self.beta = fit
        else:
            raise ValueError("Wrong format for fitted parameters!")

        self.run_data = None

    def print_msg(self, msg):
        """Print message from class with class identifier."""
        identifier = "[" + self.__class__.__name__ + "]"
        print("{0:<20}{1}".format(identifier, msg))
    
    def set_state_init(self, state_init):
        """Set the initial state for future runs."""

        if len(state_init) != 15:
            ncells = len(state_init) / 15

            self.setup['state_init'] = np.sum(
                np.reshape(state_init, (int(ncells), 15)), axis=0) / ncells
        
        else:
            self.setup['state_init'] = state_init

    def _objective_integrand(self, time, state, control):
        """Integrand of objective function, including control costs and diversity costs."""

        props = np.divide(np.array([np.sum(state[0:6]), np.sum(state[6:12]),
                                    state[12] + state[13], state[14]]),
                          np.sum(state[0:15]), out=np.zeros(4), where=(np.sum(state[0:15]) > 0.0))
        div_cost = np.sum(props * np.log(props, out=np.zeros_like(props), where=(props > 0.0)))

        integrand = np.exp(- self.params.get('discount_rate', 0.0) * time) * (
            self.params.get('cull_cost', 0.0) * self.params.get('control_rate', 0.0) * (
                control[0] * (state[1] + state[4]) + control[1] * (state[7] + state[10]) +
                control[2] * state[13] + control[3] + control[4]
            ) + self.params.get('protect_cost', 0.0) * self.params.get('control_rate', 0.0) * (
                control[5] * (state[0] + state[3]) + control[6] * (state[6] + state[9])
            ) + self.params.get('div_cost', 0.0) * div_cost
        )

        return integrand

    def _terminal_cost(self, state):
        """Payoff term in objective function"""

        payoff = - self.params.get('payoff_factor', 0.0) * np.exp(
            - self.params.get('discount_rate', 0.0) * self.setup['times'][-1]) * (
                state[6] + state[8] + state[9] + state[11])

        return payoff

    def state_deriv(self, time, state, control_func=None):
        """Return state derivative for 3 species model, including integrated objective function.

        cull_func:      Function of time giving proportion of culling effort to allocate to each
                        state (length 15 array, non-spatial strategy).
        treat_func:     Function of time giving proportion of treatment effort to allocate to each
                        tanoak susceptible age class (length 4 array, non-spatial strategy).
        """

        d_state = np.zeros(16)

        tanoak_totals = np.sum(state[0:12].reshape((4, 3)), axis=1)

        empty_space = np.max([0, (
            1.0 - np.sum(self.params['space_tanoak'] * tanoak_totals) -
            self.params['space_bay'] * (state[12] + state[13]) -
            self.params['space_redwood'] * state[14])])

        # Births
        for age in range(4):
            d_state[0] += self.params['recruit_tanoak'][age] * np.sum(
                state[3*age:3*age+3]) * empty_space
        d_state[12] += self.params['recruit_bay'] * (state[12] + state[13]) * empty_space
        d_state[14] += self.params['recruit_redwood'] * state[14] * empty_space

        # Natural deaths
        nat_mort_rates = np.append(
            np.repeat(self.params['nat_mort_tanoak'], 3),
            np.append(np.repeat(self.params['nat_mort_bay'], 2), [self.params['nat_mort_redwood']]))
        for i in range(15):
            d_state[i] -= nat_mort_rates[i] * state[i]

        # Disease induced deaths and resprouting
        for age in range(4):
            d_state[3*age+1] -= self.params['inf_mort_tanoak'][age] * state[3*age+1]
            d_state[0] += (self.params['resprout_tanoak'] *
                           self.params['inf_mort_tanoak'][age] * state[3*age+1])

        # Age transitions
        for age in range(3):
            d_state[3*age] -= self.params['trans_tanoak'][age] * state[3*age]
            d_state[3*(age+1)] += self.params['trans_tanoak'][age] * state[3*age]
            d_state[3*age+1] -= self.params['trans_tanoak'][age] * state[3*age+1]
            d_state[3*(age+1)+1] += self.params['trans_tanoak'][age] * state[3*age+1]
            d_state[3*age+2] -= self.params['trans_tanoak'][age] * state[3*age+2]
            d_state[3*(age+1)+2] += self.params['trans_tanoak'][age] * state[3*age+2]

        # Recovery
        for age in range(4):
            d_state[3*age] += self.params['recov_tanoak'] * state[3*age+1]
            d_state[3*age+1] -= self.params['recov_tanoak'] * state[3*age+1]
        d_state[12] += self.params['recov_bay'] * state[13]
        d_state[13] -= self.params['recov_bay'] * state[13]

        # Infection
        for age in range(4):
            inf_rate = (
                self.params.get("primary_inf", 0.0) +
                (self.beta[age] * np.sum(state[1:12:3]) + self.beta[4] * state[13]))
            d_state[3*age] -= state[3*age] * inf_rate
            d_state[3*age+1] += inf_rate * (
                state[3*age] + state[3*age+2] * self.params.get("treat_eff", 0.0))
            d_state[3*age+2] -= state[3*age+2] * inf_rate * self.params.get("treat_eff", 0.0)
        inf_rate = (
            self.params.get("primary_inf", 0.0) * state[12] +
            state[12] * (self.beta[5] * np.sum(state[1:12:3]) + self.beta[6] * state[13]))
        d_state[12] -= inf_rate
        d_state[13] += inf_rate

        if control_func is not None:
            control = control_func(time) * self.params.get('control_rate', 0.0)
            # control[3] = np.minimum(control[3], 10000000*(state[12] > 0))
            # control[4] = np.minimum(control[4], 10000000*(state[14] > 0))

            # Roguing
            d_state[1] -= control[0] * state[1]
            d_state[4] -= control[0] * state[4]
            d_state[7] -= control[1] * state[7]
            d_state[10] -= control[1] * state[10]
            d_state[13] -= control[2] * state[13]

            # Thinning
            d_state[12] -= control[3] * state[12]
            d_state[13] -= control[3] * state[13]
            d_state[14] -= control[4] * state[14]

            # Phosphonite protectant
            d_state[0] -= control[5] * state[0]
            d_state[2] += control[5] * state[0]
            d_state[3] -= control[5] * state[3]
            d_state[5] += control[5] * state[3]
            d_state[6] -= control[6] * state[6]
            d_state[8] += control[6] * state[6]
            d_state[9] -= control[6] * state[9]
            d_state[11] += control[6] * state[9]
        else:
            control = np.zeros(7)

        # Objective function
        d_state[-1] = self._objective_integrand(time, state, control)

        return d_state

    def run_policy(self, control_policy=None, n_fixed_steps=None):
        """Run forward simulation using a given control policy.

        Function control_policy(t)
        """

        ode = integrate.ode(self.state_deriv)
        ode.set_integrator('lsoda', nsteps=10000, atol=1e-10, rtol=1e-8)
        ode.set_initial_value(np.append(self.setup['state_init'], [0.0]), self.setup['times'][0])
        ode.set_f_params(control_policy)

        ts = [self.setup['times'][0]]
        xs = [self.setup['state_init']]

        for time in self.setup['times'][1:]:
            if n_fixed_steps is not None:
                t_old_int = ts[-1]
                state_old_int = np.append(xs[-1], [0.0])
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

                xs.append(state_old_int[:-1])
                ts.append(t_int)

            else:
                if ode.successful():
                    ode.integrate(time)
                    ts.append(ode.t)
                    xs.append(ode.y[:-1])
                else:
                    pdb.set_trace()
                    raise RuntimeError("ODE solver error!")

        X = np.vstack(xs).T
        self.run_data = X

        self.run_objective = self._terminal_cost(xs[-1]) + ode.y[-1]

        return X, self.run_objective

    def optimise(self, bocop_dir=None, verbose=True, init_policy=None, n_stages=None):
        """Run BOCOP optimisation of control.

        If n_stages is not None, then control will be piecewise constant with this number of stages
        """

        if bocop_dir is None:
            bocop_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "BOCOP")

        if init_policy is None:
            initialisation, _ = self.run_policy(None)
        else:
            initialisation, _ = self.run_policy(init_policy, n_fixed_steps=999)

        self._set_bocop_params(init=initialisation, folder=bocop_dir, n_stages=n_stages)

        if verbose is True:
            subprocess.run([os.path.join(bocop_dir, "bocop.exe")], cwd=bocop_dir)
        else:
            subprocess.run([os.path.join(bocop_dir, "bocop.exe")],
                           cwd=bocop_dir, stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL)

        state_t, _, control_t, exit_text = bocop_utils.readSolFile(
            os.path.join(bocop_dir, "problem.sol"), ignore_fail=True)

        actual_states = np.array([state_t(t) for t in self.setup['times']]).T

        if n_stages is not None:
            # Return control as piecewise constant
            actual_control = np.array([control_t(t) for t in self.setup['times'][:-1]]).T
            control_t = interp1d(
                self.setup['times'][:-1], actual_control, kind="zero", fill_value="extrapolate")

        return (actual_states, control_t, exit_text)

    def plot_hosts(self, ax=None, proportions=True, combine_ages=True, **kwargs):
        """Plot host numbers as a function of time."""

        if self.run_data is None:
            raise RuntimeError("No run has been simulated!")

        if ax is None:
            fig = plt.figure(111)
            ax = fig.add_subplot(111)

        return visualisation.plot_hosts(
            self.setup['times'], self.run_data, ax=ax, combine_ages=combine_ages, **kwargs)

    def plot_dpcs(self, ax=None, proportions=True, combine_ages=True, **kwargs):
        """Plot simulator disease progress curves as a function of time."""

        if self.run_data is None:
            raise RuntimeError("No run has been simulated!")

        if ax is None:
            fig = plt.figure(111)
            ax = fig.add_subplot(111)

        return visualisation.plot_dpcs(
            self.setup['times'], self.run_data, ax=ax, combine_ages=combine_ages, **kwargs)

    def _set_bocop_params(self, init=None, folder="BOCOP", n_stages=None):
        """Save parameters and initial conditions to file for BOCOP optimisation."""

        with open(os.path.join(folder, "problem.bounds"), "r") as infile:
            all_lines = infile.readlines()

        # Initial conditions
        for i in range(15):
            all_lines[9+i] = str(self.setup['state_init'][i]) + " " + str(
                self.setup['state_init'][i]) + " equal\n"

        # When no integrated term in objective, set bounds on integrand to zero to aid convergence
        if (self.params.get('div_cost', 0.0) == 0.0 and
                self.params.get('cull_cost', 0.0) == 0.0 and
                self.params.get('protect_cost', 0.0) == 0.0):
            all_lines[42] = "0.0 0.0 both\n"
        else:
            all_lines[42] = "-1e6 1e6 both\n"

        n_optim_vars = 0

        # Setup correct bounds for optimisation variables and path constraints
        if n_stages is None:
            all_lines[6] = "16 16 7 0 0 1\n"

            del all_lines[56:]
            all_lines.append("\n# Bounds for the path constraints :\n")
            all_lines.append("-2e+020 1 upper\n")
        else:
            n_optim_vars = n_stages * 7
            all_lines[6] = "16 16 7 0 " + str(n_optim_vars) + " 8\n"

            del all_lines[56:]
            for i in range(n_optim_vars):
                all_lines.append("0 1 both\n")

            all_lines.append("\n# Bounds for the path constraints :\n")
            all_lines.append("-2e+020 1 upper\n")
            for i in range(7):
                all_lines.append("0 0 equal\n")

        with _try_file_open(os.path.join(folder, "problem.bounds")) as outfile:
            outfile.writelines(all_lines)

        # Constant values
        with open(os.path.join(folder, "problem.constants"), "r") as infile:
            all_lines = infile.readlines()

        for i in range(7):
            all_lines[5+i] = str(self.beta[i]) + "\n"

        for i in range(4):
            all_lines[12+i] = str(self.params['space_tanoak'][i]) + "\n"
        all_lines[16] = str(self.params['space_bay']) + "\n"
        all_lines[17] = str(self.params['space_redwood']) + "\n"

        for i in range(4):
            all_lines[18+i] = str(self.params['recruit_tanoak'][i]) + "\n"
        all_lines[22] = str(self.params['recruit_bay']) + "\n"
        all_lines[23] = str(self.params['recruit_redwood']) + "\n"

        for i in range(4):
            all_lines[24+i] = str(self.params['nat_mort_tanoak'][i]) + "\n"
        all_lines[28] = str(self.params['nat_mort_bay']) + "\n"
        all_lines[29] = str(self.params['nat_mort_redwood']) + "\n"

        for i in range(4):
            all_lines[30+i] = str(self.params['inf_mort_tanoak'][i]) + "\n"

        all_lines[34] = str(self.params['resprout_tanoak']) + "\n"

        for i in range(3):
            all_lines[35+i] = str(self.params['trans_tanoak'][i]) + "\n"

        all_lines[38] = str(self.params['recov_tanoak']) + "\n"
        all_lines[39] = str(self.params['recov_bay']) + "\n"
        all_lines[40] = str(self.params.get('primary_inf', 0.0)) + "\n"
        all_lines[41] = str(self.params.get('control_rate', 0.0)) + "\n"
        all_lines[42] = str(self.params.get('treat_eff', 0.0)) + "\n"
        all_lines[43] = str(self.params.get('div_cost', 0.0)) + "\n"
        all_lines[44] = str(self.params.get('cull_cost', 0.0)) + "\n"
        all_lines[45] = str(self.params.get('protect_cost', 0.0)) + "\n"
        all_lines[46] = str(self.params.get('discount_rate', 0.0)) + "\n"
        all_lines[47] = str(self.params.get('payoff_factor', 0.0)) + "\n"

        if n_stages is None:
            all_lines[48] = "0\n"
        else:
            all_lines[48] = str(n_stages) + "\n"

        with _try_file_open(os.path.join(folder, "problem.constants")) as outfile:
            outfile.writelines(all_lines)

        with open(os.path.join(folder, "problem.def"), "r") as infile:
            all_lines = infile.readlines()

        n_steps = str(len(self.setup['times']) - 1)
        all_lines[5] = "time.initial double " + str(self.setup['times'][0]) + "\n"
        all_lines[6] = "time.final double " + str(self.setup['times'][-1]) + "\n"

        if n_stages is None:
            all_lines[12] = "parameter.dimension integer 0\n"
            all_lines[15] = "constraint.dimension integer 1\n"
        else:
            all_lines[12] = "parameter.dimension integer " + str(n_optim_vars) + "\n"
            all_lines[15] = "constraint.dimension integer 8\n"

        all_lines[18] = "discretization.steps integer " + n_steps + "\n"

        with _try_file_open(os.path.join(folder, "problem.def")) as outfile:
            outfile.writelines(all_lines)

        # # Initialisation
        # control_init = np.array([[init.control(t)[j] for j in range(2)] for t in params['times']])

        # for control in range(2):
        #     all_lines = [
        #         "#Starting point file\n",
        #         "# This file contains the values of the initial points\n",
        #         "# for variable control #{0}\n".format(control), "\n", "# Type of initialization :\n",
        #         "linear\n", "\n", "# Number of interpolation points :\n",
        #         "{0}\n".format(len(params['times'])), "\n", "# Interpolation points :\n"]

        #     for i, time in enumerate(params['times']):
        #         all_lines.append("{0} {1}\n".format(time,
        #                                             np.round(control_init[i, control], decimals=2)))

        #     with _try_file_open(os.path.join(folder, "init",
        #                                     "control." + str(control) + ".init")) as outfile:
        #         outfile.writelines(all_lines)

        for state in range(15):
            all_lines = [
                "#Starting point file\n",
                "# This file contains the values of the initial points\n",
                "# for variable state #{0}\n".format(state), "\n", "# Type of initialization :\n",
                "linear\n", "\n", "# Number of interpolation points :\n",
                "{0}\n".format(len(self.setup['times'])), "\n", "# Interpolation points :\n"]

            for i, time in enumerate(self.setup['times']):
                all_lines.append("{0} {1}\n".format(time,
                                                    np.round(init[state, i], decimals=3)))

            with _try_file_open(os.path.join(folder, "init", "state."+str(state)+".init")) as outf:
                outf.writelines(all_lines)

        optim_vars_init = [
            "#Starting point file\n",
            "# This file contains the values of the initial points\n",
            "# for the optimisation variables\n", "\n", "# Number of optimization variables : \n",
            "{0}\n".format(n_optim_vars), "\n", "Default values for the starting point : \n"
        ]

        for i in range(n_optim_vars):
            optim_vars_init.append("0.1\n")

        with _try_file_open(os.path.join(folder, "init", "optimvars.init")) as outf:
            outf.writelines(optim_vars_init)

class MixedStandFitter:
    """Fitting of MixedStandApprox model to simulation data."""

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
        self.beta = None

    def fit(self, start, bounds, show_plot=False):
        """Fit infection rate parameters, minimising sum of squared errors from simulation DPCs."""

        # First get simulator data
        simulator = ms_sim.MixedStandSimulator(self.setup, self.params)
        sim_run = simulator.run_policy(None)

        ncells = np.prod(self.setup['landscape_dims'])

        inf_idx = np.array([15*loc+np.arange(1, 14, 3) for loc in range(ncells)]).flatten()
        inf_data = np.sum(sim_run[inf_idx, :].reshape((ncells, 5, -1)), axis=0) / ncells

        scales = np.amax(inf_data, axis=1)
        scaling_matrix = np.tile(
            np.divide(1, scales, out=np.zeros_like(scales), where=(scales > 0)),
            inf_data.shape[1]).reshape((5, -1), order="F")

        start_transformed = logit_transform(start, bounds)

        approx_model = MixedStandApprox(self.setup, self.params, start)
        approx_model.params['space_tanoak'] = simulator.params['space_tanoak']
        approx_model.params['recruit_tanoak'] = simulator.params['recruit_tanoak']
        approx_model.params['recruit_bay'] = simulator.params['recruit_bay']
        approx_model.params['recruit_redwood'] = simulator.params['recruit_redwood']

        # Minimise SSE
        param_fit_transformed = minimize(
            self._sse, start_transformed, method="Nelder-Mead",
            options={'ftol': 1e-6, 'disp': True},
            args=(inf_data, approx_model, bounds, scaling_matrix))

        param_fit = reverse_logit_transform(param_fit_transformed.x, bounds)

        approx_model.beta = param_fit
        for i in range(3):
            approx_model.beta[i+1] *= approx_model.beta[0]

        if show_plot:
            model_run, objective = approx_model.run_policy(None)
            model_inf = model_run[1:14:3, :]
            fig = plt.figure()
            ax = fig.add_subplot(111)
            names = ["Tan 1", "Tan 2", "Tan 3", "Tan 4", "Bay"]
            for i, name in enumerate(names):
                ax.plot(self.setup['times'], inf_data[i, :], color="C{}".format(i), alpha=0.5,
                        label=name)
                ax.plot(self.setup['times'], model_inf[i, :], color="C{}".format(i))
            ax.legend()
            fig.tight_layout()
            plt.show()

        self.beta = param_fit
        return approx_model, param_fit

    def pairwise_scans(self, start, bounds, log=None, use_start_as_baseline=False, num=11):
        """Plot SSE as function of two fitted parameters, for each pair.

        log:        If None, linear scales used for all parameters. For log scales provide list of
                    indices for parameters to use it with.
        """

        if log is None:
            log = []

        # First get simulator data
        simulator = ms_sim.MixedStandSimulator(self.setup, self.params)
        sim_run = simulator.run_policy(None)

        ncells = np.prod(self.setup['landscape_dims'])

        inf_idx = np.array([15*loc+np.arange(1, 14, 3) for loc in range(ncells)]).flatten()
        inf_data = np.sum(sim_run[inf_idx, :].reshape((ncells, 5, -1)), axis=0) / ncells

        approx_model = MixedStandApprox(self.setup, self.params, start)
        approx_model.params['space_tanoak'] = simulator.params['space_tanoak']
        approx_model.params['recruit_tanoak'] = simulator.params['recruit_tanoak']
        approx_model.params['recruit_bay'] = simulator.params['recruit_bay']
        approx_model.params['recruit_redwood'] = simulator.params['recruit_redwood']

        if use_start_as_baseline:
            baseline_params = start
        else:
            # Need to fit model first
            approx_model, baseline_params = self.fit(start, bounds)

        param_names = [
            r"$\beta_{1,1}$", r"$\beta_{1,2}$", r"$\beta_{1,3}$", r"$\beta_{1,4}$", r"$\beta_{12}$",
            r"$\beta_{21}$", r"$\beta_{2}$"]

        # fig = plt.figure(figsize=(11.7, 8.3))
        fig = plt.figure(figsize=(8.3, 11.7))
        axis_num = 1

        for param1, param2 in itertools.combinations([0, 1, 2, 3], 2):#, 4, 6], 2):
            if param1 in log:
                param1_vals = np.geomspace(*bounds[param1], num)
            else:
                param1_vals = np.linspace(*bounds[param1], num)

            if param2 in log:
                param2_vals = np.geomspace(*bounds[param2], num)
            else:
                param2_vals = np.linspace(*bounds[param2], num)

            xx, yy = np.meshgrid(param1_vals, param2_vals)
            zz = np.zeros_like(xx)

            approx_model.beta = copy.deepcopy(baseline_params)

            for i, j in itertools.product(range(xx.shape[0]), range(xx.shape[1])):
                approx_model.beta[param1] = xx[i, j]
                approx_model.beta[param2] = yy[i, j]
                model_run, objective = approx_model.run_policy(None)
                model_inf = model_run[1:14:3, :]
                zz[i, j] = np.sum(np.square((inf_data - model_inf)))

            # ax = fig.add_subplot(3, 5, axis_num)
            ax = fig.add_subplot(3, 2, axis_num)
            axis_num += 1
            cmap = plt.get_cmap("inferno_r")
            im = ax.pcolormesh(xx, yy, zz, cmap=cmap)

            if param1 in log:
                ax.set_xscale("log")
            if param2 in log:
                ax.set_yscale("log")

            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            fig.colorbar(im, ax=ax, cax=cax)

            ax.plot(baseline_params[param1], baseline_params[param2], "rx", ms=3)
            ax.set_xlabel(param_names[param1])
            ax.set_ylabel(param_names[param2])

            print("Done {0}, {1}".format(param1, param2))

        fig.tight_layout()
        fig.savefig(os.path.join("..", "Figures", "SSESurface_Fig4b.pdf"), dpi=600)

    def _sse(self, params, sim_inf, approx_model, bounds, scales):
        params = reverse_logit_transform(params, bounds)
        beta = params[0:7]
        # powers = params[7:]

        approx_model.beta = beta
        for i in range(3):
            approx_model.beta[i+1] *= approx_model.beta[0]
        # approx_model.powers = powers

        model_run = approx_model.run_policy(None)
        model_inf = model_run[1:14:3, :]

        if sim_inf.shape != model_inf.shape:
            raise RuntimeError("Wrong shaped arrays for SSE calculation")

        sse = np.sum(np.square(scales * (sim_inf - model_inf)))
        print(sse)
        return sse

def logit_transform(params, bounds):
    """Logit transform parameters to remove bounds."""
    with np.errstate(divide="ignore", invalid="ignore"):
        ret_array = np.ma.array(
            [np.ma.log(np.true_divide((x - a), (b - x))) for x, (a, b) in zip(params, bounds)])
        ret_array.set_fill_value(0)
        return np.ma.filled(ret_array)

def reverse_logit_transform(params, bounds):
    """Reverse logit transform parameters to return bounds."""

    return np.array(
        [a + ((b-a)*np.exp(x) / (1 + np.exp(x))) for x, (a, b) in zip(params, bounds)])

def _try_file_open(filename):
    """Try repeatedly opening file for writing when permission errors occur in OneDrive."""

    while True:
        try:
            return open(filename, "w")
        except PermissionError:
            print("Permission error opening {0}. Trying again...".format(filename))
            continue
        break
