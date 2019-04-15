"""
Approximate model for tanoak-bay laurel-redwood mixed stand SOD .

Species are labelled as:
    1) Tanoak
    2) Bay Laurel
    3) Redwood
"""

import copy
import subprocess
import pickle
import os
import logging
import numpy as np
from scipy import integrate
from scipy.interpolate import interp1d
import bocop_utils

from . import utils


class MixedStandApprox:
    """
    Class to implement an approximate 3 species model of SOD within a forest stand.

    Initialisation requires the following setup dictionary of values:
        'state_init':       Length 15 array giving [S11, I11, V11 ... S2 I2 S3].
        'times':            Times to solve for.

    and parameter object. Some parameters will be taken from Cobb model, others require fitting.
    A fit object is also required
    """

    # TODO add initialisation function

    def __init__(self, setup, params, fit):
        required_keys = ['state_init', 'times']

        # Make sure required keys present
        try:
            self.setup = {'times': setup['times']}
            self.set_state_init(setup['state_init'])
        except KeyError as err:
            logging.exception("Missing required key!")
            raise err

        for key in setup:
            if key not in required_keys:
                logging.info("Unused setup parameter: %s", key)

        self.params = copy.deepcopy(params)

        if isinstance(fit, dict):
            logging.info("Getting beta values from dictionary")
            self.beta = np.zeros(7)
            self.beta[0] = fit['beta_1,1']
            self.beta[1] = fit['beta_1,2']
            self.beta[2] = fit['beta_1,3']
            self.beta[3] = fit['beta_1,4']
            self.beta[4] = fit['beta_12']
            self.beta[5] = fit['beta_21']
            self.beta[6] = fit['beta_2']
        elif len(fit) == 7:
            logging.info("Getting beta values from array")
            self.beta = fit
        else:
            logging.error("Wrong format for fitted parameters!")
            raise ValueError("Wrong format for fitted parameters!")

        self.run = {
            'state': None,
            'control': None,
            'objective': None
        }
        self.optimisation = {
            'control': None,
            'interp_kind': None
        }

    def save_optimisation(self, filename):
        """Save control optimisation and parameters to file."""

        dump_obj = {
            'optimisation': self.optimisation,
            'setup': self.setup,
            'params': self.params,
            'beta': self.beta
        }

        with open(filename, "wb") as outfile:
            pickle.dump(dump_obj, outfile)

    def save_run(self, filename):
        """Save run_data, control and run parameters to file."""

        dump_obj = {
            'run': self.run,
            'setup': self.setup,
            'params': self.params,
            'beta': self.beta
        }

        with open(filename, "wb") as outfile:
            pickle.dump(dump_obj, outfile)

    def load_optimisation(self, filename):
        """Load control optimisation and parameters."""

        with open(filename, "rb") as infile:
            load_obj = pickle.load(infile)

        self.optimisation = load_obj['optimisation']
        self.params = load_obj['params']
        self.setup = load_obj['setup']
        self.beta = load_obj['beta']

    def load_run(self, filename):
        """Load run_data, control and run parameters."""

        with open(filename, "rb") as infile:
            load_obj = pickle.load(infile)

        self.run = load_obj['run']
        self.setup = load_obj['setup']
        self.params = load_obj['params']
        self.beta = load_obj['beta']

    def set_state_init(self, state_init):
        """Set the initial state for future runs."""

        if len(state_init) != 15:
            logging.info("Setting initial state by averaging cells")
            ncells = len(state_init) / 15

            self.setup['state_init'] = np.sum(
                np.reshape(state_init, (int(ncells), 15)), axis=0) / ncells

        else:
            logging.info("Setting initial state directly")
            self.setup['state_init'] = state_init

    def state_deriv(self, time, state, control_func=None):
        """Return state derivative for 3 species model, including integrated objective function.

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

        # Vaccine decay
        for i in range(4):
            d_state[3*i+2] -= self.params.get("vaccine_decay", 0.0) * state[3*i+2]
            d_state[3*i] += self.params.get("vaccine_decay", 0.0) * state[3*i+2]

        if control_func is not None:
            control = control_func(time)

            control[0:3] *= self.params.get('rogue_rate', 0.0)
            control[3:7] *= self.params.get('thin_rate', 0.0)
            control[7:] *= self.params.get('protect_rate', 0.0)

            # Roguing
            d_state[1] -= control[0] * state[1]
            d_state[4] -= control[0] * state[4]
            d_state[7] -= control[1] * state[7]
            d_state[10] -= control[1] * state[10]
            d_state[13] -= control[2] * state[13]

            # Thinning
            for i in range(6):
                d_state[i] -= control[3] * state[i]
                d_state[i+6] -= control[4] * state[i+6]
            d_state[12] -= control[5] * state[12]
            d_state[13] -= control[5] * state[13]
            d_state[14] -= control[6] * state[14]

            # Phosphonite protectant
            d_state[0] -= control[7] * state[0]
            d_state[2] += control[7] * state[0]
            d_state[3] -= control[7] * state[3]
            d_state[5] += control[7] * state[3]
            d_state[6] -= control[8] * state[6]
            d_state[8] += control[8] * state[6]
            d_state[9] -= control[8] * state[9]
            d_state[11] += control[8] * state[9]
        else:
            control = np.zeros(9)

        # Objective function
        d_state[-1] = utils.objective_integrand(time, state[:15], control, self.params)

        return d_state

    def run_policy(self, control_policy=None, n_fixed_steps=None):
        """Run forward simulation using a given control policy.

        Function control_policy(t)

        -------
        Returns
        -------
        run_data, objective value, objective integral dynamics

        """

        ode = integrate.ode(self.state_deriv)
        ode.set_integrator('lsoda', nsteps=10000, atol=1e-10, rtol=1e-8)
        ode.set_initial_value(np.append(self.setup['state_init'], [0.0]), self.setup['times'][0])
        ode.set_f_params(control_policy)

        logging.info("Starting ODE run")

        ts = [self.setup['times'][0]]
        xs = [self.setup['state_init']]
        obj = [0.0]

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
                obj.append(state_old_int[-1])
                ts.append(time)

            else:
                if ode.successful():
                    ode.integrate(time)
                    ts.append(ode.t)
                    xs.append(ode.y[:-1])
                    obj.append(ode.y[-1])
                else:
                    logging.error("ODE solver error!")
                    raise RuntimeError("ODE solver error!")

        logging.info("ODE run completed")

        state = np.vstack(xs).T
        self.run['state'] = state
        if control_policy is None:
            self.run['control'] = None
        else:
            self.run['control'] = np.array([control_policy(t) for t in self.setup['times']]).T
        self.run['objective'] = utils.objective_payoff(ts[-1], xs[-1], self.params) + obj[-1]

        return state, self.run['objective'], obj

    def optimise(self, bocop_dir=None, verbose=True, init_policy=None, n_stages=None):
        """Run BOCOP optimisation of control.

        If n_stages is not None, then control will be piecewise constant with this number of stages
        """

        if bocop_dir is None:
            bocop_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "BOCOP")

        logging.info("Running initial policy")
        if init_policy is None:
            init_state, _, init_obj = self.run_policy(None)
        else:
            init_state, _, init_obj = self.run_policy(init_policy, n_fixed_steps=None)

        init = np.vstack((init_state, init_obj))

        self._set_bocop_params(init_state=init, init_policy=init_policy,
                               folder=bocop_dir, n_stages=n_stages)

        if verbose is True:
            logging.info("Running BOCOP verbosely")
            subprocess.run([os.path.join(bocop_dir, "bocop.exe")], cwd=bocop_dir)
        else:
            logging.info("Running BOCOP quietly")
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
            self.optimisation['control'] = actual_control
            self.optimisation['interp_kind'] = 'zero'

        else:
            actual_control = np.array([control_t(t) for t in self.setup['times'][:-1]]).T
            self.optimisation['control'] = actual_control
            self.optimisation['interp_kind'] = 'linear'

        return (actual_states, control_t, exit_text)

    def _set_bocop_params(self, init_state, init_policy=None, folder="BOCOP", n_stages=None):
        """Save parameters and initial conditions to file for BOCOP optimisation."""

        logging.info("Setting BOCOP parameters")

        logging.info("Setting problem.bounds")

        with open(os.path.join(folder, "problem.bounds"), "r") as infile:
            all_lines = infile.readlines()

        # Initial conditions
        for i in range(15):
            all_lines[9+i] = str(self.setup['state_init'][i]) + " " + str(
                self.setup['state_init'][i]) + " equal\n"

        # When no integrated term in objective, set bounds on integrand to zero to aid convergence
        if (self.params.get('div_cost', 0.0) ==
                self.params.get('cull_cost', 0.0) ==
                self.params.get('protect_cost', 0.0) == 0.0):
            all_lines[42] = "0.0 0.0 both\n"
        else:
            all_lines[42] = "-1e6 1e6 both\n"

        n_optim_vars = 0

        # Setup correct bounds for optimisation variables and path constraints
        if n_stages is None:
            all_lines[6] = "16 16 9 0 0 1\n"

            del all_lines[56:]
            all_lines.append("\n# Bounds for the path constraints :\n")
            all_lines.append("-2e+020 {} upper\n".format(self.params['max_budget']))
        else:
            n_optim_vars = n_stages * 9
            all_lines[6] = "16 16 9 0 " + str(n_optim_vars) + " 10\n"

            del all_lines[56:]
            for i in range(n_optim_vars):
                all_lines.append("0 1 both\n")

            all_lines.append("\n# Bounds for the path constraints :\n")
            all_lines.append("-2e+020 {} upper\n".format(self.params['max_budget']))
            for i in range(9):
                all_lines.append("0 0 equal\n")

        with _try_file_open(os.path.join(folder, "problem.bounds")) as outfile:
            outfile.writelines(all_lines)

        logging.info("Setting problem.constants")

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
        all_lines[41] = str(self.params.get('rogue_rate', 0.0)) + "\n"
        all_lines[42] = str(self.params.get('thin_rate', 0.0)) + "\n"
        all_lines[43] = str(self.params.get('protect_rate', 0.0)) + "\n"
        all_lines[44] = str(self.params.get('treat_eff', 0.0)) + "\n"
        all_lines[45] = str(self.params.get('div_cost', 0.0)) + "\n"
        all_lines[46] = str(self.params.get('rogue_cost', 0.0)) + "\n"
        all_lines[47] = str(self.params.get('thin_cost', 0.0)) + "\n"
        all_lines[48] = str(self.params.get('protect_cost', 0.0)) + "\n"
        all_lines[49] = str(self.params.get('discount_rate', 0.0)) + "\n"
        all_lines[50] = str(self.params.get('payoff_factor', 0.0)) + "\n"

        if n_stages is None:
            all_lines[51] = "0\n"
        else:
            all_lines[51] = str(n_stages) + "\n"

        all_lines[52] = str(self.params.get('vaccine_decay', 0.0)) + "\n"

        with _try_file_open(os.path.join(folder, "problem.constants")) as outfile:
            outfile.writelines(all_lines)

        logging.info("Setting problem.def")

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
            all_lines[15] = "constraint.dimension integer 10\n"

        all_lines[18] = "discretization.steps integer " + n_steps + "\n"

        with _try_file_open(os.path.join(folder, "problem.def")) as outfile:
            outfile.writelines(all_lines)

        # Initialisation

        # State initialisation
        for state in range(16):
            logging.info("Initialising state %i", state)
            all_lines = [
                "#Starting point file\n",
                "# This file contains the values of the initial points\n",
                "# for variable state #{0}\n".format(state), "\n",
                "# Type of initialization :\n",
                "linear\n", "\n", "# Number of interpolation points :\n",
                "{0}\n".format(len(self.setup['times'])), "\n", "# Interpolation points :\n"]

            for i, time in enumerate(self.setup['times']):
                all_lines.append("{0} {1}\n".format(time,
                                                    np.round(init_state[state, i], decimals=3)))

            with _try_file_open(os.path.join(folder, "init", "state."+str(state)+".init")) as outf:
                outf.writelines(all_lines)

        # Control initialisation
        if init_policy is not None:
            control_init = np.array([init_policy(t) for t in self.setup['times'][:-1]])

            for control in range(9):
                logging.info("Initialising control %i", control)
                all_lines = [
                    "#Starting point file\n",
                    "# This file contains the values of the initial points\n",
                    "# for variable control #{0}\n".format(control),
                    "\n", "# Type of initialization :\n", "linear\n", "\n",
                    "# Number of interpolation points :\n",
                    "{0}\n".format(len(self.setup['times'][:-1])),
                    "\n", "# Interpolation points :\n"]

                for i, time in enumerate(self.setup['times'][:-1]):
                    all_lines.append("{0} {1}\n".format(
                        time, np.round(control_init[i, control], decimals=2)))

                with _try_file_open(os.path.join(folder, "init",
                                                 "control." + str(control) + ".init")) as outfile:
                    outfile.writelines(all_lines)

        # Optimisation variables (for piecewise constant control)
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

def _try_file_open(filename):
    """Try repeatedly opening file for writing when permission errors occur in OneDrive."""

    while True:
        try:
            return open(filename, "w")
        except PermissionError:
            logging.warning("Permission error opening %s. Trying again...", filename)
            continue
        break
