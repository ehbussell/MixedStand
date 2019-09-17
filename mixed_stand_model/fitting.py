"""Tools for fitting approximate model to simulations."""

import copy
import itertools
import logging
import os
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
from scipy.optimize import minimize

from . import mixed_stand_simulator as ms_sim
from .mixed_stand_approx import MixedStandApprox


def scale_sim_model(setup, old_params, new_params, lower_bound=None, upper_bound=None,
                    time_step=None):
    """Scale infection rates in simulation to match time scales using incorrect Cobb 2012 model.

    Model using new_params is scaled to match time scale using old_params.
    The time scale is measured by the time taken for the populations of small and large tanoaks
    to be equal in size.
    """

    if lower_bound is None:
        lower_bound = 0.0
        logging.info("No lower bound set, using %f", lower_bound)

    if upper_bound is None:
        upper_bound = 10.0
        logging.info("No upper bound set, using %f", upper_bound)

    if time_step is None:
        time_step = 0.01
        logging.info("No time step set, using %f", time_step)

    model_setup = copy.deepcopy(setup)
    model_setup['times'] = np.arange(0, 50, step=time_step)

    # First find crossover time using old parameters
    base_model = ms_sim.MixedStandSimulator(model_setup, old_params)
    base_sim_run, *_ = base_model.run_policy(control_policy=None)
    base_cross_time = _get_crossover_time(base_sim_run, base_model.ncells, model_setup['times'])

    logging.info("Cross over time in base simulation model: %f", base_cross_time)

    if base_cross_time == np.inf:
        logging.error("Found no cross over in this time domain!")

    # Avoid runnning simulations for too long - round up to next multiple of 5 in time units
    model_setup['times'] = np.arange(0, 5*np.ceil(base_cross_time/5), step=time_step)

    # Now find factor to scale infection rates
    tolerance = model_setup['times'][1] / 2
    prev_upper = upper_bound
    prev_lower = lower_bound
    new_factor = (upper_bound + lower_bound) / 2

    logging.debug("Bounds now (%f, %f), now trying new factor: %f",
                  prev_lower, prev_upper, new_factor)

    # Run simulation with new factor to test
    params = copy.deepcopy(new_params)
    params['inf_tanoak_tanoak'] *= new_factor
    params['inf_bay_to_bay'] *= new_factor
    params['inf_bay_to_tanoak'] *= new_factor
    params['inf_tanoak_to_bay'] *= new_factor
    model = ms_sim.MixedStandSimulator(model_setup, params)
    sim_run, *_ = model.run_policy(control_policy=None)

    new_cross_over_time = _get_crossover_time(sim_run, model.ncells, model_setup['times'])
    diff = base_cross_time - new_cross_over_time
    logging.info("Using factor %f, new cross over time: %f", new_factor, new_cross_over_time)
    logging.debug("Using factor %f, error in cross over time: %f", new_factor, diff)

    while abs(diff) > tolerance:
        if diff > 0:
            if new_factor == prev_lower:
                prev_upper, new_factor, prev_lower = new_factor, 0.5*new_factor, 0.5*new_factor
            else:
                prev_upper, new_factor = new_factor, np.mean([prev_lower, new_factor])
        else:
            if new_factor == prev_upper:
                prev_lower, new_factor, prev_upper = new_factor, 2*new_factor, 2*new_factor
            else:
                prev_lower, new_factor = new_factor, np.mean([prev_upper, new_factor])

        logging.info("Bounds now (%f, %f), now trying new factor: %f",
                     prev_lower, prev_upper, new_factor)

        params = copy.deepcopy(new_params)
        params['inf_tanoak_tanoak'] *= new_factor
        params['inf_bay_to_bay'] *= new_factor
        params['inf_bay_to_tanoak'] *= new_factor
        params['inf_tanoak_to_bay'] *= new_factor

        model = ms_sim.MixedStandSimulator(model_setup, params)
        sim_run, *_ = model.run_policy(control_policy=None)
        new_cross_over_time = _get_crossover_time(sim_run, model.ncells, model_setup['times'])
        diff = base_cross_time - new_cross_over_time
        logging.info("Using factor %f, new cross over time: %f", new_factor, new_cross_over_time)

    logging.info("Difference %f less than tolerance %f, using factor %f", diff, tolerance,
                 new_factor)

    return new_factor

class MixedStandFitter:
    """Fitting of MixedStandApprox model to simulation data."""

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
        self.tanoak_factors = None
        self.beta = None

    def fit(self, start, bounds, dataset=None, show_plot=False, scale=True, tanoak_factors=None):
        """Fit infection rate parameters, minimising sum of squared errors from simulation DPCs.

        start:          Initial values of beta parameters for optimisation (length 7 array).
                            beta_1,1
                            beta_1,2
                            beta_1,3
                            beta_1,4
                            beta_12
                            beta_21
                            beta_2
        bounds:         Bounds for beta parameters (array of 7 (lower, upper) tuples).
        dataset:        If present use this dataset of simulation runs to fit approximate model.
                        Must be 3d array with shape (nruns, 15, ntimes). First dimension specifies
                        each run in the ensemble. Results from spatial simulations should be
                        averaged across cells.
        show_plot:      Whether to show interactive plot of fit.
        scale:          Whether to use scaling when fitting to weight smaller numbers more.
        tanoak_factors: Relative infection rate in tanoak age classes 2,3 and 4 compared to tanoak
                        age class 1. If specified this will constrain fitting

        Returns: fitted approximate model and beta array.
        """

        # Get simulator data if not provided
        ncells = np.prod(self.setup['landscape_dims'])

        if dataset is None:
            logging.info("No dataset provided; running single simulation.")
            simulator = ms_sim.MixedStandSimulator(self.setup, self.params)
            sim_run, *_ = simulator.run_policy(None)
            inf_idx = np.array([15*loc+np.arange(1, 14, 3) for loc in range(ncells)]).flatten()
            inf_data = np.expand_dims(
                np.sum(sim_run[inf_idx, :].reshape((ncells, 5, -1)), axis=0) / ncells, axis=0)

        else:
            inf_data = dataset[:, np.arange(1, 14, 3), :]

        if tanoak_factors is not None:
            # Constrain bounds on age classes 2+ and set these values based on age class 1 when
            # optimising
            for i in range(3):
                bounds[i+1] = (-1, -1)
                start[i+1] = -1
            self.tanoak_factors = tanoak_factors

        if scale:
            scales = np.amax(inf_data, axis=-1)
            scales = np.mean(scales, axis=0)
            scaling_matrix = np.tile(
                np.divide(1, scales, out=np.zeros_like(scales), where=(scales > 0)),
                inf_data.shape[-1]).reshape((5, -1), order="F")
        else:
            scaling_matrix = np.ones(inf_data.shape[-2:])

        start_transformed = _logit_transform(start, bounds)

        approx_model = MixedStandApprox(self.setup, self.params, start)

        logging.info("Starting fit minimisation")

        # Minimise SSE
        param_fit_transformed = minimize(
            self._sse, start_transformed, method="BFGS",
            options={'disp': True, 'maxiter': 2800},
            args=(inf_data, approx_model, bounds, scaling_matrix))

        logging.info("%s", param_fit_transformed)

        param_fit = _reverse_logit_transform(param_fit_transformed.x, bounds)
        if self.tanoak_factors is not None:
            for i in range(3):
                param_fit[i+1] = param_fit[0] * self.tanoak_factors[i]

        approx_model.beta = param_fit

        if show_plot:
            model_run, *_ = approx_model.run_policy(None)
            model_inf = model_run[1:14:3, :]
            fig = plt.figure()
            ax = fig.add_subplot(111)
            names = ["Tan 1", "Tan 2", "Tan 3", "Tan 4", "Bay"]
            for i, name in enumerate(names):
                ax.plot([], color="C{}".format(i), label=name)
                ax.plot(self.setup['times'], inf_data[:, i, :].T, color="C{}".format(i),
                        alpha=0.2)
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
        sim_run, *_ = simulator.run_policy(None)

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
                model_run, *_ = approx_model.run_policy(None)
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
        params = _reverse_logit_transform(params, bounds)
        beta = params

        # If factors fixed for tanoak age classes then use those
        if self.tanoak_factors is not None:
            for i in range(3):
                beta[i+1] = beta[0] * self.tanoak_factors[i]

        approx_model.beta = beta

        # Run approximate model
        model_run, *_ = approx_model.run_policy(None)
        model_inf = model_run[1:14:3, :]

        if sim_inf.shape[-2:] != model_inf.shape:
            raise RuntimeError("Wrong shaped arrays for SSE calculation")

        sse = np.sum(np.square(scales * (sim_inf - model_inf)))
        return sse

def _logit_transform(params, bounds):
    """Logit transform parameters to remove bounds."""
    with np.errstate(divide="ignore", invalid="ignore"):
        ret_array = np.ma.array(
            [np.ma.log(np.true_divide((x - a), (b - x))) for x, (a, b) in zip(params, bounds)])
        ret_array.set_fill_value(0)
        return np.ma.filled(ret_array)

def _reverse_logit_transform(params, bounds):
    """Reverse logit transform parameters to return bounds."""

    return np.array(
        [a + ((b-a)*np.exp(x) / (1 + np.exp(x))) for x, (a, b) in zip(params, bounds)])

def _get_crossover_time(sim_run, ncells, times):
    """Calculate time when small and large tanoak populations cross over."""

    small_idx = np.array([15*loc+np.arange(0, 6) for loc in range(ncells)]).flatten()
    large_idx = np.array([15*loc+np.arange(6, 12) for loc in range(ncells)]).flatten()
    small = np.sum(sim_run[small_idx, :], axis=0) / np.sum(sim_run, axis=0)
    large = np.sum(sim_run[large_idx, :], axis=0) / np.sum(sim_run, axis=0)

    crossed = np.nonzero(small >= large)[0]
    if len(crossed) > 0:
        return times[crossed[0]]
    return np.inf
