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
        self.beta = None

    def fit(self, start, bounds, dataset=None, show_plot=False, scale=True, averaged=False):
        """Fit infection rate parameters, minimising sum of squared errors from simulation DPCs.

        start:      Initial values of beta parameters for optimisation (length 7 array).
        bounds:     Bounds for beta parameters (array of 7 (lower, upper) tuples).
        dataset:    If present use this dataset of simulation runs to fit approximate model. If 2d
                    array then corresponds to single simulation run output. If 3d, first dimension
                    specifies each run in the ensemble. Can be averaged across cells.
        show_plot:  Whether to show interactive plot of fit.
        scale:      Whether to use scaling when fitting to weight smaller numbers more.
        averaged:   Whether dataset has been averaged across cells.
        """

        # Get simulator data if not provided
        ncells = np.prod(self.setup['landscape_dims'])
        inf_idx = np.array([15*loc+np.arange(1, 14, 3) for loc in range(ncells)]).flatten()

        if dataset is None:
            simulator = ms_sim.MixedStandSimulator(self.setup, self.params)
            sim_run, *_ = simulator.run_policy(None)
            inf_data = np.sum(sim_run[inf_idx, :].reshape((ncells, 5, -1)), axis=0) / ncells

        elif len(dataset.shape) == 2:
            if averaged:
                inf_data = dataset[np.arange(1, 14, 3), :]
            else:
                inf_data = np.sum(dataset[inf_idx, :].reshape((ncells, 5, -1)), axis=0) / ncells

        else:
            if averaged:
                inf_data = dataset[:, np.arange(1, 14, 3), :]
            else:
                inf_data = np.sum(dataset[:, inf_idx, :].reshape((dataset.shape[0], ncells, 5, -1)),
                                  axis=1) / ncells

        # TODO check scaling sensible
        if scale:
            scales = np.amax(inf_data, axis=-1)
            if len(scales.shape) > 1:
                scales = np.mean(scales, axis=0)
            scaling_matrix = np.tile(
                np.divide(1, scales, out=np.zeros_like(scales), where=(scales > 0)),
                inf_data.shape[-1]).reshape((5, -1), order="F")
        else:
            scaling_matrix = np.ones(inf_data.shape[-2:])

        start_transformed = logit_transform(start, bounds)

        approx_model = MixedStandApprox(self.setup, self.params, start)

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
            model_run, *_ = approx_model.run_policy(None)
            model_inf = model_run[1:14:3, :]
            fig = plt.figure()
            ax = fig.add_subplot(111)
            names = ["Tan 1", "Tan 2", "Tan 3", "Tan 4", "Bay"]
            for i, name in enumerate(names):
                if len(inf_data.shape) > 2:
                    ax.plot([], color="C{}".format(i), label=name)
                    ax.plot(self.setup['times'], inf_data[:, i, :].T, color="C{}".format(i),
                            alpha=0.2)
                else:
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
        params = reverse_logit_transform(params, bounds)
        beta = params[0:7]
        # powers = params[7:]

        approx_model.beta = beta
        for i in range(3):
            approx_model.beta[i+1] *= approx_model.beta[0]
        # approx_model.powers = powers

        model_run, *_ = approx_model.run_policy(None)
        model_inf = model_run[1:14:3, :]

        if sim_inf.shape[-2:] != model_inf.shape:
            raise RuntimeError("Wrong shaped arrays for SSE calculation")

        sse = np.sum(np.square(scales * (sim_inf - model_inf)))
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
