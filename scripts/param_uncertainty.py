"""Example run of mixed stand model."""

import pdb
import os
import numpy as np
import pickle
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.stats import truncnorm
from mixed_stand_model import mixed_stand_simulator as ms_sim
from mixed_stand_model import mixed_stand_approx as ms_approx
from mixed_stand_model import parameters
from mixed_stand_model import visualisation
from mixed_stand_model import mpc
from mixed_stand_model import utils

def even_policy(time):
    return np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])

def generate_ensemble_and_fit(params, setup, n_ensemble_runs, standard_dev):

    model = ms_sim.MixedStandSimulator(setup, params)

    ncells = np.product(setup['landscape_dims'])

    # Create parameter error distribution
    if standard_dev != 0.0:
        error_dist = truncnorm(-1.0/standard_dev, np.inf, loc=1.0, scale=standard_dev)
        error_samples = np.reshape(error_dist.rvs(size=n_ensemble_runs*7), (n_ensemble_runs, 7))
    else:
        error_samples = np.ones((n_ensemble_runs, 7))

    # Sample from parameter distribution and run simulations
    baseline_beta = np.zeros(7)
    baseline_beta[:4] = params['inf_tanoak_tanoak']
    baseline_beta[4] = params['inf_bay_to_tanoak']
    baseline_beta[5] = params['inf_tanoak_to_bay']
    baseline_beta[6] = params['inf_bay_to_bay']
    parameter_samples = error_samples * baseline_beta

    simulation_runs = np.zeros(
        (n_ensemble_runs, 15, len(setup['times'])))
    simulation_runs_no_cross_trans = np.zeros(
        (n_ensemble_runs, 15, len(setup['times'])))
    for i in range(n_ensemble_runs):
        model.params['inf_tanoak_tanoak'] = parameter_samples[i, 0:4]
        model.params['inf_bay_to_tanoak'] = parameter_samples[i, 4]
        model.params['inf_tanoak_to_bay'] = parameter_samples[i, 5]
        model.params['inf_bay_to_bay'] = parameter_samples[i, 6]

        sim_run, *_ = model.run_policy()
        simulation_runs[i] = np.sum(sim_run.reshape((ncells, 15, -1)), axis=0) / ncells

        model.params['inf_bay_to_tanoak'] = 0.0
        model.params['inf_tanoak_to_bay'] = 0.0

        sim_run, *_ = model.run_policy()
        simulation_runs_no_cross_trans[i] = np.sum(
            sim_run.reshape((ncells, 15, -1)), axis=0) / ncells

        print("Run {0} done.".format(i))

    ret_dict = {
        'params': parameter_samples,
        'sims': simulation_runs,
        'sims_no_cross_trans': simulation_runs_no_cross_trans,
        'fit': None,
        'times': setup['times']
    }

    # Fit to ensemble of simulation runs
    fitter = ms_approx.MixedStandFitter(setup, params)

    # First fit with no cross transmission to set ration within tanoak infection rates
    # Without this fit cannot disentangle tanoak from bay infection forces
    start = np.array([2.35, 0.15, 0.1, 0.1, 0.0, 0.0, 4.5])
    bounds = [(1e-10, 20), (1e-10, 20), (1e-10, 20), (1e-10, 20), (0.0, 0.0),
              (0.0, 0.0), (1e-10, 20)]

    approx_model, beta = fitter.fit(start, bounds, show_plot=False, scale=True, averaged=True,
                                    dataset=simulation_runs_no_cross_trans)
    print("Done fit within tanoak...")
    print("Beta: {0}".format(beta))

    # And now fit fixing ratio of tanoak infection rates
    start = beta
    start[4] = 3.5
    for i in range(3):
        start[i+1] /= start[0]
        bounds[i+1] = (start[i+1], start[i+1])
    bounds[4] = (1e-10, 20)
    approx_model, beta = fitter.fit(start, bounds, show_plot=False, scale=True, averaged=True,
                                    dataset=simulation_runs)
    print("Done final fit.")
    print("Beta: {0}".format(beta))

    ret_dict['fit'] = beta

    return ret_dict

def run_optimisations(ensemble_and_fit, params, setup, n_optim_runs, standard_dev, **mpc_args):
    """Run open-loop and MPC optimisations over parameter distributions"""

    approx_model = ms_approx.MixedStandApprox(setup, params, ensemble_and_fit['fit'])
    sim_model = ms_sim.MixedStandSimulator(setup, params)

    _, ol_control, _ = approx_model.optimise(n_stages=20, init_policy=even_policy)

    # Create parameter error distribution
    if standard_dev != 0.0:
        error_dist = truncnorm(-1.0/standard_dev, np.inf, loc=1.0, scale=standard_dev)
        error_samples = np.reshape(error_dist.rvs(size=n_optim_runs*7), (n_optim_runs, 7))
    else:
        error_samples = np.ones((n_optim_runs, 7))

    # Sample from parameter distribution and run simulations
    baseline_beta = np.zeros(7)
    baseline_beta[:4] = params['inf_tanoak_tanoak']
    baseline_beta[4] = params['inf_bay_to_tanoak']
    baseline_beta[5] = params['inf_tanoak_to_bay']
    baseline_beta[6] = params['inf_bay_to_bay']
    parameter_samples = error_samples * baseline_beta

    ol_objs = np.zeros(n_optim_runs)
    for i in range(n_optim_runs):
        sim_model.params['inf_tanoak_tanoak'] = parameter_samples[i, 0:4]
        sim_model.params['inf_bay_to_tanoak'] = parameter_samples[i, 4]
        sim_model.params['inf_tanoak_to_bay'] = parameter_samples[i, 5]
        sim_model.params['inf_bay_to_bay'] = parameter_samples[i, 6]

        _, obj, _ = sim_model.run_policy(control_policy=ol_control)
        ol_objs[i] = obj

        print("Open-loop run {0} done.".format(i))

    mpc_args['init_policy'] = ol_control

    mpc_objs = np.zeros(n_optim_runs)
    mpc_controls = np.zeros((n_optim_runs, 9, len(setup['times'])))
    for i in range(n_optim_runs):
        sim_model.params['inf_tanoak_tanoak'] = parameter_samples[i, 0:4]
        sim_model.params['inf_bay_to_tanoak'] = parameter_samples[i, 4]
        sim_model.params['inf_tanoak_to_bay'] = parameter_samples[i, 5]
        sim_model.params['inf_bay_to_bay'] = parameter_samples[i, 6]

        mpc_controller = mpc.Controller(sim_model, approx_model)
        _, _, mpc_control, mpc_obj = mpc_controller.run_controller(**mpc_args)
        mpc_objs[i] = mpc_obj
        mpc_controls[i] = mpc_control

        print("MPC run {0} done.".format(i))

    ret_dict = {
        'params': parameter_samples,
        'ol_control': ol_control,
        'ol_objs': ol_objs,
        'mpc_control': mpc_controls,
        'mpc_objs': mpc_objs,
        'times': setup['times']
    }

    return ret_dict

def run_all():
    params = parameters.CORRECTED_PARAMS

    # Standard deviation for parameters as proportion of parameter value
    standard_dev = 0.5

    # Number of simulation ensemble runs for fitting
    n_ensemble_runs = 10

    # Number of parameter samples for control optimisation
    n_optimisation_runs = 10

    # Initial conditions used in 2012 paper
    setup, params = utils.get_setup_params(
        parameters.CORRECTED_PARAMS, scale_inf=True, host_props=parameters.COBB_PROP_FIG4A)

    mpc_args = {
        'horizon': 100,
        'time_step': 0.5,
        'end_time': 100,
        'update_period': 5,
        'rolling_horz': False,
        'stage_len': 5,
        'init_policy': even_policy
    }

    ensemble_and_fit = generate_ensemble_and_fit(params, setup, n_ensemble_runs, standard_dev)
    np.savez_compressed("fitting_ensemble_data_0.5", **ensemble_and_fit)

    optimisations = run_optimisations(
        ensemble_and_fit, params, setup, n_optimisation_runs, standard_dev, **mpc_args)
    np.savez_compressed("optimisation_data_0.5", **optimisations)

if __name__ == "__main__":
    run_all()
