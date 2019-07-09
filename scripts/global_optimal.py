"""Rescale approximate roguing rate ucing OL control to find global optimal strategy."""

from IPython import embed
import argparse
import copy
import csv
import json
import logging
import os
import numpy as np
from scipy.optimize import minimize
from scipy.interpolate import interp1d

from mixed_stand_model import parameters
from mixed_stand_model import utils
from mixed_stand_model import mixed_stand_simulator as ms_sim
from mixed_stand_model import mixed_stand_approx as ms_approx
from mixed_stand_model import mpc


def scale_control(datapath):
    """Parameterise roguing control in approximate model to match simulations."""

    # First run simulation using open-loop policy
    setup, params = utils.get_setup_params(
        parameters.CORRECTED_PARAMS, scale_inf=True, host_props=parameters.COBB_PROP_FIG4A,
        extra_spread=True)

    with open(os.path.join("data", "scale_and_fit_results.json"), "r") as infile:
        scale_and_fit_results = json.load(infile)

    approx_model = ms_approx.MixedStandApprox.load_optimisation_class(
        os.path.join('data', 'ol_mpc_control', 'ol_control.pkl'))
    ol_control = interp1d(setup['times'][:-1], approx_model.optimisation['control'], kind="zero",
                          fill_value="extrapolate")

    ncells = np.product(setup['landscape_dims'])

    model = ms_sim.MixedStandSimulator(setup, params)

    sim_run = model.run_policy(ol_control)
    sim_state = np.sum(sim_run[0].reshape((ncells, 15, -1)), axis=0) / ncells
    sim_tans = np.sum(sim_state[[6, 8, 9, 11], -1])

    logging.info("Sim run, healthy tans: %f", sim_tans)

    beta_names = ['beta_1,1', 'beta_1,2', 'beta_1,3', 'beta_1,4',
                  'beta_12', 'beta_21', 'beta_2']
    beta = np.array([scale_and_fit_results[x] for x in beta_names])

    approx_model = ms_approx.MixedStandApprox(setup, params, beta)

    def min_func(factor):
        """Function to minimise, SSE between healthy tanoak over range of rates."""

        approx_model.params['rogue_rate'] = factor * params['rogue_rate']

        approx_run = approx_model.run_policy(ol_control)

        approx_tans = np.sum(approx_run[0][[6, 8, 9, 11], -1])

        logging.info("Approx run, Factor %f, tans: %f, rate: %f", factor,
                     approx_tans, factor * params['rogue_rate'])

        return abs(approx_tans - sim_tans)

    ret = minimize(min_func, [0.25], bounds=[(0, 2)])

    logging.info(ret)

    # Write scaling results to file
    results = {'roguing_factor': ret.x[0], 'tan_diff': ret.fun}
    with open(os.path.join(datapath, 'scaling_results.json'), "w") as outfile:
        json.dump(results, outfile, indent=4)


def run_scan(datapath):
    """Scan over control scaling factor."""

    factors = np.arange(0.05, 1.11, 0.01)

    diff_results = np.zeros_like(factors)

    # Run simulations using open-loop control policy
    setup, params = utils.get_setup_params(
        parameters.CORRECTED_PARAMS, scale_inf=True, host_props=parameters.COBB_PROP_FIG4A,
        extra_spread=True)

    approx_model = ms_approx.MixedStandApprox.load_optimisation_class(
        os.path.join('data', 'ol_mpc_control', 'ol_control.pkl'))
    ol_control = interp1d(setup['times'][:-1], approx_model.optimisation['control'], kind="zero",
                          fill_value="extrapolate")

    with open(os.path.join("data", "scale_and_fit_results.json"), "r") as infile:
        scale_and_fit_results = json.load(infile)

    ncells = np.product(setup['landscape_dims'])

    model = ms_sim.MixedStandSimulator(setup, params)

    sim_run = model.run_policy(ol_control)
    sim_state = np.sum(sim_run[0].reshape((ncells, 15, -1)), axis=0) / ncells
    sim_tans = np.sum(sim_state[[6, 8, 9, 11], -1])

    logging.info("Sim run, healthy tans: %f", sim_tans)

    beta_names = ['beta_1,1', 'beta_1,2', 'beta_1,3', 'beta_1,4',
                  'beta_12', 'beta_21', 'beta_2']
    beta = np.array([scale_and_fit_results[x] for x in beta_names])

    approx_model = ms_approx.MixedStandApprox(setup, params, beta)

    def min_func(factor):
        """Function to minimise, SSE between healthy tanoak using OL control."""

        approx_model.params['rogue_rate'] = factor * params['rogue_rate']

        approx_run = approx_model.run_policy(ol_control)

        approx_tans = np.sum(approx_run[0][[6, 8, 9, 11], -1])

        logging.info("Approx run, Factor %f, tans: %f", factor, approx_tans)

        return np.sum(np.square(approx_tans - sim_tans))

    for i, factor in enumerate(factors):
        diff = min_func(factor)
        diff_results[i] = diff

    csv_file = os.path.join(datapath, 'scan_control.csv')
    with open(csv_file, 'w', newline='') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        spamwriter.writerow(['ControlScalingFactor', 'SSE'])
        for i, factor in enumerate(factors):
            spamwriter.writerow([factor, diff_results[i]])


def run_optimisations(datapath):
    """Run OL and MPC frameworks using newly parameterised roguing rate."""

    with open(os.path.join("data", "scale_and_fit_results.json"), "r") as infile:
        scale_and_fit_results = json.load(infile)

    with open(os.path.join(datapath, "scaling_results.json"), "r") as infile:
        global_scaling = json.load(infile)

    setup, params = utils.get_setup_params(
        parameters.CORRECTED_PARAMS, scale_inf=True, host_props=parameters.COBB_PROP_FIG4A)

    beta_names = ['beta_1,1', 'beta_1,2', 'beta_1,3', 'beta_1,4', 'beta_12', 'beta_21', 'beta_2']
    beta = np.array([scale_and_fit_results[x] for x in beta_names])

    control_factor = global_scaling['roguing_factor']
    approx_params = copy.deepcopy(params)
    approx_params['rogue_rate'] *= control_factor
    approx_params['rogue_cost'] /= control_factor

    approx_model = ms_approx.MixedStandApprox(setup, approx_params, beta)

    def even_policy(time):
        return np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])

    _ = approx_model.optimise(n_stages=20, init_policy=even_policy)
    approx_model.save_optimisation(os.path.join(datapath, "OL_GloballyScaledControl.pkl"))
    ol_control = interp1d(setup['times'][:-1], approx_model.optimisation['control'], kind="zero",
                          fill_value="extrapolate")

    mpc_controller = mpc.Controller(setup, params, beta, approx_params=approx_params)
    _ = mpc_controller.optimise(
        horizon=100, time_step=0.5, end_time=100, update_period=20, rolling_horz=False, stage_len=5,
        init_policy=ol_control, use_init_first=True)
    mpc_controller.save_optimisation(os.path.join(datapath, "MPC_GloballyScaledControl.pkl"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-s", "--scan", action="store_true",
                        help="Run full scan over scaling values")
    args = parser.parse_args()

    data_path = os.path.join(os.path.realpath(__file__), '..', '..', 'data', 'global_optimal')

    os.makedirs(data_path, exist_ok=True)

    # Set up logs
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    # create file handler which logs info messages
    fh = logging.FileHandler(os.path.join(data_path, 'global_optimal.log'))
    fh.setLevel(logging.INFO)
    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(logging.WARNING)
    # create formatter and add it to the handlers
    formatter = logging.Formatter(
        '%(levelname)s | %(asctime)s | %(name)s:%(module)s:%(lineno)d | %(message)s')
    ch.setFormatter(formatter)
    fh.setFormatter(formatter)
    # add the handlers to logger
    logger.addHandler(ch)
    logger.addHandler(fh)

    if args.scan:
        run_scan(data_path)
    else:
        scale_control(data_path)
        run_optimisations(data_path)

    logging.info("Script completed")
