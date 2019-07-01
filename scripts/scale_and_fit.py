"""Script for scaling corrected version of Cobb model, and fitting of approximate model."""

import argparse
import copy
import csv
import json
import logging
import os
import numpy as np
from scipy.optimize import minimize

from mixed_stand_model import fitting
from mixed_stand_model import parameters
from mixed_stand_model import utils
from mixed_stand_model import mixed_stand_simulator as ms_sim
from mixed_stand_model import mixed_stand_approx as ms_approx

def const_rogue_policy(state):
    """Even allocation of resources across roguing controls."""
    return np.array([1.0]*3 + [0.0]*4 + [0.0]*2)

def fit_beta(setup, params, no_bay_dataset=None, with_bay_dataset=None, start=None):
    """Fit approx model beta values."""

    # First fit approximate model with Bay epidemiologically inactive
    # This ensures the within tanoak infection rates are identifiable, and we fix the relative rates
    # for the full fit.

    # First get parameters and setup
    model_setup = copy.deepcopy(setup)
    model_params = copy.deepcopy(params)

    # Now set Bay to be epidemiologically inactive
    model_params['inf_tanoak_to_bay'] = 0.0
    model_params['inf_bay_to_tanoak'] = 0.0
    model_params['inf_bay_to_bay'] = 0.0

    fitter = fitting.MixedStandFitter(model_setup, model_params)

    if start is None:
        start = np.array([0.6, 0.6, 0.5, 0.4, 0.0, 0.0, 0.0])
    bounds = [(0.05, 20)] * 4 + [(0, 0)] * 3
    _, beta = fitter.fit(start, bounds, show_plot=False, dataset=no_bay_dataset)

    logging.info("Completed fit with Bay inactive")

    tanoak_factors = beta[1:4] / beta[0]

    logging.info("Tanoak infection rates: %s", tanoak_factors)

    model_params = copy.deepcopy(params)
    fitter = fitting.MixedStandFitter(model_setup, model_params)

    if start is None:
        start = np.array([0.6, 0.6, 0.4, 0.3, 4.0, 0.15, 4.0])
    bounds = [(0.05, 20)] * 7
    _, beta = fitter.fit(start, bounds, show_plot=False, tanoak_factors=tanoak_factors,
                         dataset=with_bay_dataset)

    logging.info("Approximate model beta values found")
    logging.info("Beta: %s", beta)

    return tanoak_factors, beta

def scale_control():
    """Parameterise roguing control in approximate model to match simulations."""

    test_rates = np.linspace(0, 1.0, 51)

    # First run simulations for range of roguing rates, with constant control rates
    setup, params = utils.get_setup_params(
        parameters.CORRECTED_PARAMS, scale_inf=True, host_props=parameters.COBB_PROP_FIG4A,
        extra_spread=False)

    params['max_budget'] = 1000

    with open(os.path.join("data", "scale_and_fit_results.json"), "r") as infile:
        scale_and_fit_results = json.load(infile)

    sim_tans = np.zeros_like(test_rates)
    ncells = np.product(setup['landscape_dims'])

    for i, rate in enumerate(test_rates):
        params['rogue_rate'] = rate
        model = ms_sim.MixedStandSimulator(setup, params)

        sim_run = model.run_policy(const_rogue_policy)
        sim_state = np.sum(sim_run[0].reshape((ncells, 15, -1)), axis=0) / ncells
        sim_tans[i] = np.sum(sim_state[[6, 8, 9, 11], -1])

        logging.info("Sim run, rate: %f, healthy tans: %f", rate, sim_tans[i])

    def min_func(factor):
        """Function to minimise, SSE between healthy tanoak over range of rates."""

        approx_tans = np.zeros_like(test_rates)

        setup, params = utils.get_setup_params(
            parameters.CORRECTED_PARAMS, scale_inf=True, host_props=parameters.COBB_PROP_FIG4A,
            extra_spread=False)
        params['max_budget'] = 1000

        beta_names = ['beta_1,1', 'beta_1,2', 'beta_1,3', 'beta_1,4',
                      'beta_12', 'beta_21', 'beta_2']
        beta = np.array([scale_and_fit_results[x] for x in beta_names])

        for i, rate in enumerate(test_rates):
            params['rogue_rate'] = rate * factor
            approx_model = ms_approx.MixedStandApprox(setup, params, beta)
            approx_run = approx_model.run_policy(const_rogue_policy)

            approx_tans[i] = np.sum(approx_run[0][[6, 8, 9, 11], -1])

            logging.info("Approx run, Factor %f, Rate: %f, tans: %f", factor, rate, approx_tans[i])

        return np.sum(np.square(approx_tans - sim_tans))

    ret = minimize(min_func, [1.0], bounds=[(0, 2)])

    logging.info(ret)

    return ret.x[0]

def main(filename):
    """Run fitting process: scale simulations and fit approximate models."""

    # Set parameters and initial conditions
    _, old_params = utils.get_setup_params(
        parameters.COBB_PARAMS, scale_inf=False, host_props=parameters.COBB_PROP_FIG4A,
        extra_spread=False)
    setup, new_params = utils.get_setup_params(
        parameters.CORRECTED_PARAMS, scale_inf=False, host_props=parameters.COBB_PROP_FIG4A,
        extra_spread=False)

    # First find beta scaling that gives matching time scale using corrected model
    scaling_factor = fitting.scale_sim_model(setup, old_params, new_params, time_step=0.001)

    logging.info("Simulation scaling factor found: %f", scaling_factor)

    # Write scaling results to file
    results = {'sim_scaling_factor': scaling_factor}
    with open(filename+'.json', "w") as outfile:
        json.dump(results, outfile, indent=4)

    setup, params = utils.get_setup_params(
        parameters.CORRECTED_PARAMS, scale_inf=True, host_props=parameters.COBB_PROP_FIG4A,
        extra_spread=False)

    tanoak_factors, beta = fit_beta(setup, params)

    results['tanoak_beta_factors'] = tanoak_factors.tolist()
    beta_names = ['beta_1,1', 'beta_1,2', 'beta_1,3', 'beta_1,4', 'beta_12', 'beta_21', 'beta_2']
    for i, name in enumerate(beta_names):
        results[name] = beta[i]
        logging.info("%s: %f", name, beta[i])
    with open(filename+'.json', "w") as outfile:
        json.dump(results, outfile, indent=4)

    roguing_factor = scale_control()
    results['roguing_factor'] = roguing_factor
    with open(filename+'.json', "w") as outfile:
        json.dump(results, outfile, indent=4)

def run_scan(filename):
    """Scan over scaling factors."""

    factors = np.arange(0.5, 5.05, 0.05)

    setup, new_params = utils.get_setup_params(
        parameters.CORRECTED_PARAMS, scale_inf=False, host_props=parameters.COBB_PROP_FIG4A,
        extra_spread=False)
    setup['times'] = np.arange(0, 50, step=0.01)

    cross_over_times = []

    for factor in factors:
        params = copy.deepcopy(new_params)
        params['inf_tanoak_tanoak'] *= factor
        params['inf_bay_to_bay'] *= factor
        params['inf_bay_to_tanoak'] *= factor
        params['inf_tanoak_to_bay'] *= factor
        model = ms_sim.MixedStandSimulator(setup, params)

        sim_run, *_ = model.run_policy(control_policy=None)
        logging.info("Done sim")
        cross_over_time = fitting._get_crossover_time(sim_run, model.ncells, setup['times'])
        cross_over_times.append(cross_over_time)

        logging.info("Factor: %f, cross-over time: %f", factor, cross_over_time)

    csv_file = filename + '_scan.csv'
    with open(csv_file, 'w', newline='') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        spamwriter.writerow(['ScalingFactor', 'CrossOverTime'])
        for i, factor in enumerate(factors):
            spamwriter.writerow([factor, cross_over_times[i]])

def run_scan_control(filename):
    """Scan over control scaling factor."""

    factors = np.arange(0.5, 1.51, 0.01)
    test_rates = np.linspace(0, 1.0, 51)

    diff_results = np.zeros_like(factors)

    # Run simulations for range of roguing rates, with constant control rates
    setup, params = utils.get_setup_params(
        parameters.CORRECTED_PARAMS, scale_inf=True, host_props=parameters.COBB_PROP_FIG4A,
        extra_spread=False)

    params['max_budget'] = 1000

    with open(os.path.join("data", "scale_and_fit_results.json"), "r") as infile:
        scale_and_fit_results = json.load(infile)

    sim_tans = np.zeros_like(test_rates)
    ncells = np.product(setup['landscape_dims'])

    for i, rate in enumerate(test_rates):
        params['rogue_rate'] = rate
        model = ms_sim.MixedStandSimulator(setup, params)

        sim_run = model.run_policy(const_rogue_policy)
        sim_state = np.sum(sim_run[0].reshape((ncells, 15, -1)), axis=0) / ncells
        sim_tans[i] = np.sum(sim_state[[6, 8, 9, 11], -1])

        logging.info("Sim run, rate: %f, healthy tans: %f", rate, sim_tans[i])

    def min_func(factor):
        """Function to minimise, SSE between healthy tanoak over range of rates."""

        approx_tans = np.zeros_like(test_rates)

        setup, params = utils.get_setup_params(
            parameters.CORRECTED_PARAMS, scale_inf=True, host_props=parameters.COBB_PROP_FIG4A,
            extra_spread=False)
        params['max_budget'] = 1000

        beta_names = ['beta_1,1', 'beta_1,2', 'beta_1,3', 'beta_1,4',
                      'beta_12', 'beta_21', 'beta_2']
        beta = np.array([scale_and_fit_results[x] for x in beta_names])

        for i, rate in enumerate(test_rates):
            params['rogue_rate'] = rate * factor
            approx_model = ms_approx.MixedStandApprox(setup, params, beta)
            approx_run = approx_model.run_policy(const_rogue_policy)

            approx_tans[i] = np.sum(approx_run[0][[6, 8, 9, 11], -1])

            logging.info("Approx run, Factor %f, Rate: %f, tans: %f", factor, rate, approx_tans[i])

        return np.sum(np.square(approx_tans - sim_tans))

    for i, factor in enumerate(factors):
        diff = min_func(factor)
        diff_results[i] = diff

    csv_file = filename + '_scan_control.csv'
    with open(csv_file, 'w', newline='') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        spamwriter.writerow(['ControlScalingFactor', 'SSE'])
        for i, factor in enumerate(factors):
            spamwriter.writerow([factor, diff_results[i]])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-s", "--scan", action="store_true",
                        help="Run full scan over scaling values")
    args = parser.parse_args()

    filepath = os.path.join(os.path.realpath(__file__), '..', '..', 'data', 'scale_and_fit_results')

    # Set up logs
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    # create file handler which logs info messages
    fh = logging.FileHandler(filepath+'.log')
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
        run_scan(filepath)
        run_scan_control(filepath)
    else:
        main(filepath)

    logging.info("Script completed")
