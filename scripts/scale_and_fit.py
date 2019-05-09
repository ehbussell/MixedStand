"""Script for scaling corrected version of Cobb model, and fitting of approximate model."""

import argparse
import copy
import json
import logging
import os
import numpy as np

from mixed_stand_model import fitting
from mixed_stand_model import parameters
from mixed_stand_model import utils

def fit_beta(setup, params, no_bay_dataset=None, with_bay_dataset=None):
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

    start = np.array([2.5, 0.4, 0.4, 0.4, 0.0, 0.0, 0.0])
    bounds = [(1e-10, 20)] * 4 + [(0, 0)] * 3
    _, beta = fitter.fit(start, bounds, show_plot=False, dataset=no_bay_dataset)

    logging.info("Completed fit with Bay inactive")

    tanoak_factors = beta[1:4] / beta[0]

    logging.info("Tanoak relative infection rates: %s", tanoak_factors)

    model_params = copy.deepcopy(params)
    fitter = fitting.MixedStandFitter(model_setup, model_params)

    start = np.array([2.5, 0.4, 0.4, 0.4, 3.5, 0.3, 4.6])
    bounds = [(1e-10, 20)] * 7
    _, beta = fitter.fit(start, bounds, show_plot=False, tanoak_factors=tanoak_factors,
                         dataset=with_bay_dataset)

    logging.info("Approximate model beta values found")
    logging.info("Beta: %s", beta)

    return tanoak_factors, beta

def main(filename):
    """Run fitting process: scale simulations and fit approximate models."""

    # Set parameters and initial conditions
    _, old_params = utils.get_setup_params(
        parameters.COBB_PARAMS, scale_inf=False, host_props=parameters.COBB_PROP_FIG4A)
    setup, new_params = utils.get_setup_params(
        parameters.CORRECTED_PARAMS, scale_inf=False, host_props=parameters.COBB_PROP_FIG4A)

    # First find beta scaling that gives matching time scale using corrected model
    scaling_factor = fitting.scale_sim_model(setup, old_params, new_params, time_step=0.001)

    logging.info("Simulation scaling factor found: %f", scaling_factor)

    # Write scaling results to file
    results = {'sim_scaling_factor': scaling_factor}
    with open(filename+'.json', "w") as outfile:
        json.dump(results, outfile, indent=4)
    
    setup, params = utils.get_setup_params(
        parameters.CORRECTED_PARAMS, scale_inf=True, host_props=parameters.COBB_PROP_FIG4A)

    tanoak_factors, beta = fit_beta(setup, params)

    results['tanoak_beta_factors'] = tanoak_factors.tolist()
    beta_names = ['beta_1,1', 'beta_1,2', 'beta_1,3', 'beta_1,4', 'beta_12', 'beta_21', 'beta_2']
    for i, name in enumerate(beta_names):
        results[name] = beta[i]
        logging.info("%s: %f", name, beta[i])
    with open(filename+'.json', "w") as outfile:
        json.dump(results, outfile, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)

    args = parser.parse_args()

    filepath = os.path.join(os.path.realpath(__file__), '..', '..', 'data', 'scale_and_fit_results')

    # Set up logs
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    # create file handler which logs info messages
    fh = logging.FileHandler(filepath+'.log', mode='w')
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

    main(filepath)

    logging.info("Script completed")
