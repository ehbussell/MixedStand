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

def main(filename):
    """Run fitting process: scale simulations and fit approximate models."""

    # First find beta scaling that gives matching time scale using corrected model
    scaling_factor = fitting.scale_sim_model(time_step=0.001)

    logging.info("Simulation scaling factor found: %f", scaling_factor)

    # Write scaling results to file
    results = {'sim_scaling_factor': scaling_factor}
    with open(filename+'.json', "w") as outfile:
        json.dump(results, outfile, indent=4)

    # Now fit approximate model with Bay epidemiologically inactive
    # This ensures the within tanoak infection rates are identifiable, and we fix the relative rates
    # for the full fit.

    # First get parameters and setup
    params = copy.deepcopy(parameters.CORRECTED_PARAMS)
    inf_keys = ['inf_tanoak_tanoak', 'inf_tanoak_to_bay', 'inf_bay_to_bay', 'inf_bay_to_tanoak']
    for key in inf_keys:
        params[key] *= scaling_factor
    # Initialise using Cobb 2012 Fig 4a host proportions
    host_props = parameters.COBB_PROP_FIG4A
    params, state_init = utils.initialise_params(params, host_props=host_props)
    state_init = np.tile(state_init, 400)

    # Now set Bay to be epidemiologically inactive
    params['inf_tanoak_to_bay'] = 0.0
    params['inf_bay_to_tanoak'] = 0.0
    params['inf_bay_to_bay'] = 0.0

    # Set initial infection level
    init_inf_cells = [189]
    init_inf_factor = 0.5
    for cell_pos in init_inf_cells:
        for i in [0, 4]:
            state_init[cell_pos*15+3*i+1] = init_inf_factor * state_init[cell_pos*15+3*i]
            state_init[cell_pos*15+3*i] *= (1.0 - init_inf_factor)

    setup = {
        'state_init': state_init,
        'landscape_dims': (20, 20),
        'times': np.linspace(0, 100, 201)
    }

    logging.info("Using landscape dimensions (%d, %d)", *setup['landscape_dims'])
    logging.info("Using initial infection in cells %s", init_inf_cells)
    logging.info("Using initial infection proportion %f", init_inf_factor)

    fitter = fitting.MixedStandFitter(setup, params)

    start = np.array([2.5, 0.4, 0.4, 0.4, 0.0, 0.0, 0.0])
    bounds = [(1e-10, 20)] * 4 + [(0, 0)] *3
    _, beta = fitter.fit(start, bounds, show_plot=False)

    logging.info("Completed fit with Bay inactive")

    tanoak_factors = beta[1:4] / beta[0]

    logging.info("Tanoak relative infection rates: %s", tanoak_factors)

    results['tanoak_beta_factors'] = tanoak_factors.tolist()
    with open(filename+'.json', "w") as outfile:
        json.dump(results, outfile, indent=4)

    params = copy.deepcopy(parameters.CORRECTED_PARAMS)
    inf_keys = ['inf_tanoak_tanoak', 'inf_tanoak_to_bay', 'inf_bay_to_bay', 'inf_bay_to_tanoak']
    for key in inf_keys:
        params[key] *= scaling_factor
    # Initialise using Cobb 2012 Fig 4a host proportions
    host_props = parameters.COBB_PROP_FIG4A
    params, _ = utils.initialise_params(params, host_props=host_props)

    fitter = fitting.MixedStandFitter(setup, params)

    start = np.array([2.5, 0.4, 0.4, 0.4, 3.5, 0.3, 4.6])
    bounds = [(1e-10, 20)] * 7
    _, beta = fitter.fit(start, bounds, show_plot=False, tanoak_factors=tanoak_factors)

    logging.info("Approximate model beta values found")

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
