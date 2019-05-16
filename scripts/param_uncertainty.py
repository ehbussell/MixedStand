"""Test robustness of OL and MPC to parameter uncertainty."""

import argparse
import logging
import os
import numpy as np
from scipy.stats import truncnorm
from scipy.interpolate import interp1d
from mixed_stand_model import mixed_stand_simulator as ms_sim
from mixed_stand_model import mixed_stand_approx as ms_approx
from mixed_stand_model import parameters
from mixed_stand_model import mpc
from mixed_stand_model import utils
from scripts import scale_and_fit

def even_policy(time):
    """Even allocation across controls"""
    return np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])

def generate_ensemble_and_fit(setup, params, n_ensemble_runs, standard_dev):
    """Generate ensemble of simulation runs and fit approximate model."""

    model = ms_sim.MixedStandSimulator(setup, params)

    ncells = np.product(setup['landscape_dims'])

    # Create parameter error distribution
    if standard_dev != 0.0:
        error_dist = truncnorm(-1.0/standard_dev, np.inf, loc=1.0, scale=standard_dev)
        error_samples = np.reshape(error_dist.rvs(size=n_ensemble_runs*7), (n_ensemble_runs, 7))
    else:
        n_ensemble_runs = 1
        error_samples = np.ones((n_ensemble_runs, 7))

    # Sample from parameter distribution and run simulations
    baseline_beta = np.zeros(7)
    baseline_beta[:4] = params['inf_tanoak_tanoak']
    baseline_beta[4] = params['inf_bay_to_tanoak']
    baseline_beta[5] = params['inf_tanoak_to_bay']
    baseline_beta[6] = params['inf_bay_to_bay']
    parameter_samples = error_samples * baseline_beta

    logging.info("Generated ensemble parameters for fitting")

    simulation_runs = np.zeros((n_ensemble_runs, 15, len(setup['times'])))

    simulation_runs_no_cross_trans = np.zeros((n_ensemble_runs, 15, len(setup['times'])))

    for i in range(n_ensemble_runs):
        model.params['inf_tanoak_tanoak'] = parameter_samples[i, 0:4]
        model.params['inf_bay_to_tanoak'] = parameter_samples[i, 4]
        model.params['inf_tanoak_to_bay'] = parameter_samples[i, 5]
        model.params['inf_bay_to_bay'] = parameter_samples[i, 6]

        sim_run, *_ = model.run_policy()
        simulation_runs[i] = np.sum(sim_run.reshape((ncells, 15, -1)), axis=0) / ncells

        model.params['inf_bay_to_tanoak'] = 0.0
        model.params['inf_tanoak_to_bay'] = 0.0
        model.params['inf_bay_to_bay'] = 0.0

        sim_run, *_ = model.run_policy()
        simulation_runs_no_cross_trans[i] = np.sum(
            sim_run.reshape((ncells, 15, -1)), axis=0) / ncells

        logging.info("Run %d of %d done.", i+1, n_ensemble_runs)

    ret_dict = {
        'params': parameter_samples,
        'sims': simulation_runs,
        'sims_no_cross_trans': simulation_runs_no_cross_trans,
        'fit': None
    }

    _, beta = scale_and_fit.fit_beta(setup, params, no_bay_dataset=simulation_runs_no_cross_trans,
                                     with_bay_dataset=simulation_runs)

    ret_dict['fit'] = beta

    return ret_dict

def run_optimisations(ensemble_and_fit, params, setup, n_optim_runs, standard_dev, mpc_args, ol_pol=None):
    """Run open-loop and MPC optimisations over parameter distributions"""

    approx_model = ms_approx.MixedStandApprox(setup, params, ensemble_and_fit['fit'])
    sim_model = ms_sim.MixedStandSimulator(setup, params)

    if ol_pol is None:
        _, ol_control, _ = approx_model.optimise(n_stages=20, init_policy=even_policy)
    else:
        ol_control = ol_pol

    # Create parameter error distribution
    if standard_dev != 0.0:
        error_dist = truncnorm(-1.0/standard_dev, np.inf, loc=1.0, scale=standard_dev)
        error_samples = np.reshape(error_dist.rvs(size=n_optim_runs*7), (n_optim_runs, 7))
    else:
        n_optim_runs = 1
        error_samples = np.ones((n_optim_runs, 7))

    # Sample from parameter distribution and run simulations
    baseline_beta = np.zeros(7)
    baseline_beta[:4] = params['inf_tanoak_tanoak']
    baseline_beta[4] = params['inf_bay_to_tanoak']
    baseline_beta[5] = params['inf_tanoak_to_bay']
    baseline_beta[6] = params['inf_bay_to_bay']
    parameter_samples = error_samples * baseline_beta

    logging.info("Generated ensemble parameters for optimisation runs")

    ol_objs = np.zeros(n_optim_runs)
    for i in range(n_optim_runs):
        sim_model.params['inf_tanoak_tanoak'] = parameter_samples[i, 0:4]
        sim_model.params['inf_bay_to_tanoak'] = parameter_samples[i, 4]
        sim_model.params['inf_tanoak_to_bay'] = parameter_samples[i, 5]
        sim_model.params['inf_bay_to_bay'] = parameter_samples[i, 6]

        _, obj, _ = sim_model.run_policy(control_policy=ol_control)
        ol_objs[i] = obj

        logging.info("Open-loop run %d of %d done.", i+1, n_optim_runs)

    mpc_args['init_policy'] = ol_control

    mpc_objs = np.zeros(n_optim_runs)
    mpc_controls = np.zeros((n_optim_runs, 9, len(setup['times']) - 1))
    for i in range(n_optim_runs):
        sim_model.params['inf_tanoak_tanoak'] = parameter_samples[i, 0:4]
        sim_model.params['inf_bay_to_tanoak'] = parameter_samples[i, 4]
        sim_model.params['inf_tanoak_to_bay'] = parameter_samples[i, 5]
        sim_model.params['inf_bay_to_bay'] = parameter_samples[i, 6]

        mpc_controller = mpc.Controller(setup, sim_model.params, ensemble_and_fit['fit'])
        _, _, mpc_control, mpc_obj = mpc_controller.optimise(**mpc_args)
        mpc_objs[i] = mpc_obj
        mpc_controls[i] = mpc_control

        logging.info("MPC run %d of %d done.", i+1, n_optim_runs)

    ret_dict = {
        'params': parameter_samples,
        'ol_control': np.array([ol_control(t) for t in setup['times'][:-1]]).T,
        'ol_objs': ol_objs,
        'mpc_control': mpc_controls,
        'mpc_objs': mpc_objs
    }

    return ret_dict

def run_all(n_ens=10, n_opt=10, folder=None, append=False):
    """Run analysis for all error standard deviation values."""

    if folder is None:
        folder = os.path.join(os.path.realpath(__file__), '..', '..', 'data', 'param_uncert')

    error_std_devs = np.append(np.linspace(0.0, 0.5, 11), [0.75, 1.0])

    for std_dev in error_std_devs:
        logging.info("Starting analysis with %f standard deviation", std_dev)
        # Initial conditions used in 2012 paper
        setup, params = utils.get_setup_params(
            parameters.CORRECTED_PARAMS, scale_inf=True, host_props=parameters.COBB_PROP_FIG4A)

        mpc_args = {
            'horizon': 100,
            'time_step': 0.5,
            'end_time': 100,
            'update_period': 20,
            'rolling_horz': False,
            'stage_len': 5,
            'init_policy': even_policy,
            'use_init_first': True
        }

        if append:
            logging.info("Reading in existing fit ensemble.")
            ensemble_and_fit = {}
            filename = os.path.join(folder, "fitting_ensemble_data_" + str(std_dev) + ".npz")
            with np.load(filename) as data:
                for key in data.keys():
                    ensemble_and_fit[key] = data[key]
        else:
            ensemble_and_fit = generate_ensemble_and_fit(setup, params, n_ens, std_dev)
            np.savez_compressed(os.path.join(folder, "fitting_ensemble_data_" + str(std_dev)),
                                **ensemble_and_fit)


        if append:
            logging.info("Appending to existing dataset.")

            if std_dev == 0.0:
                logging.info("Std dev=0.0 - no repeats to run.")
                continue

            full_optimisations = {}
            filename = os.path.join(folder, "optimisation_data_" + str(std_dev) + ".npz")
            with np.load(filename) as data:
                for key in data.keys():
                    full_optimisations[key] = data[key]

            ol_pol = interp1d(setup['times'][:-1], full_optimisations['ol_control'].T,
                              kind="zero", fill_value="extrapolate")

            optimisations = run_optimisations(
                ensemble_and_fit, params, setup, n_opt, std_dev, mpc_args, ol_pol=ol_pol)

            for key in ['params', 'ol_objs', 'mpc_control', 'mpc_objs']:
                full_optimisations[key] = np.append(
                    full_optimisations[key], optimisations[key], axis=0)

            np.savez_compressed(os.path.join(folder, "optimisation_data_" + str(std_dev)), **full_optimisations)

        else:
            optimisations = run_optimisations(
                ensemble_and_fit, params, setup, n_opt, std_dev, mpc_args)

            np.savez_compressed(os.path.join(folder, "optimisation_data_" + str(std_dev)),
                                **optimisations)

        logging.info("Completed analysis with %f standard deviation", std_dev)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-f", "--folder", default='param_uncert',
                        help="Folder name to save results in data and figures directory.")
    parser.add_argument("-n", "--n_ens", default=10, type=int,
                        help="Number of parameter sets to generate for ensemble fitting")
    parser.add_argument("-m", "--n_opt", default=10, type=int,
                        help="Number of parameter sets to generate for ensemble optimisation")
    parser.add_argument("-a", "--append", action="store_true",
                        help="Flag to append to existing dataset")
    args = parser.parse_args()


    data_path = os.path.join(
        os.path.realpath(__file__), '..', '..', 'data', args.folder)

    os.makedirs(data_path, exist_ok=True)

    # Set up logs
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    # create file handler which logs info messages
    fh = logging.FileHandler(os.path.join(data_path, 'param_uncert.log'))
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

    logging.info("Starting script with args: %r", args)

    run_all(n_ens=args.n_ens, n_opt=args.n_opt, folder=data_path, append=args.append)

    logging.info("Script completed")
