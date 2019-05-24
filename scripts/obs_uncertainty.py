"""Test robustness of OL and MPC to observational uncertainty."""

import argparse
import json
import logging
import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
from sklearn.utils.random import sample_without_replacement
from mixed_stand_model import mixed_stand_simulator as ms_sim
from mixed_stand_model import mixed_stand_approx as ms_approx
from mixed_stand_model import parameters
from mixed_stand_model import mpc
from mixed_stand_model import utils

def even_policy(time):
    """Even allocation across controls"""
    return np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])

def observer_factory(pop_size, n_samples):
    """Create observation function with specified sampling characteristics."""

    def observer(state):
        """Observe simulation state with uncertainty -> approximate state."""

        # First average over cells to get non-spatial distribution
        ncells = len(state) / 15
        mixed_state = np.sum(np.reshape(state, (int(ncells), 15)), axis=0) / ncells

        # Add empty space to be sampled also
        mixed_props = np.append(mixed_state, 1.0 - np.sum(mixed_state))

        # Create population of hosts and sample appropriately
        population = np.array(pop_size * mixed_props, dtype=int)
        sample = sample_without_replacement(pop_size, n_samples)
        bins = np.append([0.0], np.cumsum(population))
        obs_state = (np.histogram(sample, bins)[0] / n_samples)[:-1]

        return obs_state

    return observer

def make_data(n_reps=10, folder=None, append=False):
    """Generate data analysiing effect of sampling effort."""

    if folder is None:
        folder = os.path.join(os.path.realpath(__file__), '..', '..', 'data', 'obs_uncert')

    setup, params = utils.get_setup_params(
        parameters.CORRECTED_PARAMS, scale_inf=True, host_props=parameters.COBB_PROP_FIG4A)

    # In Cobb (2012) average plot density was 561 stems per ha
    # This simulation is over 20 ha.

    pop_size = np.product(setup['landscape_dims']) * 561 * 20 / np.sum(setup['state_init'])
    # sampling_nums = list(map(int, pop_size * np.geomspace(0.1, 1.0, 6)))[:-1]
    sampling_nums = list(map(int, pop_size * np.power(10, [-1.6, -1.8, -2.0])))

    with open(os.path.join("data", "scale_and_fit_results.json"), "r") as infile:
        scale_and_fit_results = json.load(infile)
    beta_names = ['beta_1,1', 'beta_1,2', 'beta_1,3', 'beta_1,4', 'beta_12', 'beta_21', 'beta_2']
    beta = np.array([scale_and_fit_results[x] for x in beta_names])

    mpc_args = {
        'horizon': 100,
        'time_step': 0.5,
        'end_time': 100,
        'update_period': 20,
        'rolling_horz': False,
        'stage_len': 5,
        'use_init_first': True
    }

    approx_model = ms_approx.MixedStandApprox(setup, params, beta)
    _, baseline_ol_control, _ = approx_model.optimise(n_stages=20, init_policy=even_policy)
    mpc_args['init_policy'] = baseline_ol_control

    for n_samples in sampling_nums:
        logging.info("Running with %d/%d stems sampled, %f%%", n_samples, pop_size,
                     100*n_samples/pop_size)
        observer = observer_factory(pop_size, n_samples)

        mpc_controls = np.zeros((n_reps, 9, len(setup['times']) - 1))
        mpc_objs = []
        # For storing observed states:
        mpc_approx_states = np.zeros((n_reps, 4, 15)) # 4 as 4 update steps excluding the start

        # MPC runs - approximate model initialised correctly, & then observed at update steps
        for i in range(n_reps):
            mpc_controller = mpc.Controller(setup, params, beta)
            _, _, mpc_control, mpc_obj = mpc_controller.optimise(**mpc_args, observer=observer)

            mpc_objs.append(mpc_obj)
            mpc_controls[i] = mpc_control
            mpc_approx_states[i] = mpc_controller.approx_update_states

            logging.info("MPC run %d of %d done", i+1, n_reps)

        # Append to existing data
        if append:
            old_filename = os.path.join(folder, "sampled_data_" + str(n_samples) + ".npz")

            with np.load(old_filename) as old_dataset:
                mpc_controls = np.append(old_dataset['mpc_controls'], mpc_controls, axis=0)
                mpc_objs = np.append(old_dataset['mpc_objs'], mpc_objs, axis=0)
                mpc_approx_states = np.append(old_dataset['mpc_approx_states'], mpc_approx_states,
                                              axis=0)

        # Store data
        dataset = {
            'mpc_controls': mpc_controls,
            'mpc_objs': mpc_objs,
            'mpc_approx_states': mpc_approx_states
        }
        np.savez_compressed(os.path.join(folder, "sampled_data_" + str(n_samples)), **dataset)

def make_plots(data_folder=None, fig_folder=None):
    """Generate plots of observational uncertainty analysis."""

    if data_folder is None:
        data_folder = os.path.join(os.path.realpath(__file__), '..', '..', 'data', 'obs_uncert')

    if fig_folder is None:
        fig_folder = os.path.join(os.path.realpath(__file__), '..', '..', 'figures', 'obs_uncert')

    setup, params = utils.get_setup_params(
        parameters.CORRECTED_PARAMS, scale_inf=True, host_props=parameters.COBB_PROP_FIG4A)

    sampling_effort = np.power(10, np.arange(-2, 0.2, 0.2))
    pop_size = np.product(setup['landscape_dims']) * 561 * 20 / np.sum(setup['state_init'])
    sampling_nums = list(map(int, pop_size * sampling_effort))[:-1]

    x_data = []
    mpc_objs = []
    avg_objs = []

    for n_samples, sample_prop in zip(sampling_nums, sampling_effort[:-1]):
        filename = os.path.join(data_folder, "sampled_data_" + str(n_samples) + ".npz")

        with np.load(filename) as dataset:
            mpc_objs = np.append(mpc_objs, dataset['mpc_objs'], axis=0)
            x_data.extend([sample_prop] * len(dataset['mpc_objs']))

            avg_objs.append(np.mean(dataset['mpc_objs']))

    mpc_controller = mpc.Controller.load_optimisation(
        os.path.join(data_folder, '..', "ol_mpc_control", "mpc_control_20.pkl"))
    mpc_sim_run, _ = mpc_controller.run_control()
    x_data.append(sampling_effort[-1])
    mpc_objs = np.append(mpc_objs, [mpc_sim_run[1]], axis=0)
    avg_objs.append(mpc_sim_run[1])

    approx_model = ms_approx.MixedStandApprox.load_optimisation_class(
        os.path.join(data_folder, '..', "ol_mpc_control", "ol_control.pkl"))
    ol_control_policy = interp1d(
        approx_model.setup['times'][:-1], approx_model.optimisation['control'], kind="zero",
        fill_value="extrapolate")

    sim_model = ms_sim.MixedStandSimulator(mpc_controller.setup, mpc_controller.params)
    ol_sim_run = sim_model.run_policy(ol_control_policy)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(sampling_effort, -1*np.array(avg_objs), '-', label='MPC mean', color='C1', alpha=0.75)
    ax.plot(x_data, -mpc_objs, 'o', label='MPC', color='C1')
    ax.axhline(-1*ol_sim_run[1], label='OL', linestyle='--', color='C0')

    ax.set_xlabel("Sampling effort")
    ax.set_ylabel("Objective")
    ax.set_xscale('log')

    ax.legend()
    fig.savefig(os.path.join(fig_folder, "obs_uncert.pdf"), dpi=600, bbox_inches='tight')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-f", "--folder", default='obs_uncert',
                        help="Folder name to save results in data and figures directory.")
    parser.add_argument("-n", "--n_reps", default=10, type=int,
                        help="Number of repeats for each level of sampling effort.")
    parser.add_argument("-a", "--append", action="store_true",
                        help="Flag to append to existing dataset")
    parser.add_argument("-e", "--use_existing_data", action="store_true",
                        help="Make plots only (no new data generated)")


    args = parser.parse_args()

    data_path = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), '..', 'data', args.folder)
    fig_path = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), '..', 'figures', args.folder)

    os.makedirs(data_path, exist_ok=True)
    os.makedirs(fig_path, exist_ok=True)

    # Set up logs
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    # create file handler which logs info messages
    fh = logging.FileHandler(os.path.join(data_path, 'obs_uncert.log'))
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

    if args.use_existing_data:
        make_plots(data_path, fig_path)
    else:
        make_data(n_reps=args.n_reps, folder=data_path, append=args.append)

    logging.info("Script completed")
