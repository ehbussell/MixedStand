"""Test robustness of OL and MPC to observational uncertainty."""

import argparse
import copy
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
        ncells = int(np.round(len(state) / 15, 0))
        state_by_cell = np.reshape(state, (int(ncells), 15))

        # Species proportions
        species_by_cell = np.array([
            np.sum(state_by_cell[:, 0:3], axis=1),
            np.sum(state_by_cell[:, 3:6], axis=1),
            np.sum(state_by_cell[:, 6:9], axis=1),
            np.sum(state_by_cell[:, 9:12], axis=1),
            np.sum(state_by_cell[:, 12:14], axis=1),
            state_by_cell[:, 14]
        ]).T

        # Add empty space to be sampled also
        space = np.array(1.0 - np.sum(state_by_cell, axis=1)).reshape((400, 1))
        cell_props = np.append(species_by_cell, space, axis=1)

        # Create population of hosts and sample appropriately
        population = np.array(pop_size * cell_props)
        obs_states = []
        for i in range(ncells):
            sample = sample_without_replacement(pop_size, n_samples)
            bins = np.append([0.0], np.cumsum(population[i]))
            observed_species = np.histogram(sample, bins)[0]

            observed_state = np.zeros(15)

            # Tanoak
            for j in range(4):
                idcs = ((3*j), (3*j+3))
                inf_probs = state_by_cell[i, idcs[0]:idcs[1]] / np.sum(
                    state_by_cell[i, idcs[0]:idcs[1]])
                inf_sample = np.random.choice(3, observed_species[j], p=inf_probs)
                observed_state[idcs[0]:idcs[1]] = np.histogram(inf_sample, range(4))[0]

            # Bay
            inf_probs = state_by_cell[i, 12:14] / np.sum(state_by_cell[i, 12:14])
            inf_sample = np.random.choice(2, observed_species[4], p=inf_probs)
            observed_state[12:14] = np.histogram(inf_sample, range(3))[0]

            # Redwood
            observed_state[14] = observed_species[5]

            obs_states.append(observed_state)

        obs_states = np.array(obs_states)
        obs_state = (np.sum(obs_states, axis=0) / (n_samples * ncells))

        return obs_state

    return observer

def make_data(n_reps=10, folder=None, append=False):
    """Generate data analysing effect of sampling effort."""

    if folder is None:
        folder = os.path.join(os.path.realpath(__file__), '..', '..', 'data', 'obs_uncert')

    setup, params = utils.get_setup_params(
        parameters.CORRECTED_PARAMS, scale_inf=True, host_props=parameters.COBB_PROP_FIG4A)

    ncells = np.product(setup['landscape_dims'])

    # Use population size of 500 as 500m2 per cell
    pop_size = 500
    sampling_nums = np.array([1, 2, 3, 5, 7, 10, 15, 25, 35, 50, 70, 100, 150, 250, 350, 500])

    with open(os.path.join("data", "scale_and_fit_results.json"), "r") as infile:
        scale_and_fit_results = json.load(infile)
    beta_names = ['beta_1,1', 'beta_1,2', 'beta_1,3', 'beta_1,4', 'beta_12', 'beta_21', 'beta_2']
    beta = np.array([scale_and_fit_results[x] for x in beta_names])

    approx_params = copy.deepcopy(params)
    approx_params['rogue_rate'] *= scale_and_fit_results['roguing_factor']
    approx_params['rogue_cost'] /= scale_and_fit_results['roguing_factor']

    mpc_args = {
        'horizon': 100,
        'time_step': 0.5,
        'end_time': 100,
        'update_period': 20,
        'rolling_horz': False,
        'stage_len': 5,
        'use_init_first': True
    }

    approx_model = ms_approx.MixedStandApprox(setup, approx_params, beta)
    _, baseline_ol_control, _ = approx_model.optimise(n_stages=20, init_policy=even_policy)
    mpc_args['init_policy'] = baseline_ol_control

    for n_samples in sampling_nums:
        logging.info("Running with %d/%d stems sampled, %f%%", n_samples, pop_size,
                     100*n_samples/pop_size)
        observer = observer_factory(pop_size, n_samples)

        mpc_controls = np.zeros((n_reps, 9, len(setup['times']) - 1))
        mpc_objs = np.zeros(n_reps)

        # For storing observed states:
        mpc_approx_states = np.zeros((n_reps, 4, 15)) # 4 as 4 update steps excluding the start
        mpc_actual_states = np.zeros((n_reps, 4, 15))

        # MPC runs - approximate model initialised correctly, & then observed at update steps
        for i in range(n_reps):


            mpc_controller = mpc.Controller(setup, params, beta, approx_params=approx_params)
            _, _, mpc_control, mpc_obj = mpc_controller.optimise(**mpc_args, observer=observer)

            mpc_objs[i] = mpc_obj
            mpc_controls[i] = mpc_control
            mpc_approx_states[i] = mpc_controller.approx_update_states

            sim_run, approx_run = mpc_controller.run_control()
            sim_state = np.sum(np.reshape(sim_run[0], (ncells, 15, -1)), axis=0) / ncells
            mpc_actual_states[i] = sim_state[:, [40, 80, 120, 160]].T

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
            'mpc_approx_states': mpc_approx_states,
            'mpc_actual_states': mpc_actual_states
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

    sampling_effort = np.array([0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0])
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

    ax.legend()
    fig.savefig(os.path.join(fig_folder, "param_uncert.pdf"), dpi=600, bbox_inches='tight')


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
