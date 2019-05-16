"""Test robustness of OL and MPC to observational uncertainty."""

import argparse
import copy
import json
import logging
import os
import numpy as np
from sklearn.utils.random import sample_without_replacement
from mixed_stand_model import mixed_stand_simulator as ms_sim
from mixed_stand_model import mixed_stand_approx as ms_approx
from mixed_stand_model import parameters
from mixed_stand_model import mpc
from mixed_stand_model import utils

def even_policy(time):
    """Even allocation across controls"""
    return np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])

def observation_factory(pop_size, n_samples):
    """Create observation function with specified sampling characteristics."""

    def observation(state):
        """Observe simulation state with uncertainty -> approximate state."""

        ncells = len(state) / 15
        mixed_state = np.sum(np.reshape(state, (int(ncells), 15)), axis=0) / ncells

        density = np.sum(mixed_state)
        mixed_props = np.append(mixed_state, 1.0 - density)

        population = np.array(pop_size * mixed_props, dtype=int)
        sample = sample_without_replacement(pop_size, n_samples)
        bins = np.append([0.0], np.cumsum(population))
        obs_state = (np.histogram(sample, bins)[0] / n_samples)[:-1]

        return obs_state

    return observation

def make_data(n_reps=10, folder=None, append=False):
    """Generate data analysiing effect of sampling effort."""

    if folder is None:
        folder = os.path.join(os.path.realpath(__file__), '..', '..', 'data', 'obs_uncert')

    setup, params = utils.get_setup_params(
        parameters.CORRECTED_PARAMS, scale_inf=True, host_props=parameters.COBB_PROP_FIG4A)

    # In Cobb (2012) average plot density was 561 stems per ha
    # This simulation is over 20 ha.
    pop_size = 561 * 20 / np.sum(setup['state_init'])
    sampling_nums = list(map(int, pop_size * np.geomspace(0.01, 1.0, 11)))[:-1]

    with open(os.path.join("data", "scale_and_fit_results.json"), "r") as infile:
        scale_and_fit_results = json.load(infile)
    beta_names = ['beta_1,1', 'beta_1,2', 'beta_1,3', 'beta_1,4', 'beta_12', 'beta_21', 'beta_2']
    beta = np.array([scale_and_fit_results[x] for x in beta_names])

    setup, params = utils.get_setup_params(
        parameters.CORRECTED_PARAMS, scale_inf=True, host_props=parameters.COBB_PROP_FIG4A)

    mpc_args = {
        'horizon': 100,
        'time_step': 0.5,
        'end_time': 100,
        'update_period': 20,
        'rolling_horz': False,
        'stage_len': 5,
        'use_init_first': False
    }

    sim_model = ms_sim.MixedStandSimulator(setup, params)

    approx_model = ms_approx.MixedStandApprox(setup, params, beta)
    _, baseline_ol_control, _ = approx_model.optimise(n_stages=20, init_policy=even_policy)
    mpc_args['init_policy'] = baseline_ol_control

    for n_samples in sampling_nums:
        logging.info("Running with %d/%d stems sampled", n_samples, pop_size)
        observer = observation_factory(pop_size, n_samples)

        ol_controls = np.zeros((n_reps, 9, len(setup['times']) - 1))
        mpc_controls = np.zeros((n_reps, 9, len(setup['times']) - 1))
        ol_objs = []
        mpc_objs = []
        # For storing observed states:
        ol_approx_states = np.zeros((n_reps, 15))
        mpc_approx_states = np.zeros((n_reps, 5, 15)) # 5 as 5 update steps including the start

        # Open loop runs - approximate model initialised using observation function
        for i in range(n_reps):
            approx_setup = copy.deepcopy(setup)
            approx_setup['state_init'] = observer(setup['state_init'])
            ol_approx_states[i] = approx_setup['state_init']

            approx_model = ms_approx.MixedStandApprox(approx_setup, params, beta)
            _, ol_control, _ = approx_model.optimise(n_stages=20, init_policy=baseline_ol_control)

            _, obj, _ = sim_model.run_policy(control_policy=ol_control)

            ol_control = np.array([ol_control(t) for t in setup['times'][:-1]]).T
            ol_objs.append(obj)
            ol_controls[i] = ol_control

        # MPC runs - approximate model initialised using observation function, & at update steps
        for i in range(n_reps):
            mpc_controller = mpc.Controller(setup, params, beta)
            _, _, mpc_control, mpc_obj = mpc_controller.optimise(**mpc_args, observer=observer)

            mpc_objs.append(mpc_obj)
            mpc_controls[i] = mpc_control
            mpc_approx_states[i] = mpc_controller.approx_update_states

        # Append to existing data
        if append:
            old_filename = os.path.join(folder, "sampled_data_" + str(n_samples) + ".npz")

            with np.load(old_filename) as old_dataset:
                ol_controls = np.append(old_dataset['ol_controls'], ol_controls, axis=0)
                ol_objs = np.append(old_dataset['ol_objs'], ol_objs, axis=0)
                ol_approx_states = np.append(old_dataset['ol_approx_states'], ol_approx_states,
                                             axis=0)
                mpc_controls = np.append(old_dataset['mpc_controls'], mpc_controls, axis=0)
                mpc_objs = np.append(old_dataset['mpc_objs'], mpc_objs, axis=0)
                mpc_approx_states = np.append(old_dataset['mpc_approx_states'], mpc_approx_states,
                                              axis=0)

        # Store data
        dataset = {
            'ol_controls': ol_controls,
            'ol_objs': ol_objs,
            'ol_approx_states': ol_approx_states,
            'mpc_controls': mpc_controls,
            'mpc_objs': mpc_objs,
            'mpc_approx_states': mpc_approx_states
        }
        np.savez_compressed(os.path.join(folder, "sampled_data_" + str(n_samples)), **dataset)


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

    args = parser.parse_args()

    data_path = os.path.join(
        os.path.realpath(__file__), '..', '..', 'data', args.folder)

    os.makedirs(data_path, exist_ok=True)

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

    make_data(n_reps=args.n_reps, folder=args.folder, append=args.append)

    logging.info("Script completed")
