"""Methods for running MPC strategies."""

import argparse
import copy
import logging
import os
import pickle
import numpy as np
from scipy.interpolate import interp1d
from . import mixed_stand_simulator as ms_sim
from . import mixed_stand_approx as ms_approx

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    args = parser.parse_args()

class Controller:
    """MPC Controller"""

    def __init__(self, setup, params, beta):
        """Model predictive controller, regular optimisations on approx model control simulator."""

        self.setup = copy.deepcopy(setup)
        self.params = copy.deepcopy(params)
        self.beta = copy.deepcopy(beta)

        self.times = np.array([])
        self.control = np.array([[]])

        self.config = {}

    def optimise(self, horizon, time_step, end_time, update_period=None,
                 rolling_horz=False, stage_len=None, init_policy=None):
        """Run simulation under MPC strategy, optimising on approximate model.

        ---------
        Arguments
        ---------
            horizon:        Optimisation horizon. Optimise control over this time frame
            time_step:      Length of single time step in optimisation scheme
            end_time:       When to stop simulation
            update_period:  How often to update controller (re-optimise)
            rolling_horz:   Whether to shift horizon later at each update step
            stage_len:      Time over which control is constant. If None this feature is disabled

        """

        self.config = {
            'horizon': horizon,
            'time_step': time_step,
            'end_time': end_time,
            'update_period': update_period,
            'rolling_horz': rolling_horz,
            'stage_len': stage_len,
            'init_policy': init_policy
        }

        approx_model = ms_approx.MixedStandApprox(self.setup, self.params, self.beta)
        sim_model = ms_sim.MixedStandSimulator(self.setup, self.params)

        if update_period is None:
            next_update = np.inf
            update_period = np.inf

            if horizon < end_time:
                raise ValueError("If open-loop then horizon must be >= end_time!")

        else:
            next_update = update_period

            if horizon < update_period:
                raise ValueError("If MPC then horizon must be >= update period!")

            if stage_len is not None and update_period % stage_len != 0:
                raise ValueError("Update period must be exact multiple of stage_len!")

        if stage_len is not None and end_time % stage_len != 0:
            raise ValueError("End time must be exact multiple of stage_len!")

        current_start = 0.0
        current_end = horizon
        n_stages = None

        state_init = sim_model.setup['state_init']

        all_times = [current_start]
        all_run_data = np.empty((state_init.shape[0], 1))
        all_control_data = np.empty((9, 0))

        all_run_data[:, 0] = state_init

        sim_obj = [0.0]

        while current_start < end_time:
            logging.info("Updating control at time %f", current_start)

            current_times = np.arange(current_start, current_end+time_step, step=time_step)

            # Set initial states
            sim_model.setup['state_init'] = state_init
            approx_model.set_state_init(state_init)

            approx_model.setup['times'] = current_times
            if stage_len is not None:
                n_stages = int((current_end - current_start) / stage_len)

            _, current_control, exit_text = approx_model.optimise(n_stages=n_stages, init_policy=init_policy)

            if exit_text not in ["Optimal Solution Found.", "Solved To Acceptable Level."]:
                logging.info("Failed optimisation. Trying intialisation from previous solution.")
                filename = os.path.join(
                    os.path.dirname(os.path.realpath(__file__)), "BOCOP", "problem.def")

                with open(filename, "r") as infile:
                    all_lines = infile.readlines()
                all_lines[31] = "# " + all_lines[31]
                all_lines[32] = "# " + all_lines[32]
                all_lines[33] = all_lines[33][2:]
                all_lines[34] = all_lines[34][2:]
                with ms_approx._try_file_open(filename) as outfile:
                    outfile.writelines(all_lines)

                _, current_control, exit_text = approx_model.optimise(
                    n_stages=n_stages, init_policy=init_policy)
                
                all_lines[31] = all_lines[31][2:]
                all_lines[32] = all_lines[32][2:]
                all_lines[33] = "# " + all_lines[33]
                all_lines[34] = "# " + all_lines[34]
                with ms_approx._try_file_open(filename) as outfile:
                    outfile.writelines(all_lines)

            simulation_times = np.arange(
                current_start, np.minimum(next_update, end_time)+time_step, step=time_step)
            sim_model.setup['times'] = simulation_times
            sim_state, sim_obj_final, sim_obj = sim_model.run_policy(
                control_policy=current_control, n_fixed_steps=None, obj_start=sim_obj[-1])

            current_start = next_update
            next_update += update_period

            if rolling_horz:
                current_end += update_period

            state_init = sim_state[:, -1]

            all_times.extend(simulation_times[1:])
            all_run_data = np.hstack((all_run_data, sim_state[:, 1:]))
            all_control_data = np.hstack((
                all_control_data, np.array([current_control(t) for t in simulation_times[:-1]]).T))

        self.times = all_times
        self.control = all_control_data

        return all_times, all_run_data, all_control_data, sim_obj_final

    def run_control(self):
        """Use MPC optimised control and simulation to run approx & simulation models.

        State in approx model is updated to match simulation at regular intervals.

        Returns (sim_run, approx_run) where individual results match output from sim/approx model
        run_policy function.
        """

        approx_model = ms_approx.MixedStandApprox(self.setup, self.params, self.beta)
        sim_model = ms_sim.MixedStandSimulator(self.setup, self.params)

        if self.config['stage_len'] is None:
            control_policy = interp1d(
                self.times[:-1], self.control, kind="linear", fill_value="extrapolate")
        else:
            control_policy = interp1d(
                self.times[:-1], self.control, kind="zero", fill_value="extrapolate")

        # First run simulation model
        logging.info("Running simulation model")
        sim_model.setup['times'] = self.times
        sim_run = sim_model.run_policy(control_policy)

        # Interpolate state so can easily extract state at update times
        sim_state_t = interp1d(self.times, sim_run[0], kind="linear", fill_value="extrapolate")

        # Now loop over update periods
        current_start = 0.0
        obj_start = 0.0

        combined_states = np.empty((15, 0))
        combined_obj = np.empty((0))

        logging.info("Running approximate model")
        time_step = self.config['time_step']
        update_period = self.config['update_period']
        end_time = self.config['end_time']
        while current_start < end_time:
            logging.info("Updating state at time %f", current_start)
            current_end = min(current_start+update_period, end_time)
            current_times = np.arange(current_start, current_end+time_step, step=time_step)

            approx_model.set_state_init(sim_state_t(current_start))
            approx_model.setup['times'] = current_times

            state, final_obj, obj = approx_model.run_policy(control_policy, obj_start=obj_start)

            obj_start = obj[-1]
            current_start += update_period
            combined_states = np.hstack((combined_states, state[:, :-1]))
            combined_obj = np.hstack((combined_obj, obj[:-1]))

        combined_states = np.hstack((combined_states, state[:, -1:]))
        combined_obj = np.hstack((combined_obj, obj[-1:]))

        return (sim_run, (combined_states, final_obj, combined_obj))

    def save_optimisation(self, filename):
        """Save control optimisation to file."""

        logging.info("Saving optiimisation to file: %s", filename)

        dump_obj = {
            'times': self.times,
            'control': self.control,
            'config': self.config,
            'setup': self.setup,
            'params': self.params,
            'beta': self.beta
        }

        with open(filename, "wb") as outfile:
            pickle.dump(dump_obj, outfile)

    @classmethod
    def load_optimisation(cls, filename):
        """Load control optimisation from file."""

        logging.info("Loading optimisation from file: %s", filename)

        with open(filename, "rb") as infile:
            load_obj = pickle.load(infile)
        
        instance = cls(load_obj['setup'], load_obj['params'], load_obj['beta'])

        instance.times = load_obj['times']
        instance.control = load_obj['control']
        instance.config = load_obj['config']

        return instance
