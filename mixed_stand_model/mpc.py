"""Methods for running MPC strategies."""

import argparse
import pdb
import pickle
import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    args = parser.parse_args()

class Controller:
    """MPC Controller"""

    def __init__(self, simulator, approx_model):
        """Model predictive controller, regular optimisations on approx model control simulator."""

        self.simulator = simulator
        self.approx_model = approx_model

        self.run_times = None
        self.run_state = None
        self.run_control = None

    def run_controller(self, horizon, time_step, end_time, update_period=None,
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

        state_init = self.simulator.setup['state_init']

        all_times = [current_start]
        all_run_data = np.empty((state_init.shape[0], 1))
        all_control_data = np.empty((9, 0))

        all_run_data[:, 0] = state_init

        sim_obj = [0.0]

        while current_start < end_time:
            current_times = np.arange(current_start, current_end+time_step, step=time_step)

            # Set initial states
            self.simulator.setup['state_init'] = state_init
            self.approx_model.set_state_init(state_init)

            self.approx_model.setup['times'] = current_times
            if stage_len is not None:
                n_stages = int((current_end - current_start) / stage_len)

            _, current_control, _ = self.approx_model.optimise(
                n_stages=n_stages, init_policy=init_policy)

            simulation_times = np.arange(
                current_start, np.minimum(next_update, end_time)+time_step, step=time_step)
            self.simulator.setup['times'] = simulation_times
            sim_state, sim_obj_final, sim_obj = self.simulator.run_policy(
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

        all_control_data = np.hstack((
            all_control_data, np.array([current_control(all_times[-1])]).T))

        self.run_times = all_times
        self.run_state = all_run_data
        self.run_control = all_control_data

        return all_times, all_run_data, all_control_data, sim_obj_final

    def save_optimisation(self, filename):
        """Save control optimisation to file."""

        dump_obj = {
            'times': self.run_times,
            'state': self.run_state,
            'control': self.run_control,
        }

        with open(filename, "wb") as outfile:
            pickle.dump(dump_obj, outfile)
