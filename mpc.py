"""Methods for running MPC strategies."""

import argparse
from enum import IntEnum
import copy
import warnings
import pdb
import numpy as np
from scipy.interpolate import interp1d
from scipy import integrate
import matplotlib.pyplot as plt
from matplotlib import animation
import visualisation
import mixed_stand_simulator as ms_sim
import mixed_stand_approx as ms_approx
import parameters

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

    def run_controller(self, horizon, time_step, end_time, update_period=None,
                       rolling_horizon=False):
        """Run simulation under MPC strategy, optimising on approximate model."""

        if update_period is None:
            next_update = np.inf
            update_period = np.inf

            if horizon < end_time:
                raise ValueError("If open-loop then horizon must be >= end_time!")

        else:
            next_update = update_period

            if horizon < update_period:
                raise ValueError("If MPC then horizon must be >= update period!")

        current_start = 0.0
        current_end = horizon

        state_init = self.simulator.setup['state_init']

        all_times = [current_start]
        all_run_data = np.empty((state_init.shape[0], 1))
        all_control_data = np.empty((7, 0))

        all_run_data[:, 0] = state_init

        while current_start < end_time:
            current_times = np.arange(current_start, current_end+time_step, step=time_step)

            # Set initial states
            self.simulator.setup['state_init'] = state_init
            self.approx_model.set_state_init(state_init)

            self.approx_model.setup['times'] = current_times
            _, current_control, _ = self.approx_model.optimise()

            simulation_times = np.arange(
                current_start, np.minimum(next_update, end_time)+time_step, step=time_step)
            self.simulator.setup['times'] = simulation_times
            run_data = self.simulator.run_policy(control_policy=current_control, n_fixed_steps=100)

            current_start = next_update
            next_update += update_period

            if rolling_horizon:
                current_end += update_period

            state_init = run_data[:, -1]

            all_times.extend(simulation_times[1:])
            all_run_data = np.hstack((all_run_data, run_data[:, 1:]))
            all_control_data = np.hstack((
                all_control_data, np.array([current_control(t) for t in simulation_times[:-1]]).T))

        all_control_data = np.hstack((
            all_control_data, np.array([current_control(all_times[-1])]).T))

        return all_times, all_run_data, all_control_data
