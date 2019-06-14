"""Scan over diversity cost parameter."""

import argparse
import copy
import json
import logging
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from mixed_stand_model import mixed_stand_simulator as ms_sim
from mixed_stand_model import mixed_stand_approx as ms_approx
from mixed_stand_model import parameters
from mixed_stand_model import utils
from mixed_stand_model import mpc
from mixed_stand_model import visualisation

plt.style.use('seaborn-whitegrid')

def even_policy(time):
    """Even allocation across controls"""
    return np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])


def make_plots(data_folder=None, fig_folder=None):
    """Generate plot of diversity cost scan."""

    if data_folder is None:
        data_folder = os.path.join(os.path.realpath(__file__), '..', '..', 'data', 'div_cost_scan')

    if fig_folder is None:
        fig_folder = os.path.join(
            os.path.realpath(__file__), '..', '..', 'figures', 'div_cost_scan')

    div_props = np.array([0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0])
    n_costs = len(div_props)

    with open(os.path.join("data", "scale_and_fit_results.json"), "r") as infile:
        scale_and_fit_results = json.load(infile)

    setup, params = utils.get_setup_params(
        parameters.CORRECTED_PARAMS, scale_inf=True, host_props=parameters.COBB_PROP_FIG4A)

    beta_names = ['beta_1,1', 'beta_1,2', 'beta_1,3', 'beta_1,4', 'beta_12', 'beta_21', 'beta_2']
    beta = np.array([scale_and_fit_results[x] for x in beta_names])
    approx_model = ms_approx.MixedStandApprox(setup, params, beta)

    model = ms_sim.MixedStandSimulator(setup, params)

    ncells = np.product(setup['landscape_dims'])

    # Collect control proportions
    order = np.array([3, 4, 5, 6, 0, 1, 2, 7, 8])
    control_abs_ol = np.zeros((n_costs, 9))
    control_abs_mpc = np.zeros((n_costs, 9))
    objectives_ol = np.zeros(n_costs)
    objectives_mpc = np.zeros(n_costs)
    for i, div_prop in enumerate(div_props):
        div_cost = div_prop / (setup['times'][-1] * np.log(3))

        logging.info("Diversity cost: %f", div_cost)
        approx_model.load_optimisation(os.path.join(
            data_folder, "div_cost_" + str(div_prop) + "_OL.pkl"))
        
        control = approx_model.optimisation['control']
        control_policy = interp1d(setup['times'][:-1], control, kind="zero",
                                  fill_value="extrapolate")
        sim_run, obj, objs = model.run_policy(control_policy)

        sim_state = np.sum(np.reshape(sim_run, (ncells, 15, -1)), axis=0) / ncells

        allocation = np.array([
            sim_state[1] + sim_state[4],
            sim_state[7] + sim_state[10],
            sim_state[13],
            np.sum(sim_state[0:6], axis=0),
            np.sum(sim_state[6:12], axis=0),
            sim_state[12] + sim_state[13],
            sim_state[14],
            sim_state[0] + sim_state[3],
            sim_state[6] + sim_state[9]])[:, :-1] * control

        allocation[0:3] *= params['rogue_rate'] * params['rogue_cost']
        allocation[3:7] *= params['thin_rate'] * params['thin_cost']
        allocation[7:] *= params['protect_rate'] * params['protect_cost']
        allocation[0] *= params['rel_small_cost']
        allocation[3] *= params['rel_small_cost']

        expense = utils.control_expenditure(control, params, sim_state[:, :-1])
        for j in range(len(setup['times'])-1):
            if expense[j] > params['max_budget']:
                allocation[:, j] *= params['max_budget'] / expense[j]

        control_abs_ol[i] = (np.sum(allocation, axis=1))[order] / 200

        objectives_ol[i] = -1 * (obj - objs[-1])

        mpc_controller = mpc.Controller.load_optimisation(os.path.join(
            data_folder, "div_cost_" + str(div_prop) + "_MPC.pkl"))

        sim_run, approx_run = mpc_controller.run_control()
        control = mpc_controller.control

        sim_state = np.sum(np.reshape(sim_run[0], (ncells, 15, -1)), axis=0) / ncells

        allocation = np.array([
            sim_state[1] + sim_state[4],
            sim_state[7] + sim_state[10],
            sim_state[13],
            np.sum(sim_state[0:6], axis=0),
            np.sum(sim_state[6:12], axis=0),
            sim_state[12] + sim_state[13],
            sim_state[14],
            sim_state[0] + sim_state[3],
            sim_state[6] + sim_state[9]])[:, :-1] * control

        allocation[0:3] *= params['rogue_rate'] * params['rogue_cost']
        allocation[3:7] *= params['thin_rate'] * params['thin_cost']
        allocation[7:] *= params['protect_rate'] * params['protect_cost']
        allocation[0] *= params['rel_small_cost']
        allocation[3] *= params['rel_small_cost']

        expense = utils.control_expenditure(control, params, sim_state[:, :-1])
        for j in range(len(setup['times'])-1):
            if expense[j] > params['max_budget']:
                allocation[:, j] *= params['max_budget'] / expense[j]

        control_abs_mpc[i] = (np.sum(allocation, axis=1))[order] / 200

        objectives_mpc[i] = -1 * (sim_run[1] - sim_run[2][-1])

    # Make plot
    labels = ["Thin Tan (Small)", "Thin Tan (Large)", "Thin Bay", "Thin Red", "Rogue Tan (Small)",
              "Rogue Tan (Large)", "Rogue Bay", "Protect Tan (Small)", "Protect Tan (Large)"]

    colors = visualisation.CONTROL_COLOURS

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax2 = ax.twinx()

    color = 'tab:red'
    ax2.set_ylabel('Large healthy tanoak', color=color)
    ax2.plot(div_props, objectives_ol, '--', color=color, label='OL tanoak objective')
    ax2.tick_params(axis='y', labelcolor=color)

    bars = []

    for i in range(9):
        b = ax.bar(div_props-0.01, control_abs_ol[:, i],
                   bottom=np.sum(control_abs_ol[:, :i], axis=1), color=colors[i], width=0.015,
                   alpha=0.75)
        bars.append(b)

    ax2.plot(div_props, objectives_mpc, '-', color=color, label='MPC tanoak objective')

    bars = []

    for i in range(9):
        b = ax.bar(div_props+0.01, control_abs_mpc[:, i],
                   bottom=np.sum(control_abs_mpc[:, :i], axis=1), color=colors[i], width=0.015,
                   alpha=0.75)
        bars.append(b)

    ax.legend(bars, labels, bbox_to_anchor=(0.5, -0.15), loc="upper center", ncol=3, frameon=True)
    ax2.legend(ncol=1, loc='upper center')
    ax.set_xlabel("Diversity cost")
    ax.set_ylabel("Expenditure")
    ax.set_title('Left: OL, right: MPC', fontsize=8)

    ax.set_ylim([0, 14])
    # ax2.set_ylim([-1.6, -0.1])
    fig.tight_layout()
    fig.savefig(os.path.join(fig_folder, "DivCostScan.pdf"), dpi=300)

def make_data(folder=None):
    """Scan over budgets to analyse change in OL control"""

    if folder is None:
        folder = os.path.join(os.path.realpath(__file__), '..', '..', 'data', 'div_cost_scan')

    with open(os.path.join("data", "scale_and_fit_results.json"), "r") as infile:
        scale_and_fit_results = json.load(infile)

    setup, params = utils.get_setup_params(
        parameters.CORRECTED_PARAMS, scale_inf=True, host_props=parameters.COBB_PROP_FIG4A)

    logging.info("Parameters: %s", params)

    beta_names = ['beta_1,1', 'beta_1,2', 'beta_1,3', 'beta_1,4', 'beta_12', 'beta_21', 'beta_2']
    beta = np.array([scale_and_fit_results[x] for x in beta_names])

    approx_params = copy.deepcopy(params)
    approx_params['rogue_rate'] *= scale_and_fit_results['roguing_factor']
    approx_params['rogue_cost'] /= scale_and_fit_results['roguing_factor']

    approx_model = ms_approx.MixedStandApprox(setup, approx_params, beta)

    div_prop = np.array([0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0])
    div_costs = div_prop / (setup['times'][-1] * np.log(3))

    for div_cost, prop in zip(div_costs, div_prop):
        logging.info("Diversity cost: %f", div_cost)
        params['div_cost'] = div_cost
        approx_params['div_cost'] = div_cost
        approx_model.params['div_cost'] = div_cost

        _, control, exit_text = approx_model.optimise(n_stages=20, init_policy=even_policy)

        if exit_text not in ["Optimal Solution Found.", "Solved To Acceptable Level."]:
            logging.warning("Failed optimisation. Trying intialisation from previous solution.")
            filename = os.path.join(
                os.path.dirname(os.path.realpath(__file__)), '..', 'mixed_stand_model', "BOCOP",
                "problem.def")

            with open(filename, "r") as infile:
                all_lines = infile.readlines()
            all_lines[31] = "# " + all_lines[31]
            all_lines[32] = "# " + all_lines[32]
            all_lines[33] = all_lines[33][2:]
            all_lines[34] = all_lines[34][2:]
            with ms_approx._try_file_open(filename) as outfile:
                outfile.writelines(all_lines)

            _, control, exit_text = approx_model.optimise(n_stages=20, init_policy=even_policy)

            all_lines[31] = all_lines[31][2:]
            all_lines[32] = all_lines[32][2:]
            all_lines[33] = "# " + all_lines[33]
            all_lines[34] = "# " + all_lines[34]
            with ms_approx._try_file_open(filename) as outfile:
                outfile.writelines(all_lines)

            if exit_text not in ["Optimal Solution Found.", "Solved To Acceptable Level."]:
                logging.error("Failed optimisation in OL optimisation.")

        approx_model.save_optimisation(os.path.join(
            folder, "div_cost_" + str(prop) + "_OL.pkl"))

        mpc_controller = mpc.Controller(setup, params, beta, approx_params=approx_params)
        mpc_controller.optimise(
            horizon=100, time_step=0.5, end_time=100, update_period=20, rolling_horz=False,
            stage_len=5, init_policy=control, use_init_first=True)

        mpc_controller.save_optimisation(os.path.join(
            folder, "div_cost_" + str(prop) + "_MPC.pkl"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-f", "--folder", default='div_cost_scan',
                        help="Folder name to save results in data and figures directory.")
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
    fh = logging.FileHandler(os.path.join(data_path, 'div_cost_scan.log'))
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
        make_data(folder=data_path)
        make_plots(data_path, fig_path)

    logging.info("Script completed")
