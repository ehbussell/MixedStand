"""Scan over maximum control budget."""

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
    """Generate plot of budget scan analysis"""

    if data_folder is None:
        data_folder = os.path.join(os.path.realpath(__file__), '..', '..', 'data', 'budget_scan')

    if fig_folder is None:
        fig_folder = os.path.join(os.path.realpath(__file__), '..', '..', 'figures', 'budget_scan')

    budgets = np.array([8, 10, 12, 14, 16, 18, 20])
    n_budgets = len(budgets)

    with open(os.path.join("data", "scale_and_fit_results.json"), "r") as infile:
        scale_and_fit_results = json.load(infile)

    setup, params = utils.get_setup_params(
        parameters.CORRECTED_PARAMS, scale_inf=True, host_props=parameters.COBB_PROP_FIG4A)

    model = ms_sim.MixedStandSimulator(setup, params)

    beta_names = ['beta_1,1', 'beta_1,2', 'beta_1,3', 'beta_1,4', 'beta_12', 'beta_21', 'beta_2']
    beta = np.array([scale_and_fit_results[x] for x in beta_names])
    approx_params = copy.deepcopy(params)
    approx_params['rogue_rate'] *= scale_and_fit_results['roguing_factor']
    approx_params['rogue_cost'] /= scale_and_fit_results['roguing_factor']

    approx_model = ms_approx.MixedStandApprox(setup, approx_params, beta)

    # Collect control proportions
    order = np.array([3, 4, 5, 6, 0, 1, 2, 7, 8])
    control_abs_ol = np.zeros((n_budgets, 9))
    control_abs_mpc = np.zeros((n_budgets, 9))
    real_objectives_ol = np.zeros(n_budgets)
    real_objectives_mpc = np.zeros(n_budgets)
    approx_objectives_ol = np.zeros(n_budgets)
    approx_objectives_mpc = np.zeros(n_budgets)
    for i, budget in enumerate(budgets):
        logging.info("Budget: %f", budget)
        approx_model.load_optimisation(os.path.join(
            data_folder, "budget_" + str(int(budget)) + "_OL.pkl"))

        control = approx_model.optimisation['control']
        control_policy = interp1d(setup['times'][:-1], control, kind="zero",
                                  fill_value="extrapolate")
        model.run_policy(control_policy)
        approx_model.run_policy(control_policy)

        control[0:3] *= params['rogue_rate'] * params['rogue_cost'] * 0.5
        control[3:7] *= params['thin_rate'] * params['thin_cost'] * 0.5
        control[7:] *= params['protect_rate'] * params['protect_cost'] * 0.5
        control[0] *= params['rel_small_cost']
        control[3] *= params['rel_small_cost']

        control_abs_ol[i] = (np.sum(control, axis=1))[order] / 100

        real_objectives_ol[i] = model.run['objective']
        approx_objectives_ol[i] = approx_model.run['objective']

        mpc_controller = mpc.Controller.load_optimisation(os.path.join(
            "data", "budget_scan", "budget_" + str(int(budget)) + "_MPC.pkl"))

        sim_run, approx_run = mpc_controller.run_control()
        approx_model.run['state'] = approx_run[0]
        approx_model.run['objective'] = approx_run[1]
        model.run['state'] = sim_run[0]
        model.run['objective'] = sim_run[1]

        control = mpc_controller.control

        control[0:3] *= params['rogue_rate'] * params['rogue_cost'] * 0.5
        control[3:7] *= params['thin_rate'] * params['thin_cost'] * 0.5
        control[7:] *= params['protect_rate'] * params['protect_cost'] * 0.5
        control[0] *= params['rel_small_cost']
        control[3] *= params['rel_small_cost']

        control_abs_mpc[i] = (np.sum(control, axis=1))[order] / 100

        real_objectives_mpc[i] = model.run['objective']
        approx_objectives_mpc[i] = approx_model.run['objective']

    # Make plot
    labels = ["Thin Tan (Small)", "Thin Tan (Large)", "Thin Bay", "Thin Red", "Rogue Tan (Small)",
              "Rogue Tan (Large)", "Rogue Bay", "Protect Tan (Small)", "Protect Tan (Large)"]

    colors = visualisation.CONTROL_COLOURS

    # Open-loop plot
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax2 = ax.twinx()

    color = 'tab:red'
    ax2.set_ylabel('Objective', color=color)
    ax2.plot(budgets, -1*approx_objectives_ol, '--', color=color,
             label='OL approximate model objective')
    ax2.tick_params(axis='y', labelcolor=color)

    bars = []

    for i in range(9):
        b = ax.bar(budgets-10, control_abs_ol[:, i], bottom=np.sum(control_abs_ol[:, :i], axis=1),
                   color=colors[i], width=15, alpha=0.75)
        bars.append(b)

    ax2.legend(ncol=1)
    ax.set_xlabel("Budget")
    ax.set_ylabel("Expenditure")

    color = 'tab:red'
    ax2.set_ylabel('Objective', color=color)
    ax2.plot(budgets, -1*real_objectives_mpc, '-', color=color, label='MPC simulation objective')
    ax2.tick_params(axis='y', labelcolor=color)

    bars = []

    for i in range(9):
        b = ax.bar(budgets+10, control_abs_mpc[:, i], bottom=np.sum(control_abs_mpc[:, :i], axis=1),
                   color=colors[i], width=15, alpha=0.75)
        bars.append(b)

    ax.legend(bars, labels, bbox_to_anchor=(0.5, -0.15), loc="upper center", ncol=3, frameon=True)
    ax2.legend(ncol=1, loc='upper left')
    ax.set_xlabel("Budget")
    ax.set_ylabel("Expenditure")
    ax.set_title('Left: OL, right: MPC', fontsize=8)

    ax.set_ylim([0, 900])
    ax2.set_ylim([0.0, 1.8])
    ax.set_xticks([0, 50, 100, 200, 300, 400, 500, 600, 700])
    fig.tight_layout()
    fig.savefig(os.path.join(fig_folder, "BudgetScan.pdf"), dpi=300)

def make_data(folder=None):
    """Scan over budgets to analyse change in OL control"""

    if folder is None:
        folder = os.path.join(os.path.realpath(__file__), '..', '..', 'data', 'budget_scan')

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

    # budgets = np.array([8, 10, 12, 14, 16, 18, 20])
    budgets = np.array([36])

    for budget in budgets:
        logging.info("Budget: %f", budget)
        params['max_budget'] = budget
        approx_model.params['max_budget'] = budget
        approx_params['max_budget'] = budget

        _, control, _ = approx_model.optimise(n_stages=20, init_policy=even_policy)

        approx_model.save_optimisation(os.path.join(
            folder, "budget_" + str(int(budget)) + "_OL.pkl"))

        mpc_controller = mpc.Controller(setup, params, beta, approx_params=approx_params)
        mpc_controller.optimise(
            horizon=100, time_step=0.5, end_time=100, update_period=20, rolling_horz=False,
            stage_len=5, init_policy=control, use_init_first=True)

        mpc_controller.save_optimisation(os.path.join(
            folder, "budget_" + str(int(budget)) + "_MPC.pkl"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-f", "--folder", default='budget_scan',
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
    fh = logging.FileHandler(os.path.join(data_path, 'budget_scan.log'))
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
        # make_plots(data_path, fig_path)

    logging.info("Script completed")
