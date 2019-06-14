"""Analyse OL and MPC control strategies, and effect of MPC update frequency."""

import pdb
import copy
import json
import logging
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.interpolate import interp1d

from mixed_stand_model import mixed_stand_simulator as ms_sim
from mixed_stand_model import mixed_stand_approx as ms_approx
from mixed_stand_model import parameters
from mixed_stand_model import visualisation
from mixed_stand_model import utils
from mixed_stand_model import mpc

def even_policy(time):
    """Even allocation across controls"""
    return np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])

def make_plots():
    """Create figures."""

    plt.style.use("seaborn-whitegrid")

    os.makedirs(os.path.join('figures', 'ol_mpc_control'), exist_ok=True)

    # Open-loop control
    approx_model = ms_approx.MixedStandApprox.load_optimisation_class(
        os.path.join("data", "ol_mpc_control", "ol_control.pkl"))

    ol_control = approx_model.optimisation['control']
    ol_control_policy = interp1d(
        approx_model.setup['times'][:-1], approx_model.optimisation['control'], kind="zero",
        fill_value="extrapolate")
    ol_approx_run = approx_model.run_policy(ol_control_policy)

    mpc_controller = mpc.Controller.load_optimisation(
        os.path.join("data", "ol_mpc_control", "mpc_control_20.pkl"))
    mpc_controller.approx_params = approx_model.params

    sim_model = ms_sim.MixedStandSimulator(mpc_controller.setup, mpc_controller.params)
    nc_sim_run = sim_model.run_policy()
    ol_sim_run = sim_model.run_policy(ol_control_policy)

    ncells = np.product(sim_model.setup['landscape_dims'])

    fig = plt.figure(figsize=(6.4, 6.4))
    gs = gridspec.GridSpec(3, 2, height_ratios=[3, 1, 5], wspace=0.3, hspace=0.25)
    ax1 = fig.add_subplot(gs[0, :])
    ax2 = fig.add_subplot(gs[2, 0])
    ax3 = fig.add_subplot(gs[2, 1])
    ax_leg = fig.add_subplot(gs[1, :], frameon=False)
    ax_leg.grid = False
    ax_leg.set_xticks([])
    ax_leg.set_yticks([])

    visualisation.plot_control(
        approx_model.setup['times'][:-1], ol_control, ol_approx_run[0][:, :-1],
        approx_model.params, ax=ax1)
    ax_leg.legend(*ax1.get_legend_handles_labels(), loc="center", ncol=3)
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Control Expenditure")

    visualisation.plot_hosts(
        approx_model.setup['times'], approx_model.run['state'], ax=ax2, proportions=False,
        alpha=0.75)
    ax2.legend(bbox_to_anchor=(0.5, -0.25), loc="upper center", ncol=2, fontsize=8)
    ax2.set_xlabel("Time")
    ax2.set_ylabel("Host stems")

    ol_sim_state = np.sum(np.reshape(sim_model.run['state'], (ncells, 15, -1)), axis=0) / ncells
    nc_sim_state = np.sum(np.reshape(nc_sim_run[0], (ncells, 15, -1)), axis=0) / ncells

    visualisation.plot_hosts(
        sim_model.setup['times'], ol_sim_state, ax=ax2, proportions=False, alpha=0.75,
        linestyle='--')

    cmap = plt.get_cmap("tab20c")

    ax3.axhline(np.sum(approx_model.setup['state_init'][6:12]), linestyle='--', color='darkgreen',
                label="Disease Free")
    ax3.plot(approx_model.setup['times'], np.sum(approx_model.run['state'][[6, 8, 9, 11]], axis=0),
             alpha=0.75, color=cmap(0.025), label='Approximate model')
    ax3.plot(sim_model.setup['times'], np.sum(ol_sim_state[[6, 8, 9, 11]], axis=0), '--',
             color=cmap(0.025), label='Simulation model')
    ax3.plot(sim_model.setup['times'], np.sum(nc_sim_state[[6, 8, 9, 11]], axis=0),
             color='darkred', label='No control')
    ax3.legend(bbox_to_anchor=(0.5, -0.25), loc="upper center", ncol=2, fontsize=8)
    ax3.set_xlabel("Time")
    ax3.set_ylabel("Large Healthy\nTanoak Stems")

    gs.tight_layout(fig)

    fig.savefig(
        os.path.join("figures", "ol_mpc_control", "open_loop.pdf"), dpi=300, bbox_inches='tight')

    # Run MPC runs for 'standard' frequency (20yrs)
    mpc_sim_run, mpc_approx_run = mpc_controller.run_control()
    mpc_control_policy = interp1d(
        mpc_controller.times[:-1], mpc_controller.control, kind="zero", fill_value="extrapolate")
    mpc_sim_state = np.sum(np.reshape(mpc_sim_run[0], (ncells, 15, -1)), axis=0) / ncells

    fig = plt.figure(figsize=(6.4, 6.4))
    gs = gridspec.GridSpec(3, 2, height_ratios=[3, 1, 5], wspace=0.3, hspace=0.25)
    ax1 = fig.add_subplot(gs[0, :])
    ax2 = fig.add_subplot(gs[2, 0])
    ax3 = fig.add_subplot(gs[2, 1])
    ax_leg = fig.add_subplot(gs[1, :], frameon=False)
    ax_leg.grid = False
    ax_leg.set_xticks([])
    ax_leg.set_yticks([])

    visualisation.plot_control(
        approx_model.setup['times'][:-1], mpc_controller.control, mpc_sim_state[:, :-1],
        approx_model.params, ax=ax1)
    ax_leg.legend(*ax1.get_legend_handles_labels(), loc="center", ncol=3)
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Control Expenditure")

    visualisation.plot_hosts(
        mpc_controller.times, mpc_approx_run[0], ax=ax2, proportions=False, alpha=0.75)
    ax2.legend(bbox_to_anchor=(0.5, -0.25), loc="upper center", ncol=2, fontsize=8)
    ax2.set_xlabel("Time")
    ax2.set_ylabel("Host stems")

    visualisation.plot_hosts(
        mpc_controller.times, mpc_sim_state, ax=ax2, proportions=False, alpha=0.75,
        linestyle='--')

    cmap = plt.get_cmap("tab20c")

    ax3.axhline(np.sum(approx_model.setup['state_init'][6:12]), linestyle='--', color='darkgreen',
                label="Disease Free")
    ax3.plot(approx_model.setup['times'], np.sum(mpc_approx_run[0][[6, 8, 9, 11]], axis=0),
             alpha=0.75, color=cmap(0.025), label='Approximate model')
    ax3.plot(sim_model.setup['times'], np.sum(mpc_sim_state[[6, 8, 9, 11]], axis=0), '--',
             color=cmap(0.025), label='Simulation model')
    ax3.plot(sim_model.setup['times'], np.sum(nc_sim_state[[6, 8, 9, 11]], axis=0),
             color='darkred', label='No control')
    ax3.legend(bbox_to_anchor=(0.5, -0.25), loc="upper center", ncol=2, fontsize=8)
    ax3.set_xlabel("Time")
    ax3.set_ylabel("Large Healthy\nTanoak Stems")

    gs.tight_layout(fig)
    fig.savefig(os.path.join("figures", "ol_mpc_control", "mpc.pdf"), dpi=300, bbox_inches='tight')

    # DPCs and objective comparison
    fig = plt.figure()
    gs = gridspec.GridSpec(1, 3)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharey=ax1)
    ax3 = fig.add_subplot(gs[2])

    visualisation.plot_dpcs(sim_model.setup['times'], ol_approx_run[0], ax=ax1, proportions=False,
                            alpha=0.75)
    ax1.legend(bbox_to_anchor=(0.5, -0.25), loc="upper center", ncol=2, fontsize=8)
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Host stems")
    visualisation.plot_dpcs(sim_model.setup['times'], ol_sim_state, ax=ax1, proportions=False,
                            alpha=0.75, linestyle='--')

    visualisation.plot_dpcs(sim_model.setup['times'], mpc_approx_run[0], ax=ax2, proportions=False,
                            alpha=0.75)
    ax2.legend(bbox_to_anchor=(0.5, -0.25), loc="upper center", ncol=2, fontsize=8)
    ax2.set_xlabel("Time")
    ax2.set_ylabel("Host stems")
    visualisation.plot_dpcs(sim_model.setup['times'], mpc_sim_state, ax=ax2, proportions=False,
                            alpha=0.75, linestyle='--')

    tan_obj = np.array([-(x[1]-x[2][-1]) for x in [nc_sim_run, ol_sim_run, mpc_sim_run]])
    div_obj = np.array([-1*(x[2][-1]) for x in [nc_sim_run, ol_sim_run, mpc_sim_run]])
    b1 = ax3.bar(range(3), tan_obj, tick_label=['No control', 'OL', 'MPC'])
    b2 = ax3.bar(range(3), div_obj+tan_obj, bottom=tan_obj, tick_label=['No control', 'OL', 'MPC'])
    ax3.set_ylabel("Objective")
    ax3.legend((b1[0], b2[0]), ('Healthy large tanoak', 'Diversity'), bbox_to_anchor=(0.5, -0.25),
               loc="upper center", ncol=2, fontsize=8)

    gs.tight_layout(fig)
    fig.savefig(os.path.join("figures", "ol_mpc_control", "dpcs.pdf"), dpi=300, bbox_inches='tight')

    fig = plt.figure()
    ax = fig.add_subplot(111)

    mpc_freqs = np.arange(5, 105, 5)
    mpc_freqs_tan = np.zeros_like(mpc_freqs, dtype=float)
    mpc_freqs_div = np.zeros_like(mpc_freqs, dtype=float)
    for i, freq in enumerate(mpc_freqs):
        mpc_controller = mpc.Controller.load_optimisation(
            os.path.join("data", "ol_mpc_control", "mpc_control_{}.pkl".format(freq)))
        mpc_controller.approx_params = approx_model.params
        mpc_sim_run, mpc_approx_run = mpc_controller.run_control()

        mpc_freqs_tan[i] = -(mpc_sim_run[1] - mpc_sim_run[2][-1])
        mpc_freqs_div[i] = -1*(mpc_sim_run[2][-1])

    b1 = ax.bar(mpc_freqs, mpc_freqs_tan)
    b2 = ax.bar(mpc_freqs, mpc_freqs_div, bottom=mpc_freqs_tan)
    ax.set_ylabel("Objective")
    ax.set_xlabel("MPC update frequency")
    ax.legend((b1[0], b2[0]), ('Healthy large tanoak', 'Diversity'), bbox_to_anchor=(0.5, -0.25),
              loc="upper center", ncol=2, fontsize=8)

    fig.savefig(
        os.path.join("figures", "ol_mpc_control", "mpc_update.pdf"), dpi=300, bbox_inches='tight')

def make_data():
    """Run OL and MPC frameworks & scan over MPC update frequency."""

    os.makedirs(os.path.join('data', 'ol_mpc_control'), exist_ok=True)

    # Analysis:
    # 1. First construct default parameters (corrected and scaled Cobb)
    with open(os.path.join("data", "scale_and_fit_results.json"), "r") as infile:
        scale_and_fit_results = json.load(infile)

    setup, params = utils.get_setup_params(
        parameters.CORRECTED_PARAMS, scale_inf=True, host_props=parameters.COBB_PROP_FIG4A)

    # Setup approximate model
    beta_names = ['beta_1,1', 'beta_1,2', 'beta_1,3', 'beta_1,4', 'beta_12', 'beta_21', 'beta_2']
    beta = np.array([scale_and_fit_results[x] for x in beta_names])

    approx_params = copy.deepcopy(params)
    approx_params['rogue_rate'] *= scale_and_fit_results['roguing_factor']
    approx_params['rogue_cost'] /= scale_and_fit_results['roguing_factor']

    approx_model = ms_approx.MixedStandApprox(setup, approx_params, beta)

    # Run open-loop optimisation
    _, ol_control_policy, _ = approx_model.optimise(n_stages=20, init_policy=even_policy)
    approx_model.save_optimisation(os.path.join("data", "ol_mpc_control", "ol_control.pkl"))

    # Run MPC optimisation for range of update periods
    update_periods = np.arange(5, 105, 5)
    mpc_controller = mpc.Controller(setup, params, beta, approx_params=approx_params)
    for update_period in update_periods:
        mpc_controller.optimise(
            horizon=100, time_step=0.5, end_time=100, update_period=update_period,
            rolling_horz=False, stage_len=5, init_policy=ol_control_policy, use_init_first=True)
        mpc_controller.save_optimisation(
            os.path.join("data", "ol_mpc_control", "mpc_control_{}.pkl").format(update_period))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument("-e", "--use_existing_data", action="store_true",
                        help="Make plots only (no new data generated)")
    args = parser.parse_args()

    logger = logging.getLogger()
    logger.setLevel('INFO')
    formatter = logging.Formatter(
        '%(levelname)s | %(asctime)s | %(name)s:%(module)s:%(lineno)d | %(message)s')

    os.makedirs(os.path.join('data', 'ol_mpc_control'), exist_ok=True)

    # Create file handler with info log level
    fh = logging.FileHandler(os.path.join("data", "ol_mpc_control", "ol_mpc_control.log"))
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    # Create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(logging.WARNING)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    if not args.use_existing_data:
        make_data()
    make_plots()
