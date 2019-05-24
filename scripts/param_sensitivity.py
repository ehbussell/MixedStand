"""Test sensitivity of model and optimal control results to Cobb parameterisation."""

import copy
import json
import logging
import os
import argparse
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.interpolate import interp1d
from scipy.stats import truncnorm

from mixed_stand_model import mixed_stand_simulator as ms_sim
from mixed_stand_model import mixed_stand_approx as ms_approx
from mixed_stand_model import parameters
from mixed_stand_model import utils
from mixed_stand_model import mpc
from scripts import scale_and_fit

def shifted_color_map(cmap, start=0, midpoint=0.5, stop=1.0, name='shiftedcmap'):
    '''
    Function to offset the "center" of a colormap. Useful for
    data with a negative min and positive max and you want the
    middle of the colormap's dynamic range to be at zero.

    Input
    -----
      cmap : The matplotlib colormap to be altered
      start : Offset from lowest point in the colormap's range.
          Defaults to 0.0 (no lower offset). Should be between
          0.0 and `midpoint`.
      midpoint : The new center of the colormap. Defaults to
          0.5 (no shift). Should be between 0.0 and 1.0. In
          general, this should be  1 - vmax / (vmax + abs(vmin))
          For example if your data range from -15.0 to +5.0 and
          you want the center of the colormap at 0.0, `midpoint`
          should be set to  1 - 5/(5 + 15)) or 0.75
      stop : Offset from highest point in the colormap's range.
          Defaults to 1.0 (no upper offset). Should be between
          `midpoint` and 1.0.
    '''
    cdict = {
        'red': [],
        'green': [],
        'blue': [],
        'alpha': []
    }

    # regular index to compute the colors
    reg_index = np.linspace(start, stop, 257)

    # shifted index to match the data
    shift_index = np.hstack([
        np.linspace(0.0, midpoint, 128, endpoint=False),
        np.linspace(midpoint, 1.0, 129, endpoint=True)
    ])

    for ri, si in zip(reg_index, shift_index):
        r, g, b, a = cmap(ri)

        cdict['red'].append((si, r, r))
        cdict['green'].append((si, g, g))
        cdict['blue'].append((si, b, b))
        cdict['alpha'].append((si, a, a))

    newcmap = matplotlib.colors.LinearSegmentedColormap(name, cdict)
    plt.register_cmap(cmap=newcmap)

    return newcmap

def even_policy(time):
    """Even allocation across controls"""
    return np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])

def create_figure(objectives, controls, sort_order, setup, params):
    """Generate actual figure from dataset for control sensitivity."""

    fig = plt.figure(figsize=(6, 4))
    gs = gridspec.GridSpec(2, 4, height_ratios=[5, 1], wspace=0.7, hspace=0.4)
    gs0 = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=gs[0, 0], width_ratios=[1, 2, 1],
                                           wspace=1.5)

    cax1 = fig.add_subplot(gs[1, 0])
    ax1 = fig.add_subplot(gs0[0, 1])
    ax1.set_xticks([], [])
    ax1.set_yticks([], [])

    ax2 = fig.add_subplot(gs[0, 1], sharey=ax1)
    ax2.set_yticks([], [])
    cax2 = fig.add_subplot(gs[1, 1])

    ax3 = fig.add_subplot(gs[0, 2], sharey=ax1)
    ax3.set_yticks([], [])
    cax3 = fig.add_subplot(gs[1, 2])

    ax4 = fig.add_subplot(gs[0, 3], sharey=ax1)
    ax4.set_yticks([], [])
    cax4 = fig.add_subplot(gs[1, 3])

    x, y = np.meshgrid([0, 1], range(len(sort_order)+1))
    z = np.array([objectives[sort_order]]).T

    vmin = min(z)
    vmax = max(z)
    cmap = shifted_color_map(plt.get_cmap('PuOr'), midpoint=1 - vmax / (vmax + abs(vmin)))

    p1 = ax1.pcolormesh(x, y, z, cmap=cmap, vmin=vmin, vmax=vmax)

    opt_controls = np.array(controls)
    roguing = (np.sum(opt_controls[:, 0:3, :], axis=1) * params['rogue_rate'] *
               params['rogue_cost'] / params['max_budget'])
    thinning = (np.sum(opt_controls[:, 3:7, :], axis=1) * params['thin_rate'] *
                params['thin_cost'] / params['max_budget'])
    protecting = (np.sum(opt_controls[:, 7:, :], axis=1) * params['protect_rate'] *
                  params['protect_cost'] / params['max_budget'])

    x, y = np.meshgrid(setup['times'], range(len(sort_order)+1))

    p2 = ax2.pcolormesh(x, y, thinning, cmap='Greens', vmin=0, vmax=1)
    p3 = ax3.pcolormesh(x, y, roguing, cmap='Reds', vmin=0, vmax=1)
    p4 = ax4.pcolormesh(x, y, protecting, cmap='Blues', vmin=0, vmax=1)

    ax1.set_title("Objective")
    ax1.set_ylabel("Parameter set")

    ax2.set_title("Thinning")
    ax2.set_xlabel("Time")
    ax3.set_title("Roguing")
    ax3.set_xlabel("Time")
    ax4.set_title("Protecting")
    ax4.set_xlabel("Time")

    fig.colorbar(p1, cax=cax1, label='% Difference in\nobjective', orientation='horizontal',
                 ticks=[-25, 0.0, 25, 50], fraction=0.5)

    fig.colorbar(p2, cax=cax2, label='Thin Expense', orientation='horizontal', fraction=0.5)
    fig.colorbar(p3, cax=cax3, label='Rogue Expense', orientation='horizontal', fraction=0.5)
    fig.colorbar(p4, cax=cax4, label='Protect Expense', orientation='horizontal', fraction=0.5)

    ax1.set_yticks(np.arange(0, 101, 10) + 0.5)
    ax1.set_yticklabels(np.arange(100, -1, -10),
                        fontdict={'fontsize': 4, 'weight': 'bold'})
    plt.setp(ax2.get_yticklabels(), visible=False)
    plt.setp(ax3.get_yticklabels(), visible=False)
    plt.setp(ax4.get_yticklabels(), visible=False)

    for i in np.arange(0, 101, 10):
        ax1.axhline(y=i+0.5, xmin=0, xmax=20, c="gray", linewidth=0.25, zorder=0.0, clip_on=False)

    gs.tight_layout(fig)
    fig.canvas.draw()

    return fig

def make_plots(folder_name='param_sensitivity', run_mpc=False):
    """Create figures."""

    os.makedirs(os.path.join('figures', folder_name), exist_ok=True)

    # Load data...
    # Read summary results
    with open(os.path.join("data", folder_name, "summary.json"), "r") as infile:
        summary_results = json.load(infile)

    no_control_states = []
    ol_controls = []
    mpc_controls = []
    for i in range(len(summary_results)):
        # No control runs
        model = ms_sim.MixedStandSimulator.load_run_class(
            os.path.join("data", folder_name, "no_control_{}.pkl".format(i)))
        ncells = np.product(model.setup['landscape_dims'])
        state = np.sum(np.reshape(model.run['state'], (ncells, 15, -1)), axis=0) / ncells
        no_control_states.append(state)

        # OL controls
        approx_model = ms_approx.MixedStandApprox.load_optimisation_class(
            os.path.join("data", folder_name, "ol_control_{}.pkl".format(i)))
        ol_controls.append(approx_model.optimisation['control'])

        if run_mpc:
            # MPC controls
            mpc_controller = mpc.Controller.load_optimisation(
                os.path.join("data", folder_name, "mpc_control_{}.pkl".format(i)))
            mpc_controls.append(mpc_controller.control)

    model = ms_sim.MixedStandSimulator.load_run_class(
        os.path.join("data", folder_name, "no_control_baseline.pkl"))
    setup = model.setup
    params = model.params
    baseline_state = np.sum(np.reshape(model.run['state'], (ncells, 15, -1)), axis=0) / ncells

    # Plotting:
    plt.style.use("seaborn-whitegrid")

    # Distribution of host composition time series - show baseline, median & percentiles
    no_control_states = np.array(no_control_states)
    tan_small = np.sum(no_control_states[:, 0:6, :], axis=1)
    tan_large = np.sum(no_control_states[:, 6:12, :], axis=1)
    bay = np.sum(no_control_states[:, 12:14, :], axis=1)
    red = no_control_states[:, 14, :]

    cmap = plt.get_cmap("tab20c")
    colours = [cmap(2.5*0.05), cmap(0.5*0.05), cmap(8.5*0.05), cmap(4.5*0.05)]

    plt.rc('axes', titlesize=10)
    plt.rc('axes', labelsize=8)
    plt.rc('xtick', labelsize=8)
    plt.rc('ytick', labelsize=8)
    plt.rc('legend', fontsize=8)

    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.fill_between(setup['times'], np.percentile(tan_small, 5, axis=0),
                    np.percentile(tan_small, 95, axis=0), color=colours[0], alpha=0.5, zorder=2)
    ax.plot(setup['times'], np.percentile(tan_small, 50, axis=0), '--', color=colours[0],
            alpha=0.75, zorder=2.1)
    ax.plot(setup['times'], np.sum(baseline_state[0:6, :], axis=0), '-', color=colours[0],
            label='Small Tanoak', zorder=2.2)

    ax.fill_between(setup['times'], np.percentile(tan_large, 5, axis=0),
                    np.percentile(tan_large, 95, axis=0), color=colours[1], alpha=0.5, zorder=4)
    ax.plot(setup['times'], np.percentile(tan_large, 50, axis=0), '--', color=colours[1],
            alpha=0.75, zorder=4.1)
    ax.plot(setup['times'], np.sum(baseline_state[6:12, :], axis=0), '-', color=colours[1],
            label='Large Tanoak', zorder=4.2)

    ax.fill_between(setup['times'], np.percentile(bay, 5, axis=0),
                    np.percentile(bay, 95, axis=0), color=colours[2], alpha=0.5, zorder=3)
    ax.plot(setup['times'], np.percentile(bay, 50, axis=0), '--', color=colours[2],
            alpha=0.75, zorder=3.1)
    ax.plot(setup['times'], np.sum(baseline_state[12:14, :], axis=0), '-', color=colours[2],
            label='Bay', zorder=3.2)

    ax.fill_between(setup['times'], np.percentile(red, 5, axis=0),
                    np.percentile(red, 95, axis=0), color=colours[3], alpha=0.5, zorder=1)
    ax.plot(setup['times'], np.percentile(red, 50, axis=0), '--', color=colours[3],
            alpha=0.75, zorder=1.1)
    ax.plot(setup['times'], baseline_state[14, :], '-', color=colours[3], label='Redwood',
            zorder=1.2)

    ax.set_xlabel("Time")
    ax.set_ylabel("Host Stems")
    ax.legend(loc=2)
    fig.savefig(os.path.join("figures", folder_name, "hosts.pdf"), dpi=300, bbox_inches='tight')

    # Sorted by objective, heatmap showing control over time.

    ol_objectives = [x['ol_objective'] for x in summary_results]

    # Find baseline OL optimisation
    approx_model = ms_approx.MixedStandApprox.load_optimisation_class(
        os.path.join("data", folder_name, "opt_control_baseline.pkl"))
    control_policy = interp1d(setup['times'][:-1], approx_model.optimisation['control'],
                              kind="zero", fill_value="extrapolate")
    _, baseline_obj, _ = approx_model.run_policy(control_policy)

    # Find percentage difference from baseline
    ol_objectives = (100 * (np.array(ol_objectives) - baseline_obj) / baseline_obj)
    ol_sort_order = np.argsort(ol_objectives)

    if run_mpc:
        mpc_objectives = [x['mpc_objective'] for x in summary_results]

        # Find baseline mpc optimisation
        mpc_controller = mpc.Controller.load_optimisation(
            os.path.join("data", folder_name, "mpc_control_baseline.pkl"))
        sim_run, _ = mpc_controller.run_control()

        # Find percentage difference from baseline
        mpc_objectives = 100 * (np.array(mpc_objectives) - sim_run[1]) / sim_run[1]
        mpc_sort_order = np.argsort(mpc_objectives)

    fig = create_figure(ol_objectives, ol_controls, ol_sort_order, setup, params)
    fig.savefig(
        os.path.join("figures", folder_name, "ol_controls.png"), dpi=300, bbox_inches='tight')

    if run_mpc:
        fig = create_figure(mpc_objectives, mpc_controls, mpc_sort_order, setup, params)
        fig.savefig(
            os.path.join("figures", folder_name, "mpc_controls.png"), dpi=300, bbox_inches='tight')

def main(n_reps=10, sigma=0.1, append=False, folder_name='parameter_sensitivity', run_mpc=False):
    """Run sensitivity tests."""

    os.makedirs(os.path.join('data', folder_name), exist_ok=True)

    # Analysis:
    # 1. First construct default parameters (corrected and scaled Cobb)
    setup, params = utils.get_setup_params(
        parameters.CORRECTED_PARAMS, scale_inf=True, host_props=parameters.COBB_PROP_FIG4A)

    mpc_args = {
        'horizon': 100,
        'time_step': 0.5,
        'end_time': 100,
        'update_period': 20,
        'rolling_horz': False,
        'stage_len': 5,
        'init_policy': None,
        'use_init_first': True
    }

    # Baseline no control run
    model = ms_sim.MixedStandSimulator(setup, params)
    model.run_policy(control_policy=None, n_fixed_steps=None)

    if not append:
        model.save_run(os.path.join("data", folder_name, "no_control_baseline.pkl"))

        with open(os.path.join("data", "scale_and_fit_results.json"), "r") as infile:
            scale_and_fit_results = json.load(infile)

        beta_names = ['beta_1,1', 'beta_1,2', 'beta_1,3', 'beta_1,4',
                      'beta_12', 'beta_21', 'beta_2']
        beta = np.array([scale_and_fit_results[x] for x in beta_names])
        approx_model = ms_approx.MixedStandApprox(setup, params, beta)

        logging.info("Running baseline OL control")
        _, ol_control_policy, _ = approx_model.optimise(n_stages=20, init_policy=even_policy)
        approx_model.save_optimisation(
            os.path.join("data", folder_name, "ol_control_baseline.pkl"))

        if run_mpc:
            logging.info("Running baseline MPC control")
            mpc_args['init_policy'] = ol_control_policy
            mpc_controller = mpc.Controller(setup, params, beta)
            mpc_controller.optimise(**mpc_args)
            mpc_controller.save_optimisation(
                os.path.join("data", folder_name, "mpc_control_baseline.pkl"))

    # Which parameters to perturb:
    # First single numbers that can be perturbed
    perturbing_params_numbers = [
        'inf_bay_to_bay', 'inf_bay_to_tanoak', 'inf_tanoak_to_bay', 'nat_mort_bay',
        'nat_mort_redwood', 'recov_tanoak', 'recov_bay', 'resprout_tanoak']
    # And lists of parameters
    perturbing_params_lists = [
        'inf_tanoak_tanoak', 'nat_mort_tanoak', 'inf_mort_tanoak', 'trans_tanoak', 'recruit_tanoak']

    ncells = np.product(model.setup['landscape_dims'])

    if append:
        logging.info("Loading previous dataset to append new data to")
        # Read in data already generated
        with open(os.path.join("data", folder_name, "summary.json"), "r") as infile:
            summary_results = json.load(infile)

        n_reps = (len(summary_results), len(summary_results)+n_reps)
    else:
        # Otherwise start afresh
        summary_results = []
        n_reps = (0, n_reps)

    error_dist = truncnorm(-1.0/sigma, np.inf, loc=1.0, scale=sigma)

    for i in range(*n_reps):
        # 2. Perturb these parameters using Normal distribution, sigma 10%
        logging.info("Perturbing parameter set %d of %d with sigma %f", i+1, n_reps[1], sigma)
        new_params = copy.deepcopy(params)
        for param_key in perturbing_params_numbers:
            new_params[param_key] = new_params[param_key] * error_dist.rvs()
        for param_key in perturbing_params_lists:
            new_params[param_key] = (
                new_params[param_key] * truncnorm.rvs(size=len(new_params[param_key])))

        # 3. Recalculate space weights & recruitment rates to give dynamic equilibrium
        new_params, _ = utils.initialise_params(new_params, host_props=parameters.COBB_PROP_FIG4A)

        # 4. Run simulation model with no control policy
        model = ms_sim.MixedStandSimulator(setup, new_params)
        model.run_policy(control_policy=None, n_fixed_steps=None)
        model.save_run(os.path.join("data", folder_name, "no_control_{}.pkl".format(i)))

        # 5. Fit approximate model
        _, beta = scale_and_fit.fit_beta(setup, new_params)

        # 6. Optimise control (open-loop)
        approx_model = ms_approx.MixedStandApprox(setup, new_params, beta)
        approx_model.optimise(n_stages=20, init_policy=even_policy)
        approx_model.save_optimisation(
            os.path.join("data", folder_name, "ol_control_{}.pkl".format(i)))

        # Run OL control to get objective
        ol_control_policy = interp1d(setup['times'][:-1], approx_model.optimisation['control'],
                                     kind="zero", fill_value="extrapolate")
        approx_model.run_policy(ol_control_policy)
        ol_obj = approx_model.run['objective']

        if run_mpc:
            mpc_args['init_policy'] = ol_control_policy
            # Optimise control (MPC)
            mpc_controller = mpc.Controller(setup, params, beta)
            *_, mpc_obj = mpc_controller.optimise(**mpc_args)
            mpc_controller.save_optimisation(
                os.path.join("data", folder_name, "mpc_control_{}.pkl".format(i)))

        list_keys = ['inf_tanoak_tanoak', 'nat_mort_tanoak', 'inf_mort_tanoak', 'trans_tanoak',
                     'recruit_tanoak', 'space_tanoak']
        for key in list_keys:
            new_params[key] = new_params[key].tolist()

        summary_results.append({
            'iteration': i,
            'params': new_params,
            'beta': beta.tolist(),
            'ol_objective': ol_obj,
            'mpc_objective': mpc_obj
        })

    # Write summary results to file
    with open(os.path.join("data", folder_name, "summary.json"), "w") as outfile:
        json.dump(summary_results, outfile, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument("-n", "--n_reps", default=10, type=int,
                        help="Number of parameter sets to generate")
    parser.add_argument("-s", "--sigma", default=0.1, type=float,
                        help="Sigma to use to perturb parameter set")
    parser.add_argument("-f", "--folder", default='param_sensitivity',
                        help="Folder name to save results in data and figures directory.")
    parser.add_argument("-a", "--append", action="store_true",
                        help="Flag to append to existing dataset")
    parser.add_argument("-e", "--use_existing_data", action="store_true",
                        help="Make plots only (no new data generated)")
    parser.add_argument("-m", "--mpc", action="store_true",
                        help="Whether to run MPC optimisations also")
    args = parser.parse_args()

    logger = logging.getLogger()
    logger.setLevel('INFO')
    formatter = logging.Formatter(
        '%(levelname)s | %(asctime)s | %(name)s:%(module)s:%(lineno)d | %(message)s')

    os.makedirs(os.path.join('data', args.folder), exist_ok=True)

    # Create file handler with info log level
    fh = logging.FileHandler(os.path.join("data", args.folder, "param_sensitivity.log"))
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    # Create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(logging.WARNING)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    logging.info("Starting script with args: %r", args)

    if not args.use_existing_data:
        main(n_reps=args.n_reps, append=args.append, sigma=args.sigma, folder_name=args.folder,
             run_mpc=args.mpc)
    make_plots(folder_name=args.folder, run_mpc=args.mpc)

    logging.info("Script completed")
