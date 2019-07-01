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

def create_figure(objectives, controls, sort_order, setup, params, ticks=None):
    """Generate actual figure from dataset for control sensitivity."""

    n_reps = len(objectives)

    fig = plt.figure(figsize=(6, 4))
    gs = gridspec.GridSpec(2, 4, height_ratios=[10, 1], wspace=0.7, hspace=0.4, left=0.05, top=0.93)
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

    roguing = (np.sum(controls[:, 0:3, :], axis=1) / params['max_budget'])
    thinning = (np.sum(controls[:, 3:7, :], axis=1) / params['max_budget'])
    protecting = (np.sum(controls[:, 7:, :], axis=1) / params['max_budget'])

    x, y = np.meshgrid(setup['times'], range(len(sort_order)+1))

    p2 = ax2.pcolormesh(x, y, thinning, cmap='Greens', vmin=0, vmax=1)
    p3 = ax3.pcolormesh(x, y, roguing, cmap='Reds', vmin=0, vmax=1)
    p4 = ax4.pcolormesh(x, y, protecting, cmap='Blues', vmin=0, vmax=1)

    ax1.set_title("Objective")
    ax1.set_ylabel("Parameter set")

    ax2.set_title("Thinning")
    ax2.set_xlabel("Time / yrs")
    ax3.set_title("Roguing")
    ax3.set_xlabel("Time / yrs")
    ax4.set_title("Protecting")
    ax4.set_xlabel("Time / yrs")
    for ax in [ax1, ax2, ax3, ax4]:
        ax.grid(False)

    fig.colorbar(p1, cax=cax1, label='% Difference in\nobjective', orientation='horizontal',
                 ticks=ticks, fraction=0.5)

    fig.colorbar(p2, cax=cax2, label='Thin Expense', orientation='horizontal', fraction=0.15)
    fig.colorbar(p3, cax=cax3, label='Rogue Expense', orientation='horizontal', fraction=0.15)
    fig.colorbar(p4, cax=cax4, label='Protect Expense', orientation='horizontal', fraction=0.15)

    ax1.set_yticks(np.arange(0, n_reps+1, 20) + 0.5)
    ax1.set_yticklabels(np.arange(200, -1, -20),
                        fontdict={'fontsize': 4, 'weight': 'bold'})
    plt.setp(ax2.get_yticklabels(), visible=False)
    plt.setp(ax3.get_yticklabels(), visible=False)
    plt.setp(ax4.get_yticklabels(), visible=False)

    for i in np.arange(0, n_reps+1, 20):
        ax1.axhline(y=i+0.5, xmin=0, xmax=20, c="gray", linewidth=0.25, zorder=0.0, clip_on=False)

    fig.text(0.01, 0.98, "(a)", transform=fig.transFigure, fontsize=11, fontweight="semibold")
    fig.text(0.25, 0.98, "(b)", transform=fig.transFigure, fontsize=11, fontweight="semibold")
    fig.text(0.48, 0.98, "(c)", transform=fig.transFigure, fontsize=11, fontweight="semibold")
    fig.text(0.72, 0.98, "(d)", transform=fig.transFigure, fontsize=11, fontweight="semibold")

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

    for i in range(len(summary_results)):
        # No control runs
        model = ms_sim.MixedStandSimulator.load_run_class(
            os.path.join("data", folder_name, "no_control_{}.pkl".format(i)))
        ncells = np.product(model.setup['landscape_dims'])
        state = np.sum(np.reshape(model.run['state'], (ncells, 15, -1)), axis=0) / ncells
        no_control_states.append(state)

    # OL controls
    ol_allocations = np.load(os.path.join("data", folder_name, "ol_alloc_results.npy"))

    if run_mpc:
        # MPC controls
        mpc_allocations = np.load(os.path.join("data", folder_name, "mpc_alloc_results.npy"))

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
        os.path.join("data", folder_name, "ol_control_baseline.pkl"))
    control_policy = interp1d(setup['times'][:-1], approx_model.optimisation['control'],
                              kind="zero", fill_value="extrapolate")
    _, baseline_obj, _ = model.run_policy(control_policy)

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

    fig = create_figure(ol_objectives, ol_allocations, ol_sort_order, setup, params,
                        ticks=[0.0, 200, 400])
    fig.savefig(
        os.path.join("figures", folder_name, "ol_controls.pdf"), dpi=300, bbox_inches='tight')

    if run_mpc:
        fig = create_figure(mpc_objectives, mpc_allocations, mpc_sort_order, setup, params,
                            ticks=[-50, 0.0, 50, 100])
        fig.savefig(
            os.path.join("figures", folder_name, "mpc_controls.pdf"), dpi=300, bbox_inches='tight')

def main(n_reps=10, sigma=0.25, append=False, folder_name='parameter_sensitivity', run_mpc=False):
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

    ncells = np.product(setup['landscape_dims'])

    # Baseline no control run
    model = ms_sim.MixedStandSimulator(setup, params)
    model.run_policy(control_policy=None, n_fixed_steps=None)

    with open(os.path.join("data", "scale_and_fit_results.json"), "r") as infile:
        scale_and_fit_results = json.load(infile)

    if not append:
        model.save_run(os.path.join("data", folder_name, "no_control_baseline.pkl"))

        beta_names = ['beta_1,1', 'beta_1,2', 'beta_1,3', 'beta_1,4',
                      'beta_12', 'beta_21', 'beta_2']
        beta = np.array([scale_and_fit_results[x] for x in beta_names])

        approx_params = copy.deepcopy(params)
        approx_params['rogue_rate'] *= scale_and_fit_results['roguing_factor']
        approx_params['rogue_cost'] /= scale_and_fit_results['roguing_factor']
        approx_model = ms_approx.MixedStandApprox(setup, approx_params, beta)

        logging.info("Running baseline OL control")
        _, baseline_ol_control_policy, exit_text = approx_model.optimise(
            n_stages=20, init_policy=even_policy)
        approx_model.save_optimisation(
            os.path.join("data", folder_name, "ol_control_baseline.pkl"))

        if run_mpc:
            logging.info("Running baseline MPC control")
            mpc_args['init_policy'] = baseline_ol_control_policy
            mpc_controller = mpc.Controller(setup, params, beta, approx_params=approx_params)
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

    if append:
        logging.info("Loading previous dataset to append new data to")
        # Read in summary data already generated
        with open(os.path.join("data", folder_name, "summary.json"), "r") as infile:
            summary_results = json.load(infile)

        approx_model = ms_approx.MixedStandApprox.load_optimisation_class(
            os.path.join("data", folder_name, "ol_control_baseline.pkl"))
        baseline_ol_control_policy = interp1d(
            approx_model.setup['times'][:-1], approx_model.optimisation['control'], kind="zero",
            fill_value="extrapolate")

        n_reps = (len(summary_results), len(summary_results)+n_reps)

        ol_alloc_results = np.load(os.path.join("data", folder_name, "ol_alloc_results.npy"))
        mpc_alloc_results = np.load(os.path.join("data", folder_name, "mpc_alloc_results.npy"))
    else:
        # Otherwise start afresh
        summary_results = []
        ol_alloc_results = np.zeros((0, 9, len(setup['times']) - 1))
        mpc_alloc_results = np.zeros((0, 9, len(setup['times']) - 1))
        n_reps = (0, n_reps)

    error_dist = truncnorm(-1.0/sigma, np.inf, loc=1.0, scale=sigma)

    for i in range(*n_reps):
        # 2. Perturb these parameters using Normal distribution, sigma 25%
        logging.info("Perturbing parameter set %d of %d with sigma %f", i+1, n_reps[1], sigma)
        new_params = copy.deepcopy(params)
        for param_key in perturbing_params_numbers:
            new_params[param_key] = new_params[param_key] * error_dist.rvs()
        for param_key in perturbing_params_lists:
            new_params[param_key] = (
                new_params[param_key] * error_dist.rvs(size=len(new_params[param_key])))
        
        # Set space weights and recruitment rates to NaN so can be recaluclated for dyn equilibrium
        new_params['recruit_bay'] = np.nan
        new_params['recruit_redwood'] = np.nan
        new_params['space_tanoak'] = np.full(4, np.nan)

        # 3. Recalculate space weights & recruitment rates to give dynamic equilibrium
        new_params, _ = utils.initialise_params(new_params, host_props=parameters.COBB_PROP_FIG4A)

        # 4. Run simulation model with no control policy
        model = ms_sim.MixedStandSimulator(setup, new_params)
        model.run_policy(control_policy=None, n_fixed_steps=None)
        model.save_run(os.path.join("data", folder_name, "no_control_{}.pkl".format(i)))

        # 5. Fit approximate model
        _, beta = scale_and_fit.fit_beta(setup, new_params)

        approx_new_params = copy.deepcopy(params)
        approx_new_params['rogue_rate'] *= scale_and_fit_results['roguing_factor']
        approx_new_params['rogue_cost'] /= scale_and_fit_results['roguing_factor']

        # 6. Optimise control (open-loop)
        approx_model = ms_approx.MixedStandApprox(setup, approx_new_params, beta)
        *_, exit_text = approx_model.optimise(n_stages=20, init_policy=baseline_ol_control_policy)

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

            *_, exit_text = approx_model.optimise(
                n_stages=20, init_policy=baseline_ol_control_policy)

            all_lines[31] = all_lines[31][2:]
            all_lines[32] = all_lines[32][2:]
            all_lines[33] = "# " + all_lines[33]
            all_lines[34] = "# " + all_lines[34]
            with ms_approx._try_file_open(filename) as outfile:
                outfile.writelines(all_lines)

            if exit_text not in ["Optimal Solution Found.", "Solved To Acceptable Level."]:
                logging.error("Failed optimisation. Falling back to init policy.")

        approx_model.save_optimisation(
            os.path.join("data", folder_name, "ol_control_{}.pkl".format(i)))

        # Run OL control to get objective
        ol_control_policy = interp1d(setup['times'][:-1], approx_model.optimisation['control'],
                                     kind="zero", fill_value="extrapolate")
        sim_run = model.run_policy(ol_control_policy)
        ol_obj = model.run['objective']
        sim_state = np.sum(np.reshape(sim_run[0], (ncells, 15, -1)), axis=0) / ncells
        allocation = (np.array([
            sim_state[1] + sim_state[4],
            sim_state[7] + sim_state[10],
            sim_state[13],
            np.sum(sim_state[0:6], axis=0),
            np.sum(sim_state[6:12], axis=0),
            sim_state[12] + sim_state[13],
            sim_state[14],
            sim_state[0] + sim_state[3],
            sim_state[6] + sim_state[9]])[:, :-1] * approx_model.optimisation['control'])

        allocation[0:3] *= params['rogue_rate'] * params['rogue_cost']
        allocation[3:7] *= params['thin_rate'] * params['thin_cost']
        allocation[7:] *= params['protect_rate'] * params['protect_cost']
        allocation[0] *= params['rel_small_cost']
        allocation[3] *= params['rel_small_cost']

        expense = utils.control_expenditure(
            approx_model.optimisation['control'], new_params, sim_state[:, :-1])
        for j in range(len(setup['times'])-1):
            if expense[j] > new_params['max_budget']:
                allocation[:, j] *= new_params['max_budget'] / expense[j]

        ol_alloc_results = np.concatenate([ol_alloc_results, [allocation]], axis=0)

        if run_mpc:
            mpc_args['init_policy'] = ol_control_policy
            # Optimise control (MPC)
            mpc_controller = mpc.Controller(setup, new_params, beta,
                                            approx_params=approx_new_params)
            *_, mpc_obj = mpc_controller.optimise(**mpc_args)
            mpc_controller.save_optimisation(
                os.path.join("data", folder_name, "mpc_control_{}.pkl".format(i)))
            sim_run, _ = mpc_controller.run_control()

            sim_state = np.sum(np.reshape(sim_run[0], (ncells, 15, -1)), axis=0) / ncells
            allocation = (np.array([
                sim_state[1] + sim_state[4],
                sim_state[7] + sim_state[10],
                sim_state[13],
                np.sum(sim_state[0:6], axis=0),
                np.sum(sim_state[6:12], axis=0),
                sim_state[12] + sim_state[13],
                sim_state[14],
                sim_state[0] + sim_state[3],
                sim_state[6] + sim_state[9]])[:, :-1] * mpc_controller.control)

            allocation[0:3] *= params['rogue_rate'] * params['rogue_cost']
            allocation[3:7] *= params['thin_rate'] * params['thin_cost']
            allocation[7:] *= params['protect_rate'] * params['protect_cost']
            allocation[0] *= params['rel_small_cost']
            allocation[3] *= params['rel_small_cost']

            expense = utils.control_expenditure(
                mpc_controller.control, new_params, sim_state[:, :-1])
            for j in range(len(setup['times'])-1):
                if expense[j] > new_params['max_budget']:
                    allocation[:, j] *= new_params['max_budget'] / expense[j]

            mpc_alloc_results = np.concatenate([mpc_alloc_results, [allocation]], axis=0)

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

        # Save control allocations to file
        np.save(os.path.join("data", folder_name, "ol_alloc_results.npy"), ol_alloc_results)
        np.save(os.path.join("data", folder_name, "mpc_alloc_results.npy"), mpc_alloc_results)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument("-n", "--n_reps", default=10, type=int,
                        help="Number of parameter sets to generate")
    parser.add_argument("-s", "--sigma", default=0.25, type=float,
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
