"""Test sensitivity of model and optimal control results to Cobb parameterisation."""

import copy
import json
import logging
import os
import pdb
import argparse
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import AxesGrid
from scipy.interpolate import interp1d
from scipy.stats import norm

from mixed_stand_model import mixed_stand_simulator as ms_sim
from mixed_stand_model import mixed_stand_approx as ms_approx
from mixed_stand_model import parameters
from mixed_stand_model import visualisation
from mixed_stand_model import utils
from mixed_stand_model import mpc
from scripts import scale_and_fit

def shiftedColorMap(cmap, start=0, midpoint=0.5, stop=1.0, name='shiftedcmap'):
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
    return np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])

def make_plots(folder_name='param_sensitivity'):
    """Create figures."""

    os.makedirs(os.path.join('figures', folder_name), exist_ok=True)

    # Load data...
    # Read summary results
    with open(os.path.join("data", folder_name, "summary.json"), "r") as infile:
        summary_results = json.load(infile)

    no_control_states = []
    ol_controls = []
    for i in range(len(summary_results)):
        # No control runs
        model = ms_sim.MixedStandSimulator.load_run_class(
            os.path.join("data", folder_name, "no_control_{}.pkl".format(i)))
        ncells = np.product(model.setup['landscape_dims'])
        state = np.sum(np.reshape(model.run['state'], (ncells, 15, -1)), axis=0) / ncells
        no_control_states.append(state)

        # OL controls
        approx_model = ms_approx.MixedStandApprox.load_optimisation_class(
            os.path.join("data", folder_name, "opt_control_{}.pkl".format(i)))
        ol_controls.append(approx_model.optimisation['control'])

    model = ms_sim.MixedStandSimulator.load_run_class(
            os.path.join("data", folder_name, "no_control_baseline.pkl"))
    setup = model.setup
    params = model.params
    baseline_state = np.sum(np.reshape(model.run['state'], (ncells, 15, -1)), axis=0) / ncells

    # Plotting:
    plt.style.use("seaborn-whitegrid")
    # 7. Distribution of host composition time series - show baseline, median & percentiles
    no_control_states = np.array(no_control_states)
    tan_small = np.sum(no_control_states[:, 0:6, :], axis=1)
    tan_large = np.sum(no_control_states[:, 6:12, :], axis=1)
    bay = np.sum(no_control_states[:, 12:14, :], axis=1)
    red = no_control_states[:, 14, :]

    cmap = plt.get_cmap("tab20c")
    colours = [cmap(2.5*0.05), cmap(0.5*0.05), cmap(8.5*0.05), cmap(4.5*0.05)]

    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.fill_between(setup['times'], np.percentile(red, 5, axis=0),
                    np.percentile(red, 95, axis=0), color=colours[3], alpha=0.5)
    ax.plot(setup['times'], np.percentile(red, 50, axis=0), '--', color=colours[3], alpha=0.75)
    ax.plot(setup['times'], baseline_state[14, :], '-', color=colours[3], label='Redwood')
    
    ax.fill_between(setup['times'], np.percentile(tan_small, 5, axis=0),
                    np.percentile(tan_small, 95, axis=0), color=colours[0], alpha=0.5)
    ax.plot(setup['times'], np.percentile(tan_small, 50, axis=0), '--', color=colours[0], alpha=0.75)
    ax.plot(setup['times'], np.sum(baseline_state[0:6, :], axis=0), '-', color=colours[0],
            label='Small Tanoak')

    ax.fill_between(setup['times'], np.percentile(bay, 5, axis=0),
                    np.percentile(bay, 95, axis=0), color=colours[2], alpha=0.5)
    ax.plot(setup['times'], np.percentile(bay, 50, axis=0), '--', color=colours[2], alpha=0.75)
    ax.plot(setup['times'], np.sum(baseline_state[12:14, :], axis=0), '-', color=colours[2],
            label='Bay')

    ax.fill_between(setup['times'], np.percentile(tan_large, 5, axis=0),
                    np.percentile(tan_large, 95, axis=0), color=colours[1], alpha=0.5)
    ax.plot(setup['times'], np.percentile(tan_large, 50, axis=0), '--', color=colours[1], alpha=0.75)
    ax.plot(setup['times'], np.sum(baseline_state[6:12, :], axis=0), '-', color=colours[1],
            label='Large Tanoak')

    ax.set_xlabel("Time")
    ax.set_ylabel("Host Stems")
    ax.legend()
    fig.savefig(os.path.join("figures", folder_name, "hosts.pdf"), dpi=300, bbox_inches='tight')

    # 8. Sorted by objective, heatmap showing control over time. Colour control by proportion
    #       of control expenditure on roguing, thinning and protecting -> mapping to rgb. Hopefully
    #       see a pattern emerge.

    approx_model = ms_approx.MixedStandApprox.load_optimisation_class(
        os.path.join("data", folder_name, "opt_control_baseline.pkl".format(i)))
    ol_controls.append(approx_model.optimisation['control'])
    control_policy = interp1d(setup['times'][:-1], approx_model.optimisation['control'],
                              kind="zero", fill_value="extrapolate")
    approx_model.run_policy(control_policy)

    objectives = [x['objective'] for x in summary_results]
    objectives.append(approx_model.run['objective'])
    objectives = np.array(objectives) - approx_model.run['objective']
    sort_order = np.argsort(objectives)

    fig = plt.figure(figsize=(8, 4))
    gs = gridspec.GridSpec(1, 3, width_ratios=[0.2, 0.8, 5])
    cax = fig.add_subplot(gs[0])
    ax1 = fig.add_subplot(gs[1])
    ax1.set_xticks([], [])
    ax1.set_yticks([], [])
    ax2 = fig.add_subplot(gs[2], sharey=ax1)
    ax2.set_yticks([], [])
    dummy_axis = fig.add_subplot(gs[0:2], frameon=False)
    dummy_axis.set_axis_off()

    x, y = np.meshgrid([0, 1], range(len(sort_order)+1))
    z = np.array([objectives[sort_order]]).T

    vmin = min(z)
    vmax = max(z)
    cmap = shiftedColorMap(plt.get_cmap('bwr'), midpoint=1 - vmax / (vmax + abs(vmin)))

    p1 = ax1.pcolormesh(x, y, z, cmap=cmap, vmin=vmin, vmax=vmax)

    ol_controls = np.array(ol_controls)
    rgb = np.zeros((ol_controls.shape[0], ol_controls.shape[2], 3))
    rgb[:, :, 0] = (np.sum(ol_controls[:, 0:3, :], axis=1) * params['rogue_rate'] *
                    params['rogue_cost'] / params['max_budget'])
    rgb[:, :, 1] = (np.sum(ol_controls[:, 3:7, :], axis=1) * params['thin_rate'] *
                    params['thin_cost'] / params['max_budget'])
    rgb[:, :, 2] = (np.sum(ol_controls[:, 7:, :], axis=1) * params['protect_rate'] *
                    params['protect_cost'] / params['max_budget'])

    ax2.imshow(rgb, extent=(0, 200, 0, len(sort_order)), aspect='auto')

    dummy_axis.set_title("Objective")
    ax1.set_ylabel("Parameter set")
    ax2.set_title("Control strategy")
    ax2.set_xlabel("Time")

    cbar = fig.colorbar(p1, cax=cax, extend='both', label='Difference in objective')
    cax.yaxis.set_ticks_position('left')
    cax.yaxis.set_label_position('left')

    fig.tight_layout()
    fig.savefig(os.path.join("figures", folder_name, "controls.pdf"), dpi=300, bbox_inches='tight')

    # 9. Finally, for extremes of objective difference run OL optimisation for multiple budgets and
    #       compare patterns



def main(n_reps=10, sigma=0.1, append=False, folder_name='parameter_sensitivity'):
    """Run sensitivity tests."""

    os.makedirs(os.path.join('data', folder_name), exist_ok=True)

    # Analysis:
    # 1. First construct default parameters (corrected and scaled Cobb)
    with open(os.path.join("data", "scale_and_fit_results.json"), "r") as infile:
        scale_and_fit_results = json.load(infile)

    params = parameters.CORRECTED_PARAMS
    inf_keys = ['inf_tanoak_tanoak', 'inf_tanoak_to_bay', 'inf_bay_to_bay', 'inf_bay_to_tanoak']
    for key in inf_keys:
        params[key] *= scale_and_fit_results['sim_scaling_factor']

    # Initialise using Cobb 2012 Fig 4a host proportions
    host_props = parameters.COBB_PROP_FIG4A
    ncells = 400
    params, state_init = utils.initialise_params(params, host_props=host_props)
    state_init = np.tile(state_init, ncells)

    # Initial conditions used in 2012 paper (not quite at dynamic equilibrium) for Fig 4a
    init_inf_cells = [189]
    init_inf_factor = 0.5
    for cell_pos in init_inf_cells:
        for i in [0, 4]:
            state_init[cell_pos*15+3*i+1] = init_inf_factor * state_init[cell_pos*15+3*i]
            state_init[cell_pos*15+3*i] *= (1.0 - init_inf_factor)

    setup = {
        'state_init': state_init,
        'landscape_dims': (20, 20),
        'times': np.linspace(0, 100.0, 201)
    }

    logging.info("Using landscape dimensions (%d, %d)", *setup['landscape_dims'])
    logging.info("Using initial infection in cells %s", init_inf_cells)
    logging.info("Using initial infection proportion %f", init_inf_factor)

    params['treat_eff'] = 0.75
    params['vaccine_decay'] = 0.5
    params['rogue_rate'] = 0.5
    params['thin_rate'] = 0.5
    params['protect_rate'] = 0.25
    params['div_cost'] = 0.001
    params['rogue_cost'] = 1000
    params['thin_cost'] = 1000
    params['protect_cost'] = 200
    params['payoff_factor'] = 1.0
    params['discount_rate'] = 0.0
    params['max_budget'] = 500

    # Baseline no control run
    model = ms_sim.MixedStandSimulator(setup, params)
    model.run_policy(control_policy=None, n_fixed_steps=None)

    if not append:
        model.save_run(os.path.join("data", folder_name, "no_control_baseline.pkl"))

        with open(os.path.join("data", "scale_and_fit_results.json"), "r") as infile:
            scale_and_fit_results = json.load(infile)

        beta_names = ['beta_1,1', 'beta_1,2', 'beta_1,3', 'beta_1,4', 'beta_12', 'beta_21', 'beta_2']
        beta = np.array([scale_and_fit_results[x] for x in beta_names])
        approx_model = ms_approx.MixedStandApprox(setup, params, beta)

        approx_model.optimise(n_stages=20, init_policy=even_policy)
        approx_model.save_optimisation(
            os.path.join("data", folder_name, "opt_control_baseline.pkl"))

    # Which parameters to perturb:
    # First single numbers that can be perturbed
    perturbing_params_numbers = [
        'inf_bay_to_bay', 'inf_bay_to_tanoak', 'inf_tanoak_to_bay', 'nat_mort_bay',
        'nat_mort_redwood', 'recov_tanoak', 'recov_bay', 'resprout_tanoak']
    # And lists of parameters
    perturbing_params_lists = [
        'inf_tanoak_tanoak', 'nat_mort_tanoak', 'inf_mort_tanoak', 'trans_tanoak', 'recruit_tanoak']
    
    if append:
        # Read in data already generated
        with open(os.path.join("data", folder_name, "summary.json"), "r") as infile:
            summary_results = json.load(infile)

        no_control_states = []
        ol_controls = []
        for i in range(len(summary_results)):
            # No control runs
            model = ms_sim.MixedStandSimulator.load_run_class(
                os.path.join("data", folder_name, "no_control_{}.pkl".format(i)))
            ncells = np.product(model.setup['landscape_dims'])
            state = np.sum(np.reshape(model.run['state'], (ncells, 15, -1)), axis=0) / ncells
            no_control_states.append(state)

            # OL controls
            approx_model = ms_approx.MixedStandApprox.load_optimisation_class(
                os.path.join("data", folder_name, "opt_control_{}.pkl".format(i)))
            ol_controls.append(approx_model.optimisation['control'])

        n_reps = (len(summary_results), len(summary_results)+n_reps)
    else:
        # Otherwise start afresh
        summary_results = []
        no_control_states = []
        ol_controls = []
        n_reps = (0, n_reps)

    for i in range(*n_reps):
        # 2. Perturb these parameters using Normal distribution, sigma 10%
        logging.info("Perturbing parameter set %d of %d with sigma %f", i+1, n_reps[1], sigma)
        new_params = copy.deepcopy(params)
        for param_key in perturbing_params_numbers:
            new_params[param_key] *= np.clip(norm.rvs(loc=1, scale=sigma), 0.0, None)
        for param_key in perturbing_params_lists:
            new_params[param_key] *= np.clip(
                norm.rvs(loc=1, scale=sigma, size=len(new_params[param_key])), 0.0, None)

        # 3. Recalculate space weights & recruitment rates to give dynamic equilibrium and get
        #       initial state
        new_params, _ = utils.initialise_params(new_params, host_props=host_props)

        # 4. Run simulation model with no control policy
        model = ms_sim.MixedStandSimulator(setup, new_params)
        model.run_policy(control_policy=None, n_fixed_steps=None)
        model.save_run(os.path.join("data", folder_name, "no_control_{}.pkl".format(i)))
        state = np.sum(np.reshape(model.run['state'], (ncells, 15, -1)), axis=0) / ncells
        no_control_states.append(state)

        # 5. Fit approximate model
        _, beta = scale_and_fit.fit_beta(setup, new_params)

        # 6. Optimise control (open-loop)
        approx_model = ms_approx.MixedStandApprox(setup, new_params, beta)
        approx_model.optimise(n_stages=20, init_policy=even_policy)
        approx_model.save_optimisation(
            os.path.join("data", folder_name, "opt_control_{}.pkl".format(i)))
        
        # Run OL control to get objective
        control_policy = interp1d(setup['times'][:-1], approx_model.optimisation['control'],
                                  kind="zero", fill_value="extrapolate")
        ol_controls.append(approx_model.optimisation['control'])
        approx_model.run_policy(control_policy)

        list_keys = ['inf_tanoak_tanoak', 'nat_mort_tanoak', 'inf_mort_tanoak', 'trans_tanoak',
                     'recruit_tanoak', 'space_tanoak']
        for key in list_keys:
            new_params[key] = new_params[key].tolist()

        summary_results.append({
            'iteration': i,
            'params': new_params,
            'beta': beta.tolist(),
            'objective': approx_model.run['objective']
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
    parser.add_argument("-f", "--folder", default='',
                        help="Folder name to save results in data and figures directory.")
    parser.add_argument("-a", "--append", action="store_true",
                        help="Flag to append to existing dataset")
    parser.add_argument("-e", "--use_existing_data", action="store_true",
                        help="Make plots only (no new data generated)")
    args = parser.parse_args()

    logger = logging.getLogger()
    logger.setLevel('INFO')
    formatter = logging.Formatter(
        '%(levelname)s | %(asctime)s | %(name)s:%(module)s:%(lineno)d | %(message)s')

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

    if not args.use_existing_data:
        main(n_reps=args.n_reps, append=args.append, sigma=args.sigma, folder_name=args.folder)
    make_plots(folder_name=args.folder)
