"""Visualisation of results."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from scipy.interpolate import interp1d


def plot_hosts(times, model_run, ax=None, combine_ages=True, proportions=True, **kwargs):
    """Plot host proportions.

    model_run should have 15 rows and the same number of columns as times.
    Can be simulation or approximate model.
    """

    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)

    if proportions:
        host_totals = np.sum(model_run[0:15, :], axis=0)
    else:
        host_totals = 1.0

    tan1 = np.sum(model_run[0:3, :], axis=0) / host_totals
    tan2 = np.sum(model_run[3:6, :], axis=0) / host_totals
    tan3 = np.sum(model_run[6:9, :], axis=0) / host_totals
    tan4 = np.sum(model_run[9:12, :], axis=0) / host_totals
    bay = np.sum(model_run[12:14, :], axis=0) / host_totals
    red = np.sum(model_run[14:15, :], axis=0) / host_totals

    cmap = plt.get_cmap("tab20c")

    if combine_ages:
        states = [tan1+tan2, tan3+tan4, bay, red]
        colours = [cmap(2.5*0.05), cmap(0.5*0.05), cmap(8.5*0.05), cmap(4.5*0.05)]
        names = ["Small Tanoak", "Large Tanoak", "Bay", "Redwood"]
    else:
        states = [tan1, tan2, tan3, tan4, bay, red]
        colours = [cmap(3.5*0.05), cmap(2.5*0.05), cmap(1.5*0.05), cmap(0.5*0.05),
                   cmap(8.5*0.05), cmap(4.5*0.05)]
        names = ["Tanoak 1-2cm", "Tanoak 2-10cm", "Tanoak 10-30cm", "Tanoak >30cm",
                 "Bay", "Redwood"]

    if 'labels' in kwargs:
        names = kwargs.pop('labels')

    for state, colour, name in zip(states, colours, names):
        ax.plot(times, state, color=colour, label=name, **kwargs)

    return ax

def plot_dpcs(times, model_run, ax=None, combine_ages=True, proportions=True, **kwargs):
    """Plot disease progress curves.

    model_run should have 15 rows and the same number of columns as times.
    Can be simulation or approximate model.
    """

    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)

    if proportions:
        host_totals = np.sum(model_run[0:15, :], axis=0)
    else:
        host_totals = 1.0

    tan1 = model_run[1, :] / host_totals
    tan2 = model_run[4, :] / host_totals
    tan3 = model_run[7, :] / host_totals
    tan4 = model_run[10, :] / host_totals
    bay = model_run[13, :] / host_totals

    cmap = plt.get_cmap("hot")
    cmap_min = 0.2
    cmap_max = 0.7

    if combine_ages:
        states = [tan1+tan2, tan3+tan4, bay]
        colours = [cmap(cmap_min+x*(cmap_max-cmap_min)) for x in [0.125, 0.625, 1.0]]
        names = ["Small Tanoak", "Large Tanoak", "Bay"]
    else:
        states = [tan1, tan2, tan3, tan4, bay]
        colours = [cmap(cmap_min+x*(cmap_max-cmap_min)) for x in [0.0, 0.25, 0.5, 0.75, 1.0]]
        names = ["Tanoak 1-2cm", "Tanoak 2-10cm", "Tanoak 10-30cm", "Tanoak >30cm", "Bay"]

    if 'labels' in kwargs:
        names = kwargs.pop('labels')

    for state, colour, name in zip(states, colours, names):
        ax.plot(times, state, color=colour, label=name, **kwargs)

    return ax

def plot_control(times, control_policy, params, ax=None, labels=None, colors=None, **kwargs):
    """Plot given control strategy."""

    if ax is None:
        fig = plt.figure(111)
        ax = fig.add_subplot(111)

    order = np.array([3, 4, 5, 6, 0, 1, 2, 7, 8])

    if labels is None:
        labels = [
            "Thin Tan (Small)", "Thin Tan (Large)", "Thin Bay", "Thin Red",
            "Rogue Tan (Small)", "Rogue Tan (Large)", "Rogue Bay", "Protect Tan (Small)",
            "Protect Tan (Large)"
        ]

    if colors is None:
        cmap = plt.get_cmap("tab20c")
        # colors = [mpl.colors.to_rgba(col, alpha=alph) for col, alph in zip(
        #     ["r", "r", "r", "b", "b", "b", "b", "purple", "purple"],
        #     [0.75, 0.5, 0.25, 1.0, 0.6, 0.4, 0.15, 0.6, 0.3]
        # )]
        # colors = ([cmap(0.025 + x*0.05) for x in [7, 6, 5, 4]] +
        #           [cmap(0.025 + x*0.05) for x in [14, 13, 12]] +
        #           [cmap(0.025 + x*0.05) for x in [18, 16]])
        colors = ([cmap(0.025 + x*0.05) for x in [8, 9, 10, 11]] +
                  [cmap(0.025 + x*0.05) for x in [4, 5, 6]] +
                  [cmap(0.025 + x*0.05) for x in [12, 14]])
    all_controls = np.array([control_policy(t)[order] for t in times]).T

    all_controls[0:4] *= params['thin_rate'] * params['thin_cost']
    all_controls[4:7] *= params['rogue_rate'] * params['rogue_cost']
    all_controls[7:] *= params['protect_rate'] * params['protect_cost']

    all_controls[[0, 4]] *= params['rel_small_cost']

    ax.stackplot(times, *all_controls, labels=labels, colors=colors, **kwargs)

    return ax

class MixedStandAnimator:
    """Plotting object for MixedStandSimulator results."""

    def __init__(self, simulator):
        self.simulator = simulator

    @staticmethod
    def _default_plot_func(state):
        """Default plot proportion of hosts infected."""

        total_inf = np.sum(state[1::3])
        total = np.sum(state)
        return total_inf / total

    def make_animation(self, plot_function=None, video_length=10, save_file=None, **kwargs):
        """Plot spatial animation of diseased proportion over time.

        plot_function:  If specified this function takes current state of a single cell and returns
                        the desired attribute to plot on the map. By default plots proportion of all
                        hosts that are infected.
        kwargs:         Keyword arguments passed to pcolormesh
        """

        if self.simulator.run['state'] is None:
            raise RuntimeError("No run has been simulated!")

        if plot_function is None:
            plot_function = self._default_plot_func

        run_data = interp1d(self.simulator.setup['times'], self.simulator.run['state'])
        fps = 30
        nframes = fps * video_length
        times = np.linspace(
            self.simulator.setup['times'][0], self.simulator.setup['times'][-1], nframes)

        # Setup plotting data
        dataset = np.zeros((nframes, *self.simulator.setup['landscape_dims']))
        for i, time in enumerate(times):
            dataset[i] = np.apply_along_axis(
                plot_function, 1, run_data(time).reshape((self.simulator.ncells, 15))).reshape(
                    self.simulator.setup['landscape_dims'])

        fig = plt.figure()
        ax = fig.add_subplot(111)
        vmin = kwargs.pop('vmin', 0)
        vmax = kwargs.pop('vmax', 1)

        im = ax.pcolormesh(dataset[0, :, :], vmin=vmin, vmax=vmax, **kwargs)
        fig.colorbar(im, ax=ax)
        fig.tight_layout()

        time_template = 'time = {0:.1f}'
        time_text = ax.text(0.05, 0.055, time_template.format(times[0]), transform=ax.transAxes,
                            bbox={'facecolor':'w', 'alpha':0.5, 'pad':5})
        time_text.set_animated(True)

        def update(frame_number):
            im.set_array(dataset[frame_number].ravel())
            time_text.set_text(time_template.format(times[frame_number]))

            return im, time_text

        im_ani = animation.FuncAnimation(fig, update, interval=1000*video_length/nframes,
                                         frames=nframes, blit=True, repeat=True)

        if save_file is not None:
            Writer = animation.writers['ffmpeg']
            writer = Writer(fps=fps, metadata=dict(artist='Me'), bitrate=1800, codec="h264")
            im_ani.save(save_file+'.mp4', writer=writer, dpi=300)

        return im_ani

    def plot_hosts(self, ax=None, combine_ages=True, **kwargs):
        """Plot simulator host numbers as a function of time."""

        if self.simulator.run['state'] is None:
            raise RuntimeError("No run has been simulated!")

        if ax is None:
            fig = plt.figure(111)
            ax = fig.add_subplot(111)

        ncells = np.product(self.simulator.setup['landscape_dims'])

        return plot_hosts(
            self.simulator.setup['times'],
            np.sum(np.reshape(self.simulator.run['state'], (ncells, 15, -1)), axis=0) / ncells,
            ax=ax, combine_ages=combine_ages, **kwargs)

    def plot_dpcs(self, ax=None, combine_ages=True, **kwargs):
        """Plot simulator disease progress curves as a function of time."""

        if self.simulator.run['state'] is None:
            raise RuntimeError("No run has been simulated!")

        if ax is None:
            fig = plt.figure(111)
            ax = fig.add_subplot(111)

        ncells = np.product(self.simulator.setup['landscape_dims'])

        return plot_dpcs(
            self.simulator.setup['times'],
            np.sum(np.reshape(self.simulator.run['state'], (ncells, 15, -1)), axis=0) / ncells,
            ax=ax, combine_ages=combine_ages, **kwargs)
