"""Visualisation of results."""

import pdb
import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import mixed_stand_simulator as ms_sim
import mixed_stand_approx as ms_approx
import parameters

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

def plot_control(times, control_policy, ax=None, labels=None, colors=None, **kwargs):
    """Plot given control strategy."""

    if ax is None:
        fig = plt.figure(111)
        ax = fig.add_subplot(111)

    if labels is None:
        labels = [
            "Rogue Tan (Small)", "Rogue Tan (Large)", "Rogue Bay", "Thin Bay", "Thin Red",
            "Protect Tan (Small)", "Protect Tan (Large)"
        ]

    if colors is None:
        colors = [mpl.colors.to_rgba(col, alpha=alph) for col, alph in zip(
            ["r", "r", "r", "b", "b", "purple", "purple"], [0.75, 0.5, 0.25, 0.6, 0.3, 0.6, 0.3]
        )]
    all_controls = np.array([control_policy(t) for t in times]).T

    ax.stackplot(times, *all_controls, labels=labels, colors=colors, **kwargs)

    return ax
