# -*- coding: utf-8 -*-
"""Miscellaneous utilities."""

import itertools
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


# ----------------------------------------------------------------------------------------------------------------------
def plot_init(nrows=1, ncols=1, figsize=(4,4), wspace=0, hspace=0, sharex=False, sharey=False):
    """Initializes a multirow or multicolumn plot."""

    fig, ax = plt.subplots(nrows, ncols, figsize=figsize, sharex=sharex, sharey=sharey)
    plt.subplots_adjust(wspace=wspace, hspace=hspace)
    return (fig, ax)

def plot_series(a, indices=None, cols=None, title=None, xlabel=None, figsize=(16,4), do_legend=True):
    """Plots multiple multivariate time series."""

    indices = indices or range(len(a))

    fig, ax = plot_init(len(indices), figsize=figsize, sharex=True)
    for (i,idx) in enumerate(indices):
        _,_ = plot_ts(a[idx], cols, title=title, xlabel=xlabel, do_legend=do_legend, ax=ax[i])
        ax[i].set_ylabel(indices[i])
    return (fig, ax)

def plot_ts(x, cols=None, do_subplots=False, figsize_w=16, figsize_h=2, hspace=0, title=None, xlabel=None, ylabel=None, colors=plt.get_cmap('tab10').colors, highlight_rect_xw=None, highlight_rect_kwargs={ 'alpha': 0.2, 'color': '#aaaaaa' }, do_legend=True, ylim=None, xlim_adj=(0,0), ylim_adj=(-0.2,0.2), ax=None, line_kwargs_lst=[]):
    """Plots a multivariate time series.

    Resources:
    - https://matplotlib.org/stable/tutorials/colors/colormaps.html
    """

    cols = cols or range(len(x))

    if colors:
        colors = itertools.cycle(colors)
    else:
        if len(x) <= 148:
            colors = itertools.cycle(mpl.colors.cnames.values())  # n=148
        elif len(x) <= 949:
            colors = itertools.cycle(mpl.colors.XKCD_COLORS.values())  # n=949
        else:
            colors = itertools.cycle(mpl.colors._colors_full_map.values())  # n=1163

    if ax is None:
        if do_subplots:
                fig, ax = plt.subplots(nrows=len(cols), ncols=1, sharex=True, figsize=(figsize_w, len(cols) * figsize_h))
                ax = [ax] if len(cols) == 1 else ax
        else:
            fig, ax = plt.subplots(figsize=(figsize_w, figsize_h))
            ax = [ax] * len(cols)
    else:
        fig = None
        ax = [ax] * len(cols)

    # Line plots:
    if len(line_kwargs_lst) == 0:
        line_kwargs_lst = [{ 'lw': 1.0 }] * len(x)
    elif len(line_kwargs_lst) == 1:
        line_kwargs_lst = line_kwargs_lst * len(x)

    y_min = np.min(x) if ylim is None else ylim[0]
    y_max = np.max(x) if ylim is None else ylim[1]

    for i in range(len(x)):
        ax[i].plot(np.arange(len(x[i])), x[i], label=cols[i], color=next(colors), **line_kwargs_lst[i])
        ax[i].set_xlim(0     + xlim_adj[0], len(x[i]) - 1 + xlim_adj[1])
        ax[i].set_ylim(y_min + ylim_adj[0], y_max         + ylim_adj[1])
        ax[i].set_ylabel(ylabel)
        if do_subplots:
            ax[i].set_ylabel(cols[i])
        if do_legend:
            ax[i].legend(cols)
    plt.subplots_adjust(hspace=hspace)

    # Rectangle highlights:
    if highlight_rect_xw:
        for (i,hr) in enumerate(highlight_rect_xw):
            y = ax[i].get_ylim()
            ax[i].add_patch(mpl.patches.Rectangle((hr[0], y[0]), hr[1], y[1] - y[0], **highlight_rect_kwargs))

    plt.title(title)
    plt.xlabel(xlabel)

    # if not do_subplots and do_legend:
    #     plt.legend()

    return (fig, ax)
