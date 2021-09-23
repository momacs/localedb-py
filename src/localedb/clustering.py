# -*- coding: utf-8 -*-
"""LocaleDB plotting functionality."""

import math
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import pywt
import scipy
import scipy.signal
import sklearn
import sklearn.cluster
import sklearn.decomposition
import sklearn.metrics
import sklearn.preprocessing
import sklearn_extra.cluster
import tslearn
import tslearn.clustering
import tslearn.metrics
import tslearn.preprocessing

from .util import plot_init, plot_series, plot_ts


# ----[ Plotting ]------------------------------------------------------------------------------------------------------

def _idices2slices(a, a0=0, a1=99):
    """Converts an iterable of split-point indices into an array of range indices.

    E.g.,
        []      --> [(a0,a1)]
        [4]     --> [(a0, 3), (4, a1)]
        [2,3,6] --> [(a0, 1), (2, 2), (3, 5), (6, a1)]
    """

    if a is None or len(a) == 0:
        return [(a0,a1)]
    return [(a[i-1] if i > 0 else a0, a[i]-1) for i in range(len(a))] + [(a[-1], a1)]

def plot_cluster_centers_kmeans(model, cols=None, title=None, xlabel=None, figsize=(16,4), do_legend=True):
    return plot_series(np.transpose(model.cluster_centers_, (0,2,1)), cols=cols, title=title, xlabel=xlabel, figsize=figsize)

def plot_cluster_centers_kmedoids(model, n_ts, cols=None, title=None, xlabel=None, figsize=(16,4), do_legend=True):
    """Need to reshape the cluster centers based on the number of time series `n_ts`."""

    if model.cluster_centers_ is None:
        return None
    return plot_series(model.cluster_centers_.reshape((model.cluster_centers_.shape[0], n_ts, -1)), cols=cols, title=title, xlabel=xlabel, figsize=figsize)

def plot_dendrogram(model, y_label=None, title=None, figsize=(16,6), **kwargs):
    """Create linkage matrix and then plot the dendrogram.

    https://scikit-learn.org/stable/auto_examples/cluster/plot_agglomerative_dendrogram.html#sphx-glr-auto-examples-cluster-plot-agglomerative-dendrogram-py
    """

    # Create the counts of samples under each node:
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack([model.children_, model.distances_, counts]).astype(float)

    # Plot the corresponding dendrogram:
    fig, ax = plt.subplots(1,1, figsize=figsize)
    scipy.cluster.hierarchy.dendrogram(linkage_matrix, labels=np.arange(len(linkage_matrix) + 1), ax=ax, **kwargs)
    ax.set_ylabel(y_label or 'Distance')
    ax.set_title(title)
    plt.show()
    return (fig, ax)

def plot_dist_mat(d, figsize=(12,12), title=None, ylabel=None, ax=None, cmap='Greys'):
    """Plots distance matrix."""

    if d is None:
        return
    if ax is None:
        fig, ax = plt.subplots(1,1, figsize=figsize)
    else:
        fig = None
    ax.title.set_text(title)
    ax.pcolormesh(np.arange(0, d.shape[0]), np.arange(0, d.shape[0]), np.abs(d), cmap=cmap, shading='gouraud')
    ax.set_ylabel(ylabel)
    return (fig, ax)

def plot_dist_mat_multi_col(D, figsize=(3,3), titles=None, ylabel=None, wspace=0, sharey=False, cmap='Greys'):
    """Plots distaince matrices in a multi column single row manner."""

    titles = titles or [None] * len(D)
    fig, ax = plot_init(1, len(D), (len(D) * figsize[0] + (len(D) - 1) * wspace, figsize[1]), wspace=wspace, sharey=sharey)
    for (i,d) in enumerate(D):
        plot_dist_mat(d, title=titles[i], ax=ax[i], cmap=cmap)
        if i == 0:
            ax[i].set_ylabel(ylabel)
    return (fig, ax)

def plot_dist_mat_multi_row(D, figsize=(3,3), titles=None, ylabels=None, wspace=0, hspace=0, sharex=False, sharey=False, cmap='Greys'):
    """Plots distaince matrices in a multi column and multi row manner (square matrix)."""

    nr = len(D)
    nc = len(D[0])

    titles = titles or [None] * nc
    ylabels = ylabels or [None] * nr

    if len(ylabels) != nr:
        raise ValueError(f'Number of rows ({nr}) and ylabels ({len(ylabels)}) has to match.')
    if len(titles) != nc:
        raise ValueError(f'Number of columns ({nc}) and titles ({len(titles)}) has to match.')

    fig, ax = plot_init(nr, nc, (nc * figsize[0] + (nc-1) * wspace, nr * figsize[1] + (nr-1) * hspace), wspace, hspace, sharex, sharey)
    for i in range(nr):
        for j in range(nc):
            plot_dist_mat(D[i][j], ax=ax[i][j], cmap=cmap)
            if i == 0:
                ax[i][j].title.set_text(titles[j])
            if j == 0:
                ax[i][j].set_ylabel(ylabels[i])
    return (fig, ax)

def plot_dist_mat_hist(d, n=None, min_max=None, figsize=(3,3), title=None, ylabel=None, ax=None, color='Gray'):
    """Plots distance matrix histogram."""

    if d is None:
        return
    if ax is None:
        fig, ax = plt.subplots(1,1, figsize=figsize)
    else:
        fig = None
    ax.title.set_text(title)
    n, bins, patches = ax.hist(np.real(d[np.triu_indices(d.shape[0] - 1)]), n, range=min_max, color=color)
    ax.set_ylabel(ylabel)
    return (fig, ax, (n, bins, patches))

def plot_dist_mat_hist_multi_col(D, n=None, figsize=(3,3), titles=None, ylabel=None, wspace=0, sharey=False, fix_ranges=True, color='Gray'):
    """Plots distaince matrice histograms in a multi column single row manner."""

    titles = titles or [None] * len(D)

    if fix_ranges:
        min_max = [0,0]
        for d in D:
            min_max[1] = max(min_max[1], np.real(np.max(d)))
    else:
        min_max = None

    fig, ax = plot_init(1, len(D), (len(D) * figsize[0] + (len(D) - 1) * wspace, figsize[1]), wspace=wspace, sharey=sharey)
    for (i,d) in enumerate(D):
        plot_dist_mat_hist(d, n, min_max, title=titles[i], ax=ax[i], color=color)
        if i == 0:
            ax[i].set_ylabel(ylabel)
    return (fig, ax)

def plot_dist_mat_hist_multi_row(D, n=None, figsize=(3,3), titles=None, ylabels=None, wspace=0, hspace=0, sharex=False, sharey=False, fix_ranges=True, color='Grey'):
    """Plots distaince matrice histograms in a multi column and multi row manner (square matrix)."""

    nr = len(D)
    nc = len(D[0])

    titles = titles or [None] * nc
    ylabels = ylabels or [None] * nr

    if fix_ranges:
        min_max = [0,0]
        for i in range(nr):
            for j in range(nc):
                min_max[1] = max(min_max[1], np.real(np.max(D[i][j])))
    else:
        min_max = None

    if len(ylabels) != nr:
        raise ValueError(f'Number of rows ({nr}) and ylabels ({len(ylabels)}) has to match.')
    if len(titles) != nc:
        raise ValueError(f'Number of columns ({nc}) and titles ({len(titles)}) has to match.')

    fig, ax = plot_init(nr, nc, (nc * figsize[0] + (nc-1) * wspace, nr * figsize[1] + (nr-1) * hspace), wspace, hspace, sharex, sharey)
    for i in range(nr):
        for j in range(nc):
            plot_dist_mat_hist(D[i][j], n, min_max, title=titles[i], ax=ax[i][j], color=color)
            if i == 0:
                ax[i][j].title.set_text(titles[j])
            if j == 0:
                ax[i][j].set_ylabel(ylabels[i])
    return (fig, ax)

def plot_cluster_perf_eval_heatmap(res_dict, metric='ari', values_fontsize=8, figsize=(12,9)):
    ds_lst = list(res_dict.keys())
    ds_name_lst = [v['name'] for v in res_dict.values()]
    method_lst = res_dict[list(res_dict.keys())[0]]['perf'][metric].keys()

    scores = np.array([[res_dict[ds]['perf'][metric][m] for m in method_lst] for ds in ds_lst])

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(scores)

    ax.set_xticks(np.arange(len(method_lst)))
    ax.set_yticks(np.arange(len(ds_name_lst)))

    ax.set_xticklabels(method_lst)
    ax.set_yticklabels(ds_name_lst)

    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')

    if values_fontsize > 0:
        for i in range(len(ds_name_lst)):
            for j in range(len(method_lst)):
                text = ax.text(j, i, np.round(scores[i,j], 2), ha='center', va='center', color='w', size=values_fontsize)

    ax.set_title(metric)
    fig.tight_layout()
    plt.show()

def plot_scalogram_cwt(cwt, scales, plot_type=0, figsize=(16,3), ax=None, title=None, ylabel=None, cmap='viridis'):
    """Plots a scalogram of the CWT specified.
    """

    if ax is None:
        fig, ax = plt.subplots(1,1, figsize=figsize)
    else:
        fig = None
    ax.set_title(title)
    if ylabel is not None:
        ax.set_ylabel(ylabel)  # fontsize=18
    if plot_type == 0:
        ax.pcolormesh(np.arange(0, len(cwt.T)), scales, np.abs(cwt), cmap=cmap, shading='gouraud')  # https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.morlet2.html#scipy.signal.morlet2
    elif plot_type == 1:
        ax.imshow(cwt, extent=[0, len(a), 1, max(scales)], cmap=cmap, aspect='auto', vmax=abs(cwt).max(), vmin=-abs(cwt).max())  # https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.cwt.html
    return (fig, ax)

def plot_cluster_perf_eval_heatmap_split(res_dict, metric='ari', vsplit=[], hsplit=[], wspace=0.15, hspace=0.15, values_fontsize=8, figsize=(0.5,0.5), cmap=None, lst_rem=['a_covid_pa_pop_k8', 'a_covid_pa_adu_obe_k8', 'a_covid_pa_chi_pov_k8', 'a_covid_pa_flu_vax_k8', 'a_covid_pa_mam_scr_k8', 'a_covid_pa_d_pop_k8', 'a_covid_pa_d_adu_obe_k8', 'a_covid_pa_d_chi_pov_k8', 'a_covid_pa_d_flu_vax_k8', 'a_covid_pa_d_mam_scr_k8']):
    res_dict = { k:v for (k,v) in res_dict.items() if k not in lst_rem }

    ds_lst = list(res_dict.keys())
    ds_name_lst = [v['name'] for v in res_dict.values()]
    method_lst = list(res_dict[list(res_dict.keys())[0]]['perf'][metric].keys())

    nh = len(method_lst)
    nv = len(ds_lst)

    scores = np.array([[res_dict[ds]['perf'][metric][m] for m in method_lst] for ds in ds_lst])

    if len(vsplit) > 0:
        figsize = (figsize[0], figsize[1] + 0.05)
    if len(hsplit) > 0:
        figsize = (figsize[0] + 0.05, figsize[1])

    vsplit = _idices2slices(vsplit, 0, nv - 1)
    hsplit = _idices2slices(hsplit, 0, nh - 1)

    fig, axes = plot_init(1, 1, (figsize[0] * nh, figsize[1] * nv), wspace, hspace)
    images = []

    # Plot heatmaps:
    for (vi,v) in enumerate(vsplit):
        for (hi,h) in enumerate(hsplit):
            mh = h[1] - h[0] + 1
            mv = v[1] - v[0] + 1

            ax = plt.subplot2grid((nv, nh), (v[0], h[0]), colspan=mh, rowspan=mv)  # sharex
            ax.invert_yaxis()

            images.append(ax.pcolor(scores[v[0]:v[1]+1, h[0]:h[1]+1], cmap=cmap))

            ax.set_xticks(np.arange(mh) + 0.5)
            ax.set_yticks(np.arange(mv) + 0.5)

            if vi == len(vsplit) - 1:
                ax.set_xticklabels(method_lst[h[0]:h[1]+1])
            else:
                ax.set_xticklabels([''] * mh)

            if hi == 0:
                ax.set_yticklabels(ds_name_lst[v[0]:v[1]+1])
            else:
                ax.set_yticklabels([''] * mv)

            plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')

            if values_fontsize > 0:
                for i in range(mv):
                    for j in range(mh):
                        text = ax.text(j+0.5, i+0.5, np.round(scores[v[0]+i, h[0]+j], 2), ha='center', va='center', color='w', size=values_fontsize)

    # Normalize heatmaps:
    vmin = min(i.get_array().min() for i in images)
    vmax = max(i.get_array().max() for i in images)
    norm = mpl.colors.Normalize(vmin, vmax)
    for i in images:
        i.set_norm(norm)

    fig.tight_layout()
    plt.show()

def plot_scalogram(a, wavelet, scales=None, ax=None, title=None, ylabel=None, **kwargs):
    """Computes CWT and plots its scalogram.
    """

    if scales is None:
        scales = np.arange(1, len(a))
    elif not isinstance(scales, Iterable):
        scales = np.arange(1, scales)

    if isinstance(wavelet, str):
        cwt = pywt.cwt(a, scales, wavelet)[0]
    elif callable(wavelet):
        cwt = scipy.signal.cwt(a, wavelet, scales)
    else:
        raise ValueError("Unknown type of the 'wavelet' argument.")

    return plot_scalogram_cwt(cwt, scales, ax=ax, title=title, ylabel=ylabel, **kwargs)


def plot_ts_clusters_sample(a, labels, m=2, n=0, ts_labels=None, do_split_cols=False, do_match_ylim=False, grid_style=':', figsize_w=8, figsize_h=1, wspace=0.1, hspace=0.1, bg_color=(0.97, 0.97, 0.97), highlight_rect_xw=None, highlight_rect_kwargs={'alpha': 0.2, 'color': '#aaaaaa'}, colors=plt.get_cmap('tab10').colors, line_kwargs_lst=[]):
    """Plots a random sample of multivariate time series drawn from each of the clusters based on the cluster labels provided.

    Args:
        a (numpy array): Dataset of multivariate time series with shape of (n_units, n_timeseries, n_values).
        labels (array-like): Cluster labels.
        m (int): Number of time series to plot per cluster.
        n (int, optional): Number of clusters to plot; if ``0`` all clusters are plotted.
        ts_labels (array-like, optional): Labels of the univariate time series for plot legend.
        do_split_cols (boolean): Split multivariate time series into columns?
        grid_style (string, optional): Grid linestyle (e.g., ``:``, ``--`` or ``-``); ``None`` for no grid.
        figsize_w (float): Width of a single column.
        figsize_h (float): Height of a single row.
        wspace (float): Space between subplots width-wise.
        hspace (float): Space between subplots height-wise.
        bg_color (color): Subplots background color.
        colors (array-like): Series colors.
    """

    n = n if n != 0 else np.unique(labels).shape[0]
    nm = sum([min(m, len(np.argwhere(labels == l))) for l in np.unique(labels)[:n]])  # total number of rows; depends on the max number of samples in each cluster

    ts_labels = ts_labels if ts_labels is not None else np.arange(a.shape[1]).tolist()

    # (1) Init figure and axes:
    # fig, _ = plot_init(nm, 1 if not do_split_cols else a.shape[1], (figsize_w * (1 if not do_split_cols else a.shape[1]), figsize_h * n * m), sharex=True)
    fig = plt.figure(figsize=(figsize_w * (1 if not do_split_cols else a.shape[1]), figsize_h * n * m), constrained_layout=False)
    outer_grid = fig.add_gridspec(n, 1, wspace=0, hspace=hspace)

    # (1.1) Y axes limits:
    ylim = None
    if do_match_ylim:
        if not do_split_cols:
            ylim = (np.min(a), np.max(a))
        else:
            ylim = [(np.min(a[:,i]), np.max(a[:,i])) for i in range(a.shape[1])]

    # (2) Plot:
    for (outer_ir, l) in enumerate(np.unique(labels)[:n]):  # for each cluster label
        # (2.1) Sample items from the current cluster:
        l_idx = np.argwhere(labels == l).T[0]
        l_idx_sample = np.random.choice(l_idx, min(m, len(l_idx)), replace=False)
        a_l = a[l_idx_sample]

        # (2.2) Prepare axes:
        inner_grid = outer_grid[outer_ir,0].subgridspec(a_l.shape[0], 1 if not do_split_cols else a.shape[1], wspace=wspace, hspace=0)
        axes = inner_grid.subplots()
        axes = [ax for _,ax in np.ndenumerate(axes)]
        if grid_style:
            for ax in axes:
                ax.grid('on', linestyle=grid_style)
        for ax in axes:
            ax.set_facecolor(bg_color)

        # (2.3) Plot the time series:
        i_ax = 0  # axis index
        for (i,ts) in enumerate(a_l):
            if not do_split_cols:
                ax = axes[i_ax]
                ylim_ = ylim if do_match_ylim else (np.min(a_l), np.max(a_l))
                legend = [f'{ts_labels[0]} ({l_idx.shape[0]})'] + ts_labels[1:]
                plot_ts(ts, cols=legend, ylabel=f'{l}:{l_idx_sample[i]}', colors=colors, highlight_rect_xw=highlight_rect_xw, highlight_rect_kwargs=highlight_rect_kwargs, do_legend=(i == 0), ylim=ylim_, ax=ax, line_kwargs_lst=line_kwargs_lst)
                if outer_ir + i < n:
                    ax.get_xaxis().set_ticklabels([])
                i_ax += 1
            else:
                for ic in range(a.shape[1]):
                    ax = axes[i_ax]
                    ylim_ = ylim[ic] if do_match_ylim else (np.min(a_l[:,ic]), np.max(a_l[:,ic]))
                    cols = [f'{ts_labels[ic]}' + (f' ({l_idx.shape[0]})' if ic == 0 else '')]
                    plot_ts([ts[ic]], cols=cols, ylabel=(f'{l}:{l_idx_sample[i]}' if ic == 0 else ''), colors=[colors[ic]], highlight_rect_xw=highlight_rect_xw, highlight_rect_kwargs=highlight_rect_kwargs, do_legend=(i == 0), ylim=ylim_, ax=ax, line_kwargs_lst=line_kwargs_lst)
                    if outer_ir + i < n + 1:
                        ax.get_xaxis().set_ticklabels([])
                    i_ax += 1
    return fig


def plot_ts_clusters_all(a, labels, n=0, ts_labels=None, grp_color=None, do_color_clusters=False, do_split_cols=False, do_plot_mean=False, do_match_ylim=False, grid_style=':', figsize_w=8, figsize_h=1, wspace=0.1, hspace=0, bg_color=(0.97, 0.97, 0.97), highlight_rect_xw=None, highlight_rect_kwargs={'alpha': 0.2, 'color': '#aaaaaa'}, colors=plt.get_cmap('tab10').colors, line_alpha_base=1.0, line_kwargs_lst=[]):
    """Plots all multivariate time series from each of the clusters based on the cluster labels provided.

    Args:
        a (numpy array): Dataset of multivariate time series with shape of (n_units, n_timeseries, n_values).
        labels (array-like): Cluster labels.
        n (int): Number of clusters to plot; if ``0`` all clusters are plotted.
        ts_labels (Iterable): Labels of the univariate time series for plot legend.
        grp_color (IterableIterable[int]): Ids of groups. Used only when ``do_split_cols`` is ``True``.
        do_split_cols (boolean): Split multivariate time series into columns?
        grid_style (string): Grid linestyle (e.g., ``:``, ``--`` or ``-``); ``None`` for no grid.
        figsize_w (float): Width of a single column.
        figsize_h (float): Height of a single row.
        wspace (float): Space between subplots width-wise.
        hspace (float): Space between subplots height-wise.
        bg_color (color): Subplots background color.
        highlight_rect_xw (Iterable): X-axis rectagles to highlight.
        highlight_rect_kwargs: (Mapping or Iterable): X-axis highlight rectagles style(s).
        colors (Iterable): Series colors.
        line_kwargs_lst (Iterable): Keyword args of line style.
    """

    n = n if n != 0 else np.unique(labels).shape[0]

    ts_labels = ts_labels if ts_labels is not None else np.arange(a.shape[1])

    # (1) Init figure and axes:
    fig, axes = plot_init(n, 1 if not do_split_cols else a.shape[1], (figsize_w * (1 if not do_split_cols else a.shape[1]), figsize_h * n), wspace=wspace, hspace=hspace, sharex=True)
    axes = [ax for _,ax in np.ndenumerate(axes)]

    # (1.1) Axes styles:
    if grid_style:
        for ax in axes:
            ax.grid('on', linestyle=grid_style)
    for ax in axes:
        ax.set_facecolor(bg_color)

    # (1.2) Y axes limits:
    ylim = None
    if do_match_ylim:
        if not do_split_cols:
            ylim = (np.min(a), np.max(a))
        else:
            ylim = [(np.min(a[:,i]), np.max(a[:,i])) for i in range(a.shape[1])]

    # (2) Plot:
    i_ax = 0  # axis index (incremented in the loop below)
    legend = []
    for (ir,l) in enumerate(np.unique(labels)[:n]):  # for each cluster label
        # (2.1) Subset the dataset from the current cluster:
        l_idx = np.argwhere(labels == l).T[0]
        a_l = a[l_idx]
        grp_color_l = None if grp_color is None else grp_color[l_idx]

        # (2.1) Plot the mean of each time series:
        if do_plot_mean:
            line_kwargs_lst_ = [{ 'lw': 1.5, 'ls': '--', 'alpha': 1.00 }]
            if not do_split_cols:
                plot_ts(np.mean(a_l, 0), colors=colors if not do_color_clusters else [colors[ir]], ax=axes[i_ax], line_kwargs_lst=line_kwargs_lst_)
            else:
                for ic in range(a.shape[1]):
                    if do_color_clusters:
                        plot_ts([np.mean(a_l[:,ic], 0)], colors=[colors[ir]], ax=axes[i_ax + ic], line_kwargs_lst=line_kwargs_lst_)
                    else:
                        plot_ts([np.mean(a_l[:,ic], 0)], colors=[colors[ic] if grp_color is None else 'gray'], ax=axes[i_ax + ic], line_kwargs_lst=line_kwargs_lst_)

        # (2.2) Plot the individual time series:
        for (i,ts) in enumerate(a_l):
            if not do_split_cols:
                ax = axes[i_ax]
                ylim_ = ylim if do_match_ylim else (np.min(a_l), np.max(a_l))
                legend = [f'{ts_labels[0]} ({l_idx.shape[0]})'] + ts_labels[1:]
                line_kwargs_lst_ = [{ 'lw': 1.0, 'alpha': line_alpha_base / max(1.5, math.log(l_idx.shape[0])) }]
                plot_ts(ts, cols=legend, ylabel=f'{l}', colors=colors if not do_color_clusters else [colors[ir]], highlight_rect_xw=highlight_rect_xw if i == 0 else None, highlight_rect_kwargs=highlight_rect_kwargs, do_legend=(i == 0), ylim=ylim_, ax=ax, line_kwargs_lst=line_kwargs_lst_)
            else:
                for ic in range(a.shape[1]):
                    ax = axes[i_ax + ic]
                    ylim_ = ylim[ic] if do_match_ylim else (np.min(a_l[:,ic]), np.max(a_l[:,ic]))
                    cols = [f'{ts_labels[ic]}' + (f' ({l_idx.shape[0]})' if ic == 0 else '')]
                    if do_color_clusters:
                        colors_ = [colors[ir]]
                    else:
                        colors_ = [colors[ic]] if grp_color_l is None else [colors[grp_color_l[i]]]
                    line_kwargs_lst_ = [{ 'lw': 1.0, 'alpha': line_alpha_base / max(1.5, math.log(l_idx.shape[0])) }]
                    plot_ts([ts[ic]], cols=cols, ylabel=(f'{l}' if ic == 0 else ''), colors=colors_, highlight_rect_xw=highlight_rect_xw if i == 0 else None, highlight_rect_kwargs=highlight_rect_kwargs, do_legend=(i == 0), ylim=ylim_, ax=ax, line_kwargs_lst=line_kwargs_lst_)

        i_ax += 1 if not do_split_cols else a.shape[1]

    if grp_color is not None and do_split_cols:
        lines = [mpl.lines.Line2D([0], [0], color=colors[i], lw=1) for i in np.unique(grp_color)]
        fig.legend(lines, np.unique(grp_color), loc='center right')

    return fig


# ----[ Clustering: Distance matrices ]---------------------------------------------------------------------------------

def standardize_complex(a):
    """Standardizes an array of complex numbers.

    SRC: https://datascience.stackexchange.com/questions/55795/how-to-normalize-complex-valued-data
    """

    a_re = np.real(a)
    a_im = np.imag(a)

    a_re = (a_re - a_re.mean()) / a_re.std()
    a_im = (a_im - a_im.mean()) / a_im.std()

    return a_re + 1j * a_im

def standardize_ts(a):
    if np.iscomplexobj(a):
        return np.array([standardize_complex(a[i,:,:].T).T for i in range(a.shape[0])])
    else:
        return np.array([sklearn.preprocessing.StandardScaler().fit_transform(a[i,:,:].T).T for i in range(a.shape[0])])

def standardize_ts_wt(a):
    if np.iscomplexobj(a):
        return np.array([[standardize_complex(a[i,j,:,:].T).T for j in range(a.shape[1])] for i in range(a.shape[0])])
    else:
        return np.array([[sklearn.preprocessing.StandardScaler().fit_transform(a[i,j,:,:].T).T for j in range(a.shape[1])] for i in range(a.shape[0])])

def pdist(a, metric='euclidean', is_condensed=True, *args, **kwargs):
    d = scipy.spatial.distance.pdist(a.reshape((a.shape[0], -1)), metric, *args, **kwargs)
    return d if is_condensed else scipy.spatial.distance.squareform(d)

def pca_svd(x, n_components=0.95):
    """Principal Component Analysis (PCA) by Singular Value Decomposition (SVD).

    The `sklearn.decomposition.PCA` only handles real numbers.  This function handles complex numbers as well which is necessary for dealing with complex wavelets
    like the complex Morlet (scipy.signal.morlet2).

    Returns:
        (singular values, principal components, explained variance ratio)
    """

    # w,v = eig(x.T @ x)  # eigenvalues, eigenvectors

    # x = (x - x.mean(axis=0)) / x.std(axis=0)
    # u,s,v = np.linalg.svd(x, False)  # eigenvalues, singluar values, eigenvectors, ...
    u,s,v = scipy.linalg.svd(x, False)  # eigenvalues, singluar values, eigenvectors, ...

    s2 = s**2
    s2_sum = sum(s2)
    var_expl = [(i / s2_sum) for i in sorted(s2, reverse=True)]

    if n_components >= 1:  # target the number of components
        n_components = min(n_components, len(s))
        s = s[:n_components]
        v = v[:n_components]
        var_expl = var_expl[:n_components]
    elif n_components > 0 and n_components < 1:  # target the total amount of variability explained
        i = 0
        var_expl_target = 0
        while var_expl_target < n_components and i < len(s):
            var_expl_target += var_expl[i]
            i += 1
        s = s[:i]
        v = v[:i]
        var_expl = var_expl[:i]

    return (s, v, var_expl)

def pca_eig(x, n_components=0.95):
    """Principal Component Analysis (PCA) by Eigendecomposition.

    The `sklearn.decomposition.PCA` only handles real numbers.  This function handles complex numbers as well which is necessary for dealing with complex wavelets
    like the complex Morlet (scipy.signal.morlet2).

    This method is much slower than the SVD based one, at least for the inputs I've been feeding the two.

    Returns:
        (singular values, principal components, explained variance ratio)
    """

    x -= np.mean(x, axis=0)

    # w,v = np.linalg.eig(np.cov(x, rowvar=False))  # eigenvalues, eigenvectors
    w,v = np.linalg.eig(x.T @ x)  # eigenvalues, eigenvectors
    v = v.T

    w_sum = sum(w)
    var_expl = [(i / w_sum) for i in sorted(w, reverse=True)]

    if n_components >= 1:  # target the number of components
        n_components = min(n_components, len(w))
        w = w[:n_components]
        v = v[:n_components]
        var_expl = var_expl[:n_components]
    elif n_components > 0 and n_components < 1:  # target the total amount of variability explained
        i = 0
        var_expl_target = 0
        while var_expl_target < n_components and i < len(w):
            var_expl_target += var_expl[i]
            i += 1
        w = w[:i]
        v = v[:i]
        var_expl = var_expl[:i]

    return (np.sqrt(w), v, var_expl)

def pca(a, n_components=0.95, **kwargs):
    """Compute real numbers PCA or dispatch for complex numbers."""

    if np.iscomplexobj(a):
        return np.array([pca_svd(a[i,j,:,:], n_components)[1] for j in range(a.shape[0])])
    else:
        return np.array([sklearn.decomposition.PCA(n_components=n_components, **kwargs).fit(a[i,:,:]).components_ for i in range(a.shape[0])])

def pca_wt(a, n_components=0.95, **kwargs):
    """Compute real numbers PCA or dispatch for complex numbers; wavelet transform input array version."""

    a = standardize_ts_wt(a)

    if np.iscomplexobj(a):
        return np.array([[pca_svd(a[i,j,:,:], n_components)[1] for j in range(a.shape[1])] for i in range(a.shape[0])])
    else:
        return np.array([[sklearn.decomposition.PCA(n_components=n_components, **kwargs).fit(a[i,j,:,:]).components_ for j in range(a.shape[1])] for i in range(a.shape[0])])

def spca(L,M):
    """PCA similarity measure (SPCA; Krzanowski, 1979)."""

    return (L @ M.T @ M @ L.T).trace() / L.shape[0]

def cdist_spca(a):
    return np.array([[spca(a[i], a[j]) if i != j else 1.0 for j in range(a.shape[0])] for i in range(a.shape[0])])
    return np.array([[spca(a[i], a[j]) for j in range(a.shape[0])] for i in range(a.shape[0])])  # if i != j else 1.0

def dist_mat(a, metric='euclidean', ts_weights=None, wavelet=scipy.signal.ricker, scales=None, scales_w=None, n_components=0.95, n_jobs=max(1, os.cpu_count() - 2), ret_all=False, is_verbose=False):
    """Compute distance matrix for multivariate time series based on multiscale continuous wavelet transform.

    Args:
        a (Iterable): Array-like of shape (n_units, n_time_series, n_samples)
        ...

    Multivariate distance measures
        DTW: Generalization of discrete sequences comparison algorithms (e.g., minimum string edit distance) to continuous values sequences (Cormen, Leiserson, Rivest, 1990).
        PCSA: Uses the first k eigenvectors (which corresponding eigenvalues explain the desired amount of variance); does not satisfy the trangle inequality.
        Eros: Uses both eigenvectors and eigenvalues; satisfies the trangle inequality.

    arr | idx-0   idx-1   idx-2   idx-3  shape e.g.        array content
    ----+---------------------------------------------------------------------
    a   | unit    var     t              (3, 2, 200     )  data (in)
    b   | unit    var     scale   t      (3, 2,   4, 200)  CWT (scipy)
    b   | scale   unit    var     t      (4, 3,   2, 200)  CWT (pywt)
    c   | scale   unit    t       var    (4, 3, 200,   2)  CWT per wavelet scale
    d   | scale   unit    pc      t      (4, 3,   1,   2)  PCA per wavelet scale
    e   | scale   unit-i  unit-j         (4, 3,   3     )  SPCA per wavelet scale
    f   | unit-i  unit-j                 (3, 3          )  distance matrix (out)
    ----+---------------------------------------------------------------------
    Notes
        A unit is a multivariate time series composed of variables
        A variable is an univariate time series

    TODO
        Optimization
            https://stackoverflow.com/questions/40682041/multidimensional-matrix-multiplication-in-python
            https://pypi.org/project/fastdtw/
    """

    if is_verbose: print(f'a: {a.shape}')

    ts_weights = ts_weights or [1] * a.shape[1]

    # (1) Untransformed data based methods:
    # if method == 'euc':
    #     return tslearn.clustering.TimeSeriesKMeans(np.transpose(a, (0,2,1)), n_jobs=n_jobs)

    if metric in ['braycurtis', 'canberra', 'chebyshev', 'cityblock', 'correlation', 'cosine', 'dice', 'euclidean', 'hamming', 'jaccard', 'jensenshannon', 'kulsinski', 'mahalanobis', 'matching', 'minkowski', 'rogerstanimoto', 'russellrao', 'seuclidean', 'sokalmichener', 'sokalsneath', 'sqeuclidean', 'yule']:
        return pdist(a, metric, is_condensed=False)

    if metric == 'dtw':
        return tslearn.metrics.cdist_dtw(np.transpose(a, (0,2,1)), n_jobs=n_jobs)

    if metric == 'softdtw':
        return tslearn.metrics.cdist_soft_dtw_normalized(np.transpose(a, (0,2,1)))

    # (2) PCA based methods (non-CWT):
    if metric == 'spca':
        return 1 - np.clip(cdist_spca(pca(standardize_ts(a), min(n_components, a.shape[1]))),0,1)

    # (3) CWT based methods (non-PCA):
    if scales is None:
        scales = np.arange(1, a.shape[len(a.shape) - 1])
    elif not isinstance(scales, Iterable):
        scales = np.arange(1, scales)

    if isinstance(wavelet, str):
        b = pywt.cwt(a, scales, wavelet)[0]
        if is_verbose: print(f'b: {b.shape}')
        c = np.transpose(b, axes=(0,1,3,2))
    elif callable(wavelet):
        b = np.array([[scipy.signal.cwt(a[i,j], wavelet, scales) for j in range(a.shape[1])] for i in range(a.shape[0])])  # i-observation, j-variable
        if is_verbose: print(f'b: {b.shape}')
        c = np.transpose(b, axes=(2,0,3,1))
    else:
        raise ValueError("Invalid type of the 'wavelet' argument.")
    if is_verbose: print(f'c: {c.shape}')

    if metric == 'cwt-dtw':
        c_dtw = np.array([tslearn.metrics.cdist_dtw(c[i], c[i], n_jobs=n_jobs) for i in range(c.shape[0])])
        return np.average(c_dtw, axis=0, weights=scales_w)

    if metric == 'cwt-soft-dtw':
        c_dtw = np.array([tslearn.metrics.cdist_soft_dtw_normalized(c[i], c[i]) for i in range(c.shape[0])])
        return np.average(c_dtw, axis=0, weights=scales_w)

    # (4) CWT + PCA based methods:
    d = pca_wt(c, min(n_components, b.shape[1]))
    if is_verbose: print(f'd: {d.shape}')

    if metric == 'cwt-spca':
        e = np.array([cdist_spca(d[scale]) for scale in range(d.shape[0])])
    elif metric == 'cwt-pca-fro':
        e = np.array([[[scipy.spatial.distance.euclidean(d[scale,ts_i].flatten(), d[scale,ts_j].flatten()) if ts_i != ts_j else 1.0 for ts_j in range(d.shape[1])] for ts_i in range(d.shape[1])] for scale in range(d.shape[0])])
    e = np.clip(e,0,1)  # correct rounding errors
    if is_verbose: print(f'e: {e.shape}')

    f = np.average(e, axis=0, weights=scales_w)

    return (1-f) if not ret_all else (b,c,d,e,1-f)


# ----[ Clustering: With k ]--------------------------------------------------------------------------------------------

def cluster_kmeans(a, k, metric='euclidean', **kwargs):  # euclidean,dtw,softdtw
    return tslearn.clustering.TimeSeriesKMeans(k, metric=metric, **kwargs).fit(np.transpose(a, (0,2,1)))

def cluster_kmedoids(a, k, metric='euclidean', init='k-medoids++', **kwargs):
    """Cluster centers won't be available because a distance matrix is the input."""

    model = sklearn_extra.cluster.KMedoids(k, metric, init=init, **kwargs)
    if metric == 'precomputed':
        model.fit(a)
    else:
        model.fit(a.reshape((a.shape[0], -1)))
    return model

def cluster_kshape(a, k, **kwargs):
    """Cross-correlation based.

    https://tslearn.readthedocs.io/en/stable/gen_modules/clustering/tslearn.clustering.KShape.html
    """

    return tslearn.clustering.KShape(n_clusters=k, **kwargs).fit(a)

def cluster_spectral(dm, k, **kwargs):
    return sklearn.cluster.SpectralClustering(n_clusters=k, affinity='precomputed', **kwargs).fit(dm)

def cluster_agglo(dm, k, linkage='single', **kwargs):
    return sklearn.cluster.AgglomerativeClustering(n_clusters=k, affinity='precomputed', linkage=linkage, compute_distances=True, **kwargs).fit(dm)


# ----[ Clustering: Without k ]-----------------------------------------------------------------------------------------

def cluster_dbscan(dm, eps=0.02, min_samples=1, **kwargs):
    return sklearn.cluster.DBSCAN(metric='precomputed', eps=eps, min_samples=min_samples, **kwargs).fit(dm)

def cluster_optics(dm, min_samples=1, **kwargs):
    return sklearn.cluster.OPTICS(metric='precomputed', min_samples=min_samples, **kwargs).fit(dm)

def cluster_aff_prop(dm, **kwargs):
    return sklearn.cluster.AffinityPropagation(affinity='precomputed').fit(dm)
