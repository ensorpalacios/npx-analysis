#!/usr/bin/env python
"""
Script to plot example whisker (wsk) and spike (spk) trace of selected clusters
from choosen recording. Contains the following:

- plot_wskspk: loads metadata, retrieve whisker and spike data through
align_fun, then plot clusters of interest and example of whisking bout;
"""

# Ensure /data_analysis is in path
import os
import sys
if not ('data_analysis' in os.getcwd()):
    sys.path.append(os.path.join(os.getcwd(), 'data_analysis'))

# Import modules
import pdb
import itertools
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from random import sample, randint
from meta_data import load_meta
from align_fun import align_spktrain


def plot_wskspk(rec_idx, whisker='whisk2', t_wind=[-2, 3], var='angle', cgs=2, surr=False, pre=None, discard=False):
    """ Function for plotting spikes from selected clusters clu_idx in
    choosen recording rec_idx; additionally specify whisking variable of
    interest, time window around whisking bout start to consider, cluster
    group (default 2==good), whether to use surrogate data, and filter clusters
    by min fr in each bin (don't use here: binl too small).
    - Note 1: rec_idx can be 0-18 gCNO, 19-23 wCNO, 24-32
    - Note 2: wsknpx_data contains: whiskd, spktr_sorted, cids_sorted, a_wvar,
    a_wvar_m, a_spktr, a_spktr_m.

    """
    # load metadata
    wskpath, npxpath, aid, cortex_depth, t_drop, clu_idx, pethdiscard_idx, bout_idxs, wmask, conditions, mli_depth = load_meta()

    # Load Data specific for rec_idx: 0-18 gCNO, 19-23 wCNO, 24-32 PBS
    wskpath = wskpath[rec_idx]
    npxpath = npxpath[rec_idx]
    aid = aid[rec_idx]
    cortex_depth = cortex_depth[rec_idx]
    t_drop = t_drop[rec_idx]
    conditions = conditions[rec_idx]
    clu_idx = clu_idx[rec_idx]
    bout_idxs = bout_idxs[rec_idx]
    clu_idx = []
    # wsknpx_data contains: whiskd, spktr_sorted, cids_sorted, a_wvar, a_wvar_m, a_spktr, a_spktr_m
    wsknpx_data = align_spktrain(wskpath, npxpath, cortex_depth, cgs=cgs, t_wind=t_wind, var=var, surr=surr, t_drop=t_drop, pre=pre, clu_idx=clu_idx, discard=discard)
    clu_idx = wsknpx_data.cids_sorted
    # pdb.set_trace()
    # Plot
    # Initialise parameters for figures
    sr = wsknpx_data.whiskd.sr
    figsize = (14, 12)
    x_len = [0, np.diff(t_wind)[0] * sr]
    x_ticks = np.arange(0, np.diff(t_wind) * sr, sr)
    x_labels = np.round(np.arange(t_wind[0] * sr, t_wind[1] * sr, 299) / sr, 2)
    n_rows = len(clu_idx)
    color = 'lavenderblush'
    lcolor = 'darkgoldenrod'
    sns.set_context("poster", font_scale=1)
    sns.color_palette("deep")
    # plt.rcParams['axes.facecolor'] = 'lavenderblush'
    # plt.rcParams['axes.facecolor'] = 'white'
    plt.rcParams["figure.edgecolor"] = 'black'

    # deg = np.array([0.4, 0.4 + 10 / (180 / np.pi)])
    sec = np.array([100, 100 + 0.5 * 299])
    bout_idxs = np.arange(1, 40)
    # bout_idxs = [25]
    bout_idxs = [23]
    for bout_idx in bout_idxs:
        # Plot spikes and wsk var
        fig1, axs = plt.subplots(n_rows, 1, sharey=True, gridspec_kw={'height_ratios':  [20] + [1] * (n_rows - 1)}, figsize=figsize)
        # with sns.axes_style('dark'):
        for idx, cl in enumerate(clu_idx):
            if idx == 0:
                # Plot whisker activity
                wvar = wsknpx_data.a_wvar[whisker, bout_idx]
                # ax = sns.lineplot(data=wsknpx_data.a_wvar[whisker, bout_idx], linewidth=4, color=lcolor', ax=axs[idx])
                axs[idx].plot(wvar, linewidth=2, color=lcolor)
                # ax.set_ylabel(f'whisker {var}')
                axs[idx].set_xlim(x_len)
                # axs[idx].set_yticks([])
                axs[idx].set_xticks(x_ticks)
                axs[idx].set_xticklabels(x_labels)
                axs[idx].set_xticks([])
                axs[idx].set_ylim([wvar.min() - 0.1, wvar.max() + 0.1])
                axs[idx].set_ylabel(f'whisker {var}')
                # midpoint = (wsknpx_data.a_wvar[whisker, bout_idx].max() + wsknpx_data.a_wvar[whisker, bout_idx].min()) / 2
                # deg = np.array([midpoint, midpoint + 10 / (180 / np.pi)])
                # Convert rad to degrees
                deg = np.array([wvar.max() - 10 / (180 / np.pi), wvar.max()])
                axs[idx].vlines(sec[0], deg[0], deg[1], colors=lcolor)
                axs[idx].hlines(deg[0], sec[0], sec[1], colors=lcolor)
                axs[idx].text(sec[0], deg[1] + 0.01, '10 deg')
                axs[idx].text(sec.mean(), deg[0] + 0.01, '0.5 sec')
                axs[idx].set_facecolor(color)
                axs[idx].spines['bottom'].set_visible(False)
            else:
            # Plot spike activity
                cdix = np.where(wsknpx_data.cids_sorted == clu_idx[idx])[0][0]
                # __ = wsknpx_data.a_spktr[whisker, f'{cdix}', bout_idx]
                __ = wsknpx_data.a_spktr[whisker, cdix, bout_idx]
                axs[idx].set_xlim(x_len)
                axs[idx].eventplot(__.index[__ > 0].tolist(), linewidth=1, colors='black')
                # sns.despine(left=False, bottom=False, right=False)
                axs[idx].spines['top'].set_visible(False)
                # axs[idx].spines['bottom'].set_visible(False)
                axs[idx].set_yticks([])
                axs[idx].set_xticks([])
                if idx == n_rows // 2:
                    axs[idx].set_ylabel('clusters')
                if idx < n_rows - 1:
                    axs[idx].spines['bottom'].set_visible(False)
                # axs[idx].set_facecolor(color)

        left = 0.05
        # right = 0.95
        right = 0.90
        bottom = 0.05
        top = 0.9
        wspace = 0.0
        hspace = 0

        plt.subplots_adjust(left=left,
                            right=right,
                            bottom=bottom,
                            top=top,
                            wspace=wspace,
                            hspace=hspace)

        fig1.suptitle('Raster plot')

        if os.path.basename(os.getcwd()) != 'data_analysis':
            os.chdir('./data_analysis')

        if not surr:
            surr = 'nosurr'
        try:
            plt.savefig(os.getcwd() + f'/images/raster/{os.path.basename(wskpath[:-1])}/{os.path.basename(wskpath[:-1])}_{var}_{surr}_pred{pre}_bout{bout_idx}.svg', format='svg')
        except FileNotFoundError:
            print(f'created dir images/raster/{os.path.basename(wskpath[:-1])}')
            os.makedirs(os.getcwd() + f'/images/raster/{os.path.basename(wskpath[:-1])}')
            plt.savefig(os.getcwd() + f'/images/raster/{os.path.basename(wskpath[:-1])}/{os.path.basename(wskpath[:-1])}_{var}_{surr}_pred{pre}_bout{bout_idx}.svg', format='svg')


if __name__ == '__main__':
    """ Run script if spkcount_analysis.py module is main programme
    """
    # rec_idx = 8               # EP_GLYT2_220506_CNO_Drop_A
    # rec_idx = 22              # EP_WT_220425_CNO_Drop_A
    rec_idx = 27              # EP_WT_220602_A
    whisker = 'whisk2'                 # which whisker "" ""
    # t_wind = [-2, 3]            # t_wind around whisking bout start
    t_wind = [-5, 25]            # t_wind around whisking bout start
    # t_wind = [-10, 10]            # t_wind around whisking bout start
    var = 'angle'
    # var = 'setpoint'
    # var = 'amplitude'           # which whisker var "" ""
    cgs = 2                     # specify type of clusters to use (2=good, 1=mua, 0=bad)
    surr = False
    # surr = 'shuffle'
    pre = None
    # pre = False
    discard = False             # discard clu with fr < 0.1 Hz
    plot_wskspk(rec_idx, whisker=whisker, t_wind=t_wind, var=var, cgs=cgs, surr=surr, pre=pre, discard=discard)
