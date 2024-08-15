#!/usr/bin/env python
"""
Script to visualise spatiotemporal evolution of activity

Based on the following modules:
Load npx data (load_npx.py) as module SpikeStruct with following variables:
     - st                spike times
     - spikeTemplates    template identity for each spike
     - clu               clusted identity for each spike
     - tempScalingAmp    scaling of template used to fit trace by KS2
     - cgs               cluster lables (good=2, mua=1,unsorted=3); noise
                         is excluded
     - cids              clusted ids used to retrieve lable from cgs (e.g.
                         cgs[cids==1000] returns label of cluster 1000)
     - nclu              number of clusters
     - cidsSorted        clusted ids sorted by depth (output of
                         "process_spiketimes")
     - depth             depth clusters
     - sodtedDepth       sorted depth clusters (output of "process_spiketimes")
     - xcoords           channel x position
     - ycoords           channel y position
     - temps             template shape (commented)
     - winv              inverse whitening matrix used by KS2 (commented)
     - pcFeat            spike scores; the channels that those features came
                         from are specified in pc_features_ind.npy.; e.g.
                         the value at pc_features[123, 1, 5] is the projection
                         of the 123rd spike onto the 1st PC on the channel
                         given by pc_feature_ind[5]
     - pcFeatInd         matrix specifying which pcFeatures are included in
                         the pc_features matrix

This module performes three types of analysis:
1) Spatiotemporal visualisation of population spike counts for one recording:
plot population spike count for every section of the probe (given a spatial bin
widht sb_width) for different time intervals (given by time time windows
start tw_start and width tw_width); position bin in μm from tip of probe; also
plot cluster distribution (number of clusters detected for each section of the
probe).
2) Temporal visualisation of cortical population spike count for all
recordings: plot total population spike count for each recording across time
(specify time bins with tw_start and tw_width); select section of probe in
cerebellar cortex (specify cortex_depth); scale spike counts in each time bin
by spike count in baseline (if tw_width=5, use -5 baseline); in addition, fit
model of spike counts to data.
3) Temporal visualisation of cortical cluster spike count for all recordings:
compute spike count for each cluster, scaled by spike count in baseline (default
-5 min); by default select only good clusters (can be mua), and select those whose
firing rate does not drop before 1 Hz (try alos <1 spike per time bin, or <0.5 Hz);
plot (absolute) spike count difference between selected time windows and (-5 min)
baseline; alternatively, compute scaled firing rate (by baseline); fit model
of scaled firing rates (gamma function).

If using elpy python shell, add cwd with the following
"""
import os
import sys

sys.path.append(os.path.join(os.getcwd(), 'data_analysis'))

import pdb
import itertools
import load_npx as lnpx
import pandas as pd
import seaborn as sns
import numpy as np
from matplotlib import pyplot as plt
from meta_data import load_meta


def spk_train_nw(npx_data, *args, binl=1, surr=False, **kwarg):
    """ Compute spike train of clusters sorted by depth; not for
    whisker allignment as in other modules; binl (defaul 1) is
    in ms. Returns binned spike train and length of train.

    Attention 1: now solved double-spike count by deleting spikes too close,
    but in rare cases possibly deleting good spikes when many consecutive
    spikes have diff<1.

    Attention 2: spk_hist has possibly variable bin length, spk_data has always
    bin 1 ms long;
    Note: use **kwarg to select good clusters (=2); use *args to select
    specific clusters.
    """
    spktr_sorted = []
    spk_times = np.round(npx_data.st * 1000)  # ms
    end_time = np.ceil(spk_times[-1]).astype(int)
    edges = np.arange(0, end_time + binl, binl)  # + binl include last frame

    if bool(kwarg):
        # if kwarg set to noise, take both good and mua units
        if kwarg['cgs'] != 0:
            cids_sorted = npx_data.cidsSorted[npx_data.cgsSorted ==
                                              kwarg['cgs']]
            npx_data.sortedDepth = npx_data.sortedDepth[npx_data.cgsSorted == kwarg['cgs']]
        else:
            cids_sorted = npx_data.cidsSorted[
                npx_data.cgsSorted != kwarg['cgs']]
    else:
        cids_sorted = npx_data.cidsSorted

    if bool(args):
        cids_sorted = cids_sorted.iloc[list(args[0])]

    # Get spike array (sorted by spatial location)
    for cl in range(len(cids_sorted)):
        spk_data = spk_times[npx_data.clu == cids_sorted.iloc[cl]]
        # force >=1 ms refractory period (delete double-counted spikes)
        spk_data = spk_data[:-1][np.diff(spk_data) >= 1]

        if surr == 'shuffle':
            spk_data = sample(range(int(end_time)), spk_data.size)

        if surr == 'ditter':
            spk_data = np.array(
                [randint(t - 5000, t + 5000) for t in spk_data])
        spk_hist, __ = np.histogram(spk_data, bins=edges)
        spktr_sorted.append(spk_hist)

    return spktr_sorted


def spkcount_cortex(spktr_sorted,
                    sortedDepth,
                    cortex_depth,
                    tw_start=[1, 13, 25, 37, 48],
                    tw_width=10,
                    tw_scale=2):
    """ Compute spike count within cerebellar cortex for each
    time window; input is spatially sorted spike trains (output of
    spk_train_nw function); sortedDepth come from npx_data; cortex_depth
    specifies which channels are to be considered cerebellar cortex and
    is distance from tip of the probe.
    Attention: used only when provided multiple datasets
    """
    # Select clusters within cortex (e.g. >=1000μm from probe tip)...
    ctx_start = np.sum(sortedDepth < cortex_depth)
    spktr_sorted = np.array(spktr_sorted[ctx_start:])
    print(np.sum(~(sortedDepth < cortex_depth)))

    # Remove trailing time interval if necessary
    spktr_sorted = spktr_sorted[:, :len(tw_start)]

    # Don't use if plotting 'Row'
    # # If cgs=2, discard clu with <0.5 Hz in tw_scale
    # mask = ~(spktr_sorted[:, int(tw_scale)] < 150)
    # # mask = ~(spktr_sorted[:, tw_scale] < 30).any(axis=1)
    # spktr_sorted = spktr_sorted[mask, :]
    # # cidsS = cidsSorted[mask]

    # ... and compress (rename in tw_spkcnt)
    tw_spkcnt = spktr_sorted.sum(axis=0).tolist()

    return tw_spkcnt


def plot_tanalysis(npxpath,
                   aid,
                   cortex_depth,
                   t_drop,
                   conditions,
                   tw_start=[1, 13, 25, 37, 48],
                   tw_width=10,
                   cgs=2,
                   plot_pop='yes'):
    """
    Plot distribution of spike counts across datasets for each time window;
    input is list of directories to datasets, depth of recording above
    which clusters are considered, time of CNO dropping time windows start
    and length; plot both original and scaled data by baseline (last time
    interval before drop).
    Attention 1: only used when providing multiple datasets.
    Attention 2: because some datasets have different t_drop, I need to expand
    the data adding nan, then group data belonging to same time interval
    excluding nan (necessary to use boxplot function).
    Attention 3:  only for tw_width=5, compute absolute spike counts difference
    between baseline (both -10 and -5 min) and rest of time intervals (included
    other baseline).
    """
    # Sanity check paths, depths and t_drops
    if (len(npxpath) != len(cortex_depth)) or (len(npxpath) != len(t_drop)):
        print('Attention: data length is different')
        exit()

    # Time interval at drop and reference tw (for scaling)
    drop_int = min(t_drop) // tw_width
    tw_scale = drop_int - 0

    ctx_spkcnt = []
    pdb.set_trace()
    for idx, dataset in enumerate(npxpath):
        # Get npx data
        spk = lnpx.loadKsDir(dataset)
        spk = lnpx.process_spiketimes(spk)
        print(dataset)

        # Compute spike trains for each cluster (sorted by depth)
        spktr_sorted = spk_train_nw(spk, binl=tw_width * 1000 * 60, surr=False, cgs=cgs)
        # Count cortical spikes for every time window
        ctx_d = cortex_depth[idx]
        tw_spkcnt = spkcount_cortex(spktr_sorted,
                                    spk.sortedDepth,
                                    cortex_depth=ctx_d,
                                    tw_start=tw_start,
                                    tw_width=tw_width,
                                    tw_scale=tw_scale,
                                    )

        # Expand data with nan if t_drop different across datasets
        if t_drop.count(t_drop[0]) != len(t_drop):
            if t_drop[idx] == 10:
                for i in range(10 // tw_width):
                    tw_spkcnt.insert(0, np.nan)
            elif t_drop[idx] == 20:
                for i in range(10 // tw_width):
                    tw_spkcnt.append(np.nan)

        ctx_spkcnt.append(tw_spkcnt)

    # Dataframe for cortex spike count (csc) - remove incomplete time intervals
    df_csc = pd.DataFrame(ctx_spkcnt)
    df_csc = df_csc.dropna(axis=1)

    # Rename colums with time
    df_csc.columns = np.arange(df_csc.columns.size) * tw_width - min(t_drop)

    # Scale by baseline (time interval just before drop)
    df_csc_scl = df_csc.div(df_csc.iloc[:, tw_scale], axis=0)

    # Add conditions to each dataset (used for fit and hues)
    df_csc['condition'] = conditions
    df_csc['aid'] = aid
    df_csc['idx_rec'] = df_csc.index
    df_csc_scl['aid'] = aid
    df_csc_scl['condition'] = conditions
    df_csc_scl['idx_rec'] = df_csc.index

    # Melt datagrame
    df_csc_m = df_csc.melt(id_vars=['condition', 'aid','idx_rec'])
    df_csc_scl_m = df_csc_scl.melt(id_vars=['condition', 'aid', 'idx_rec'])

    # Rename variable to time and convert to int
    df_csc_m.rename(columns={'variable':'time'}, inplace=True)
    df_csc_scl_m.rename(columns={'variable':'time'}, inplace=True)

    df_csc_m.time = df_csc_m.time.astype(int)
    df_csc_scl_m.time = df_csc_scl_m.time.astype(int)

    # Plot data
    # Set style
    pdb.set_trace()
    # sns.set(style='darkgrid')
    sns.set_context("poster", font_scale=1)
    sns.color_palette("deep")
    # sns.set(rc={'axes.facecolor':'lavenderblush'})
    plt.rcParams['axes.facecolor'] = 'lavenderblush'
    plt.rcParams["figure.edgecolor"] = 'black'
    figsize = (14, 12)
    size = 'x-large'
    style = 'italic'


    if plot_pop == 'mus':
        # Lineplot
        fig1, ax1 = plt.subplots(figsize=figsize)
        sns.lineplot(x=df_csc_scl_m.time,
                     y=df_csc_scl_m.value,
                     hue=df_csc_scl_m.aid,
                     palette="flare",
                     ax=ax1)

        fig1.suptitle('Population spike count', style=style)
        plt.ylabel('scaled spike count')
        plt.xlabel('Time from muscimol drop (min)')

        if os.path.basename(os.getcwd()) != 'data_analysis':
            os.chdir('./data_analysis')

        fig1.savefig(os.getcwd() + f'/images/spkcnt/Muscimol/pop_spkcount_dynamics.svg', format='svg')
    else:
        # Boxplot
        fig1, ax1 = plt.subplots(figsize=(12, 12))
        sns.boxplot(x=df_csc_m.time,
                    y=df_csc_m.value,
                    hue=df_csc_m.condition,
                    ax=ax1)
        sns.swarmplot(x=df_csc_m.time,
                      y=df_csc_m.value,
                      hue=df_csc_m.condition,
                      dodge=True,
                      palette='pastel',
                      ax=ax1)
        fig1.suptitle('Population spike count')
        plt.ylabel('spike count')
        plt.xlabel('Time from drop (min)')
        xticks_labels = (ax1.get_xticks() - drop_int) * tw_width
        ax1.set_xticklabels(xticks_labels)
        handles, labels = ax1.get_legend_handles_labels()
        ax1.legend(handles[:3], labels[:3], loc='upper right')

        # Boxplot scaled value
        fig3, ax3 = plt.subplots(figsize=(12, 12))
        sns.boxplot(x=df_csc_scl_m.time,
                    y=df_csc_scl_m.value,
                    hue=df_csc_scl_m.condition,
                    ax=ax3)
        sns.stripplot(x=df_csc_scl_m.time,
                      y=df_csc_scl_m.value,
                      hue=df_csc_scl_m.condition,
                      dodge=True,
                      palette='pastel',
                      ax=ax3)

        fig3.suptitle('Population spike count')
        plt.ylabel('scaled spike count')
        plt.xlabel('Time from drop (min)')
        ax3.set_xticklabels(xticks_labels)
        handles, labels = ax3.get_legend_handles_labels()
        ax3.legend(handles[:3], labels[:3], loc='upper right')

        return df_csc_m, df_csc_scl_m


def clu_absdiff(spktr_sorted, sortedDepth, cidsSorted, cortex_depth, tw_int, tw_start=np.arange(0, 60, 5)):
    """
    Compute for each cluster within one dataset absolute difference
    of spike counts between two baselines and between baseline and after-drop
    time interval; tw_width is 5 min to allow two baselines pre-drop;
    time windows of interest (tw_int) are time intervals starting at -10, -5,
    15 and 30 min from CNO/PBS drop; drop clusters with <0.5 Hz fr during any
    tw_int; reference time interval is tw_int[1] (5 min before t_drop);
    also return firing rates (spike count/ 5 min is sec).
    """
    # Select clusters within cortex (e.g. >=1000μm from probe tip)
    ctx_start = np.sum(sortedDepth < cortex_depth)
    spktr_sorted = np.array(spktr_sorted[ctx_start:])
    cidsSorted = np.array(cidsSorted[ctx_start:])

    # Remove trailing time interval if necessary
    spktr_sorted = spktr_sorted[:, :len(tw_start)]

    # Discard clu with <0.5 Hz in tw_int (<150 5 min)
    mask = ~(spktr_sorted[:, tw_int] < 150).any(axis=1)
    spktr_sorted = spktr_sorted[mask, :]
    cidsS = cidsSorted[mask]
    print(len(spktr_sorted))

    # Dataframe clusters firing rate in each 5 min bin
    df_fr = pd.DataFrame(spktr_sorted).iloc[:, tw_int]
    df_fr.columns = ['-10', '-5', '0', '15', '20', '25']
    # df_fr.columns = ['-10', '-5', '0', '5', '10', '15', '20', '25', '30', '35']

    # Scale spikecount by baseline (time of dropping)
    spktr_sorted = spktr_sorted / np.expand_dims(spktr_sorted[:, tw_int[2]], axis=1)

    # Compute abs differences
    adiff_pre = abs(spktr_sorted[:, tw_int[1]] - spktr_sorted[:, tw_int[0]])
    adiff_post_0 = abs(spktr_sorted[:, tw_int[1]] - spktr_sorted[:, tw_int[2]])
    adiff_post_1 = abs(spktr_sorted[:, tw_int[1]] - spktr_sorted[:, tw_int[3]])
    adiff_post_2 = abs(spktr_sorted[:, tw_int[1]] - spktr_sorted[:, tw_int[4]])
    adiff_post_3 = abs(spktr_sorted[:, tw_int[1]] - spktr_sorted[:, tw_int[5]])

    # Compute differences (No Abs) ATTENTION
    diff_pre = spktr_sorted[:, tw_int[0]] - spktr_sorted[:, tw_int[1]]
    diff_post_0 = spktr_sorted[:, tw_int[2]] - spktr_sorted[:, tw_int[1]]
    diff_post_1 = spktr_sorted[:, tw_int[3]] - spktr_sorted[:, tw_int[1]]
    diff_post_2 = spktr_sorted[:, tw_int[4]] - spktr_sorted[:, tw_int[1]]
    diff_post_3 = spktr_sorted[:, tw_int[5]] - spktr_sorted[:, tw_int[1]]


    # Organise in dataframe
    adiff = pd.DataFrame(np.array([adiff_pre, adiff_post_0, adiff_post_1, adiff_post_2, adiff_post_3]).T, columns=['baseline', '0tw', 'epost', 'mpost', 'lpost'])
    diff = pd.DataFrame(np.array([diff_pre, diff_post_0, diff_post_1, diff_post_2, diff_post_3]).T, columns=['baseline', '0tw', 'epost', 'mpost', 'lpost'])

    return adiff, diff, df_fr, cidsS


def plot_cluanalysis(npxpath,
                     aid,
                     cortex_depth,
                     t_drop,
                     conditions,
                     tw_start=np.arange(0, 60, 5),
                     tw_width=5,
                     cgs=2):
    """
    Plot distribution across clusters of absolute spike count difference
    between two baseline time intervals and between baseline and after-drop
    time intervals; tw_width must be 5 to allow two baselines pre-drop; ;
    default use only good units (cgs=2); remove units which do not fire
    during time intervals of interest.
    Attention: used binl=tw_width * 1000 * 60 for spk_train_nw to get
    spike count over 5 min intervals.
    """
    # Sanity check paths, depths and t_drops
    if (len(npxpath) != len(cortex_depth)) or (len(npxpath) != len(t_drop)):
        print('Attention: data length is different')
        exit()

    # Initialise abs difference and fr dataframes
    df_adiff = pd.DataFrame()
    df_diff = pd.DataFrame()
    df_fr = pd.DataFrame()
    df_fr_scl = pd.DataFrame()

    # Get npx data
    for idx, dataset in enumerate(npxpath):
        spk = lnpx.loadKsDir(dataset)
        spk = lnpx.process_spiketimes(spk)

        # Time window of interest
        if t_drop[idx] == 10:
            tw_int = np.array([0, 1, 2, 5, 6, 7])
        elif t_drop[idx] == 20:
            tw_int = np.array([2, 3, 4, 7, 8, 9])
        # if t_drop == 10:
        #     tw_int = np.arange(10)
        # elif t_drop == 20:
        #     tw_int = np.arange(2, 12)

        # Compute spike trains for each cluster (sorted by depth)
        spktr_sorted, __, __, __ = spk_train_nw(spk, binl=tw_width * 1000 * 60, surr=False, cgs=cgs)

        # Compute spkcnt abs difference for each cluster
        adiff, diff, fr, cidsS = clu_absdiff(spktr_sorted, spk.sortedDepth, spk.cidsSorted[spk.cgsSorted == cgs], cortex_depth[idx], tw_int, tw_start=tw_start)

        # Scale fr by baseline before drop
        fr_scl = fr.div(fr['0'], axis=0)

        # Add description
        adiff['condition'] = conditions[idx]
        adiff['aid'] = aid[idx]
        adiff['idx_rec'] = idx

        diff['condition'] = conditions[idx]
        diff['aid'] = aid[idx]
        diff['idx_rec'] = idx

        fr['condition'] = conditions[idx]
        fr['aid'] = aid[idx]
        fr['idx_rec'] = idx

        fr_scl['condition'] = conditions[idx]
        fr_scl['aid'] = aid[idx]
        fr_scl['idx_rec'] = idx

        # Concatenate recordings
        df_adiff = pd.concat([df_adiff, adiff])
        df_diff = pd.concat([df_diff, diff])
        df_fr = pd.concat([df_fr, fr])
        df_fr_scl = pd.concat([df_fr_scl, fr_scl])

    # Melt df (long format)
    df_adiff_m = df_adiff.melt(id_vars=['condition', 'aid', 'idx_rec'])
    df_diff_m = df_diff.melt(id_vars=['condition', 'aid', 'idx_rec'])
    df_fr_m = df_fr.melt(id_vars=['condition', 'aid', 'idx_rec'])
    df_fr_scl_m = df_fr_scl.melt(id_vars=['condition', 'aid', 'idx_rec'])

    # Rename variable to time and convert to int
    df_fr_m.rename(columns={'variable':'time'}, inplace=True)
    df_fr_scl_m.rename(columns={'variable':'time'}, inplace=True)

    df_fr_m.time = df_fr_m.time.astype(int)
    df_fr_scl_m.time = df_fr_scl_m.time.astype(int)

    # # Save df as csv_file
    # df_fr_m.to_csv(os.path.join(os.getcwd(), 'data_analysis/df_spkcnt_clu_1hz'), index=False)

    # Set style
    sns.set(style='whitegrid')

    # Plot
    # Boxplot differences
    fig1, ax1 = plt.subplots(figsize=(12, 12))
    sns.boxplot(x=df_adiff_m.variable, y=df_adiff_m.value, hue=df_adiff_m.condition, ax=ax1)
    sns.stripplot(x=df_adiff_m.variable, y=df_adiff_m.value,hue=df_adiff_m.condition, dodge=True, ax=ax1)
    handles, labels = ax1.get_legend_handles_labels()
    ax1.legend(handles[3:], labels[3:])
    # ax1.set_ylim([0, 100])
    fig1.suptitle('Absolute spike count difference')

    # Boxplot firing rates
    fig2, ax2 = plt.subplots(figsize=(12, 12))
    sns.boxplot(x=df_fr_m.time, y=df_fr_m.value, hue=df_fr_m.condition, ax=ax2)
    sns.stripplot(x=df_fr_m.time, y=df_fr_m.value, hue=df_fr_m.condition, dodge=True, ax=ax2)
    handles, labels = ax2.get_legend_handles_labels()
    ax2.legend(handles[3:], labels[3:])
    fig2.suptitle('Firing rate clusters')

    # Boxplot scaled firing rates
    fig3, ax3 = plt.subplots(figsize=(12, 12))
    sns.boxplot(x=df_fr_scl_m.time, y=df_fr_scl_m.value, hue=df_fr_scl_m.condition, ax=ax3)
    sns.stripplot(x=df_fr_scl_m.time, y=df_fr_scl_m.value, hue=df_fr_scl_m.condition, dodge=True, ax=ax3)
    handles, labels = ax3.get_legend_handles_labels()
    ax3.legend(handles[3:], labels[3:])
    fig3.suptitle('Scaled firing rate clusters')


if __name__ == '__main__':
    """ Run script if spkcount_analysis.py module is main programme; if
    cgs != 0 take only mua or good units, if cgs == 0 take both mua and
    good units; select which analysis to run by defining data path: if
    path to one dataset use string for directory and plot st_analysis,
    otherwise use list and plot t_analysis.
    """
    
#     # Path to Muscimol data
    npxpath_mus = ['/media/bunaken/Ensor/npx/Muscimol/EP_PCP_220204_Muscimol_noC\
am_g0/catgt_EP_PCP_220204_Muscimol_noCam_g0/kilosort3/',
                   '/media/bunaken/Ensor/npx/Muscimol/EP_PCP_220209_Muscimol_noC\
am_g0/catgt_EP_PCP_220209_Muscimol_noCam_g0//kilosort3/',
                   '/media/bunaken/Ensor/npx/Muscimol/EP_PCP_220210_Muscimol_noC\
am_g0/catgt_EP_PCP_220210_Muscimol_noCam_g0/kilosort3/']

#     # Path CNO data: glyt2 CNO (GC), wild type CNO (WC), glyt2 PBS (GP)
#     npxpath_gCNO = ['/media/bunaken/Ensor/npx/CNO/Drop/EP_GLYT2_220316_CNO_Drop_A_g0/\
# catgt_EP_GLYT2_220316_CNO_Drop_A_g0',
#                     '/media/bunaken/Ensor/npx/CNO/Drop/EP_GLYT2_220316_CNO_Drop_B_g0/\
# catgt_EP_GLYT2_220316_CNO_Drop_B_g0',
#                     '/media/bunaken/Ensor/npx/CNO/Drop/EP_GLYT2_220318_CNO_Drop_A_g0/\
# catgt_EP_GLYT2_220318_CNO_Drop_A_g0',
#                     '/media/bunaken/Ensor/npx/CNO/Drop/EP_GLYT2_220318_CNO_Drop_B_g0/\
# catgt_EP_GLYT2_220318_CNO_Drop_B_g0',
#                     '/media/bunaken/Ensor/npx/CNO/Drop/EP_GLYT2_220325_CNO_Drop_A_g0/\
# catgt_EP_GLYT2_220325_CNO_Drop_A_g0',
#                     '/media/bunaken/Ensor/npx/CNO/Drop/EP_GLYT2_220325_CNO_Drop_B_g0/\
# catgt_EP_GLYT2_220325_CNO_Drop_B_g0',
#                     '/media/bunaken/Ensor/npx/CNO/Drop/EP_GLYT2_220429_CNO_Drop_B_g0/\
# catgt_EP_GLYT2_220429_CNO_Drop_B_g0',
#                     '/media/bunaken/Ensor/npx/CNO/Drop/EP_GLYT2_220502_CNO_Drop_A_g0/\
# catgt_EP_GLYT2_220502_CNO_Drop_A_g0',
#                     '/media/bunaken/Ensor/npx/CNO/Drop/EP_GLYT2_220506_CNO_Drop_A_g0/\
# catgt_EP_GLYT2_220506_CNO_Drop_A_g0',
#                     '/media/bunaken/Ensor/npx/CNO/Drop/EP_GLYT2_220510_CNO_Drop_A_g0/\
# catgt_EP_GLYT2_220510_CNO_Drop_A_g0',
#                     '/media/bunaken/Ensor/npx/CNO/Drop/EP_GLYT2_220510_CNO_Drop_B_g0/\
# catgt_EP_GLYT2_220510_CNO_Drop_B_g0',
#                     '/media/bunaken/Ensor/npx/CNO/Drop/EP_GLYT2_220512_CNO_Drop_A_g0/\
# catgt_EP_GLYT2_220512_CNO_Drop_A_g0',
#                     '/media/bunaken/Ensor/npx/CNO/Drop/EP_GLYT2_220512_CNO_Drop_B_g0/\
# catgt_EP_GLYT2_220512_CNO_Drop_B_g0',
#                     '/media/bunaken/Ensor/npx/CNO/Drop/EP_GLYT2_220517_CNO_Drop_A_g0/\
# catgt_EP_GLYT2_220517_CNO_Drop_A_g0',
#                     '/media/bunaken/Ensor/npx/CNO/Drop/EP_GLYT2_220517_CNO_Drop_B_g0/\
# catgt_EP_GLYT2_220517_CNO_Drop_B_g0',
#                     '/media/bunaken/Ensor/npx/CNO/Drop/EP_GLYT2_220519_CNO_Drop_A_g0/\
# catgt_EP_GLYT2_220519_CNO_Drop_A_g0',
#                     '/media/bunaken/Ensor/npx/CNO/Drop/EP_GLYT2_220519_CNO_Drop_B_g0/\
# catgt_EP_GLYT2_220519_CNO_Drop_B_g0',
#                     '/media/bunaken/Ensor/npx/CNO/Drop/EP_GLYT2_220527_CNO_Drop_A_g0/\
# catgt_EP_GLYT2_220527_CNO_Drop_A_g0',
#                     '/media/bunaken/Ensor/npx/CNO/Drop/EP_GLYT2_220527_CNO_Drop_B_g0/\
# catgt_EP_GLYT2_220527_CNO_Drop_B_g0']
#     npxpath_wCNO = ['/media/bunaken/Ensor/npx/CNO/Drop/EP_WT_220303_CNO_Drop_g0/\
# catgt_EP_WT_220303_CNO_Drop_g0',
#                     '/media/bunaken/Ensor/npx/CNO/Drop/EP_WT_220422_CNO_Drop_A_g0/\
# catgt_EP_WT_220422_CNO_Drop_A_g0',
#                     '/media/bunaken/Ensor/npx/CNO/Drop/EP_WT_220422_CNO_Drop_B_g0/\
# catgt_EP_WT_220422_CNO_Drop_B_g0',
#                     '/media/bunaken/Ensor/npx/CNO/Drop/EP_WT_220425_CNO_Drop_A_g0/\
# catgt_EP_WT_220425_CNO_Drop_A_g0',
#                     '/media/bunaken/Ensor/npx/CNO/Drop/EP_WT_220425_CNO_Drop_B_g0/\
# catgt_EP_WT_220425_CNO_Drop_B_g0']
#     npxpath_aPBS = ['/media/bunaken/Ensor/npx/CNO/Drop/EP_GLYT2_220429_PBS_Drop_A_g0/\
# catgt_EP_GLYT2_220429_PBS_Drop_A_g0',
#                     '/media/bunaken/Ensor/npx/CNO/Drop/EP_GLYT2_220502_PBS_Drop_B_g0/\
# catgt_EP_GLYT2_220502_PBS_Drop_B_g0',
#                     '/media/bunaken/Ensor/npx/CNO/Drop/EP_GLYT2_220506_PBS_Drop_B_g0/\
# catgt_EP_GLYT2_220506_PBS_Drop_B_g0',
#                     '/media/bunaken/Ensor/npx/WT/EP_WT_220602_A_g0/\
# catgt_EP_WT_220602_A_g0',
#                     '/media/bunaken/Ensor/npx/WT/EP_WT_220602_B_g0/\
# catgt_EP_WT_220602_B_g0',
#                     '/media/bunaken/Ensor/npx/WT/EP_WT_220607_A_g0/\
# catgt_EP_WT_220607_A_g0',
#                     '/media/bunaken/Ensor/npx/WT/EP_WT_220607_B_g0/\
# catgt_EP_WT_220607_B_g0',
#                     '/media/bunaken/Ensor/npx/WT/EP_WT_220609_A_g0/\
# catgt_EP_WT_220609_A_g0',
#                     '/media/bunaken/Ensor/npx/WT/EP_WT_220609_B_g0/\
# catgt_EP_WT_220609_B_g0']

#     # Animal id
    aid_mus = [0, 1, 2]
#     aid_gCNO = [0, 0, 1, 1, 2, 2, 3, 4, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10]
#     aid_wCNO = [11, 12, 12, 13, 13]
#     aid_aPBS = [3, 4, 5, 14, 14, 15, 15, 16, 16]

#     # Depth from which to consider cortex of interest
    cortex_depth_mus = [1000, 1160, 2000]
#     cortex_depth_gCNO = [580, 740, 620, 1040, 2600, 760, 1080, 880, 1000, 1000, 1000, 1000, 900, 1000, 1000, 1000, 1000, 840, 1000]
#     cortex_depth_wCNO = [1680, 880, 1100, 1180, 1140]
#     cortex_depth_aPBS = [1100, 1000, 1000, 700, 700, 500, 800, 1080, 1060]

#     # Time of CNO/PBS dropping
    t_drop_mus = [10, 10, 10]
#     t_drop_gCNO = [10, 10, 10, 10, 10, 10, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20]
#     t_drop_wCNO = [10, 20, 20, 20, 20]
#     t_drop_aPBS = [20, 20, 20, 20, 20, 20, 20, 20, 20]

#     # Select or concatenate paths, aid, cortex_depths and t_drops
#     npxpath = npxpath_gCNO + npxpath_wCNO + npxpath_aPBS
#     aid = aid_gCNO + aid_wCNO + aid_aPBS
#     cortex_depth = cortex_depth_gCNO + cortex_depth_wCNO + t_drop_aPBS
#     t_drop = t_drop_gCNO + t_drop_wCNO + t_drop_aPBS

#     # Lable condition of each recording (compulsory)
    condition_mus = np.repeat('muscimol', len(npxpath_mus))
#     conditions = np.concatenate(
#         (np.repeat('gCNO', len(npxpath_gCNO)), np.repeat('wCNO', len(npxpath_wCNO)),
#          np.repeat('aPBS', len(npxpath_aPBS))))

    # Info time windows
    tw_width = 1
    tw_start = np.arange(0, 60, tw_width)

    # Load muscimol or CNO data
    mus = True

    # Load metadata
    if mus:
        cgs = 0
        npxpath, aid, cortex_depth, t_drop, clu_idx, pethdiscard_idx, conditions = load_meta(mus=True)
        plot_pop = 'mus'
    else:
        cgs = 2
        _, npxpath, aid, cortex_depth, t_drop, clu_idx, pethdiscard_idx, conditions = load_meta()
        # Choose to plot pop or single cluster data
        plot_pop = 'yes'

    # Check path existence
    for path in npxpath:
        print(os.path.exists(path))

    if (plot_pop == 'yes') or (plot_pop == 'mus'):
        plot_tanalysis(npxpath,
                       aid,
                       cortex_depth,
                       t_drop,
                       conditions,
                       tw_start=tw_start,
                       tw_width=tw_width,
                       cgs=cgs,
                       plot_pop=plot_pop)
    else:
        plot_cluanalysis(npxpath,
                         aid,
                         cortex_depth,
                         t_drop,
                         conditions,
                         tw_start=tw_start,
                         tw_width=tw_width,
                         cgs=cgs)

    plt.show()
