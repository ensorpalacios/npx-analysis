"""
Script to compute and plot tuning curves, that is, cluters' activity against
discretised whisker variable. Contains the following:

- tc_fun: function to calculate firing rate assotiaced to each discrete
wvar value; uses load_data, spike_train (from align_fun module) and
tbin_wvar functions;

- tbin_wvar: compute average z-scored wvar value within time bins of length
fxb * frame length in ms.

- plot_tc: get firing rate per wvar bin with tc_fun; plot results. Function
for data without manipulation.

- plot_tc_compare: get firing rate per wvar bin with tc_fun; plot results. Function
for data with manipulation (compare pre-post drug drop periods.)

- plot_tc_clusters: plot typical tc for clusters of tc (clusters got from k-means
clustering). Function for data without manipulation.

- plot_tc_clusters: plot typical tc for clusters of tc (clusters got from k-means
clustering). Function for data with manipulation (compare -pre-post drug drop periods).

- plot_tc_clusters_compare_pre: plot typical tc for clusters of tc (clusters got
from k-means clustering). Function for data with manipulation (compare -pre-post
drug drop periods). Attention: this always consider clustering based on -pre drop
data.

- info_transmitted: compute information transfer of the confusion matrix.

- plot_confusion: compute and plot confution matrix, namely, the matrix
condensing info about change of tc clusters identity by tc pre- to post-drop.

- entropy_tc_compare: compute entropy of all tuning curves for cluster (unit), for
pre- and post-drop data; compute difference between pre-post etropies, compute
skewness, compute kl-divergence between tc's pre and post drop (tc normalised to 1).

- fit_bspline: fit b-spline model to reduce noise in tc (assume that fr for adjacent
wsk positions are correlated).

- run_tc: run script for data without manipulation.

- run_tc_compare: run script for data with manipulation.

ATTENTION:
set pre or PrePost to run analysis on controls only (fig 1/2) GlyT2+CNO vs controls
(fig 4)
"""

# Ensure /data_analysis is in path
import os
import sys

if not ('data_analysis' in os.getcwd()):
    sys.path.append(os.path.join(os.getcwd(), 'data_analysis'))

# Import modules
import os
import itertools
import pdb
import align_fun
import pickle
import matplotlib.pyplot as plt
import matplotlib
import warnings
import seaborn as sns
import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
import aesara.tensor as at
import scipy.linalg as la


from scipy import stats
from meta_data import load_meta
from patsy import dmatrix
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from scipy.stats import entropy, skew
from scipy.optimize import curve_fit


def tbin_wvar(whiskdata,
              numBins,
              edges,
              whisker=2,
              fxb=15,
              whisking=True,
              var="angle",
              sr=299):
    """Take the average of the wvar (default angle) over time bins of length
    fxb * length frame in ms. Whisker data are z-scored. If whisking=True, the
    avg_var is only calculated during active whisking bouts.
    - Minor point: I manually correct for bins=edges whith 14 or 16 whisking
    points (due to numerical imprecision), although not necessary."""

    if var == 'angle':
        wvar = stats.zscore(whiskdata.angle[whisker])
        # wvar_ = whiskdata.angle[whisker]
    elif var == 'amplitude':
        wvar = stats.zscore(whiskdata.amplitude[whisker])
    elif var == 'setpoint':
        wvar = stats.zscore(whiskdata.setpoint[whisker])
    elif var == 'phase':
        wvar = whiskdata.phase[whisker]

    # Reshape whisker variable over fxb frames per bin (e.g. to average over
    # fxb=15); last frames of recording are chopped (same as edges for
    # np.hist(npx_recording,edges)); it's a view of whiskvar
    # pdb.set_trace()
    wvar_r = np.reshape(wvar[:(wvar.size // fxb) * fxb], (-1, fxb))
    # Check length npx and whisker binned data
    if not len(wvar_r) == numBins:
        print('Whisking and neural data mismatch length. ')
        # print(len(wvar_r), numBins)
        # wvar_r = wvar_r[:-1]

    # Average wvar per bin
    wvar_rm = np.mean(wvar_r, axis=1)

    # If interested in whisker attribute only during active whisking bouts:
    if whisking:
        flat_isw = [
            item / sr * 1000 for sublist in whiskdata.isw_ifrm[whisker]
            for item in sublist
        ]
        hist_isw, _ = np.histogram(flat_isw, bins=edges)
        # Compensate for numerical precision when comparing hist_isw and edges
        hist_isw[np.logical_or(hist_isw == fxb + 1, hist_isw == fxb - 1)] = fxb
        isw_bin = list(map(
            bool, hist_isw))  # Boolean iswhisking (yes/no) for each bin
        wvar_rm = wvar_rm[isw_bin]  # wvar_rm only during whisking
    else:
        isw_bin = []

    return wvar_rm, isw_bin  # default average position, nbins*1whisker


def tc_fun(rec_idx,
           fxb=15,
           whisking=True,
           cgs=2,
           var="angle",
           w_bin=12,
           surr=False,
           pre=None,
           pethdiscard=True):
    """ Compute clusters' firing rate tuning to discretised wvar; used
    np.digitize to retrieve index during wvar time series in which each
    wvar bin occur; use index to get firing rate associated to each wvar
    bin;  w_bin is initialised because the right edge is cancelled; use
    pre to select specific part of whisker and npx recordings: if
    pre==None, take full recordings, if pre is True, take recordings pre-
    -drop, if pre is False take recording post-drop.
    Note 1: align_fun.spike_train just returns spike train (not aligned)
    Note 2: if pre==False, take recording of length 0:t_drop and starting
    from t_drop + 5 minutes!
    """
    # load metadata
    wskpath, npxpath, aid, cortex_depth, t_drop, clu_idx, pethdiscard_idx, __, __, conditions, mli_depth = load_meta()

    # Load Data specific for rec_idx: 0-18 gCNO, 19-23 wCNO, 24-32
    wskpath = wskpath[rec_idx]
    npxpath = npxpath[rec_idx]
    aid = aid[rec_idx]
    cortex_depth = cortex_depth[rec_idx]
    t_drop = t_drop[rec_idx]
    conditions = conditions[rec_idx]
    clu_idx = clu_idx[rec_idx]
    pethdiscard_idx = pethdiscard_idx[rec_idx]

    # Initialise w_bin
    w_bin = w_bin + 1

    # Load and preprocess whisker and npx data; possible to get subset of
    # data: either pre or post CNO/PBS drop
    if pre is None:
        whiskd, spk = align_fun.load_data(wskpath, npxpath, t_drop=False)
    else:
        whiskd, spk = align_fun.load_data(wskpath, npxpath, t_drop=t_drop, pre=pre)

    # Get spike train binned to match frame length (0.001 ms precision)
    t_bin = fxb / whiskd.sr * 1000  # frames per bin * frame length in ms

    spktr_sorted, endt, cidsSorted, edges = align_fun.spike_train(spk,
                                                                  whiskd,
                                                                  binl=t_bin,
                                                                  surr=surr,
                                                                  cgs=cgs)
    # if pethdiscard:
        
    #                                                                   # clu_idx=clu_idx)
    # else:
    #     spktr_sorted, endt, cidsSorted, edges = align_fun.spike_train(spk,
    #                                                                   whiskd,
    #                                                                   binl=t_bin,
    #                                                                   surr=surr,
    #                                                                   cgs=cgs)

    # pdb.set_trace()
    print(f'# include cortex+dcn: {len(spktr_sorted)}' )
    # Get only cortex data
    spktr_sorted, cidsSorted, sortedDepth = align_fun.spkcount_cortex(spktr_sorted,
                                                                      spk.sortedDepth,
                                                                      cidsSorted,
                                                                      cortex_depth,
                                                                      binl=t_bin,
                                                                      discard=False)
    # pdb.set_trace()
    if pethdiscard:
        spktr_sorted = np.delete(spktr_sorted, pethdiscard_idx, axis=0)
        cidsSorted = np.delete(cidsSorted, pethdiscard_idx)

    print(f'# after pethdiscard: {spktr_sorted.shape[0]}')
    # pdb.set_trace()
        
    # wsknpx_data = align_spktrain(wskpath, npxpath, cortex_depth, cgs=cgs, t_wind=t_wind, var=var, surr=surr, t_drop=t_drop, pre=pre, clu_idx=clu_idx, discard=discard)

    # Bin wvar and get corresponding firing rate
    wvar_tc = []  # list (whiskers) of lists (clusters) of wvar tuning
    wvar_tc_m = []  # list (whiskers) of lists (clusters) of wvar tuning curves
    wvar_hist = []

    for whisker in range(whiskd.nwhisk):  # for each whisker
        # Calculate z-scored average angle per time bin
        avg_wvar, isw_bin = tbin_wvar(whiskd,
                                      len(spktr_sorted[0]),
                                      edges,
                                      whisker=whisker,
                                      fxb=fxb,
                                      whisking=whisking,
                                      var=var,
                                      sr=whiskd.sr)

        # pdb.set_trace()
        # Discretise avg_angle range (in Space)
        wvar_edges = np.linspace(min(avg_wvar), max(avg_wvar), w_bin)
        whichbin = np.digitize(avg_wvar, wvar_edges)  # get idx of wvar values
        whichbin[whichbin == np.unique(whichbin)[-1]] = np.unique(whichbin)[
            -2]  # avoid edge problem for max wvar value
        wvar_hist_, _ = np.histogram(avg_wvar, wvar_edges, density=True)
        wvar_hist.append(wvar_hist_)
        # Get firing rate per discretised avg_wvar; save tot spkcount
        tc_cl = []  # cluster specific wvar tuning
        tc_cl_m = []  # cluster specific average wvar tuning curve
        tot_spkcount = []                  # total spike count per clu
        for clu in range(cidsSorted.size):  # for each cluster
            spike_count = spktr_sorted[clu]

            if whisking:
                spike_count = spike_count[isw_bin]
                # whichbin__ = whichbin[len(whichbin) // 2:]
                # spike_count__ = spike_count[len(spike_count) // 2:]
                # whichbin__ = whichbin[:len(whichbin) // 2]
                # spike_count__ = spike_count[:len(spike_count) // 2]

            # Save tot spike count per clu
            tot_spkcount.append(np.sum(spike_count))

            # Sort spk_count by avg_wvar; compute firing rate per avg_wvar;
            # t_bin / 1000 from ms to sec
            # wvar_fr = [
            #     spike_count[whichbin == wvar_bin] / (t_bin / 1000)
            #     for wvar_bin in np.unique(whichbin)
            # ]
            # wvar_fr_m = [
            #     np.mean(spike_count[whichbin == wvar_bin] / (t_bin / 1000))
            #     for wvar_bin in np.unique(whichbin)
            # ]
            # pdb.set_trace()
            wvar_fr = [
                spike_count[whichbin == wvar_bin] / (t_bin / 1000)
                for wvar_bin in np.arange(1, w_bin)
            ]
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', category=RuntimeWarning)
                wvar_fr_m = [
                    np.mean(spike_count[whichbin == wvar_bin] / (t_bin / 1000))
                    for wvar_bin in np.arange(1, w_bin)
                ]
            wvar_fr_m = np.nan_to_num(wvar_fr_m)
            tc_cl.append(wvar_fr)
            tc_cl_m.append(wvar_fr_m)

        wvar_tc.append(tc_cl)  # save whisker specific cluters' tuning
        wvar_tc_m.append(
            tc_cl_m)  # save whisker specific cluters' tuning curve

    return wvar_tc, wvar_tc_m, tot_spkcount, cidsSorted, wvar_hist, wskpath


def plot_tc(wvar_tc,
            wvar_tc_m,
            wvar_tc_shuff,
            bfit,               # from tc_fit
            tot_spkcount,
            cidsSorted,
            wvar_hist,
            path,
            kmsclusters_tsne3,
            whisker,
            pethdiscard=True,
            plot_single=False,
            clu_id=[],
            surr=False,
            cluster=True):
    ''' Plot tc for single recording
    '''
    # pdb.set_trace()
    # Figure parameters
    lcolor = 'darkgoldenrod'
    style = 'italic'
    figsize = (14, 12)          # for single unit tc plot! 
    sns.set_context("poster", font_scale=1)
    # sns.color_palette("deep")
    # sns.color_palette("flare")
    plt.rcParams['axes.facecolor'] = 'lavenderblush'
    plt.rcParams["figure.edgecolor"] = 'black'
    plt.rcParams['axes.linewidth'] = matplotlib.rcParamsDefault['axes.linewidth']

    # Figure structure (rows and columns)
    n_clu = len(cidsSorted)
    n_col = np.round(np.sqrt(n_clu)).astype(int)
    n_row = n_clu // n_col if (n_clu // n_col *
                               n_col) == n_clu else n_clu // n_col + 1
    # szfig_y = 2 * n_row # use for png with no added text!!!!
    # szfig_x = 3 * n_col
    szfig_y = 4 * n_row
    szfig_x = 6 * n_col

    whisker_ = int(whisker[-1])

    # Group colors
    if cluster:
        gcolors = kmsclusters_tsne3[0].map({0:'blue', 1:'red', 2:'green', 3:'violet', 4:'yellow', 5:'pink', 6:'brown', 7:'black', 8:'cyan'})

    # Plot
    if not (var == 'phase'):
        fig, axs = plt.subplots(n_row,
                                n_col,
                                figsize=(szfig_x, szfig_y),
                                sharex=False)
    else:
        fig, axs = plt.subplots(n_row,
                                 n_col,
                                 figsize=(szfig_x, szfig_y),
                                 sharex=False,
                                 subplot_kw={'projection': 'polar'})

    # Turn off all axes and turn on one-by-on to avoid empty axes
    for axis in axs.flat:
        axis.set_axis_off()

    # pdb.set_trace()
    # If var is phase plot polar
    if not (var == 'phase'):
        for idx, clu in zip(
                itertools.product(np.arange(n_row), np.arange(n_col)),
                range(n_clu)):

            # Get posterior samples
            sf = bfit['sf'][whisker_][clu]
            ppc = bfit['ppc'][whisker_][clu].posterior_predictive.tc_val * sf
            

            if n_col == 1:
                idx = idx[0]

            # Plot tuning curves
            axs[idx].set_axis_on()
            sns.barplot(ax=axs[idx], data=wvar_tc_shuff[whisker_][clu], errorbar='se', saturation=0.1, edgecolor=".5", facecolor=(0, 0, 0, 0))
            sns.barplot(ax=axs[idx], data=wvar_tc[whisker_][clu], errorbar='se', alpha=0.7)
            az.plot_hdi(np.arange(11), ppc, hdi_prob=0.68, ax=axs[idx])

            if idx[1] == 0:
                # axs[idx].set_ylabel('fr', size=15)
                axs[idx].set_ylabel('fr')
            if idx[0] == n_row - 1:
                # axs[idx].set_xlabel(var, size=15)
                axs[idx].set_xlabel(var)

            ax2 = axs[idx].twinx()
            ax2.plot(wvar_hist[whisker_], 'r', label=var)
            if idx[1] == n_col -1:
                ax2.set_ylabel(f'{var} density')
                # ax2.set_ylabel(f'{var} density', size=15)

            # Add text
            max_y = axs[idx].get_ylim()[1]
            axs[idx].set_title(f'cluster {cidsSorted[clu]} - spk count {tot_spkcount[clu]}', size=15)

            # axs[idx].text(w_bin - 1,
            #               max_y * (5 / 6),
            #               'id {}'.format(cidsSorted[clu]),
            #               fontsize='large',
            #               fontweight='semibold')

            # Color based on kmean cluster
            if cluster:
                plt.setp(axs[idx].spines.values(), edgecolor=gcolors.loc['8'][clu], linewidth=10)

    else:
        # theta = np.linspace(0.0, 2 * np.pi, w_bin)
        theta = np.linspace(-np.pi, np.pi, w_bin)
        for idx, clu in zip(
                itertools.product(np.arange(n_row), np.arange(n_col)),
                range(n_clu)):

            if n_col == 1:
                idx = idx[0]

            # Plot tuning curves
            axs[idx].set_axis_on()
            axs[idx].plot(theta, wvar_tc_m[whisker_][clu], linewidth=7.0)
            max_y = axs[idx].get_ylim()[1]
            axs[idx].set_title(f'cluster {cidsSorted[clu]}', size=15)
            # axs[idx].text(1,
            #               max_y + max_y / 5,
            #               'id {}'.format(cidsSorted[clu]),
            #               fontsize='xx-large',
            #               fontweight='semibold')
    # fig.suptitle('Tuning curves', size=20)
    fig.suptitle('Tuning curves')

    # plt.tight_layout()

    left = 0.05
    right = 0.95
    bottom = 0.05
    top = 0.9
    wspace = 0.2
    hspace = 0.4

    #
    plt.subplots_adjust(left=left,
                        right=right,
                        bottom=bottom,
                        top=top,
                        wspace=wspace,
                        hspace=hspace)

    # Save
    if os.path.basename(os.getcwd()) != 'data_analysis':
        os.chdir('./data_analysis')
    try:
        # plt.savefig(os.getcwd() + f'/images/tc/{os.path.basename(path[6:-1])}/{os.path.basename(path[6:-1])}_{var}_{surr}_whisking{whisking}_pred{pre}_pethdiscard{pethdiscard}_whisker{whisker_}_gcolor5_3d.svg', format='svg')
        # plt.savefig(os.getcwd() + f'/images/tc/{os.path.basename(path[6:-1])}/{os.path.basename(path[6:-1])}_{var}_{surr}_whisking{whisking}_pred{pre}_pethdiscard{pethdiscard}_whisker{whisker_}_nocluster.svg', format='svg')
        # plt.savefig(os.getcwd() + f'/images/tc/{os.path.basename(path[6:-1])}/{os.path.basename(path[6:-1])}_{var}_{surr}_whisking{whisking}_pred{pre}_pethdiscard{pethdiscard}_whisker{whisker_}_gcolor7_rowbspline.svg', format='svg')
        plt.savefig(os.getcwd() + f'/images/tc/{os.path.basename(path[6:-1])}/{os.path.basename(path[6:-1])}_{var}_whisking{whisking}_pred{pre}_pethdiscard{pethdiscard}_whisker{whisker_}_withshuffle.svg', format='svg')
    except FileNotFoundError:
        print(f'created dir images/tc/{os.path.basename(path[6:-1])}')
        os.makedirs(os.getcwd() + f'/images/tc/{os.path.basename(path[6:-1])}')
        # plt.savefig(os.getcwd() + f'/images/tc/{os.path.basename(path[6:-1])}/{os.path.basename(path[6:-1])}_{var}_{surr}_whisking{whisking}_pred{pre}_pethdiscard{pethdiscard}_whisker{whisker_}_gcolor4_3d.svg', format='svg')
        # plt.savefig(os.getcwd() + f'/images/tc/{os.path.basename(path[6:-1])}/{os.path.basename(path[6:-1])}_{var}_{surr}_whisking{whisking}_pred{pre}_pethdiscard{pethdiscard}_whisker{whisker_}_gcolor7_rowbspline.svg', format='svg')
        plt.savefig(os.getcwd() + f'/images/tc/{os.path.basename(path[6:-1])}/{os.path.basename(path[6:-1])}_{var}_whisking{whisking}_pred{pre}_pethdiscard{pethdiscard}_whisker{whisker_}_withshuffle.svg', format='svg')

    # Plot single
    if plot_single:
        # clu_id = 19
        # clu_id = 33
        # clu_id = 69
        clu_id = clu_id
        # clu_id = 58
        # clu_id = 65
        # clu_id = 67

        fig3, ax3 = plt.subplots(tight_layout=True, figsize=figsize)
        sns.barplot(ax=ax3, data=wvar_tc[whisker_][clu_id], sd='se')
        ax3_1 = ax3.twinx()
        ax3_1.plot(wvar_hist[whisker_], 'darkgoldenrod', linewidth=5, label=var)

        # Aesthetics
        ax3.set_xticks([])
        ax3.set_xticklabels([])
        ax3.set_xlabel(f' {var} binned')
        ax3.set_ylabel('fr (Hz)')
        ax3.set_xlabel(f' {var} (bins)')
        ax3_1.set_ylabel(f'{var} density')
        # fig3.legend(prop={'size': 6})


        if os.path.basename(os.getcwd()) != 'data_analysis':
            os.chdir('./data_analysis')

        if not surr:
            surr = 'nosurr'
        try:
            plt.savefig(os.getcwd() + f'/images/tc/{os.path.basename(path[6:-1])}/{os.path.basename(path[6:-1])}_{var}_{surr}_whisking{whisking}_pred{pre}_CluIdx{cidsSorted[clu_id]}.svg', format='svg')
        except FileNotFoundError:
            print(f'created dir images/tc/{os.path.basename(path[6:-1])}')
            os.makedirs(os.getcwd() + f'/images/tc/{os.path.basename(path[6:-1])}')
            plt.savefig(os.getcwd() + f'/images/tc/{os.path.basename(path[6:-1])}/{os.path.basename(path[6:-1])}_{var}_{surr}_whisking{whisking}_pred{pre}_CluIdx{cidsSorted[clu_id]}.svg', format='svg')


def plot_tc_compare(wvar_tc,
                    wvar_tc_m,
                    wvar_tc_shuff,
                    bfit,               # from tc_fit
                    tot_spkcount,
                    cidsSorted,
                    wvar_hist,
                    path,
                    kmsclusters_tsne3,
                    whisker,
                    pethdiscard=True,
                    plot_single=False,
                    clu_id=[],
                    surr=False,
                    cluster=True):
    ''' Plot tc for single recorgind
    '''
    # pdb.set_trace()
    cluster = True                # Color code tc by cluster
    ngroup = '6'                    # use clusters grouped in k=6 groups
    # pdb.set_trace()
    
    idxsli = pd.IndexSlice
    # Figure parameters
    lcolor = 'darkgoldenrod'
    style = 'italic'
    figsize = (14, 12)          # for single unit tc plot! 
    sns.set_context("poster", font_scale=1)
    # sns.color_palette("deep")
    # sns.color_palette("flare")
    plt.rcParams['axes.facecolor'] = 'lavenderblush'
    plt.rcParams["figure.edgecolor"] = 'black'
    plt.rcParams['axes.linewidth'] = matplotlib.rcParamsDefault['axes.linewidth']

    # Figure structure (rows and columns)
    for time_idx, time in enumerate(['pre', 'post']):
        n_clu = len(cidsSorted[time_idx])
        n_col = np.round(np.sqrt(n_clu)).astype(int)
        n_row = n_clu // n_col if (n_clu // n_col *
                                   n_col) == n_clu else n_clu // n_col + 1
        # szfig_y = 2 * n_row # use for png with no added text!!!!
        # szfig_x = 3 * n_col
        szfig_y = 4 * n_row
        szfig_x = 6 * n_col

        whisker_ = int(whisker[-1])

        # Group colors
        if cluster:
            gcolors = kmsclusters_tsne3[0].map({0:'blue', 1:'red', 2:'green', 3:'violet', 4:'yellow', 5:'pink', 6:'brown', 7:'black', 8:'cyan'})

        # Plot
        if not (var == 'phase'):
            fig, axs = plt.subplots(n_row,
                                    n_col,
                                    figsize=(szfig_x, szfig_y),
                                    sharex=False)
        else:
            fig, axs = plt.subplots(n_row,
                                     n_col,
                                     figsize=(szfig_x, szfig_y),
                                     sharex=False,
                                     subplot_kw={'projection': 'polar'})

        # Turn off all axes and turn on one-by-on to avoid empty axes
        for axis in axs.flat:
            axis.set_axis_off()

        # pdb.set_trace()
        # If var is phase plot polar
        if not (var == 'phase'):
            for idx, clu in zip(
                    itertools.product(np.arange(n_row), np.arange(n_col)),
                    range(n_clu)):

                # Get posterior samples
                sf = bfit[f'sf_{time}'][whisker_][clu]  # scaling factor
                ppc = bfit[f'ppc_{time}'][whisker_][clu].posterior_predictive.tc_val * sf


                if n_col == 1:
                    idx = idx[0]

                # Plot tuning curves
                axs[idx].set_axis_on()
                sns.barplot(ax=axs[idx], data=wvar_tc_shuff[time_idx][whisker_][clu], errorbar='se', saturation=0.1, edgecolor=".5", facecolor=(0, 0, 0, 0))
                sns.barplot(ax=axs[idx], data=wvar_tc[time_idx][whisker_][clu], errorbar='se', alpha=0.7)
                az.plot_hdi(np.arange(11), ppc, hdi_prob=0.68, ax=axs[idx])

                if idx[1] == 0:
                    # axs[idx].set_ylabel('fr', size=15)
                    axs[idx].set_ylabel('fr')
                if idx[0] == n_row - 1:
                    # axs[idx].set_xlabel(var, size=15)
                    axs[idx].set_xlabel(var)

                ax2 = axs[idx].twinx()
                ax2.plot(wvar_hist[time_idx][whisker_], 'r', label=var)
                if idx[1] == n_col -1:
                    ax2.set_ylabel(f'{var} density')
                    # ax2.set_ylabel(f'{var} density', size=15)

                # Add text
                max_y = axs[idx].get_ylim()[1]
                axs[idx].set_title(f'cluster {cidsSorted[time_idx][clu]} - spk count {tot_spkcount[time_idx][clu]}', size=15)

                # axs[idx].text(w_bin - 1,
                #               max_y * (5 / 6),
                #               'id {}'.format(cidsSorted[clu]),
                #               fontsize='large',
                #               fontweight='semibold')

                # Color based on kmean cluster
                if cluster:
                    plt.setp(axs[idx].spines.values(), edgecolor=gcolors.loc[idxsli[ngroup, time, clu]], linewidth=10)

        else:
            # theta = np.linspace(0.0, 2 * np.pi, w_bin)
            theta = np.linspace(-np.pi, np.pi, w_bin)
            for idx, clu in zip(
                    itertools.product(np.arange(n_row), np.arange(n_col)),
                    range(n_clu)):

                if n_col == 1:
                    idx = idx[0]

                # Plot tuning curves
                axs[idx].set_axis_on()
                axs[idx].plot(theta, wvar_tc_m[time_idx][whisker_][clu], linewidth=7.0)
                max_y = axs[idx].get_ylim()[1]
                axs[idx].set_title(f'cluster {cidsSorted[clu]}', size=15)
                # axs[idx].text(1,
                #               max_y + max_y / 5,
                #               'id {}'.format(cidsSorted[clu]),
                #               fontsize='xx-large',
                #               fontweight='semibold')
        # fig.suptitle('Tuning curves', size=20)
        fig.suptitle(f'Tuning curves {time}drop')

        # plt.tight_layout()

        left = 0.05
        right = 0.95
        bottom = 0.05
        top = 0.9
        wspace = 0.2
        hspace = 0.4

        #
        plt.subplots_adjust(left=left,
                            right=right,
                            bottom=bottom,
                            top=top,
                            wspace=wspace,
                            hspace=hspace)

        # Save
        if os.path.basename(os.getcwd()) != 'data_analysis':
            os.chdir('./data_analysis')
        try:
            # plt.savefig(os.getcwd() + f'/images/tc/{os.path.basename(path[6:-1])}/{os.path.basename(path[6:-1])}_{var}_{surr}_whisking{whisking}_pred{pre}_pethdiscard{pethdiscard}_whisker{whisker_}_gcolor5_3d.svg', format='svg')
            # plt.savefig(os.getcwd() + f'/images/tc/{os.path.basename(path[6:-1])}/{os.path.basename(path[6:-1])}_{var}_{surr}_whisking{whisking}_pred{pre}_pethdiscard{pethdiscard}_whisker{whisker_}_nocluster.svg', format='svg')
            # plt.savefig(os.getcwd() + f'/images/tc/{os.path.basename(path[6:-1])}/{os.path.basename(path[6:-1])}_{var}_{surr}_whisking{whisking}_pred{pre}_pethdiscard{pethdiscard}_whisker{whisker_}_gcolor7_rowbspline.svg', format='svg')
            plt.savefig(os.getcwd() + f'/images/tc/{os.path.basename(path[time_idx][6:-1])}/{os.path.basename(path[time_idx][6:-1])}_{var}_whisking{whisking}_{time}drop_pethdiscard{pethdiscard}_whisker{whisker_}_withshuffle_ngroup{ngroup}.svg', format='svg')
        except FileNotFoundError:
            print(f'created dir images/tc/{os.path.basename(path[6:-1])}')
            os.makedirs(os.getcwd() + f'/images/tc/{os.path.basename(path[6:-1])}')
            # plt.savefig(os.getcwd() + f'/images/tc/{os.path.basename(path[6:-1])}/{os.path.basename(path[6:-1])}_{var}_{surr}_whisking{whisking}_pred{pre}_pethdiscard{pethdiscard}_whisker{whisker_}_gcolor4_3d.svg', format='svg')
            # plt.savefig(os.getcwd() + f'/images/tc/{os.path.basename(path[6:-1])}/{os.path.basename(path[6:-1])}_{var}_{surr}_whisking{whisking}_pred{pre}_pethdiscard{pethdiscard}_whisker{whisker_}_gcolor7_rowbspline.svg', format='svg')
            plt.savefig(os.getcwd() + f'/images/tc/{os.path.basename(path[time_idx][6:-1])}/{os.path.basename(path[time_idx][6:-1])}_{var}_whisking{whisking}_{time}drop_pethdiscard{pethdiscard}_whisker{whisker_}_withshuffle_ngroup{ngroup}.svg', format='svg')
    plt.close()

    # Plot single
    if plot_single:
        # clu_id = 19
        # clu_id = 33
        # clu_id = 69
        clu_id = clu_id
        # clu_id = 58
        # clu_id = 65
        # clu_id = 67

        fig3, ax3 = plt.subplots(tight_layout=True, figsize=figsize)
        sns.barplot(ax=ax3, data=wvar_tc[whisker_][clu_id], sd='se')
        ax3_1 = ax3.twinx()
        ax3_1.plot(wvar_hist[whisker_], 'darkgoldenrod', linewidth=5, label=var)

        # Aesthetics
        ax3.set_xticks([])
        ax3.set_xticklabels([])
        ax3.set_xlabel(f' {var} binned')
        ax3.set_ylabel('fr (Hz)')
        ax3.set_xlabel(f' {var} (bins)')
        ax3_1.set_ylabel(f'{var} density')
        # fig3.legend(prop={'size': 6})


        if os.path.basename(os.getcwd()) != 'data_analysis':
            os.chdir('./data_analysis')

        try:
            plt.savefig(os.getcwd() + f'/images/tc/{os.path.basename(path[6:-1])}/{os.path.basename(path[6:-1])}_{var}_whisking{whisking}_pred{pre}_CluIdx{cidsSorted[clu_id]}.svg', format='svg')
        except FileNotFoundError:
            print(f'created dir images/tc/{os.path.basename(path[6:-1])}')
            os.makedirs(os.getcwd() + f'/images/tc/{os.path.basename(path[6:-1])}')
            plt.savefig(os.getcwd() + f'/images/tc/{os.path.basename(path[6:-1])}/{os.path.basename(path[6:-1])}_{var}_whisking{whisking}_pred{pre}_CluIdx{cidsSorted[clu_id]}.svg', format='svg')


def plot_tc_clusters(labeled_tc, labeled_tc_shuffle, kmclusters, wsk_hist):
    """ Plot the typical tuning curves for each clusters; plot for each
    number of clusters (ngroups)
    """
    # pdb.set_trace()
    idxsli = pd.IndexSlice
    if os.path.basename(os.getcwd()) != 'data_analysis':
        os.chdir('./data_analysis')

    # Figure parameters
    lcolor = 'darkgoldenrod'
    style = 'italic'
    figsize = (14, 12)          # for single unit tc plot! 
    sns.set_context("poster", font_scale=1)
    # sns.color_palette("deep")
    # sns.color_palette("flare")
    plt.rcParams['axes.facecolor'] = 'lavenderblush'
    plt.rcParams["figure.edgecolor"] = 'black'
    plt.rcParams['axes.linewidth'] = matplotlib.rcParamsDefault['axes.linewidth']

    wsk_hist['kgroup_idx'] = labeled_tc.index.get_level_values(3)
    wsk_hist.set_index('kgroup_idx', append=True, inplace=True)

    # Plot & save
    for ngroups in labeled_tc.index.get_level_values(0).unique():
        nrow = np.sqrt(ngroups + 1).astype(int)
        # ncol = (ngroups + 1) // nrow if nrow ** 2 == ngroups + 1 else (ngroups + 1) // nrow + (ngroups + 1 - ((ngroups + 1) // nrow) * nrow)
        ncol = np.ceil((ngroups + 1) / nrow).astype(int)
        fig, ax = plt.subplots(nrow, ncol, figsize=(14, 12), sharex=True, sharey=True)
        for idx in range(ngroups + 1):
            # ax2 = ax.flatten()[idx].twinx()
            # ax2.set_ylim([0, 0.65])
            # sns.lineplot(data=labeled_tc.loc[(ngroups, slice(None), slice(None), idx)].melt(), x='wbin', y='value', ax=ax.flatten()[idx])
            sns.pointplot(data=labeled_tc.loc[(ngroups, slice(None), slice(None), idx)].melt(), x='wbin', y='value', errorbar='se', ax=ax.flatten()[idx])
            # sns.barplot(data=labeled_tc.loc[(ngroups, slice(None), slice(None), idx)].values, ax=ax.flatten()[idx], errorbar='se', palette='flare', alpha=0.7)
            # sns.lineplot(data=wsk_hist.loc[idxsli[str(ngroups), :, :, idx]].melt(), x='variable', y='value', errorbar='se', ax=ax2)
            ax.flatten()[idx].set_ylim(bottom=0)

            ax.flatten()[idx].set_title(f'{idx}')
            
        try:
            plt.savefig(os.getcwd() + f'/images/tc/typicalTC_clu{ngroups}.svg', format='svg')
        except FileNotFoundError:
            print('created dir images/tc')
            os.makedirs(os.getcwd() + '/images/tc')
            plt.savefig(os.getcwd() + f'/images/tc/typicalTC_clu{ngroups}.svg', format='svg')

    # Plot distribution clusters across recordings
    for ngroups in kmclusters.index.get_level_values(0).unique():
        # Plot
        fig, ax = plt.subplots(figsize=figsize)
        sns.histplot(data=kmclusters.loc[ngroups], x='rec', hue=0, multiple='stack', bins=kmclusters.index.get_level_values(1).unique().size)
        # Save
        try:
            plt.savefig(os.getcwd() + f'/images/tc/kmclu_dist_ngroups{ngroups}.svg', format='svg')
        except FileNotFoundError:
            print('created dir images/tc')
            os.makedirs(os.getcwd() + '/images/tc')
            plt.savefig(os.getcwd() + f'/images/tc/kmclu_dist_ngroups{ngroups}.svg', format='svg')

    
def plot_tc_clusters_compare(labeled_tc, labeled_tc_shuffle, kmclusters, clu_distribution):
    """ Plot the typical tuning curves for each clusters; plot for each
    number of clusters (ngroups); compare pre vs post drop periods;
    km clusters computed separately for pre and post periods.
    """
    idxsli = pd.IndexSlice
    # pdb.set_trace()

    labeled_tc_copy = labeled_tc.copy()
    labeled_tc = labeled_tc.div(labeled_tc.sum(axis=1), axis=0)

    if os.path.basename(os.getcwd()) != 'data_analysis':
        os.chdir('./data_analysis')

    # Figure parameters
    lcolor = 'darkgoldenrod'
    style = 'italic'
    figsize = (14, 12)          # for single unit tc plot! 
    sns.set_context("poster", font_scale=1)
    # sns.color_palette("deep")
    # sns.color_palette("flare")
    plt.rcParams['axes.facecolor'] = 'lavenderblush'
    plt.rcParams["figure.edgecolor"] = 'black'
    plt.rcParams['axes.linewidth'] = matplotlib.rcParamsDefault['axes.linewidth']

    # Typical tc
    for ngroups in labeled_tc.index.get_level_values(0).unique():
        ngroups = int(ngroups)
        nrow = np.sqrt(ngroups).astype(int)
        # ncol = (ngroups + 1) // nrow if nrow ** 2 == ngroups + 1 else (ngroups + 1) // nrow + (ngroups + 1 - ((ngroups + 1) // nrow) * nrow)
        ncol = np.ceil((ngroups) / nrow).astype(int)
        fig, ax = plt.subplots(nrow, ncol, figsize=(14, 12))
        for group_idx in range(ngroups):
            sns.barplot(data=labeled_tc.loc[idxsli[str(ngroups), :, :, :, group_idx]].values, ax=ax.flatten()[group_idx], errorbar='se', palette='flare', alpha=0.7)
            # sns.barplot(data=labeled_tc_shuffle.loc[idxsli[str(ngroups), :, :, :, group_idx]].values, ax=ax.flatten()[group_idx], errorbar='se', saturation=0.1, edgecolor=".5", facecolor=(0, 0, 0, 0), errcolor='.5')
            ax.flatten()[group_idx].set_title(f'{group_idx}')

        try:
            plt.savefig(os.getcwd() + f'/images/tc/compare/typicalTC_clu{ngroups}_compare.svg', format='svg')
        except FileNotFoundError:
            print('created dir images/tc')
            os.makedirs(os.getcwd() + '/images/tc/compare')
            plt.savefig(os.getcwd() + f'/images/tc/compare/typicalTC_clu{ngroups}_compare.svg', format='svg')

    # labeled_tc_copy = labeled_tc.copy()
    # labeled_tc = labeled_tc_copy.copy()
    # labeled_tc_copy_scaled = labeled_tc.copy()

    # Typical tc (split by pre and post)
    for ngroups in labeled_tc.index.get_level_values(0).unique():
        ngroups = int(ngroups)
        nrow = np.sqrt(ngroups).astype(int)
        # ncol = (ngroups + 1) // nrow if nrow ** 2 == ngroups + 1 else (ngroups + 1) // nrow + (ngroups + 1 - ((ngroups + 1) // nrow) * nrow)
        ncol = np.ceil((ngroups) / nrow).astype(int)
        fig = plt.figure(figsize=(14, 12))
        subfig = fig.subfigures(2)
        for time_idx, time in enumerate(['pre', 'post']):
            ax = subfig[time_idx].subplots(nrow, ncol)
            # for idx in range(ngroups + 1):
            for group_idx in range(ngroups):
                # sns.lineplot(data=avg_clu_tc.loc[(ngroups, idx)].values, ax=ax.flatten()[idx])
                sns.barplot(data=labeled_tc.loc[idxsli[str(ngroups), :, time, :, group_idx]].values, ax=ax.flatten()[group_idx], errorbar='se', palette='flare', alpha=0.7)
                # sns.barplot(data=labeled_tc_shuffle.loc[idxsli[str(ngroups), :, time, :, group_idx]].values, ax=ax.flatten()[group_idx], errorbar='se', saturation=0.1, edgecolor=".5", facecolor=(0,  0, 0, 0), errcolor='.5')
                ax.flatten()[group_idx].set_title(f'{group_idx}')
            subfig[time_idx].suptitle(f'{time}')

            try:
                plt.savefig(os.getcwd() + f'/images/tc/compare/typicalTC_clu{ngroups}_compare_split.svg', format='svg')
            except FileNotFoundError:
                print('created dir images/tc/compare')
                os.makedirs(os.getcwd() + '/images/tc/compare')
                plt.savefig(os.getcwd() + f'/images/tc/compare/typicalTC_clu{ngroups}_compare_split.svg', format='svg')

    # Plot distribution clusters across recordings
    for ngroups in kmclusters.index.get_level_values(0).unique():
        # Plot
        fig = plt.figure(figsize=(14, 12))
        subfig = fig.subfigures(2)
        for time_idx, time in enumerate(['pre', 'post']):
            ax = subfig[time_idx].subplots()
            sns.histplot(data=kmclusters.loc[idxsli[ngroups, :, :, time, :]], x='rec', hue=0, multiple='stack', ax=ax, bins=kmclusters.index.get_level_values(2).unique().size)
            subfig[time_idx].suptitle(f'{time}')

        # Save
        try:
            plt.savefig(os.getcwd() + f'/images/tc/compare/kmclu_dist_ngroups{ngroups}_compare.svg', format='svg')
        except FileNotFoundError:
            print('created dir images/tc/compare')
            os.makedirs(os.getcwd() + '/images/tc/compare')
            plt.savefig(os.getcwd() + f'/images/tc/compare/kmclu_dist_ngroups{ngroups}_compare.svg', format='svg')

    # Plot distribution kgroups
    diff_clu_dist = clu_distribution.groupby(by=['ngroups', 'cond']).diff(periods=-1)
    diff_clu_dist = diff_clu_dist.iloc[::2].droplevel(2)
    for ngroups in ['6', '7', '8']:
        fig, ax = plt.subplots(2, figsize=figsize)
        for cond_idx, cond in enumerate(['gCNO', 'control']):
            sns.histplot(data=kmclusters.loc[idxsli[ngroups, cond, :, :, :]], x=0, bins=int(ngroups), hue='time', multiple='dodge', hue_order=['pre', 'post'], shrink=.8, ax=ax[cond_idx])
            ax[cond_idx].set_title(f'{cond}_{diff_clu_dist.loc[idxsli[ngroups, cond]]}')
            ax[cond_idx].set_xlabel(None)
            # ax[cond_idx].text(5, ax[cond_idx].get_ylim()[1], f'{diff_clu_dist.loc[idxsli[ngroups, cond]]}')
        # save
        try:
            plt.savefig(os.getcwd() + f'/images/tc/compare/kmclu_prepost_distribution_ngroups{ngroups}.svg', format='svg')
        except FileNotFoundError:
            print('created dir images/tc/compare')
            os.makedirs(os.getcwd() + '/images/tc/compare')
            plt.savefig(os.getcwd() + f'/images/tc/compare/kmclu_prepost_distribution_ngroups{ngroups}.svg', format='svg')


    # clu_distribution.groupby(by=['ngroups', 'cond']).apply(lambda x: x.loc[idxsli[:, :, 'post']] - x.loc[idxsli[:, :, 'pre']])
    # diff_clu_dist = clu_distribution.groupby(by=['ngroups', 'cond']).apply(lambda x: x.loc[(slice(None), slice(None), 'post')] - x.loc[(slice(None), slice(None), 'pre')])
    diff_clu_dist = clu_distribution.groupby(by=['ngroups', 'cond']).diff(periods=-1)
    diff_clu_dist = diff_clu_dist.iloc[::2].droplevel(2)

    for ngroups in kmclusters.index.get_level_values(0).unique():
        fig, ax = plt.subplots(figsize=figsize)
        __ = [diff_clu_dist.loc[ngroups]['control'], diff_clu_dist.loc[ngroups]['gCNO']]
        __ = pd.DataFrame(__, index=['control', 'gCNO'])
        __ = __.melt(ignore_index=False).reset_index()
        # __.reset_index(inplace=True)
        # __ = __.melt()
        sns.barplot(data=__, x='variable', y='value', hue='index')


    # Plot change of clusters pre-post drop
    for ngroups in ['6', '7', '8']:
        gCNO_data = kmclusters.loc[idxsli[ngroups, 'gCNO', :, :, :]]
        control_data = kmclusters.loc[idxsli[ngroups, 'control', :, :, :]]
        all_data = [gCNO_data, control_data]

        fig = plt.figure(figsize=figsize)
        subfig = fig.subfigures(2)
        for cond_idx, cond in enumerate(['gCNO', 'control']):
            ax = subfig[cond_idx].subplots(1, int(ngroups), sharey=False)
            for kgroup, axis in enumerate(ax):
                __ = all_data[cond_idx]
                __ = __.loc[idxsli[:, 'post', :]][(__.loc[idxsli[:, 'pre', :]] == kgroup).values]
                histplot = sns.histplot(data=__, ax=axis, bins=int(ngroups), color=['r', 'r', 'r', 'r', 'r', 'r'], discrete=True, stat='percent')
                histplot.patches[kgroup].set_facecolor('salmon')
                axis.legend([], [], frameon=False)
            subfig[cond_idx].suptitle(f'{cond}')

        try:
            plt.savefig(os.getcwd() + f'/images/tc/compare/kmclu_prepost_change_ngroups{ngroups}.svg', format='svg')
        except FileNotFoundError:
            print('created dir images/tc/compare')
            os.makedirs(os.getcwd() + '/images/tc/compare')
            plt.savefig(os.getcwd() + f'/images/tc/compare/kmclu_dist_ngroups{ngroups}.svg', format='svg')


    # Plot proportion of tc that remain in same kmclu
    # Compute proportion for each condition; proportion compared to
    # total number of units in condition.
    ngroups = '8'
    hist_data = []
    for cond_idx, cond in enumerate(['gCNO', 'control']):
        __ = kmclusters.loc[idxsli[ngroups, cond, :, :, :]]
        count_all_pre, bins = np.histogram(__.loc[idxsli[:, 'pre', :]], bins=np.arange(-0.5, int(ngroups) + 0.5, 1))
        mask_same = __.loc[idxsli[:, 'pre', :]] == __.loc[idxsli[:, 'post', :]]
        __ = __.loc[idxsli[:, 'pre', :]][mask_same].dropna().astype('int')
        count, bins = np.histogram(__, bins=np.arange(-0.5, int(ngroups) + 0.5, 1))
        # count = count / mask_same.shape[0] * 100
        count = count / count_all_pre * 100

        count = pd.DataFrame(count)
        count = pd.concat([count], keys=[f'{cond}'])
        count.reset_index(inplace=True)
        count.columns = ['cond', 'kmclu', 'values']
        hist_data.append(count)

    hist_data = pd.concat(hist_data)
    prop_same_gCNO = [np.round(hist_data[hist_data.cond=='gCNO']['values'].mean(), 2), np.round(hist_data[hist_data.cond=='gCNO']['values'].std(), 2)]
    prop_same_control = [np.round(hist_data[hist_data.cond=='control']['values'].mean(), 2), np.round(hist_data[hist_data.cond=='control']['values'].std(), 2)]

    hist_data.rename(columns={'values':'% per cluster'}, inplace=True)

    fig, ax = plt.subplots(figsize=figsize)
    sns.barplot(data=hist_data, x='kmclu', y='% per cluster', hue='cond', ax=ax)
    fig.suptitle(f'Proportion same gCNO ={prop_same_gCNO}% control ={prop_same_control}%')
    try:
        plt.savefig(os.getcwd() + f'/images/tc/compare/kmclu_prepost_same_ngroups{ngroups}.svg', format='svg')
    except FileNotFoundError:
        print('created dir images/tc/compare')
        os.makedirs(os.getcwd() + '/images/tc/compare')
        plt.savefig(os.getcwd() + f'/images/tc/compare/kmclu_prepost_same_ngroups{ngroups}.svg', format='svg')

    # Compute proportion for only gCNO; proportion now is within cluter, not within condition
    # (total number of units within each cluster, not within condition)
    ngroup = ['8']
    cond = 'gCNO'
    __ = kmclusters.loc[idxsli[ngroups, cond, :, :, :]]
    count_all_pre, bins = np.histogram(__.loc[idxsli[:, 'pre', :]], bins=np.arange(-0.5, int(ngroups) + 0.5, 1))
    mask_same = __.loc[idxsli[:, 'pre', :]] == __.loc[idxsli[:, 'post', :]]
    __ = __.loc[idxsli[:, 'pre', :]][mask_same].dropna().astype('int')
    count, bins = np.histogram(__, bins=np.arange(-0.5, int(ngroups) + 0.5, 1))
    hist_data = count / count_all_pre * 100
    hist_data = pd.DataFrame(hist_data)
    hist_data = pd.concat([hist_data], keys=[f'{cond}'])
    hist_data.reset_index(inplace=True)
    hist_data.columns = ['cond', 'kmclu', 'values']
    hist_data.rename(columns={'values':'%'})
    
    fig, ax = plt.subplots(figsize=figsize)
    sns.barplot(data=hist_data, x='kmclu', y='values', hue='cond', ax=ax)
    fig.suptitle(f'Proportion same gCNO (all clusters) ={prop_same_gCNO}%')  # use from previous figure!
    try:
        plt.savefig(os.getcwd() + f'/images/tc/compare/kmclu_prepost_same_ngroups{ngroups}_gCNO.svg', format='svg')
    except FileNotFoundError:
        print('created dir images/tc/compare')
        os.makedirs(os.getcwd() + '/images/tc/compare')
        plt.savefig(os.getcwd() + f'/images/tc/compare/kmclu_prepost_same_ngroups{ngroups}_gCNO.svg', format='svg')


def plot_tc_clusters_compare_pre(labeled_tc, kmclusters, wsk_hist):
    """ Plot the typical tuning curves for each clusters; plot for each
    number of clusters (ngroups); compare pre vs post drop periods;
    km clusters computed based on pre.
    """
    idxsli = pd.IndexSlice
    # pdb.set_trace()

    if os.path.basename(os.getcwd()) != 'data_analysis':
        os.chdir('./data_analysis')

    # Figure parameters
    lcolor = 'darkgoldenrod'
    style = 'italic'
    figsize = (14, 12)          # for single unit tc plot! 
    sns.set_context("poster", font_scale=1)
    # sns.color_palette("deep")
    # sns.color_palette("flare")
    plt.rcParams['axes.facecolor'] = 'lavenderblush'
    plt.rcParams["figure.edgecolor"] = 'black'
    plt.rcParams['axes.linewidth'] = matplotlib.rcParamsDefault['axes.linewidth']

    wsk_hist['kgroup_idx'] = labeled_tc.index.get_level_values(5)
    wsk_hist.set_index('kgroup_idx', append=True, inplace=True)
    

    # Typical tc (split by pre and post)
    for ngroups in labeled_tc.index.get_level_values(0).unique():
        ngroups = int(ngroups)
        nrow = np.sqrt(ngroups).astype(int)
        # ncol = (ngroups + 1) // nrow if nrow ** 2 == ngroups + 1 else (ngroups + 1) // nrow + (ngroups + 1 - ((ngroups + 1) // nrow) * nrow)
        ncol = np.ceil((ngroups) / nrow).astype(int)
        fig = plt.figure(figsize=figsize)
        subfig = fig.subfigures(2, 2)
        for time, cond in itertools.product(enumerate(['pre', 'post']), enumerate(['gCNO', 'control'])):
            # ax = subfig[time[0], cond[0]].subplots(nrow, ncol, sharex=True, sharey=True)
            ax = subfig[time[0], cond[0]].subplots(nrow, ncol, sharex=True, sharey=True)
            ax_twin = []
            # for idx in range(ngroups + 1):
            for group_idx in range(ngroups):
                ax2 = ax.flatten()[group_idx].twinx()
                ax_twin.append(ax2)
                sns.barplot(data=labeled_tc.loc[idxsli[str(ngroups), cond[1], :, time[1], :, group_idx]].values, ax=ax.flatten()[group_idx], errorbar='se', palette='flare', alpha=0.7, errwidth=2)
                sns.lineplot(data=wsk_hist.loc[idxsli[str(ngroups), cond[1], :, time[1], :, group_idx]].melt(), x='variable', y='value', errorbar='se', ax=ax2)
                # sns.barplot(data=labeled_tc_shuffle.loc[idxsli[str(ngroups), :, time, :, group_idx]].values, ax=ax.flatten()[group_idx], errorbar='se', saturation=0.1, edgecolor=".5", facecolor=(0, 0, 0, 0), errcolor='.5')
                ax.flatten()[group_idx].set_title(f'{group_idx}')
                ax.flatten()[group_idx].set(xticklabels=[])
                ax2.set(ylabel=None)
                ax2.set_ylim([0, 0.7])
                if (group_idx + 1 != ngroups) & (group_idx + 1 != np.ceil(ngroups/2)):
                    ax2.yaxis.set_tick_params(labelright=False)
    
            subfig[time[0], cond[0]].suptitle(f'{time[1]}_{cond[1]}')
        try:
            plt.savefig(os.getcwd() + f'/images/tc/compare/typicalTC_clu{ngroups}_compare_split_pre.svg', format='svg')
        except FileNotFoundError:
            print('created dir images/tc/compare')
            os.makedirs(os.getcwd() + '/images/tc/compare')
            plt.savefig(os.getcwd() + f'/images/tc/compare/typicalTC_clu{ngroups}_compare_split_pre.svg', format='svg')
        fig.suptitle('Typical tc, based on pre-drop labels')

    # Plot difference pre - post
    labeled_tc_diff = labeled_tc.groupby(by=['ngroups', 'cond', 'rec', 'clu', 'kgroup_idx']).diff(periods=1, axis=0)
    labeled_tc_diff = labeled_tc_diff.loc[idxsli[:, :, :, 'post', :, :]]

    for ngroups in labeled_tc.index.get_level_values(0).unique():
        ngroups = int(ngroups)
        nrow = np.sqrt(ngroups).astype(int)
        # ncol = (ngroups + 1) // nrow if nrow ** 2 == ngroups + 1 else (ngroups + 1) // nrow + (ngroups + 1 - ((ngroups + 1) // nrow) * nrow)
        ncol = np.ceil((ngroups) / nrow).astype(int)
        fig = plt.figure(figsize=figsize)
        subfig = fig.subfigures(1, 2)
        for cond_idx, cond in enumerate(['gCNO', 'control']):
            tot_clu = labeled_tc_diff.loc[idxsli[str(ngroups), cond, :, :, :]].shape[0]
            ax = subfig[cond_idx].subplots(nrow, ncol, sharex=True, sharey=True)
            # for idx in range(ngroups + 1):
            for group_idx in range(ngroups):
                sli = labeled_tc_diff.loc[idxsli[str(ngroups), cond, :, :, group_idx]]
                sns.barplot(data=sli.values, ax=ax.flatten()[group_idx], errorbar='se', palette='flare', alpha=0.7)
                # sns.barplot(data=labeled_tc_shuffle.loc[idxsli[str(ngroups), :, time, :, group_idx]].values, ax=ax.flatten()[group_idx], errorbar='se', saturation=0.1, edgecolor=".5", facecolor=(0, 0, 0, 0), errcolor='.5')
                ax.flatten()[group_idx].set_title(f'{group_idx}')
                ax.flatten()[group_idx].set_ylim([-1.4, 1.4])
                ax.flatten()[group_idx].set(title=f'{sli.shape[0]}; {np.round(sli.shape[0] / tot_clu, 2)}')
            subfig[cond_idx].suptitle(f'{cond}')
        fig.suptitle('Difference pre-post typical tc, based on pre-drop labels')

        try:
            plt.savefig(os.getcwd() + f'/images/tc/compare/typicalTC_clu{ngroups}_compare_split_pre_diff.svg', format='svg')
        except FileNotFoundError:
            print('created dir images/tc/compare')
            os.makedirs(os.getcwd() + '/images/tc/compare')
            plt.savefig(os.getcwd() + f'/images/tc/compare/typicalTC_clu{ngroups}_compare_split_pre_diff.svg', format='svg')


def info_transmitted(cmat):
    """ Information transmitted, from pre to post drop time, as a
    measure of stablity of tc assignment to clusters before and
    after drup application.
    ref: 1)Idow and Sono, Japanese Psychological research, 1969,
    'the amount of transmitted information in confusion matrix';
    2) Sagi and Svirsky, J acoust Soc Am, 2008, 'Information transfer
    analysis: a first look at estimation bias'.
    """
    N = cmat.sum()
    
    p_pre = np.sum(cmat, axis=1) / N
    sum_pre = np.sum(p_pre * np.log(p_pre))
    p_post = np.sum(cmat, axis=0) / N
    sum_post = np.sum(p_post * np.log(p_post))
    p_prepost = np.sum(cmat.reshape(-1, 1)) / N
    sum_prepost = np.sum(p_prepost * np.log(p_prepost))

    It = -sum_pre - sum_post + sum_prepost
    
    return It

            
        
def plot_confusion(kmclusters, n_kmgroups='6'):
    """ Function to compute and plot confusion matrix: row is pre-drop tc assignment,
    column is post-drop tc assignment.
    Compute information transmitted between pre- and post-drop tc assignment as a
    measure of stability of tc assignment.
    """
    # Figure parameters
    figsize = (14, 12)          # for single unit tc plot! 
    sns.set_context("poster", font_scale=1)
    # sns.color_palette("deep")
    # sns.color_palette("flare")
    plt.rcParams['axes.facecolor'] = 'lavenderblush'
    plt.rcParams["figure.edgecolor"] = 'black'
    plt.rcParams['axes.linewidth'] = matplotlib.rcParamsDefault['axes.linewidth']

    
    # pdb.set_trace()
    idxsli = pd.IndexSlice
    gCNO_pre = kmclusters.loc[idxsli[n_kmgroups, 'gCNO', :, 'pre']]
    gCNO_post = kmclusters.loc[idxsli[n_kmgroups, 'gCNO', :, 'post']]
    control_pre = kmclusters.loc[idxsli[n_kmgroups, 'control', :, 'pre']]
    control_post = kmclusters.loc[idxsli[n_kmgroups, 'control', :, 'post']]

    cm_gCNO = confusion_matrix(gCNO_pre, gCNO_post)
    cm_control = confusion_matrix(control_pre, control_post)
    __ = [cm_gCNO, cm_control]

    # Information transfer
    it_gCNO = info_transmitted(cm_gCNO)
    it_control = info_transmitted(cm_control)

    it_all = {'gCNO':it_gCNO, 'control':it_control}

    with open(f'/home/yq18021/Documents/github_gcl/data_analysis/save_data/tc_confmat_infotransfer_{n_kmgroups}groups', 'wb') as f:
        pickle.dump(it_all, f)

    np.round(confusion_matrix(gCNO_pre, gCNO_post), 3)

    n_groups = int(n_kmgroups)
    fig, ax = plt.subplots(1, 2, figsize=(14, 12), sharex=True, sharey=True)
    for cond_idx, cond_data in enumerate(__):
        ax[cond_idx].matshow(cond_data)
        for i in range(n_groups):
            for j in range(n_groups):
                ax[cond_idx].text(j, i, '{:.1f}'.format(cond_data[i, j]),
                        ha="center", va="center", color="w")
        ax[cond_idx].set_xlabel('post')
        ax[cond_idx].set_ylabel('pre')

    ax[0].set_title('gCNO')
    ax[1].set_title('control')

    if os.path.basename(os.getcwd()) != 'data_analysis':
        os.chdir('./data_analysis')

    try:
        fig.savefig(os.getcwd() + f'/images/tc/confusion_matrix_{n_kmgroups}groups.svg', format='svg')
    except FileNotFoundError:
        print('created dir images/tc')
        os.makedirs(os.getcwd() + '/images/tc')
        fig.savefig(os.getcwd() + f'/images/tc/confusion_matrix_{n_kmgroups}groups.svg', format='svg')


def entropy_tc_compare(tc_m, tc_β, km_labels, wsk=2):
    """ Function to compute entropy of tuning curve for each cluster (pre and post):
    entropy as a measure of information about whisking position present in cluster's
    tc; high entropy means less informative tc.
    Take difference post - pre tc entropy for each cluster and plot histogram: positive
    values indicate that post tc is less informative than pre tc. Compute skewness
    of entropy difference distribution for two conditions (gCNO and control).Compute
    KL-divergence -pre-post tc's (normalised to 1).
    
    - Attention: tc_m list with len=24 rec, len=2 time (0='pre' and 1='post').
    """
    # pdb.set_trace()
    idxsli = pd.IndexSlice

    # Index for df (attention: post here comes before pre time)
    midx_ = km_labels.loc[idxsli['8']].index  # ngroup ('8') does not matter

    # Organise row tc in df (attention to time!) and take difference (post-pre)
    df_tcm = []
    for rec_idx in midx_.get_level_values(1).unique():
        for time_idx in [1, 0]:
            __ = tc_m[rec_idx][time_idx][wsk]
            n_clu = len(__)
            for clu in range(n_clu):
                df_tcm.append(__[clu] / np.sum(__[clu]))

    df_tcm = pd.DataFrame(df_tcm)
    df_tcm.index = midx_
    df_tcm_diff = df_tcm.groupby(by=['cond', 'rec', 'clu']).diff(-1)  # post-pre
    df_tcm_diff = df_tcm_diff.loc[idxsli[:, :, 'post', :]]

    # Organise b-spline fit to tc in df and take difference (post-pre)
    df_tcβ = []
    for rec_idx, rec in enumerate(midx_.get_level_values(1).unique()):
        for time_idx, time in enumerate(midx_.get_level_values(2).unique()):
            __ = tc_β.loc[idxsli[rec, time, :]]
            n_clu = len(__)
            for clu in range(n_clu):
                df_tcβ.append(__.loc[clu] / np.sum(__.loc[clu]))
    
    df_tcβ = pd.DataFrame(df_tcβ)
    df_tcβ.index = midx_
    df_tcβ_diff = df_tcβ.groupby(by=['cond', 'rec', 'clu']).diff(-1)  # post-pre
    df_tcβ_diff = df_tcβ_diff.loc[idxsli[:, :, 'post', :]]

    # Compute entropy row tc
    df_entropy = []
    # for rec_idx, rec in enumerate(midx_.get_level_values(1).unique()):
    #     for time_idx, time in enumerate(midx_.get_level_values(2).unique()):
    for rec_idx in midx_.get_level_values(1).unique():
        for time_idx in [1, 0]:
            __ = tc_m[rec_idx][time_idx][wsk]
            n_clu = len(__)
            for clu in range(n_clu):
                df_entropy.append(entropy(__[clu]))

    df_entropy = pd.DataFrame(df_entropy)
    df_entropy.index = midx_
    df_entropy_diff = df_entropy.groupby(by=['cond', 'rec', 'clu']).diff(-1) # post-pre
    df_entropy_diff = df_entropy_diff.loc[idxsli[:, :, 'post', :]]

    # # Compute entropy b-spline fit to tc
    # df_entropy_β = []
    # for rec_idx, rec in enumerate(midx_.get_level_values(1).unique()):
    #     for time_idx, time in enumerate(midx_.get_level_values(2).unique()):
    #         __ = tc_β.loc[idxsli[rec,time, :]]
    #         n_clu = len(__)
    #         for clu in range(n_clu):
    #             df_entropy_β.append(entropy(__.loc[clu]))

    # df_entropy_β = pd.DataFrame(df_entropy_β)
    # df_entropy_β.index = midx_
    # df_entropy_β_diff = df_entropy_β.groupby(by=['cond', 'rec', 'clu']).diff()
    # df_entropy_β_diff = df_entropy_β_diff.loc[idxsli[:, :, 'pre', :]]


    # Compute Skewness distribution of entropy differences (post-pre)
    pdb.set_trace()
    skewness = {}
    for cond in ['gCNO', 'control']:
        __ = skew(df_entropy_diff.loc[idxsli[f'{cond}', :, :]][0])
        __ = np.round(__, 4)
        skewness[f'{cond}'] = '{:0.4f}'.format(__)  # reformat to leave 4 number after 0

    # Compute KL-divergence D(pre|post): true distribution is pre (changes compared
    # to pre-drop).
    df_kl = []
    for rec_idx, rec in enumerate(midx_.get_level_values(1).unique()):
        pre__ = tc_m[rec_idx][0][wsk]
        post__ = tc_m[rec_idx][1][wsk]
        n_clu = len(pre__)
        for clu in range(n_clu):
            pre_nozero = pre__[clu].copy()
            pre_nozero[pre_nozero==0] = 0.01
            post_nozero = post__[clu].copy()
            post_nozero[post_nozero==0] = 0.01
            
            df_kl.append(entropy(pre_nozero, qk=post_nozero))
            # print(rec_idx, clu, entropy(pre_nozero, qk=post_nozero))
            

    df_kl = pd.DataFrame(df_kl)
    df_kl.index = df_entropy_diff.index
    # pdb.set_trace()

    # Exponential function for fitting kl distribution
    def expfun(x, a, b):
        y = a*np.exp(b*(-x))
        return y

    # Figure parameters
    figsize = (14, 12)          # for single unit tc plot! 
    sns.set_context("poster", font_scale=1)
    # sns.color_palette("deep")
    # sns.color_palette("flare")
    plt.rcParams['axes.facecolor'] = 'lavenderblush'
    plt.rcParams["figure.edgecolor"] = 'black'
    plt.rcParams['patch.edgecolor'] = 'none'
    plt.rcParams['axes.linewidth'] = matplotlib.rcParamsDefault['axes.linewidth']

    # Plot figures
    # Plot difference row tc's
    fig, ax = plt.subplots(1, 2, sharex=True, sharey=True)
    for idx_cond, cond in enumerate(['gCNO', 'control']):
        __ = df_tcm_diff.loc[idxsli[cond, :, :]]
        sns.barplot(data=__, ax=ax[idx_cond], errorbar='se')
        ax[idx_cond].set_title(f'{cond}')
    fig.suptitle('Difference post-pre row tc')

    # Plot difference b-spline fit to tc's
    fig1, ax1 = plt.subplots(1, 2, sharex=True, sharey=True)
    for idx_cond, cond in enumerate(['gCNO', 'control']):
        __ = df_tcβ_diff.loc[idxsli[cond, :, :]]
        sns.barplot(data=__, ax=ax1[idx_cond], errorbar='se')
        ax1[idx_cond].set_title(f'{cond}')
    fig1.suptitle('Difference post-pre tc (b-spline fit)')

    # Plot entropy row tc's
    fig2, ax2 = plt.subplots(2, 1, sharex=True, sharey=False, figsize=figsize)
    for cond_idx, cond in enumerate(['gCNO', 'control']):
        __ = df_entropy.loc[idxsli[cond, :, :, :]].reset_index()
        # sns.histplot(data=__, x=0, hue='time', ax=ax2[cond_idx], element='step')
        sns.histplot(data=__, x=0, hue='time', ax=ax2[cond_idx])
        ax2[cond_idx].set_title(f'{cond}')
    fig2.suptitle('entropy tc')

    fig3, ax3 = plt.subplots(2, 1, sharex=True, sharey=False, figsize=figsize)
    for idx_cond, cond in enumerate(['gCNO', 'control']):
        __ = df_entropy.loc[idxsli[cond, :, :]]
        __ = __.reset_index()
        sns.histplot(data=__, x=0, hue='time', stat='density', cumulative=True, ax=ax3[idx_cond], common_norm=False, element='step', fill=False, bins=300)
        ax3[idx_cond].set_title(f'{cond}')
    fig3.suptitle('entropy tc (cumulative distribution)')
    

    fig4, ax4 = plt.subplots(figsize=figsize)
    for cond_idx, cond in enumerate(['gCNO', 'control']):
        __ = df_entropy_diff.reset_index()
        sns.histplot(data=df_entropy_diff, x=0, hue='cond', stat='probability', element='step', ax=ax4)
    fig4.suptitle('difference post-pre entropy tc')

    fig5, ax5 = plt.subplots(figsize=figsize)
    # sns.histplot(data=df_kl.reset_index(), x=0, hue='cond', bins=100, element='step', stat='density', ax=ax5)
    cond_color = ['blue', 'orange']
    bins = np.linspace(0, df_kl.max(), 100)[:, 0]
    bwidth = np.diff(bins)[0]
    for cond_idx, cond in enumerate(['gCNO', 'control']):
        __ = df_kl.loc[idxsli[cond, :, :]][0]
        hist_diff, bin_edges = np.histogram(__, density=True, bins=bins)
        # Fit exponential
        # x_ = bin_edges[:-1]
        hist_maxval = np.argwhere(hist_diff)[-1][0] + 1
        x_ = bin_edges[:hist_maxval]
        params, cov = curve_fit(expfun, x_, hist_diff[:hist_maxval], p0=[100, 8])
        # Plot histogram and fit
        ax5.bar(bin_edges[:-1], hist_diff, width=bwidth, label='data', color=cond_color[cond_idx], alpha=0.5)
        ax5.plot(x_, expfun(x_, params[0], params[1]), c=cond_color[cond_idx], label='exp fit')
        # Adornments
        if cond_idx == 0:
            htext = ax5.get_ylim()[1] / 2
        else:
            htext = ax5.get_ylim()[1] / 2 + 1
        ax5.text(2, htext, 'intercept_{}={:0.3f} \n slope={:0.3f}'.format(cond, params[0], params[1]))

    ax5.legend()
    fig5.suptitle('kl divergence pre-post with exponential fit')

    # Plot post against pre entropy
    fig6, ax6 = plt.subplots(1, 2, figsize=figsize, sharex=True, sharey=True)
    for cond_idx, cond in enumerate(['gCNO', 'control']):
        pre_entropy = df_entropy.loc[cond, :, 'pre', :]
        post_entropy = df_entropy.loc[cond, :, 'post', :]
        ax6[cond_idx].scatter(pre_entropy, post_entropy, s=5)
        ax6[cond_idx].set_title(cond)
        ax6[cond_idx].plot([-0.0,2.5], [-0.0,2.5], color='red')
        ax6[cond_idx].set_xlabel('pre')
        ax6[cond_idx].set_ylabel('post')
        ax6[cond_idx].set_box_aspect(aspect=1)


    if os.path.basename(os.getcwd()) != 'data_analysis':
        os.chdir('./data_analysis')

    try:
        fig.savefig(os.getcwd() + f'/images/tc/compare/diff_tc_row.svg', format='svg')
        fig1.savefig(os.getcwd() + f'/images/tc/compare/diff_tc_bspline.svg', format='svg')
        fig2.savefig(os.getcwd() + f'/images/tc/compare/entropy_clu.svg', format='svg')
        fig3.savefig(os.getcwd() + f'/images/tc/compare/entropy_clu_cumulative.svg', format='svg')
        fig4.savefig(os.getcwd() + f'/images/tc/compare/entropy_clu_diff.svg', format='svg')
        fig5.savefig(os.getcwd() + f'/images/tc/compare/kl_divergence_prepost.svg', format='svg')
        fig6.savefig(os.getcwd() + f'/images/tc/compare/entropy_preVSpost.svg', format='svg')
        # fig7.savefig(os.getcwd() + f'/images/tc/compare/kl_expdist_ppc.svg', format='svg')
    except FileNotFoundError:
        print('created dir images/tc')
        os.makedirs(os.getcwd() + '/images/tc/compare')
        fig.savefig(os.getcwd() + f'/images/tc/compare/diff_tc_row.svg', format='svg')
        fig1.savefig(os.getcwd() + f'/images/tc/compare/diff_tc_bspline.svg', format='svg')
        fig2.savefig(os.getcwd() + f'/images/tc/compare/entropy_clu.svg', format='svg')
        fig3.savefig(os.getcwd() + f'/images/tc/compare/entropy_clu_cumulative.svg', format='svg')
        fig4.savefig(os.getcwd() + f'/images/tc/compare/entropy_clu_diff.svg', format='svg')
        fig5.savefig(os.getcwd() + f'/images/tc/compare/kl_divergence_prepost.svg', format='svg')
        fig6.savefig(os.getcwd() + f'/images/tc/compare/entropy_preVSpost.svg', format='svg')
        # fig7.savefig(os.getcwd() + f'/images/tc/compare/kl_expdist_ppc.svg', format='svg')


def fit_bspline(tc_m, wbin, cidsSorted, deg=3):
    ''' Fit a basis spline function of order 3 to average tuning curve data; this
    function is a piecewise polynomial function; the B-splines are the basis
    functions that can be combined to form the function; noise model is positively
    truncated Gaussian.
    Important: scale data to 1
    '''
    # Create design matrix of B-splines
    n_bs = int(wbin // 1.5)
    dmat = dmatrix(f'bs(x, df=n_bs, degree={deg}, include_intercept=True) - 1', {'x':np.arange(wbin)})
    dmat = np.asarray(dmat)
    df_dmat = pd.DataFrame(dmat).reset_index(names='wbins').melt(id_vars='wbins', var_name='splines')

    # Define model (fake obs)
    coords_ = {'obs':np.arange(wbin), 'splines':df_dmat.splines.unique()} 
    with pm.Model(coords=coords_) as spline_model:
        tc_obs = pm.MutableData('tc_obs', np.arange(wbin, dtype=np.float64), dims='obs')
        # tc_obs = pm.MutableData('tc_obs', data_s, dims='obs')
        σ = pm.Exponential('σ', lam=.5)
        α = pm.Exponential('α', lam=.5)
        β = pm.Exponential('β', lam=.5, dims='splines')
        μ = pm.Deterministic('μ', α + at.dot(dmat, β))
        # tc_val = pm.Normal('tc_val', μ, σ, observed=tc_obs)
        tc_val = pm.TruncatedNormal('tc_val', μ, σ, lower=0, observed=tc_obs)
        # trace_spline = pm.sample()

    # Run for each cluster
    save_trace = []
    save_ppc = []
    save_sf = []
    for wsk in range(3):
        save_trace_wsk = []
        save_ppc_wsk = []
        save_sf_wsk = []
        for __, tc_data in enumerate(tc_m[wsk]):
            data_s = np.array(tc_data)
            sf = data_s.std()  # scaling factor
            data_s = data_s / sf

            with spline_model:
                pm.set_data({'tc_obs': data_s})
                # prior_samples = pm.sample_prior_predictive()
                trace_spline = pm.sample()
                post_samples = pm.sample_posterior_predictive(trace_spline)

            save_trace_wsk.append(trace_spline)
            save_ppc_wsk.append(post_samples)
            save_sf_wsk.append(sf)
        save_trace.append(save_trace_wsk)
        save_ppc.append(save_ppc_wsk)
        save_sf.append(save_sf_wsk)
        # pdb.set_trace()

    return save_trace, save_ppc, save_sf, df_dmat


def plot_avg_tc(avg_tc_data,
                wvar_hist_data,
                whisker,
                avg_tc_data_shuffle,
                saveplot=False,
                compare=False,
                **kwargs):
    """ Plot average across (control) recordings average tc; second
    average is across trials.
    Attention: if compare=True then compare control and glyt2 condition
    using all data (not divided in pre vs post)
    """
    # Initialise common parameters
    wsk = int(whisker[-1])
    # Plot
    figsize = (14, 12)          # for single unit tc plot! 
    sns.set_context("poster", font_scale=1)
    # sns.color_palette("deep")
    # sns.color_palette("flare")
    plt.rcParams['axes.facecolor'] = 'lavenderblush'
    plt.rcParams["figure.edgecolor"] = 'black'
    plt.rcParams['axes.linewidth'] = matplotlib.rcParamsDefault['axes.linewidth']

    if compare:
        cond_list = kwargs['cond_list']
        df_tc = []
        df_index = []
        df_hist = []
        df_cond = []
        df_time = []
        df_time_hist = []
        for idx_rec, tc_rec in enumerate(avg_tc_data):
            for idx_time, time in enumerate(['pre', 'post']):
                __ = tc_rec[idx_time][wsk]
                for tc_clu in __:                
                    df_tc.append(tc_clu)
                    df_index.append(idx_rec)
                    df_cond.append(cond_list[idx_rec])
                    df_time.append(time)
                df_hist.append(wvar_hist_data[idx_rec][idx_time][wsk])
                df_time_hist.append(time)
        df_tc = pd.DataFrame(df_tc, index=[df_index, df_cond, df_time])
        df_tc.index.names = ['rec', 'cond', 'time']
        df_hist = pd.DataFrame(df_hist, index=df_time_hist)
        df_hist = df_hist.reset_index().melt(id_vars=['index']).set_index('index')

        # Plot figure
        # aaa = df_tc.reset_index().melt(id_vars=['rec', 'cond', 'time']).set_index(['rec', 'cond'])
        # sns.barplot(data=aaa[aaa.cond=='gCNO'], x='variable', y='value', hue='time', errorbar='se'
        # )
        fig, ax = plt.subplots(1, 2, figsize=figsize, sharey=True)
        ax2 = [ax[0].twinx(), ax[1].twinx()]
        for idx_time, time in enumerate(['pre', 'post']):
            # sns.barplot(data=df_tc.loc[slice(None), 'gCNO', time], errorbar='se', ax=ax[idx_time])
            # sns.barplot(data=avg_tc_data_shuffle.loc[:, time, 2, :, :].reset_index(-1), errorbar='se', x='wbin', y=0, ax=ax[idx_time], saturation=0.1, edgecolor=".5", facecolor=(0, 0, 0, 0), errcolor='.5')
            sns.pointplot(data=df_tc.loc[slice(None), 'gCNO', time], errorbar='se', ax=ax[idx_time])
            sns.pointplot(data=avg_tc_data_shuffle.loc[:, time, 2, :, :].reset_index(-1), errorbar='se', x='wbin', y=0, ax=ax[idx_time], color='grey')
            sns.lineplot(data=df_hist.loc[time], x='variable', y='value', errorbar='se', ax=ax2[idx_time])
            ax[idx_time].set_ylim(23, 35)
            ax[idx_time].set_ylabel('Hz')
            ax2[idx_time].set_ylabel('Freq of time')
            ax2[idx_time].set_title(time)
        fig.tight_layout()
        
    else:
        # avg_tc_data = wvar_tc_m.copy()
        avg_tc_data = avg_tc_data[13:]
        # wvar_hist_data = wvar_hist.copy()
        wvar_hist_data = wvar_hist_data[13:]

        df_tc = []
        df_index = []
        df_hist = []
        for idx_rec, tc_rec in enumerate(avg_tc_data):
            # Tc
            __ = tc_rec[wsk]
            for tc_clu in __:
                df_tc.append(tc_clu)
                df_index.append(idx_rec)
                # Wsk
            df_hist.append(wvar_hist_data[idx_rec][wsk])

        df_tc = pd.DataFrame(df_tc, index=df_index)
        df_hist = pd.DataFrame(df_hist)


        fig, ax = plt.subplots(figsize=figsize)
        ax2 = ax.twinx()
        # sns.barplot(data=df_tc, errorbar='se', ax=ax)
        # sns.barplot(data=avg_tc_data_shuffle.loc[:, 2, :, :].reset_index(-1), x='wbin', y=0, ax=ax, saturation=0.1, edgecolor=".5", facecolor=(0, 0, 0, 0), errcolor='.5')
        sns.pointplot(data=df_tc.melt(), x='variable', y='value', errorbar='se', ax=ax)
        sns.pointplot(data=avg_tc_data_shuffle.loc[:, 2, :, :].reset_index(-1), x='wbin', y=0, ax=ax, color='grey')
        sns.lineplot(data=df_hist.melt(), x='variable', y='value', errorbar='se', ax=ax2 )
        ax.set_ylim(20, 35)
        ax.set_ylabel('Hz')
        ax2.set_ylabel('Freq of time')


    if saveplot:
        if os.path.basename(os.getcwd()) != 'data_analysis':
            os.chdir('./data_analysis')
        if compare:
            fig.savefig(os.getcwd() + f'/images/tc/average_tc_Controls_compare.svg', format='svg')
        else:
            fig.savefig(os.getcwd() + f'/images/tc/average_tc_Controls.svg', format='svg')


def run_tc(drop_rec=[0, 10, 15, 16, 25],
           whisker='whisk2',
           fxb=15,
           whisking=True,
           cgs=2,
           var='angle',
           w_bin=11,
           surr=False,
           runtimes=10,
           pre=None,
           pethdiscard=True,
           save_data=False,
           plot=True,
           plot_avgtc=True,
           cluster=True):
    """ Plot tuning curves; default only good clusters; seaborn.barplot plots
    confidence intervals using bootstrapping.
    """
    # Load metadata, prepare second mask and final good list & get rec
    wskpath, npxpath, aid, cortex_depth, t_drop, clu_idx, pethdiscard_idx, bout_idxs, wmask, conditions, mli_depth = load_meta()
    good_list = np.arange(len(wskpath))
    good_list = good_list[wmask]
    wmask2 = ~np.in1d(np.arange(len(good_list)), np.array(drop_rec))
    good_list = good_list[wmask2]

    ### Checks
    # good_list = good_list[13:]
    
    # pdb.set_trace()
    if save_data:               # either compute and save data...
        if not surr:
            runtimes = 1

        for run in range(runtimes):
            save_wvar_tc = []
            save_wvar_tc_m = []
            save_tot_spkcount = []
            save_cidsSorted = []
            save_wvar_hist = []
            save_path = []
            for idx, rec_idx in enumerate(good_list):
                # Compute firing rate per wvar bin
                wvar_tc, wvar_tc_m, tot_spkcount, cidsSorted, wvar_hist, path = tc_fun(rec_idx, fxb=fxb, whisking=whisking, cgs=cgs, var=var, w_bin=w_bin, surr=surr, pre=pre, pethdiscard=pethdiscard)

                # Save data for each rec
                save_wvar_tc.append(wvar_tc)
                save_wvar_tc_m.append(wvar_tc_m)
                save_tot_spkcount.append(tot_spkcount)
                save_cidsSorted.append(cidsSorted)
                save_wvar_hist.append(wvar_hist)
                save_path.append(path)

            # Save in .pickle
            save_all_tc_nowhisk = [save_wvar_tc, save_wvar_tc_m, save_tot_spkcount, save_cidsSorted, save_wvar_hist, save_path]
            if not surr:
                with open(f'/home/yq18021/Documents/github_gcl/data_analysis/save_data/tc_all_data_nowhisk.pickle', 'wb') as f:
                    pickle.dump(save_all_tc_nowhisk, f)
            else:
                with open(f'/home/yq18021/Documents/github_gcl/data_analysis/save_data/tc_all_data_nowhisk_shuffle{run}.pickle', 'wb') as f:
                    pickle.dump(save_all_tc_nowhisk, f)
                

    else: # load data
        if not var=='phase':
            with open(f'/home/yq18021/Documents/github_gcl/data_analysis/save_data/tc_all_data_nowhisk.pickle', 'rb') as f:
                save_all_tc_nowhisk = pickle.load(f)
        else:
            with open(f'/home/yq18021/Documents/github_gcl/data_analysis/save_data/tc_all_data_nowhisk_phase.pickle', 'rb') as f:
                save_all_tc_nowhisk = pickle.load(f)

        wvar_tc = save_all_tc_nowhisk[0]
        wvar_tc_m = save_all_tc_nowhisk[1]
        tot_spkcount = save_all_tc_nowhisk[2]
        cidsSorted = save_all_tc_nowhisk[3]
        wvar_hist = save_all_tc_nowhisk[4]
        path = save_all_tc_nowhisk[5]

        # Load shuffled data
        all_shuffled_data = []
        for run in range(runtimes):
            with open(f'/home/yq18021/Documents/github_gcl/data_analysis/save_data/tc_all_data_nowhisk_shuffle{run}.pickle', 'rb') as f:
                __  = pickle.load(f)
                all_shuffled_data.append(__)

    if save_data:
        # Fit spline function to tc
        tc_fit = []
        for rec in range(13, 25):
            __ = fit_bspline(wvar_tc_m[rec], w_bin, cidsSorted, deg=3)
            data_fit = {'trace':__[0], 'ppc':__[1], 'sf':__[2], 'df_dmat':__[3]}
            tc_fit.append(data_fit)

        # Save spline fits
        with open(f'/home/yq18021/Documents/github_gcl/data_analysis/save_data/tc_fit_nowhisk_24.pickle', 'wb') as f:
            pickle.dump(tc_fit, f)
    else:                       # load spline fits
        with open(f'/home/yq18021/Documents/github_gcl/data_analysis/save_data/tc_fit_nowhisk_24.pickle', 'rb') as f:
            tc_fit = pickle.load(f)



    if var=='phase':
        if plot:
            for rec in range(13, 25):
                plot_tc(wvar_tc[rec],
                        wvar_tc_m[rec],
                        None,
                        None,
                        tot_spkcount[rec],
                        cidsSorted[rec],
                        wvar_hist[rec],
                        path[rec],
                        None,
                        whisker,
                        pethdiscard=pethdiscard,
                        plot_single=False,
                        clu_id=[],
                        surr=surr,
                        cluster=False)

    else:
        # Pull together shuffled data
        wsk = int(whisker[-1])
        wvar_tc_shuff = all_shuffled_data[0][0].copy()
        for rec in range(13, 25):
            for clu in range(len(wvar_tc_shuff[rec][wsk])):
                for bins in range(w_bin):
                    wvar_tc_shuff[rec][wsk][clu][bins] = np.append(wvar_tc_shuff[rec][wsk][clu][bins], [all_shuffled_data[1][0][rec][wsk][clu][bins], all_shuffled_data[2][0][rec][wsk][clu][bins], all_shuffled_data[3][0][rec][wsk][clu][bins], all_shuffled_data[4][0][rec][2][clu][bins], all_shuffled_data[5][0][rec][2][clu][bins], all_shuffled_data[6][0][rec][2][clu][bins], all_shuffled_data[7][0][rec][2][clu][bins], all_shuffled_data[8][0][rec][2][clu][bins], all_shuffled_data[9][0][rec][2][clu][bins]])
            
        # DataFrame posterior means from b-spline fit
        mean_tc = pd.DataFrame()
        for rec in range(13, 25):
            rec = rec - 13
            for wsk in range(3):
                for clu_idx, clu_data in enumerate(tc_fit[rec]['trace'][wsk]):
                    __ = clu_data.posterior['μ'].mean(dim=['chain', 'draw']).values
                    midx = pd.MultiIndex.from_product([[rec], [wsk], [clu_idx], np.arange(w_bin)], names=['rec', 'wsk', 'clu', 'wbin'] )
                    __ = pd.DataFrame(__, index=midx)
                    mean_tc = pd.concat([mean_tc, __])


        # DataFrame b-spline coefficients
        df_tc = pd.DataFrame()
        for rec in range(13, 25):
            rec = rec - 13
            for wsk in range(3):
                __ = tc_fit[rec]['trace'][wsk]
                for idx, fitclu in enumerate(__):
                    bpline_fit_α = fitclu.posterior['α'].stack(z=['chain', 'draw'])
                    bpline_fit_β = fitclu.posterior['β'].stack(z=['chain', 'draw'])
                    bpline_fit = np.append(bpline_fit_α.mean(dim=['z']), bpline_fit_β.mean(dim=['z']))
                    # Into df
                    midx = pd.MultiIndex.from_product([[rec], [wsk], [idx], np.arange(8)], names=['rec', 'wsk', 'clu', 'features'])
                    df_tc = pd.concat([df_tc, pd.DataFrame(bpline_fit, index=midx)], axis=0)

        # Add parameter level and change condition ('control')
        df_tc['parameters'] = df_tc.index.get_level_values(3).values
        df_tc['parameters'] = np.where(df_tc['parameters']==0, 'α', 'β')
        df_tc.set_index('parameters', append=True, inplace=True)
        df_tc.index = df_tc.index.swaplevel(-2, -1)

        ####### Dimensionality reduction on 'control' ############
        # Use spline coefficients (β parameters) for classification
        df_tc_β = df_tc.loc[(slice(None), int(whisker[-1]), slice(None), 'β', slice(None))]

        # t-SNE on b-splines coefficients
        embed = []
        df_tc_β_unstack = df_tc_β.unstack()
        for i in range(2, 4):
            embed.append(TSNE(n_components=i, perplexity=10, learning_rate='auto', init='random').fit_transform(df_tc_β_unstack))

        # K-means of b-spline data
        kmclusters = []
        kmclusters_inertia = []  # sum of squared distance of samples from cluster centres
        for i in range(1, 11):
            kmclusters.append(KMeans(n_clusters=i, n_init=10, random_state=3).fit(df_tc_β_unstack))
            kmclusters_inertia.append(kmclusters[i-1].inertia_)

        # Remap cluster id
        remap = [{0:0, 2:1, 1:2}, {1:0, 0:1, 2:2, 3:3}, {0:0, 2:1, 4:2, 3:3, 1:4}, {2:0, 1:1, 5:2, 0:3, 4:4, 3:5}, {6:0, 3:1, 5:2, 4:3, 2:4, 1:5, 0:6}, {0:0, 2:1, 6:2, 7:3, 4:4, 3:5, 1:6, 5:7}]
        df_kmclu_labels = pd.DataFrame()
        for idx, ngroups in enumerate(range(2, 8)):  # index starts from 0
            # df_labels = pd.DataFrame(kmclusters_tsne3[ngroups].labels_, index=__.index)
            df_labels = pd.DataFrame(kmclusters[ngroups].labels_, index=df_tc_β_unstack.index)
            df_labels[0] = df_labels[0].map(remap[idx])
            df_labels = pd.concat([df_labels], keys=[f'{ngroups}'], names=['ngroups'])
            df_kmclu_labels = pd.concat([df_kmclu_labels, df_labels])

        # DataFrame of row tc with k-means labels
        mean_tc_row = pd.DataFrame()
        for rec in range(13, 25):
            for wsk in range(3):
                for clu_idx, clu_data in enumerate(wvar_tc[rec][wsk]):
                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore', category=RuntimeWarning)
                        __ = np.array(tuple(map(np.nanmean, clu_data)))
                    __ = np.nan_to_num(__)
                    midx = pd.MultiIndex.from_product([[rec], [wsk], [clu_idx], np.arange(w_bin)], names=['rec', 'wsk', 'clu', 'wbin'] )
                    __ = pd.DataFrame(__, index=midx)
                    mean_tc_row = pd.concat([mean_tc_row, __])

        label_mean_tc_row = pd.DataFrame()
        save_mean_groups = []
        for ngroups in range(2, 8):
            __ = mean_tc_row.loc[slice(None), int(whisker[-1]), slice(None), slice(None)].copy()
            __ = __.unstack()
            kgroup_idx = tuple(df_kmclu_labels.loc[f'{ngroups}'].values[:, 0])
            __['kgroup_idx'] = kgroup_idx
            __.set_index('kgroup_idx', append=True, inplace=True)
            # __ = __.groupby('kgroup_idx').mean()
            __['ngroups'] = ngroups
            __.set_index('ngroups', append=True, inplace=True)
            # __ = __.swaplevel()
            __ = __.reorder_levels(['ngroups', 'rec', 'clu', 'kgroup_idx'])
            __ = __.droplevel(axis=1, level=0)
            save_mean_groups.append(__.groupby(by='kgroup_idx', group_keys=False).mean().mean(axis=1))
            __ = __.groupby(by='kgroup_idx', group_keys=False).apply(lambda x: x / x.mean().mean())
            
            label_mean_tc_row = pd.concat([label_mean_tc_row, __])


        # DataFrame of shuffled data with same k-means labels
        mean_tc_shuffle = pd.DataFrame()
        for rec in range(13, 25):
            for wsk in range(3):
                for clu_idx, clu_data in enumerate(wvar_tc_shuff[rec][wsk]):
                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore', category=RuntimeWarning)
                        __ = np.array(tuple(map(np.nanmean, clu_data)))
                    __ = np.nan_to_num(__)
                    midx = pd.MultiIndex.from_product([[rec], [wsk], [clu_idx], np.arange(w_bin)], names=['rec', 'wsk', 'clu', 'wbin'] )
                    __ = pd.DataFrame(__, index=midx)
                    mean_tc_shuffle = pd.concat([mean_tc_shuffle, __])

        label_mean_tc_shuffle = pd.DataFrame()
        for ngroups in range(2, 8):
            __ = mean_tc_shuffle.loc[slice(None), int(whisker[-1]), slice(None), slice(None)].copy()
            __ = __.unstack()
            kgroup_idx = tuple(df_kmclu_labels.loc[f'{ngroups}'].values[:, 0])
            __['kgroup_idx'] = kgroup_idx
            __.set_index('kgroup_idx', append=True, inplace=True)
            # __ = __.groupby('kgroup_idx').mean()
            __['ngroups'] = ngroups
            __.set_index('ngroups', append=True, inplace=True)
            # __ = __.swaplevel()
            __ = __.reorder_levels(['ngroups', 'rec', 'clu', 'kgroup_idx'])
            __ = __.droplevel(axis=1, level=0)
            for kgroups in __.index.get_level_values('kgroup_idx').unique():
                idx_slice = pd.IndexSlice
                sli = __.loc[idx_slice[:, :, :, kgroups], :]
                sli = sli / save_mean_groups[ngroups - 2].loc[kgroups]
                __.loc[idx_slice[:, :, :, kgroups], :] = sli
            label_mean_tc_shuffle = pd.concat([label_mean_tc_shuffle, __])

        # Sort whisker density in df
        idxsli = pd.IndexSlice
        df_wvar_hist = pd.DataFrame()
        df_wvar_hist = []
        midx = df_kmclu_labels.index
        for ngroup in midx.get_level_values(0).unique():
            for rec_idx, rec in enumerate(midx.get_level_values(1).unique()):
                rec_idx = rec_idx + 13
                # for time_idx, time in enumerate(midx.get_level_values(2).unique()):
                n_clu = df_kmclu_labels.loc[idxsli[ngroup, rec, :]].size
                for clu in range(n_clu):
                    __ = wvar_hist[rec_idx][wsk]
                    df_wvar_hist.append(__)

        df_wvar_hist = pd.DataFrame(df_wvar_hist)
        df_wvar_hist.index = midx

        # Plot and save average tc
        if plot_avgtc:
            plot_avg_tc(wvar_tc_m, wvar_hist, whisker, mean_tc_shuffle, saveplot=False)

        if not cluster:         # plot tc + shuffle data without cluster assignment
            if plot:
                for rec in range(13, 25):
                    rec_bspline = rec - 13
                    plot_tc(wvar_tc[rec],
                            wvar_tc_m[rec],
                            wvar_tc_shuff[rec],
                            tc_fit[rec_bspline],
                            tot_spkcount[rec],
                            cidsSorted[rec],
                            wvar_hist[rec],
                            path[rec],
                            df_kmclu_labels.loc[(slice(None), rec_bspline, slice(None))],
                            whisker,
                            pethdiscard=pethdiscard,
                            plot_single=False,
                            clu_id=[],
                            surr=surr,
                            cluster=cluster)

        else:
            # Plot average tuning curves per cluster
            plot_tc_clusters(label_mean_tc_row, label_mean_tc_shuffle, df_kmclu_labels, df_wvar_hist)

            # Plot all data clustered for one recording (ad hoc, not great!!!)        
            gcolors = df_kmclu_labels[0].map({0:'blue', 1:'red', 2:'green', 3:'violet', 4:'yellow', 5:'pink', 6:'brown', 7:'black', 8:'cyan'})
            fig1 = plt.figure(figsize=(14, 12))
            ax1 = fig1.add_subplot(projection='3d')
            ax1.scatter(embed[1][:, 0], embed[1][:, 1], embed[1][:, 2], c=gcolors.loc['4'].values)
            ax1.set_title('row_bspline')

            fig2, ax2 = plt.subplots(figsize=(14, 12))
            ax2.plot(kmclusters_inertia)
            ax2.set_title('row_bspline')

            if os.path.basename(os.getcwd()) != 'data_analysis':
                os.chdir('./data_analysis')

            if not surr:
                surr = 'nosurr'
            try:
                fig1.savefig(os.getcwd() + f'/images/tc/alltc_clustered.svg', format='svg')
                fig2.savefig(os.getcwd() + f'/images/tc/sumsquareddistance.svg', format='svg')
            except FileNotFoundError:
                print(f'created dir images/tc')
                fig1.savefig(os.getcwd() + f'/images/tc/alltc_clustered.svg', format='svg')
                fig2.savefig(os.getcwd() + f'/images/tc/sumsquareddistance.svg', format='svg')


def run_tc_compare(drop_rec=[0, 10, 15, 16, 25],
                   whisker='whisk2',
                   fxb=15,
                   whisking=True,
                   cgs=2,
                   var='angle',
                   w_bin=11,
                   surr=False,
                   runtimes=10,
                   pre=None,
                   pethdiscard=True,
                   save_data=False,
                   plot=True,
                   cluster=True):
    """ Plot tuning curves; default only good clusters; seaborn.barplot plots
    confidence intervals using bootstrapping. Compare pre- vs post-drop periods
    """
    # Load metadata, prepare second mask and final good list & get rec
    wskpath, npxpath, aid, cortex_depth, t_drop, clu_idx, pethdiscard_idx, bout_idxs, wmask, conditions, mli_depth = load_meta()
    good_list = np.arange(len(wskpath))
    good_list = good_list[wmask]
    wmask2 = ~np.in1d(np.arange(len(good_list)), np.array(drop_rec))
    good_list = good_list[wmask2]

    if save_data:               # either compute and save data...
        if not surr:
            runtimes = 1

        for run_ in range(runtimes):
            save_wvar_tc = []
            save_wvar_tc_m = []
            save_tot_spkcount = []
            save_cidsSorted = []
            save_wvar_hist = []
            save_path = []
            for idx, rec_idx in enumerate(good_list):
                wvar_tc = []
                wvar_tc_m = []
                tot_spkcount = []
                cidsSorted = []
                wvar_hist = []
                path = []
                for time in [True, False]:
                    # Compute firing rate per wvar bin (pre and post t_drop)
                    pre = time
                    wvar_tc, wvar_tc_m, tot_spkcount, cidsSorted, wvar_hist, path = tc_fun(rec_idx, fxb=fxb, whisking=whisking, cgs=cgs, var=var, w_bin=w_bin, surr=surr, pre=pre, pethdiscard=pethdiscard)
                    wvar_tc.append(wvar_tc)
                    wvar_tc_m.append(wvar_tc_m)
                    tot_spkcount.append(tot_spkcount)
                    cidsSorted.append(cidsSorted)
                    wvar_hist.append(wvar_hist)
                    path.append(path)

                # Save data for each rec
                save_wvar_tc.append(wvar_tc)
                save_wvar_tc_m.append(wvar_tc_m)
                save_tot_spkcount.append(tot_spkcount)
                save_cidsSorted.append(cidsSorted)
                save_wvar_hist.append(wvar_hist)
                save_path.append(path)

            # Save in .pickle
            save_all_tc_nowhisk = [save_wvar_tc, save_wvar_tc_m, save_tot_spkcount, save_cidsSorted, save_wvar_hist, save_path]
            if not surr:
                with open(f'/home/yq18021/Documents/github_gcl/data_analysis/save_data/tc_all_data_nowhisk_compare.pickle', 'wb') as f:
                    pickle.dump(save_all_tc_nowhisk, f)
            else:
                with open(f'/home/yq18021/Documents/github_gcl/data_analysis/save_data/tc_all_data_nowhisk_compare_shuffle{run_}.pickle', 'wb') as f:
                    pickle.dump(save_all_tc_nowhisk, f)
                

    else: # load data
        if not var=='phase':
            with open(f'/home/yq18021/Documents/github_gcl/data_analysis/save_data/tc_all_data_nowhisk_compare.pickle', 'rb') as f:
                save_all_tc_nowhisk = pickle.load(f)
        else:
            with open(f'/home/yq18021/Documents/github_gcl/data_analysis/save_data/tc_all_data_nowhisk_compare_phase.pickle', 'rb') as f:
                save_all_tc_nowhisk = pickle.load(f)

        wvar_tc = save_all_tc_nowhisk[0]
        wvar_tc_m = save_all_tc_nowhisk[1]
        tot_spkcount = save_all_tc_nowhisk[2]
        cidsSorted = save_all_tc_nowhisk[3]
        wvar_hist = save_all_tc_nowhisk[4]
        path = save_all_tc_nowhisk[5]

        # Load shuffled data
        all_shuffled_data = []
        for run_ in range(runtimes):
            with open(f'/home/yq18021/Documents/github_gcl/data_analysis/save_data/tc_all_data_nowhisk_compare_shuffle{run_}.pickle', 'rb') as f:
                __  = pickle.load(f)
                all_shuffled_data.append(__)                


    if save_data:
        # Fit spline function to tc
        tc_fit = []
        for rec in range(25):
            __ = fit_bspline(wvar_tc_m[rec], w_bin, cidsSorted, deg=3)
            data_fit = {'trace':__[0], 'ppc':__[1], 'sf':__[2], 'df_dmat':__[3]}
            tc_fit.append(data_fit)

        # Save spline fits
        with open(f'/home/yq18021/Documents/github_gcl/data_analysis/save_data/tc_fit_nowhisk_24.pickle', 'wb') as f:
            pickle.dump(tc_fit, f)
    else:                       # load spline fits
        tc_fit = []
        for rec in range(25):
            with open(f'/home/yq18021/Documents/github_gcl/data_analysis/save_data/tc_fit_nowhisk_compare_bspline{rec}.pickle', 'rb') as f:
                __ = pickle.load(f)
                tc_fit.append(__)



    if var=='phase':
        if plot:
            for rec in range(25):
                plot_tc_compare(wvar_tc[rec],
                                wvar_tc_m[rec],
                                None,
                                None,
                                tot_spkcount[rec],
                                cidsSorted[rec],
                                wvar_hist[rec],
                                path[rec],
                                None,
                                whisker,
                                pethdiscard=pethdiscard,
                                plot_single=False,
                                clu_id=[],
                                surr=surr,
                                cluster=False)

    else:
        # Pull together shuffled data
        wsk = int(whisker[-1])
        wvar_tc_shuff = all_shuffled_data[0][0].copy()
        for rec in range(25):
            for time, __ in enumerate(['pre', 'post']):
                for clu in range(len(wvar_tc_shuff[rec][time][wsk])):
                    for bins in range(w_bin):
                        wvar_tc_shuff[rec][time][wsk][clu][bins] = np.append(wvar_tc_shuff[rec][time][wsk][clu][bins], [all_shuffled_data[1][0][rec][time][wsk][clu][bins], all_shuffled_data[2][0][rec][time][wsk][clu][bins], all_shuffled_data[3][0][rec][time][wsk][clu][bins], all_shuffled_data[4][0][rec][time][wsk][clu][bins], all_shuffled_data[5][0][rec][time][wsk][clu][bins], all_shuffled_data[6][0][rec][time][wsk][clu][bins], all_shuffled_data[7][0][rec][time][wsk][clu][bins], all_shuffled_data[8][0][rec][time][wsk][clu][bins], all_shuffled_data[9][0][rec][time][wsk][clu][bins], all_shuffled_data[10][0][rec][time][wsk][clu][bins], all_shuffled_data[11][0][rec][time][wsk][clu][bins], all_shuffled_data[12][0][rec][time][wsk][clu][bins], all_shuffled_data[13][0][rec][time][wsk][clu][bins], all_shuffled_data[14][0][rec][time][wsk][clu][bins], all_shuffled_data[15][0][rec][time][wsk][clu][bins], all_shuffled_data[16][0][rec][time][wsk][clu][bins], all_shuffled_data[17][0][rec][time][wsk][clu][bins], all_shuffled_data[18][0][rec][time][wsk][clu][bins], all_shuffled_data[19][0][rec][time][wsk][clu][bins]])
            
        # pdb.set_trace()
        # DataFrame posterior means from b-spline fit
        mean_tc = pd.DataFrame()
        for rec in range(25):
            for wsk in range(3):
                for time_idx, time in enumerate(['pre', 'post']):
                    for clu_idx, clu_data in enumerate(tc_fit[rec][f'trace_{time}'][wsk]):
                        __ = clu_data.posterior['μ'].mean(dim=['chain', 'draw']).values
                        midx = pd.MultiIndex.from_product([[rec], [time], [wsk], [clu_idx], np.arange(w_bin)], names=['rec', 'time', 'wsk', 'clu', 'wbin'] )
                        __ = pd.DataFrame(__, index=midx)
                        mean_tc = pd.concat([mean_tc, __])


        # DataFrame b-spline coefficients
        df_tc = pd.DataFrame()
        for rec in range(25):
            for wsk in range(3):
                for time_idx, time in enumerate(['pre', 'post']):
                    __ = tc_fit[rec][f'trace_{time}'][wsk]
                    for idx, fitclu in enumerate(__):
                        bpline_fit_α = fitclu.posterior['α'].stack(z=['chain', 'draw'])
                        bpline_fit_β = fitclu.posterior['β'].stack(z=['chain', 'draw'])
                        bpline_fit = np.append(bpline_fit_α.mean(dim=['z']), bpline_fit_β.mean(dim=['z']))
                        # Into df
                        midx = pd.MultiIndex.from_product([[rec], [time], [wsk], [idx], np.arange(8)], names=['rec', 'time', 'wsk', 'clu', 'features'])
                        df_tc = pd.concat([df_tc, pd.DataFrame(bpline_fit, index=midx)], axis=0)

        # Add parameter level and change condition ('control')
        df_tc['parameters'] = df_tc.index.get_level_values(4).values
        df_tc['parameters'] = np.where(df_tc['parameters']==0, 'α', 'β')
        df_tc.set_index('parameters', append=True, inplace=True)
        df_tc.index = df_tc.index.swaplevel(-2, -1)

        ####### Dimensionality reduction on 'control' ############
        # Use spline coefficients (β parameters) for classification
        df_tc_β = df_tc.loc[(slice(None), slice(None), int(whisker[-1]), slice(None), 'β', slice(None))]

        # t-SNE on b-splines coefficients
        idxsli = pd.IndexSlice
        df_tc_β_unstack = df_tc_β.unstack()
        embed = []
        for i in range(2, 4):
            __ = {}
            # for time_idx, time in enumerate(['pre', 'post']):
                # __[time] = TSNE(n_components=i, perplexity=10, learning_rate='auto', init='random').fit_transform(df_tc_β_unstack.loc[idxsli[:, time, :], :])
            __ = TSNE(n_components=i, perplexity=10, learning_rate='auto', init='random').fit_transform(df_tc_β_unstack)

            embed.append(__)

        # Kmeans on pre and post data
        kmclusters_inertia = []
        # df_kmclu_labels = pd.DataFrame()
        df_kmclu_labels = []
        remap = [{2:0, 1:1, 0:2}, {3:0, 1:1, 2:2, 0:3}, {1:0, 2:1, 3:2, 0:3, 4:4}, {2:0, 3:1, 4:2, 5:3, 1:4, 0:5}, {0:0, 6:1, 4:2, 5:3, 3:4, 2:5, 1:6}, {6:0, 7:1, 3:2, 1:3, 2:4, 5:5, 4:6, 0:7}]
        for ngroups in range(3, 9):
            km__ = KMeans(n_clusters=ngroups, n_init=10, random_state=3).fit(df_tc_β_unstack)
            kmclusters_inertia.append(km__.inertia_)
            __ = pd.DataFrame(km__.labels_, index=df_tc_β_unstack.index)
            __ = pd.concat([__], keys=[f'{ngroups}'], names=['ngroups'])
            __[0] = __[0].map(remap[ngroups - 3] )
            # df_kmclu_labels = pd.concat([df_kmclu_labels, __])
            df_kmclu_labels.append(__)
        df_kmclu_labels = pd.concat(df_kmclu_labels)

        # Kmeans on pre data
        kmclusters_inertia_pre = []
        # df_kmclu_labels_pre = pd.DataFrame()
        df_kmclu_labels_pre = []
        for ngroups in range(3, 9):
            sli = df_tc_β_unstack.loc[idxsli[:, 'pre', :]]
            km__ = KMeans(n_clusters=ngroups, n_init=10, random_state=3).fit(sli)
            kmclusters_inertia_pre.append(km__.inertia_)
            __ = pd.DataFrame(km__.labels_, index=sli.index)
            __ = pd.concat([__], keys=[f'{ngroups}'], names=['ngroups'])
            __[0] = __[0].map(remap[ngroups - 3] )
            # df_kmclu_labels_pre = pd.concat([df_kmclu_labels_pre, __])
            df_kmclu_labels_pre.append(__)
        df_kmclu_labels_pre = pd.concat(df_kmclu_labels_pre)

        # DataFrame of row tc with k-means labels
        # mean_tc_row = pd.DataFrame()
        mean_tc_row = []
        for rec in range(25):
            for time_idx, time in enumerate(['pre', 'post']):
                for wsk in range(3):
                    for clu_idx, clu_data in enumerate(wvar_tc[rec][time_idx][wsk]):
                        with warnings.catch_warnings():
                            warnings.simplefilter('ignore', category=RuntimeWarning)
                            __ = np.array(tuple(map(np.nanmean, clu_data)))
                            __ = np.nan_to_num(__)
                        midx = pd.MultiIndex.from_product([[rec], [time], [wsk], [clu_idx], np.arange(w_bin)], names=['rec', 'time', 'wsk', 'clu', 'wbin'] )
                        __ = pd.DataFrame(__, index=midx)
                        # mean_tc_row = pd.concat([mean_tc_row, __])
                        mean_tc_row.append(__)
        mean_tc_row = pd.concat(mean_tc_row)

        # label_mean_tc_row = pd.DataFrame()
        label_mean_tc_row = []
        save_mean_groups = {}
        for ngroups in df_kmclu_labels.index.get_level_values(0).unique():
            for time_idx, time in enumerate(['pre', 'post']):
                __ = mean_tc_row.loc[idxsli[:, time, wsk, :, :], :].copy()
                __ = __.droplevel(wsk)
                __ = __.unstack()
                # kgroup_idx = tuple(df_kmclu_labels.loc[idxsli[f'{ngroups}']].values[:, 0])
                kgroup_idx = tuple(df_kmclu_labels.loc[idxsli[ngroups, :, time, :]].values[:, 0])
                __['kgroup_idx'] = kgroup_idx
                __.set_index('kgroup_idx', append=True, inplace=True)
                # __ = __.groupby('kgroup_idx').mean()
                __['ngroups'] = ngroups
                __.set_index('ngroups', append=True, inplace=True)
                # __ = __.swaplevel()
                __ = __.reorder_levels(['ngroups', 'rec', 'time', 'clu', 'kgroup_idx'])
                __ = __.droplevel(axis=1, level=0)
                # save_mean_groups.append(__.groupby(by='kgroup_idx', group_keys=False).mean().mean(axis=1))
                save_mean_groups[f'{ngroups}_{time}'] = __.groupby(by='kgroup_idx', group_keys=False).mean().mean(axis=1)
                __ = __.groupby(by='kgroup_idx', group_keys=False).apply(lambda x: x / x.mean().mean()) # ATTENTION: scaling by mean of km cluster
                # __ = __.div(__.mean(axis=1), axis=0).sum(axis=1)
            
                label_mean_tc_row.append(__)
        label_mean_tc_row = pd.concat(label_mean_tc_row)

        # Same but use only pre-drop tc group assignment
        label_mean_tc_row_pre = []
        for ngroups in df_kmclu_labels.index.get_level_values(0).unique():
            for time_idx, time in enumerate(['pre', 'post']):
                __ = mean_tc_row.loc[idxsli[:, time, wsk, :, :], :].copy()
                __ = __.droplevel(wsk)
                __ = __.unstack()
                kgroup_idx = tuple(df_kmclu_labels.loc[idxsli[ngroups, :, 'pre', :]].values[:, 0])
                __['kgroup_idx'] = kgroup_idx
                __.set_index('kgroup_idx', append=True, inplace=True)
                # __ = __.groupby('kgroup_idx').mean()
                __['ngroups'] = ngroups
                __.set_index('ngroups', append=True, inplace=True)
                # __ = __.swaplevel()
                __ = __.reorder_levels(['ngroups', 'rec', 'time', 'clu', 'kgroup_idx'])
                __ = __.droplevel(axis=1, level=0)
                __ = __.groupby(by='kgroup_idx', group_keys=False).apply(lambda x: x / x.mean().mean())
                # __ = __.div(__.mean(axis=1), axis=0).sum(axis=1)
            
                # label_mean_tc_row_pre = pd.concat([label_mean_tc_row_pre, __])
                label_mean_tc_row_pre.append(__)
        label_mean_tc_row_pre = pd.concat(label_mean_tc_row_pre)

        # DataFrame of shuffled data with same k-means labels
        mean_tc_shuffle = pd.DataFrame()
        for rec in range(25):
            for time_idx, time in enumerate(['pre', 'post']):
                for wsk in range(3):
                    for clu_idx, clu_data in enumerate(wvar_tc_shuff[rec][time_idx][wsk]):
                        with warnings.catch_warnings():
                            warnings.simplefilter('ignore', category=RuntimeWarning)
                            __ = np.array(tuple(map(np.nanmean, clu_data)))
                            __ = np.nan_to_num(__)
                        midx = pd.MultiIndex.from_product([[rec], [time], [wsk], [clu_idx], np.arange(w_bin)], names=['rec', 'time', 'wsk', 'clu', 'wbin'] )
                        __ = pd.DataFrame(__, index=midx)
                        mean_tc_shuffle = pd.concat([mean_tc_shuffle, __])

        label_mean_tc_shuffle = pd.DataFrame()
        for ngroups in df_kmclu_labels.index.get_level_values(0).unique():
            for time_idx, time in enumerate(['pre', 'post']):
                __ = mean_tc_shuffle.loc[idxsli[:, time, wsk, :, :], :].copy()
                __ = __.droplevel(wsk)
                __ = __.unstack()
                kgroup_idx = tuple(df_kmclu_labels.loc[idxsli[ngroups, :, time, :]].values[:, 0])
                __['kgroup_idx'] = kgroup_idx
                __.set_index('kgroup_idx', append=True, inplace=True)
                # __ = __.groupby('kgroup_idx').mean()
                __['ngroups'] = ngroups
                __.set_index('ngroups', append=True, inplace=True)
                # __ = __.swaplevel()
                __ = __.reorder_levels(['ngroups', 'rec', 'time', 'clu', 'kgroup_idx'])
                __ = __.droplevel(axis=1, level=0)
                for kgroups in __.index.get_level_values('kgroup_idx').unique():
                    sli = __.loc[idxsli[:, :, :, :, kgroups], :]
                    # sli = sli / save_mean_groups[int(ngroups) - 1].loc[kgroups]
                    sli = sli / save_mean_groups[f'{ngroups}_{time}'].loc[kgroups]
                    __.loc[idxsli[:, :, :, :, kgroups], :] = sli
                label_mean_tc_shuffle = pd.concat([label_mean_tc_shuffle, __])
               
        # pdb.set_trace()

        # Add condition
        map_cond = {}
        for id_rec, rec in enumerate(path):
            rec = os.path.basename(rec[0][:-1])
            if ('GLYT2' in rec) & ('CNO' in rec):
                map_cond[id_rec] = 'gCNO'
            else:
                map_cond[id_rec] = 'control'

        # kmclusters['cond'] =  kmclusters.index.get_level_values(1)
        # kmclusters['cond'] = kmclusters['cond'].map(map_cond)
        # kmclusters.set_index('cond', append=True, inplace=True)
        # kmclusters = kmclusters.reorder_levels(['ngroups', 'cond', 'rec', 'time', 'clu'])
        df_kmclu_labels['cond'] =  df_kmclu_labels.index.get_level_values(1)
        df_kmclu_labels['cond'] = df_kmclu_labels['cond'].map(map_cond)
        df_kmclu_labels.set_index('cond', append=True, inplace=True)
        df_kmclu_labels = df_kmclu_labels.reorder_levels(['ngroups', 'cond', 'rec', 'time', 'clu'])
        df_kmclu_labels_pre['cond'] =  df_kmclu_labels_pre.index.get_level_values(1)
        df_kmclu_labels_pre['cond'] = df_kmclu_labels_pre['cond'].map(map_cond)
        df_kmclu_labels_pre.set_index('cond', append=True, inplace=True)
        df_kmclu_labels_pre = df_kmclu_labels_pre.reorder_levels(['ngroups', 'cond', 'rec', 'clu'])

        label_mean_tc_row_pre['cond'] = label_mean_tc_row_pre.index.get_level_values(1)
        label_mean_tc_row_pre['cond'] = label_mean_tc_row_pre['cond'].map(map_cond)
        label_mean_tc_row_pre.set_index('cond', append=True, inplace=True)
        label_mean_tc_row_pre = label_mean_tc_row_pre.reorder_levels(['ngroups', 'cond', 'rec', 'time', 'clu', 'kgroup_idx'])

        if save_data:
            with open(f'/home/yq18021/Documents/github_gcl/data_analysis/save_data/km_clustered_data.pickle', 'wb') as f:
                pickle.dump(df_kmclu_labels, f)
            with open(f'/home/yq18021/Documents/github_gcl/data_analysis/save_data/km_clustered_data_pre.pickle', 'wb') as f:
                pickle.dump(df_kmclu_labels_pre, f)



        # Compute cluster distribution
        clu_distribution = df_kmclu_labels.groupby(by=['ngroups', 'cond', 'time']).apply(lambda x: np.histogram(x, bins=x[0].unique().size)[0])
        # pdb.set_trace()

        # Sort whisker density in df
        df_wvar_hist = pd.DataFrame()
        df_wvar_hist = []
        midx = df_kmclu_labels.index
        for ngroup in midx.get_level_values(0).unique():
            for cond in midx.get_level_values(1).unique():
                for rec_idx, rec in enumerate(midx.get_level_values(2).unique()):
                    for time_idx, time in enumerate(midx.get_level_values(3).unique()):
                        n_clu = df_kmclu_labels.loc[idxsli[ngroup, cond, rec, time, :]].size
                        for clu in range(n_clu):
                            __ = wvar_hist[rec_idx][time_idx][wsk]
                            df_wvar_hist.append(__)

        df_wvar_hist = pd.DataFrame(df_wvar_hist)
        df_wvar_hist.index = midx
        
        # Plot and save average tc
        if plot_avgtc:
            plot_avg_tc(wvar_tc_m, wvar_hist, whisker, mean_tc_shuffle, saveplot=False, compare=True, cond_list=conditions)

        if not cluster:         # plot tc + shuffle data without cluster assignment
            if plot:
                for rec in range(25):
                    rec=5
                    plot_tc_compare(wvar_tc[rec],
                                    wvar_tc_m[rec],
                                    wvar_tc_shuff[rec],
                                    tc_fit[rec],
                                    tot_spkcount[rec],
                                    cidsSorted[rec],
                                    wvar_hist[rec],
                                    path[rec],
                                    df_kmclu_labels.loc[idxsli[:, :, rec, :, :]],
                                    whisker,
                                    pethdiscard=pethdiscard,
                                    plot_single=False,
                                    clu_id=[],
                                    surr=surr,
                                    cluster=cluster)
            # pdb.set_trace()

        else:
            # pdb.set_trace()
            plot_tc_clusters_compare(label_mean_tc_row, label_mean_tc_shuffle, df_kmclu_labels, clu_distribution)

            entropy_tc_compare(wvar_tc_m, df_tc_β_unstack, df_kmclu_labels, wsk=2)

            # Plot average tuning curves per cluster (based on pre labels)
            plot_tc_clusters_compare_pre(label_mean_tc_row_pre, df_kmclu_labels, df_wvar_hist)

            # # Confusion matrix
            # plot_confusion(df_kmclu_labels, n_kmgroups='8')


if __name__ == '__main__':
    """ Run module and plot graphs from python shell; pre can be either True,
    False or None; use latter to use the whole npx and whisk recordings
    """
    drop_rec = [0, 10, 15, 16, 25]
    whisker = 'whisk2'
    fxb = 10
    whisking = False
    cgs = 2
    var = 'angle'
    w_bin = 11
    surr = False
    pre = None                  # Use to get whole rec
    # pre = 'PrePost'               # Get both pre and post data
    if not pre:
        runtimes = 10
    else:
        runtimes = 20
    pethdiscard = True          # Discard clusters based on blank PETH
    save_data = False            # Compute and save data, otherwise load pre-existing ones
    plot = True
    cluster = True             # Plot tc and their clusters (color-code)
    plot_avgtc = True          # Plot average tc for control conditions

    if pre == None:
        run_tc(drop_rec=drop_rec, whisker=whisker, fxb=fxb, whisking=whisking, cgs=cgs, var=var, w_bin=w_bin, surr=surr, runtimes=runtimes, pre=pre, pethdiscard=pethdiscard,save_data=save_data, plot=plot,plot_avgtc=plot_avgtc, cluster=cluster)
    else:
        run_tc_compare(drop_rec=drop_rec, whisker=whisker, fxb=fxb, whisking=whisking, cgs=cgs, var=var, w_bin=w_bin, surr=surr, runtimes=runtimes, pre=pre, pethdiscard=pethdiscard,save_data=save_data, plot=plot, cluster=cluster)
