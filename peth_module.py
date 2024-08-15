"""
Script for computing/analysis and plotting peri-event-time histogram.
Contains the following:

align_spktrain function; puts together aligned whisker variable and spike
counts (binsize equal to frame) of each cluster; it returns data for
plot_psth function.

spike_train function: used by align_spktrain to compute spike count for
each cluster; algn_spktrn_m is avg spike count across trials with time bin
equal frame length in ms.

align_to_whisking function: used by align_spktrain to align spike counts
and whisker variable to whiker onset (default takes window from -2 up
to 3 sec).

plot_peth function: plots peri-stimulus-time histogram using output from
of align_spktrain.

peth_bmt function: computes Brunner Munzel test (alternative to Mann-Whitney
U test - no asumption of equal variance) to compare baseline neural activity
to subsequent bins; used by plot_peth.

load_data fucntion: combines output of load_npx.py and whisk_analysis.py to
prepare data for subsequent functions.

Order: load whiskd and spk data with load_data; get spike train binned to match
frame length (default take only good clusters); align whisker variable and
spk_trains; use aligned data to plot peth; optional, run Brunner Munzel test
to check significance.

Based on the following modules:
Load npx data (load_npx.py) as module spikeStruct with following variables:
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
 

Load and analyse whisker data (whisk_analysis.py) with following functions:
     - load_data         returns dataframe of whisker position and bodyparts labels
     - calc_angle        returns angle_trace, calculate raw angle using arctan(sin(θ)/cos(θ)); 
                         θ=0 depends on angle of highspeedcamera
     - calc_phase        return analytic signal and phase of angle_trace based on Hilbert
                         transform; raw angle is first bandpass filtered (default 6-30 Hz)
                         then negative freq are removed, which is equivalent to phase-shifting
                         the time-domain signal by 90 degres at all frequences and add this as 
                         imaginary part to obtain analytic signal; whisker phase is the angle
                         of the analytic signal 
     - get_slow_var      return output of function handle, tops and bottoms; tops and bottoms are
                         frames in which phase goes from <0 to >0 and >= π/2 to <= π/2,
                         respectively (turning points); tops and bottoms times are used to chop
                         whisker trace and calculate function handle within segments (e.g. angle 
                         setpoint and amplitude); each output is associated to the idx of middle 
                         frame and sorted accordingly; sorted values are linearly interpolated. 
     - loadWhiskerData   returns class whiskdata with attributes of type df, array or list:
                         - df            dlc dataframe without top level
                         - frame         array of frames
                         - nframe        number of frames
                         - time          array of times in sec
                         - nwhisk        number of whiskers
                         ** list where each element contains whisker-specific data:
             
                         - phase         phase entire trace
                         - amplitude     amplitude entire trace
                         - setpoint      setpoint entire trace
                         - angle         angle entire trace
                         - tops          top idx when whisking
                         - iswhisking    array of 1's and 0's for whisking/non-whisking frames
                         - ACall         autocorrelation whisker angle (empty for now)
                         - spikes        empty for now

                         ** list (whisker specific) of sublists containing info specific for 
                           each whisking bout *** 

                         - isw_ifrm      list of frame ids when is whisking
                         - isw_itop      array of frame ids when whisking 
                                         occur at top phase position
                         - isw_angl      array of angles when is whisking
                         - isw_phs       array of angles when is whisking
                         - isw_amp       array of angles when is whisking
                         - isw_spt       array of angles when is whisking
                         - event_times    list of times in sec when whisking bout starts 
                         - ACisw         autocorrelation whisker angle during whisking
                      
                         inside loadWhiskerData, detection of whisker bouts is done from 
                         amplitude trace obtained using get_slow_var function with function handle
                         np.ptp; the amplitde trace is further bandpass-filtered before using 
                         np.heaviside function with threshold 10*np.pi/180 (10 degrees to rad) 
                         to extract periods of whisking bouts. Using amplitude trace for epoch
                         detection allows to directly account for slow changes in setpoint!
     - group_consecu-    returns list of lists of frames with consecutinve whisking
       tive 
     - whiskingpos       returns avg_var average whisker variable (default angle or amplitude) 
                         per bin of length fxb (# of frames per bin); returns also isw_bin list of
                         boolean True if iswhisking in bin. Variable is first z-scored 
                         and reshaped to match edges of binned neural data; if whisking=True, the 
                         avg_var is only calculated during active whisking bouts.
"""
# Ensure /data_analysis is in path
import os
import sys
if not ('data_analysis' in os.getcwd()):
    sys.path.append(os.path.join(os.getcwd(), 'data_analysis'))
import itertools

import pdb
from scipy.ndimage import gaussian_filter1d
from scipy import stats
from align_fun import align_spktrain, load_data, spike_train, spkcount_cortex, align_to_whisking, Data_container
from meta_data import load_meta
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as colors


def peth(rec_idx,
         whisker='whisk2',
         var='angle',
         t_wind=[-2, 3],
         subwind=[-0.5, 1],
         cgs='good',
         bmt=False,
         pre=None,
         surr=False,
         discard=False,
         selclu=False,
         peth_discard=True):
    """ Compute peth, calculate point within subwind where max occures.
    Attention: added discard_clu to allow to discard clues based on blank peth;
    only plot in peth_compare uses this.
    """

    # load metadata
    wskpath, npxpath, aid, cortex_depth, t_drop, clu_idx, pethdiscard_idx, __, wmask, conditions, __ = load_meta()

    # Load Data specific for rec_idx: 0-18 gCNO, 19-23 wCNO, 24-32 PBS
    wskpath = wskpath[rec_idx]
    npxpath = npxpath[rec_idx]
    aid = aid[rec_idx]
    cortex_depth = cortex_depth[rec_idx]
    t_drop = t_drop[rec_idx]
    conditions = conditions[rec_idx]
    if selclu:
        clu_idx = clu_idx[rec_idx]
    else:
        clu_idx = []

    print(wskpath)
    # wsknpx_data contains: whiskd, spktr_sorted, cids_sorted, a_wvar, a_wvar_m, a_spktr, a_spktr_m
    wsknpx_data = align_spktrain(wskpath, npxpath, cortex_depth, cgs=cgs, t_wind=t_wind, var=var, surr=surr, t_drop=t_drop, pre=pre, clu_idx=clu_idx, discard=discard)

    sr = wsknpx_data.whiskd.sr

    # Get spike and wsk data; drop bad wsk trilas (DLC wrong)
    # ad hoc solution, not great
    if (wskpath == '/media/bunaken/Ensor/videos/CNO/Drop/EP_GLYT2_220318_CNO_Drop_B/') and (pre):
        spktr_data = wsknpx_data.a_spktr.loc[:, (whisker)].copy()
        spktr_data = spktr_data.drop([5, 18, 50, 51, 52], level=1, axis=1)
        spktr_data = spktr_data.groupby(axis=1, level=0).mean()
        df_wvar = wsknpx_data.a_wvar.loc[:, whisker].copy()
        df_wvar = df_wvar.drop([5, 18, 50, 51, 52], axis=1)

    elif (wskpath == '/media/bunaken/Ensor/videos/CNO/Drop/EP_GLYT2_220510_CNO_Drop_B/') and (not pre):
        spktr_data = wsknpx_data.a_spktr.loc[:, (whisker)].copy()
        spktr_data = spktr_data.drop([9], level=1, axis=1)
        spktr_data = spktr_data.groupby(axis=1, level=0).mean()
        df_wvar = wsknpx_data.a_wvar.loc[:, whisker].copy()
        df_wvar = df_wvar.drop([9], axis=1)

    else:
        spktr_data = wsknpx_data.a_spktr_m.loc[:, (whisker)].copy()
        df_wvar = wsknpx_data.a_wvar.loc[:, whisker].copy()


    # Z-score and smooth spiking data
    std = 20

    if peth_discard:
        drop_idx = pethdiscard_idx[rec_idx]
        print(rec_idx, drop_idx)
        spktr_data.drop(drop_idx, axis=1, inplace=True)

    spktr_data = (spktr_data - spktr_data.mean(axis=0)) / spktr_data.std(axis=0)
    spktr_data = gaussian_filter1d(spktr_data, sigma=std, axis=0).T
    # n_clu = wsknpx_data.a_spktr_m[whisker].shape[1]
    n_clu = spktr_data.shape[0]

    # Long df for whisking variable
    df_wvar['idx'] = df_wvar.index
    df_wvar = pd.melt(df_wvar, id_vars='idx')

    # Peak PETH in subwind
    subwind = ((np.array(subwind) - t_wind[0]) * sr).astype(int)
    peak = abs(spktr_data[:, subwind[0]:subwind[1]]).argmax(axis=1) + subwind[0]

    return(spktr_data, df_wvar, peak, n_clu, wskpath)


def plot_peth(rec_idx,
              whisker='whisk2',
              var='angle',
              t_wind=[-2, 3],
              subwind=[-0.5, 1],
              cgs='good',
              plot_peak=True,
              bmt=False,
              surr=False,
              pre=None,
              discard=False,
              selclu=False,
              plotwisk=False):
    """ Plot peth; by default plot activity of good clusters against
    angle of whisker 2 (anterior one); activity is converted from spike
    count to fr and additionally Z-scored and convolved with gaussian
    kernel (for explanation see https://tmramalho.github.io/blog/2013/04/05/an-
    introduction-to-smoothing-time-series-in-python-part-i-filtering-theory/);
    binsize is frame size in ms (~3.3 ms), gaussian sd=10; selclu allows to
    plot only subset of clusters (selected based on tuning curve)
    """
    spktr_data, df_wvar, peak, n_clu, wskpath = peth(rec_idx, whisker=whisker, var=var, t_wind=t_wind, subwind=subwind, cgs=cgs, pre=pre, surr=surr, discard=discard, selclu=selclu)
    pdb.set_trace()
    # Initialise figure
    # fcolor = 'lavenderblush'
    lcolor = 'darkgoldenrod'
    style = 'italic'
    deg = np.array([0.6, 0.6 + 5 / (180 / np.pi)])
    sec = np.array([200, 200 + 0.5 * 299])
    figsize = (14, 12)
    sns.set_context("poster", font_scale=1)
    sns.color_palette("deep")
    plt.rcParams['axes.facecolor'] = 'lavenderblush'
    plt.rcParams["figure.edgecolor"] = 'black'


    # Plot
    # fig1, ((ax1, __), (ax2, cbar_ax)) = plt.subplots(2, 2, sharex='col', gridspec_kw={'height_ratios': [1, 3], 'width_ratios': [10, 1]}, facecolor=fcolor, figsize=figsize)
    fig1, ((ax1, __), (ax2, cbar_ax)) = plt.subplots(2, 2, sharex='col', gridspec_kw={'height_ratios': [1, 3], 'width_ratios': [10, 1]}, frameon=True, figsize=figsize)
    # fig1, (ax1, ax2) = plt.subplots(2, 1, sharex='col', figsize=figsize, gridspec_kw={'height_ratios': [1, 3]})

    # Mean whisker activity
    # ax1.plot(wsknpx_data.a_wvar_m.loc[:, whisker], linewidth=2, color=lcolor)
    sns.lineplot(x='idx', y='value', data=df_wvar, color=lcolor, errorbar='se', ax=ax1)

    # Add scale bars
    ax1.vlines(sec[0], deg[0], deg[1], colors=lcolor)
    ax1.hlines(deg[0], sec[0], sec[1], colors=lcolor)
    ax1.text(sec[0], deg[1] + 0.01, '5 deg')
    ax1.text(sec.mean(), deg[0] + 0.01, '0.5 sec')

    # Add 0 line
    ax1.vlines(t_wind[0] * -299, ax1.get_ylim()[0], ax1.get_ylim()[1])

    # Clear axis except facecolor, add y label
    # ax1.set_facecolor(fcolor)
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.spines['bottom'].set_visible(False)
    # for spine in ax1.spines.values():
    #     spine.set_visible(False)
    ax1.set_ylabel(f'whisker {var}')

    # Mean cluster activity
    # hmap = sns.heatmap(data=spktr_data, cbar_ax=cbar_ax, linecolor='b', center=0, ax=ax2)
    hmap = ax2.matshow(spktr_data, norm=colors.CenteredNorm(), cmap='coolwarm', aspect='auto')
    if plot_peak:
        ax2.scatter(peak, np.arange(n_clu))
    # x2, y2 = np.mgrid[0:spktr_data.shape[1], spktr_data.shape[0]:0:-1]
    # hmap = ax2.pcolormesh(x2.T, y2.T, spktr_data.astype('float32'), norm=colors.CenteredNorm(), cmap='coolwarm')

    # Clear axis, add y label
    ax2.set_xticks([])
    ax2.set_yticks([])
    # ax2.set_yticks(np.arange(n_clu, 0, -1))
    # ax2.set_yticklabels(wsknpx_data.cids_sorted)
    # for spine in ax2.spines.values():
    #     spine.set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax2.set_ylabel('Clusters')

    # Add colorbar
    cbar_ax.axis('off')
    # cbar_ax.set_facecolor(fcolor)
    # cbar_ax.set_xticks([])
    # cbar_ax.set_yticks([])
    # for spine in cbar_ax.spines.values():
    #     spine.set_visible(False)

    # cbar_ax.set_facecolor(fcolor)
    plt.colorbar(hmap, ax=cbar_ax, label='activity (sd)')


    # Clear axis dummy subplot
    __.axis('off')
    # __.set_facecolor(fcolor)
    # __.set_xticks([])
    # __.set_yticks([])
    for spine in __.spines.values():
        spine.set_visible(False)

    # __.set_facecolor(fcolor)


    # Set subplots spacing
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
    # fig1.suptitle('Peri-event time histogram')
    # fig1.suptitle('Peri-event time histogram', style=style)
    fig1.suptitle('Peri-event time histogram', style=style)
    pdb.set_trace()

    #     # If muwt True, plot baseline, edges, p-values, effect size
    #     if bmt:
    #         # Baseline shade
    #         axs[idx].fill_between(fill_x,
    #                               0,
    #                               max_y,
    #                               color=(.8, .8, 1),
    #                               alpha=0.5)

    #         # Vertical line at bin edges
    #         for vl in edges:
    #             axs[idx].axvline(vl, dashes=(5, 2, 1, 2), alpha=0.5)

    #         # Add p-values and effect size
    #         for idx_t, pos in enumerate(pos_text):
    #             if bm_test[clu][idx_t, 1] <= alpha:
    #                 axs[idx].text(pos,
    #                               max_y * (9 / 10),
    #                               '*',
    #                               fontsize='xx-large',
    #                               fontweight='black')

    # Save figures
    # - name recording
    # - var: whisker variable (default angle)
    # - surr: shuffle spiket times or not
    # - pred: pre drop True, False (post) or None (all rec)
    # - selclu: select subset clusters (based on tc)
    if os.path.basename(os.getcwd()) != 'data_analysis':
        os.chdir('./data_analysis')

    if not surr:
        surr = 'nosurr'
    try:
        fig1.savefig(os.getcwd() + f'/images/peth/{os.path.basename(wskpath[:-1])}/{os.path.basename(wskpath[:-1])}_{var}_surr{surr}_pred{pre}.svg', format='svg')
    except FileNotFoundError:
        print(f'created dir images/peth/{os.path.basename(wskpath[:-1])}')
        os.makedirs(os.getcwd() + f'/images/peth/{os.path.basename(wskpath[:-1])}')
        fig1.savefig(os.getcwd() + f'/images/peth/{os.path.basename(wskpath[:-1])}/{os.path.basename(wskpath[:-1])}_{var}_surr{surr}_pred{pre}.svg', format='svg')



def peth_bmt(algn_spktrn_m, sr=299, whisker=2, t_wind=[-2, 3]):
    """ Compute Brunner Munzel test (alternative to Mann-Whitney U test)
    to check significance of changes in neural activity aligned to whisking
    onset; compare binned neural data to baseline (spike count trace from
    onset to -0.5 sec before whisking onset); whole neural data trace is
    z-scored using statistics (mean and std) computed on baseline;
    Brunner Munzel test is then calculated between z-scored mean spike count
    of baseline and subsequent bins.
    Returns results of Brunner Munzel test, end-frame of baseline and edges of
    subsequent bins
    """
    # # Get aligned whisker, npx data, metadata
    # algn_wvar_m, algn_spktrn, algn_spktrn_m, cids_sorted, sr = align_spktrain(
    #     whiskpath, npxpath, t_wind, cgs)

    # Initialise
    binsize = int(0.5 * sr)
    # n_bins = 5
    stop_bsl = int(np.diff([t_wind[0], -0.5])[0] * sr)
    edges = np.arange(stop_bsl, np.diff(t_wind)[0] * sr, binsize)
    len_bins = np.diff(edges)[0]
    n_bins = len(edges) - 1

    bm_test = []
    for spkcount_m in algn_spktrn_m[whisker]:
        # Get z-score statistics
        bsl_m = np.mean(spkcount_m[:stop_bsl])  # mean
        bsl_sd = np.std(spkcount_m[:stop_bsl])  # st. dev.

        # Z-score spkcount in baseline and subsequent bins
        zs_bsl = (spkcount_m[:stop_bsl] - bsl_m) / bsl_sd
        zs_spc_m = np.empty([n_bins, len_bins])  # n_bins, len_bins

        for idx_b, [start, end] in enumerate(zip(edges[:-1], edges[1:])):
            zs_spc_m[idx_b, :] = (spkcount_m[start:end] - bsl_m) / bsl_sd

        # Mann-Whitney U test for each bin
        bmt = np.empty([n_bins, 2])
        for idx_b in range(n_bins):
            # mwut[idx_b, :] = stats.mannwhitneyu(zs_bsl,
            #                                     zs_spc_m[idx_b, :],
            #                                     alternative='two-sided')
            bmt[idx_b, :] = stats.brunnermunzel(zs_bsl,
                                                zs_spc_m[idx_b, :],
                                                alternative='two-sided',
                                                distribution='normal')

        bm_test.append(bmt)
    # NEED TO DEVIDE U-statistics by n*m bins

    return bm_test, stop_bsl, edges


def whisk_autocorr(whiskpath, npxpath, length=50, var='angle', whisker=2):

    # Load data
    whisk_data, _ = load_data(whiskpath, npxpath)

    # Compute autocorrelation
    acorr_fun = lambda wvar: np.array([1] + [
        np.corrcoef(wvar[:-t], wvar[t:])[0, 1] for t in range(1, length)
    ])
    wvar = getattr(whisk_data, var)[whisker]
    wvar_acorr = acorr_fun(wvar)

    # Plot autocorrelation
    _, ax = plt.subplots(1, figsize=(7, 6))
    ax.plot(wvar_acorr)
    ax.set_xticklabels(np.round(ax.get_xticks() / whisk_data.sr, 3))
    ax.set_xlabel('sec')
    ax.vlines(15, ax.get_ylim()[0], ax.get_ylim()[1])

    plt.savefig(os.getcwd() +
                '/save_images/{}_peth_wvarAutocor.png'.format(var))

def fr_count(spktr_array, sr=299):
    """ Compute firing rate for each cluster; exclude clusters with fr>50Hz
    and fr<2. Original data have bin of len=frame (1/299 Hz), therefore
    every 299 bins = 1 sec length; use np.cumsum to sum each consecutive
    element, and ther retain each elements every 299 steps.
    """
    spktr_sec = np.cumsum(spktr_array, axis=1)
    spktr_sec = spktr_sec[:, sr::sr] - spktr_sec[:, :-sr:sr]
    spktr_fr = spktr_sec.mean(axis=1)

    return spktr_fr

def plot_peth_mli(spkdata, wskdata, frm_len, t_wind=[-2, 3], std=5, sr=299):
    """ Plot and save peth for each putative mli; plot together with average
    whisker angle. Scale to fr (divide by len bin = frm_len in sec) and smooth.
    """
    # Scale to fr and smooth
    spkdata = spkdata.div(frm_len / 1000)
    spkdata.iloc[:, :] = gaussian_filter1d(spkdata, sigma=std, axis=0)
    figsize = (14, 12)
    # wvar_m = wsknpx_data.a_wvar_m[whisker, 0]
    xticks = np.arange(0, spkdata.index.values[-1], sr)
    xtick_labels = np.arange(t_wind[0], t_wind[1])
    # deg = np.array([wvar_m.max() - 5 / (180 / np.pi), wvar_m.max() + 5 / (180 / np.pi)])
    # sec = np.array([200, 200 + 0.5 * 299])
    # fcolor = 'lavenderblush'
    # lcolor = 'darkgoldenrod'
    # mcolor = 'copper'
    # mcolor_pre = 'autumn'
    # mcolor_post = 'winter'
    # linewidth = 2
    sns.set_context("poster", font_scale=1)
    sns.color_palette("deep")
    plt.rcParams["figure.edgecolor"] = 'black'
    plt.rcParams['axes.facecolor'] = 'lavenderblush'

    # Save directory
    if os.path.basename(os.getcwd()) != 'data_analysis':
        os.chdir('./data_analysis')

    if not os.path.isdir(os.getcwd() + '/images/peth_mli'):
        print('created dir /images/peth_mli')
        os.makedirs(os.getcwd() + '/images/peth_mli')


    idxslice = pd.IndexSlice
    for rec in spkdata.columns.get_level_values(0).unique():
        spk__ = spkdata.loc[:, idxslice[rec, :, :]]
        wsk__ = wskdata.loc[:, idxslice[rec, :]]
        for clu in spk__.columns.get_level_values(1).unique():
            fig, ax = plt.subplots(2, figsize=figsize, sharex=True, gridspec_kw={'height_ratios': [1, 3]}, frameon=True)
            sns.lineplot(data=wsk__, errorbar='se', ax=ax[0], palette=['brown'])
            sns.lineplot(data=spk__.loc[:, idxslice[:, clu, :]], errorbar='se', ax=ax[1])

            ax[0].set_ylabel('angle (a.u.)')
            ax[1].set_ylabel('firing rate (Hz)')
            ax[1].set_xlabel('time')

            ax[1].set_xticks(xticks)
            ax[1].set_xticklabels(xtick_labels)
            for axis in ax:
                axis.legend().set_visible(False)

            ax[0].set_title('whisking')
            ax[1].set_title('peth')

            # Save
            fig.savefig(os.getcwd() + f'/images/peth_mli/mli_rec{rec}_clu{clu}.svg', format='svg')


def run_mli(whisker='whisker2',
            var='angle',
            t_wind=[-2, 3],
            cgs=2,
            bmt=False,
            plotwisk=False,
            surr=False,
            pre=None,
            selclu=False,
            std=5,
            save_data=False):
    """ Select clusters with <50H located within 200μm from survace; plot peth.
    To get data, follow align_fun script, but adapt to needs:
    - fr for bins=1 second
    - depth<200μm from surface
    """
    # Select control datasets
    drop_rec = [0, 10, 15, 16, 25]
    wskpath, npxpath, aid, cortex_depth, t_drop, clu_idx, pethdiscard_idx, bout_idxs, wmask, conditions, mli_depth = load_meta()
    good_list = np.arange(len(wskpath))
    good_list = good_list[wmask]
    wmask2 = ~np.in1d(np.arange(len(good_list)), np.array(drop_rec))
    good_list = good_list[wmask2]
    good_list = good_list[13:]

    pdb.set_trace()
    # Length bin (in ms)
    frm_len = 1 / 299 * 1000  # frame length in ms

    # Get firing rates and wsk data (follow align_fun)
    if save_data:
        all_data = []
        for idx, rec_idx in enumerate(good_list):
            whiskd, spk = load_data(wskpath[rec_idx], npxpath[rec_idx], t_drop=False)

            # Get spikes for all clusters
            spktr_sorted, endt, cids_sorted, _ = spike_train(spk,
                                                             whiskd,
                                                             binl=frm_len,
                                                             surr=surr,
                                                             cgs=cgs)
            # Retain only clusters in ml (~200μm depth)
            cortex_depth = 200      # μm
            spktr_sorted, cids_sorted = spkcount_cortex(spktr_sorted,
                                                        spk.sortedDepth,
                                                        cids_sorted,
                                                        mli_depth[rec_idx],
                                                        binl=frm_len,
                                                        discard=discard)

            # Compute cluster firing rate and exclude clusters
            # with fr<2 fr>50 Hz
            pdb.set_trace()
            if spktr_sorted.size!=0:
                clufr = fr_count(spktr_sorted, sr=whiskd.sr)
                fr_mask = (clufr>2)&(clufr < 50)
                spktr_sorted = spktr_sorted[fr_mask, :]        

                # Align spike trains to whisking activity
                algn_wvar = []
                algn_wvar_m = []
                algn_spktr = []
                algn_spktr_m = []
                for wsk in range(whiskd.nwhisk):
                    a_wvar, a_wvar_m, a_spktr, a_spktr_m = align_to_whisking(whiskd,
                                                                             spktr_sorted,
                                                                             t_wind,
                                                                             whisker=wsk,
                                                                             var=var)
                    algn_wvar.append(a_wvar)
                    algn_wvar_m.append(a_wvar_m)
                    algn_spktr.append(a_spktr)
                    algn_spktr_m.append(a_spktr_m)

                wsknpx_data = Data_container(whiskd=whiskd,
                                             spktr_sorted=spktr_sorted,
                                             a_wvar=algn_wvar,
                                             a_wvar_m=algn_wvar_m,
                                             a_spktr=algn_spktr,
                                             a_spktr_m=algn_spktr_m,
                                             cids_sorted=cids_sorted)
                pdb.set_trace()
            else:
                wsknpx_data = ['no_clusters']

            pdb.set_trace()
            all_data.append(wsknpx_data)


            # wsknpx_data = align_spktrain(wskpath[rec_idx], npxpath[rec_idx], cortex_depth[rec_idx], cgs=cgs, t_wind=t_wind, var=var, surr=surr, t_drop=t_drop, pre=pre, clu_idx=[], discard=discard)

        all_data_clean = all_data.copy()
        for data in all_data_clean:
            delattr(al_data_clean[10], 'whiskd')

        pdb.set_trace()
        with open('/home/yq18021/Documents/github_gcl/data_analysis/save_data/mli_data.pickle', 'wb') as f:
            pickle.dump(all_data_clean, f)
    else:
        with open('/home/yq18021/Documents/github_gcl/data_analysis/save_data/mli_data.pickle', 'rb') as f:
            all_data_clean = pickle.load(f)


    # # Organise mean mli activity in df
    # a_spktr_m_mli = pd.DataFrame()
    # for data in all_data_clean:
    #     if not isinstance(data, list):
    #         __ = data.a_spktr_m[f'whisk{whisker[-1]}']
    #         __ = pd.DataFrame(__)
    #         a_spktr_m_mli = pd.concat([a_spktr_m_mli, __], axis=1)

    # Organise mli activity in df    
    df_a_spktr_mli = pd.DataFrame()
    for idx, data in enumerate(all_data_clean):
        if not isinstance(data, list):
            __ = data.a_spktr[f'whisk{whisker[-1]}']
            __ = pd.DataFrame(__)
            __ = pd.concat([__], keys=[idx], axis=1)
            df_a_spktr_mli = pd.concat([df_a_spktr_mli, __], axis=1)

    # Organise wsk angle in df
    df_a_wvar = pd.DataFrame()
    for idx, data in enumerate(all_data_clean):
        if not isinstance(data, list):
            __ = data.a_wvar[f'whisk{whisker[-1]}']
            __ = pd.DataFrame(__)
            __ = pd.concat([__], keys=[idx], axis=1)
            df_a_wvar = pd.concat([df_a_wvar, __], axis=1)

    # Plot and save peth
    plot_peth_mli(df_a_spktr_mli, df_a_wvar, frm_len, t_wind=t_wind, std=std, sr=299)



if __name__ == '__main__':
    """ Run script if peth_module.py module is main programme
    """
    # rec_idx = 0               # EP_GLYT2_220316_CNO_Drop_A
    # rec_idx = 1               # EP_GLYT2_220316_CNO_Drop_B
    # rec_idx = 2               # EP_GLYT2_220318_CNO_Drop_A
    # rec_idx = 3               # EP_GLYT2_220318_CNO_Drop_B
    # rec_idx = 4               # EP_GLYT2_220325_CNO_Drop_A
    # rec_idx = 5               # EP_GLYT2_220325_CNO_Drop_B
    # rec_idx = 6               # EP_GLYT2_220429_CNO_Drop_B
    # rec_idx = 8               # EP_GLYT2_220506_CNO_Drop_A !!!!
    # rec_idx = 9               # EP_GLYT2_220510_CNO_Drop_A
    # rec_idx = 10               # EP_GLYT2_220510_CNO_Drop_B
    # rec_idx = 12               # EP_GLYT2_220512_CNO_Drop_B
    # rec_idx = 13               # EP_GLYT2_220517_CNO_Drop_A
    # rec_idx = 14               # EP_GLYT2_220517_CNO_Drop_B
    # rec_idx = 15               # EP_GLYT2_220519_CNO_Drop_A
    # rec_idx = 16               # EP_GLYT2_220519_CNO_Drop_B
    # rec_idx = 17               # EP_GLYT2_220527_CNO_Drop_A
    # rec_idx = 18               # EP_GLYT2_220527_CNO_Drop_B

    # rec_idx = 19              # EP_WT_220303_CNO_Drop
    # rec_idx = 20              # EP_WT_220422_CNO_Drop_A
    # rec_idx = 21              # EP_WT_220422_CNO_Drop_B
    # rec_idx = 22              # EP_WT_220425_CNO_Drop_A
    # rec_idx = 23              # EP_WT_220425_CNO_Drop_B
    
    # rec_idx = 24              # EP_GLYT2_220429_PBS_Drop_A
    # rec_idx = 26              # EP_GLYT2_220506_PBS_Drop_B
    rec_idx = 27              # EP_WT_220602_A !!!!
    # rec_idx = 28              # EP_WT_220602_B
    # rec_idx = 29              # EP_WT_220607_A
    # rec_idx = 30              # EP_WT_220607_B
    # rec_idx = 31              # EP_WT_220609_A
    # rec_idx = 32              # EP_WT_220609_B

    whisker = 'whisk2'          # which whisker "" ""
    t_wind = [-2, 3]            # t_wind around whisking bout start
    subwind = [-0.5, 1]           # window where to consider PETH max
    # t_wind = [-10, 10]            # t_wind around whisking bout start
    var = 'angle'
    # var = 'setpoint'
    # var = 'amplitude'           # which whisker var "" ""
    cgs = 2
    bmt = False                 # Run Brunner Munzel test
    plotwisk = False
    surr = False
    # surr = 'shuffle'
    pre = None                  # Use None to get whole rec
    # pre = True                  # True for pre, False for post drop
    discard = False             # discard clu with fr < 0.1 Hz
    selclu = False
    run_mli_fun = False
    std = 5                     # length window used to smooth data
    save_data = False           # Compute and save data, otherwise load
    plot_peak = False

    if not run_mli_fun:
        plot_peth(rec_idx, whisker=whisker, var=var, t_wind=t_wind, subwind=subwind, cgs=cgs, bmt=bmt, plotwisk=plotwisk, surr=surr, pre=pre, selclu=selclu, plot_peak=plot_peak)
    else:
        run_mli(whisker=whisker, var=var, t_wind=t_wind, cgs=cgs, bmt=bmt, plotwisk=plotwisk, surr=surr, pre=pre, selclu=selclu, std=std, save_data=save_data)
