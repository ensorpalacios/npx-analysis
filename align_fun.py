"""
Module to align spike count trains to whisking bouts. This module includes then
following functions:

load_data: load whisker and npx data.
Attention 1: now rescaling spike times by ratio of whisker and npx rec
Attention 2: now if t_drop is not False, select npx and whisker data based
of time of dropping; can be either before or after

spike_train: compute spike count in bins of length binl; use binl = range frame
in ms; use **kwarg to select goo clusters (=2); use *args to select specific
clusters

spkcount_cortex: select clusters in cortex based on cortex_depth; possible to
remove clusters whose firing rate drops below x (e.g.0.1) Hz at any time bin, but
don't use for small time bins (e.g. <5 min).

align_to_whisking: align wvar and spk_train to whisking onset.

align_spktrain: put together all functions and return algn_wvar, algn_wvar_m,
algn_spktr, algn_spktr_m, cids_sorted, sr.

Important: because I use whiskvar to bin data, whiskvar has an additional
bin, therefore whisk_var needs to be sliced with [:-1]!!!!
"""
from random import sample, randint
import numpy as np
import whisk_analysis as wa
import load_npx as lnpx
import pdb
import pandas as pd


def load_data(wskpath, npxpath, t_drop=False, pre=True):
    """ Load and process whisker and npx data"""
    df, body, tot_wskdlen = wa.load_wdata(wskpath, t_drop=t_drop, pre=pre)
    angle_trace = wa.calc_angle(df, body)
    _, phase_trace, filtered_sig = wa.calc_phase(angle_trace)
    whiskd = wa.loadWhiskerData(df, angle_trace, phase_trace, filtered_sig)
    spk = lnpx.loadKsDir(npxpath)
    spk = lnpx.process_spiketimes(spk)

    # Correct misalignment wsk and npx times
    resc_fact = tot_wskdlen / spk.rec_len
    print(resc_fact)
    spk.st = spk.st * resc_fact

    if t_drop is not False:
        if pre:
            mask = spk.st < t_drop * 60
            reset = 0
        else:
            # mask = spk.st >= t_drop * 60
            mask = (spk.st >= (t_drop + 5) * 60) & (spk.st < (t_drop * 2 + 5) * 60)
            reset = (t_drop + 5) * 60

        spk.st = spk.st[mask]
        spk.st = spk.st - reset  # reset time
        spk.spikeTemplates = spk.spikeTemplates[mask]
        spk.clu = spk.clu[mask]
        spk.tempScalingAmps = spk.tempScalingAmps[mask]

    return whiskd, spk


# def spike_train(npx_data, whisk_data, binl=1, **kwarg):
#     """ Compute spike count in bin for period of length whisk_data; binl
#     (defaul 1 but normally used with length frame) is in ms. Returns binned
#     spike train and length of train. Attention: possible to find more than
#     one spike in 1 ms bin because of not perfect clustering.
#     """
#     spk_train = []
#     spk_times = np.round(npx_data.st * 1000)  # ms
#     end_time = whisk_data.time[-1] * 1000  # ms
#     edges = np.arange(0, end_time + binl, binl)  # +binl to include last frame

#     if bool(kwarg):
#         cids_sorted = npx_data.cidsSorted[npx_data.cgsSorted == kwarg['cgs']]
#     else:
#         cids_sorted = npx_data.cidsSorted

#     # Get spike array
#     for cl in range(len(cids_sorted)):
#         spk_hist, _ = np.histogram(
#             spk_times[npx_data.clu == cids_sorted.iloc[cl]], bins=edges)
#         spk_train.append(spk_hist)

#     return spk_train, end_time, cids_sorted, edges


def spike_train(npx_data, whisk_data, *args, binl=1, surr=False, **kwarg):
    """ Compute spike count in bin for period of length whisk_data; binl
    (defaul 1 but normally used with length frame) is in ms. Returns binned
    spike train and length of train. Attention: possible to find more than
    one spike in 1 ms bin because of not perfect clustering.

    Attention: problems with number precision when creating edges for binning:
    try to round binl.

    Attention 1: now solved double-spike count by deleting spikes too close,
    but in rare cases possibly deleting good spikes when many consecutive
    spikes have diff<1.

    Attention 2: spk_hist has bin frame long, spk_data has bin 1 ms long;
    Note: use **kwarg to select goo clusters (=2); use *args to select specific
    clusters
    """
    spk_train = []
    spk_train_msec = []
    spk_times = np.round(npx_data.st * 1000)  # ms
    end_time = whisk_data.time[-1] * 1000  #

    # Create edges and control for mismatch binning methods
    binl = np.round(binl, 5)
    edges = np.arange(0, end_time + binl, binl)  # +binl to include last frame
    if (len(edges) - 1) != (whisk_data.nframe // np.round(binl * 299 / 1000).astype(int)):
        print('discard last bin in spike raster edges')
        edges = np.arange(0, end_time, binl)
    if 'clu_idx' in kwarg:
        cids_sorted = npx_data.cidsSorted[npx_data.cidsSorted.isin(kwarg['clu_idx'])]
        npx_data.sortedDepth = npx_data.sortedDepth[npx_data.cidsSorted.isin(kwarg['clu_idx'])]
    else:
        # if kwarg set to noise, take both good and mua units
        if kwarg['cgs'] != 0:
            cids_sorted = npx_data.cidsSorted[npx_data.cgsSorted ==
                                              kwarg['cgs']]
            npx_data.sortedDepth = npx_data.sortedDepth[npx_data.cgsSorted ==
                                                        kwarg['cgs']]
        else:
            cids_sorted = npx_data.cidsSorted[
                npx_data.cgsSorted != kwarg['cgs']]
    # else:
    #     cids_sorted = npx_data.cidsSorted

    # Get spike array (sorted by spatial location)
    for cl in range(len(cids_sorted)):  # for each cluster
        spk_data = spk_times[npx_data.clu == cids_sorted.iloc[cl]]
        # force >=1 ms refractory period (delete double-counted spikes)
        spk_data = spk_data[:-1][np.diff(spk_data) >= 1]

        # if surr == 'shuffle':
        #     spk_data = sample(range(int(end_time)), spk_data.size)
        if surr == 'shuffle':
            spk_data = sample(range(int(edges[-1])), sum(~(spk_data >= edges[-1])))

        if surr == 'ditter':
            spk_data = np.array(
                [randint(t - 5000, t + 5000) for t in spk_data])

        spk_hist, _ = np.histogram(spk_data, bins=edges)
        spk_train.append(spk_hist)
        if 'msec' in kwarg:
            __ = np.histogram(spk_data, bins=np.arange(0, int(end_time) + 1))
            spk_train_msec.append(__[0])

    if 'msec' in kwarg:
        return spk_train, end_time, cids_sorted, edges, spk_train_msec
    else:
        return spk_train, end_time, cids_sorted, edges


def spkcount_cortex(spktr_sorted,
                    sortedDepth,
                    cidsSorted,
                    cortex_depth,
                    binl=1,
                    discard=False):
    """
    Select clusters in cortex based on cortex_depth; remove clusters whose
    firing rate drops below 0.1 Hz at any time bin (might relax or improve
    in future).
    """
    # Select clusters within cortex (e.g. >=1000μm from probe tip)
    ctx_start = np.sum(sortedDepth < cortex_depth)
    spktr_sorted = np.array(spktr_sorted[ctx_start:])
    cidsSorted = np.array(cidsSorted.iloc[ctx_start:])
    sortedDepth = np.array(sortedDepth[ctx_start:])

    print(f'# before discard (HZ): {len(spktr_sorted)}')
    # Discard clu with <0.1 Hz in tw_int (e.g. <6 in 1 min)
    if discard:
        mask = ~(spktr_sorted[:, ] < 0.1 * binl * 60).any(axis=1)
        spktr_sorted = spktr_sorted[mask, :]
        cidsSorted = cidsSorted[mask]
        sortedDepth = sodtedDepth[mask]

    print(f'# before discard (HZ): {len(spktr_sorted)}')

    # Organise in dataframe

    return spktr_sorted, cidsSorted, sortedDepth


def align_to_whisking(whisk_data,
                      spk_trains,
                      t_wind=[-2, 3],
                      whisker=2,
                      var='angle'):
    """ Align whisker variable and spike count (list of clusters) based on
    whisking bout onset; spk_trains is cluster list with spk_tr; t_wind
    centered on whisking onset; time is then converted from seconds to frames.
    Important: because I use whiskvar to bin data, whiskvar has an additional
    bin, therefore whisk_var needs to be sliced with [:-1]!!!!
    Returns aligned whisker variable and mean as array and aligned spikes and
    means as list of clusters.
    """
    sr = whisk_data.sr  # sampling rate
    t_wind = (np.array(t_wind) * sr).astype(int)  # sec to frame
    start_bout = np.array(whisk_data.event_times[whisker]) * sr  # sec to frame
    aligned_wvar = np.empty([len(start_bout), np.diff(t_wind)[0]])
    aligned_wvar[:] = np.nan
    aligned_spktr = []
    aligned_spktr_m = []
    # whisk_var = getattr(whisk_data, var)[whisker][:-1]
    whisk_var = getattr(whisk_data, var)[whisker]

    for idx_c, _ in enumerate(spk_trains):  # for all clusters
        a_spktr = np.empty([len(start_bout), np.diff(t_wind)[0]])
        a_spktr[:] = np.nan

        for idx_b, stb in enumerate(start_bout):  # for all whisking bouts
            isw_edge = t_wind + int(stb)

            # Within edges of time
            if (isw_edge[0] >= 0) & (isw_edge[1] <= len(whisk_var)):

                # Align whisker variable (only once)
                if idx_c == 0:
                    aligned_wvar[idx_b, :] = whisk_var[isw_edge[0]:isw_edge[1]]

                # Align spike counts
                a_spktr[idx_b, :] = spk_trains[idx_c][isw_edge[0]:isw_edge[1]]

            # Account for neg time
            elif isw_edge[0] < 0:

                # Align whisker variable (only once)
                if idx_c == 0:
                    aligned_wvar[idx_b,
                                 abs(isw_edge[0]):] = whisk_var[:isw_edge[1]]

                # Align spike counts
                a_spktr[idx_b,
                        abs(isw_edge[0]):] = spk_trains[idx_c][:isw_edge[1]]

            # Account for excess time
            elif isw_edge[1] > len(whisk_var):

                # Align whisker variable (only once)
                if idx_c == 0:
                    aligned_wvar[idx_b, :(
                        len(whisk_var) -
                        isw_edge[1])] = whisk_var[isw_edge[0]:]

                # Align spike counts
                a_spktr[idx_b, :(
                    len(whisk_var) -
                    isw_edge[1])] = spk_trains[idx_c][isw_edge[0]:]

        aligned_spktr.append(a_spktr)
        a_spktr_m = np.nanmean(a_spktr, 0)
        aligned_spktr_m.append(a_spktr_m)

    aligned_wvar_m = np.nanmean(aligned_wvar, 0)
    return aligned_wvar, aligned_wvar_m, aligned_spktr, aligned_spktr_m


class Data_container():
    ''' Class to organise whisker and neuropixels data; where necessary create
    dataframe with whisker as keys
    '''
    def __init__(self, **kwargs):
        self.whiskd = kwargs['whiskd']
        self.spktr_sorted = kwargs['spktr_sorted']
        self.cids_sorted = kwargs['cids_sorted']

        # Convert whisker variables in df with
        # level=0 whiskers, level=1 whisking bouts
        self.a_wvar = pd.concat([
            pd.DataFrame(kwargs['a_wvar'][0].T),
            pd.DataFrame(kwargs['a_wvar'][1].T),
            pd.DataFrame(kwargs['a_wvar'][2].T)
        ],
                                axis=1,
                                keys=('whisk0', 'whisk1', 'whisk2'))
        self.a_wvar.index.name = 'time'
        self.a_wvar_m = pd.concat([
            pd.DataFrame(kwargs['a_wvar_m'][0].T),
            pd.DataFrame(kwargs['a_wvar_m'][1].T),
            pd.DataFrame(kwargs['a_wvar_m'][2].T)
        ],
                                  axis=1,
                                  keys=('whisk0', 'whisk1', 'whisk2'))
        self.a_wvar_m.index.name = 'time'

        # Convert spktr data in df with
        # level=0 whiskers, level=1 clusters, level=2 whisking bouts
        ___ = []                # for whiskers
        for wsk_ in range(3):
            __ = pd.DataFrame()  # for clusters
            for clu, spk_data in enumerate(kwargs['a_spktr'][wsk_]):
                _ = pd.DataFrame(spk_data.T)  # single cluster
                # _.columns = pd.MultiIndex.from_product([[f'{clu}'], _.columns])
                _.columns = pd.MultiIndex.from_product([[clu], _.columns])
                __ = pd.concat([__, _], axis=1)
            ___.append(__)

        self.a_spktr = pd.concat([___[0], ___[1], ___[2]],
                                 axis=1,
                                 keys=('whisk0', 'whisk1', 'whisk2'))
        self.a_spktr.index.name = 'time'

        ___ = []                # for whiskers
        for wsk_ in range(3):
            __ = pd.DataFrame()  # for clusters
            for clu, spk_data in enumerate(kwargs['a_spktr_m'][wsk_]):
                _ = pd.DataFrame(spk_data.T)  # single cluster
                # _.columns = pd.MultiIndex.from_product([[clu], _.columns])
                _.columns = [clu]
                __ = pd.concat([__, _], axis=1)
            ___.append(__)

        self.a_spktr_m = pd.concat([___[0], ___[1], ___[2]],
                                   axis=1,
                                   keys=('whisk0', 'whisk1', 'whisk2'))
        self.a_spktr_m.index.name = 'time'


def align_spktrain(wskpath,
                   npxpath,
                   cortex_depth,
                   cgs=2,
                   t_wind=[-2, 3],
                   var='angle',
                   surr=False,
                   t_drop=20,
                   pre=None,
                   clu_idx=[],
                   discard=False,
                   pethdiscard_idx=None):
    """ Helper function  to compute aligned spike count trains aligned to
    whisking bout onset; bin is of length video frame in ms (~3.3 ms);
    compute spike count per bin aligned to each frame for all clusters
    (default only good=2); extract whisker variable (default angle) and
    spike count per bin aligned whisking bout onset.
    Returns average whisker variable and aligned average spike count (average
    across trials/whisking bounts).
    """
    # Load and preprocess whisker and npx data; possible to get subset of
    # data: either pre or post CNO/PBS drop
    if pre is None:
        whiskd, spk = load_data(wskpath, npxpath, t_drop=False)
    else:
        whiskd, spk = load_data(wskpath, npxpath, t_drop=t_drop, pre=pre)

    # Get spike train binned to match frame length
    frm_len = 1 / whiskd.sr * 1000  # frame length in ms

    if clu_idx == []:
        spktr_sorted, endt, cids_sorted, _ = spike_train(spk,
                                                         whiskd,
                                                         binl=frm_len,
                                                         surr=surr,
                                                         cgs=cgs)
    else:
        spktr_sorted, endt, cids_sorted, _ = spike_train(spk,
                                                         whiskd,
                                                         binl=frm_len,
                                                         surr=surr,
                                                         clu_idx=clu_idx,
                                                         cgs=cgs)

    # if cgs == 'mua':
    # spktr_sorted, endt, cids_sorted, _ = spike_train(spk,
    #                                                  whiskd,
    #                                                  binl=frm_len,
    #                                                  surr=surr,
    #                                                  cgs=1)

    # Get only cortex data
    spktr_sorted, cids_sorted, sortedDepth = spkcount_cortex(spktr_sorted,
                                                             spk.sortedDepth,
                                                             cids_sorted,
                                                             cortex_depth,
                                                             binl=frm_len,
                                                             discard=discard)

    if pethdiscard_idx != None:
        spktr_sorted = np.delete(spktr_sorted, pethdiscard_idx, axis=0)
        cids_sorted = np.delete(cids_sorted, pethdiscard_idx)

    print(f'# after pethdiscard: {spktr_sorted.shape[0]}')


    # Align whisker variable and spk_trains to whisking onset
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

    return wsknpx_data  # added spktr_sorted
