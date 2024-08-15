""" Code for neuropixel analysis (output kilosort/phy2) """

# look script_dualPhase3-2.m from http://data.cortexlab.net/dualPhase3/data/script_dualPhase3.m

# Also time is ok as Marie says: -1 sec at beginning (- 1 at the end)

import numpy as np
import pandas as pd
from pathlib import Path
import os.path
import pdb
import re

def loadKsDir(ksdir, *args):
    varargin = args
    if varargin:
        params = varargin
    else:
        params = type(
            '', (), {})()  # create an empty class (see https://realpython.com/
        # python-metaclasses/)

    if not hasattr(params, 'excludeNoise'):
        params.excludeNoise = True

    if not hasattr(params, 'loadPCs'):
        params.loadPCs = False

    import sys
    full_path = os.path.join(
        ksdir)  # could use Path(ksdir), so no need of Path(full_path) later
    if full_path not in sys.path:
        sys.path.append(full_path)

    import params as spikeStruct  # spikeStruct is params.py module, not class params above

    # Get recording length
    with open(full_path + '/' + Path(full_path).name[6:] + '_tcat.imec0.ap.meta') as meta:
        for line in meta:
            if re.findall(r'fileTimeSecs*', line):
                rec_len = float(re.split('=', line)[1])

    #  breakpoint()
    ss = np.load(
        Path(full_path, "spike_times.npy")
    )  # use Path to read spike_times.npy (use .open() under the hood)

    # Get recording sample rate
    with open(full_path + '/' + Path(full_path).name[6:] + '_tcat.imec0.ap.meta') as meta:
        for line in meta:
            if re.findall(r'imSampRate*', line):
                sample_rate = float(re.split('=', line)[1])

    # st = np.double(ss) / spikeStruct.sample_rate
    st = np.double(ss) / sample_rate
    spikeTemplates = np.load(Path(full_path, "spike_templates.npy"))
    if Path(full_path, "spike_clusters.npy").exists():
        clu = np.load(Path(full_path, "spike_clusters.npy"))
    else:
        clu = spikeTemplates

    tempScalingAmps = np.load(Path(full_path, "amplitudes.npy"))
    if params.loadPCs:
        pcFeat = np.load(
            Path(full_path,
                 'pc_features.npy'))  # % nSpikes x nFeatures x nLocalChannels
        pcFeatInd = np.load(Path(
            full_path, 'pc_feature_ind.npy'))  # % nTemplates x nLocalChannels
    else:
        pcFeat = []
        pcFeatInd = []

    cgsFileref = ''
    if Path(full_path, "cluster_groups.csv").exists():
        cgsFileref = Path(full_path, "cluster_groups.csv")
    if Path(full_path, "cluster_group.tsv").exists():
        cgsFileref = Path(full_path, "cluster_group.tsv")
    if Path(full_path, "cluster_info.tsv").exists():
        cinfo = Path(full_path, "cluster_info.tsv")

    if cgsFileref:
        clinfo = pd.read_csv(cinfo, delimiter='\t')
        clinfo.group.fillna('unsorted', inplace=True)
        clinfo['cgs'] = clinfo.group.map({
            'noise': 0,
            'mua': 1,
            'good': 2,
            'unsorted': 3
        })

        if params.excludeNoise:
            noiseclusters = clinfo.id[clinfo.cgs == 0]
            st = st[np.in1d(clu, noiseclusters,
                            invert=True)]  # remove noise clusters
            spikeTemplates = spikeTemplates[np.in1d(clu,
                                                    noiseclusters,
                                                    invert=True)]
            tempScalingAmps = tempScalingAmps[np.in1d(clu,
                                                      noiseclusters,
                                                      invert=True)]

            depth = clinfo.depth[np.in1d(clinfo.id, noiseclusters,
                                         invert=True)]
            clu = clu[np.in1d(clu, noiseclusters, invert=True)]
            cgs = clinfo.cgs[np.in1d(clinfo.id, noiseclusters, invert=True)]
            cids = clinfo.id[np.in1d(clinfo.id, noiseclusters, invert=True)]

    coords = np.load(Path(full_path, "channel_positions.npy"))
    ycoords = coords[:, 1]
    xcoords = coords[:, 0]

    ncl = cids.size

    # temps = np.load(Path(full_path, "templates.npy"))
    # winv = np.load(Path(full_path, "whitening_mat_inv.npy"))

    # Save into class
    spikeStruct.rec_len = rec_len
    spikeStruct.st = st.flatten()  # spike times
    spikeStruct.spikeTemplates = spikeTemplates  # template identity for each spike
    spikeStruct.clu = clu  # cluster identity of each spike
    spikeStruct.tempScalingAmps = tempScalingAmps  # scaling of template
    spikeStruct.cgs = cgs  # cluster lable (good, mua)
    spikeStruct.cids = cids  # cluster ids associated to cgs
    spikeStruct.ncl = ncl  # number of clusters
    spikeStruct.depth = depth  # added by Marie Tolkiehn
    spikeStruct.xcoords = xcoords
    spikeStruct.ycoords = ycoords
    # spikeStruct.temps = temps
    # spikeStruct.winv = winv
    spikeStruct.pcFeat = pcFeat
    spikeStruct.pcFeatInd = pcFeatInd

    return spikeStruct


def process_spiketimes(spikeStruct):
    """The default for SpikeGLX recordings is to start writing to file 1 s before the
     trigger started (SpikeGLX feature). This means that neural data precedes triggered
    inputs by 1s. This function subtracts the second and resets signals to start at t=0.
    It also sorts depth information by depth, and rearranges sorted cids
    from deepest to most superficial cluster.
     sp.st are spike times in seconds
     sp.clu are cluster identities
     spikes from clusters labeled "noise" have already been omitted
    """
    spikeStruct.st = spikeStruct.st - 1  # recording started 1s before video (SpikeGLX feature)
    spikeStruct.spikeTemplates = spikeStruct.spikeTemplates[spikeStruct.st > 0]
    spikeStruct.clu = spikeStruct.clu[spikeStruct.st > 0]
    # spikeStruct.tempScalingAmps[spikeStruct.st > 0] # Marie original
    spikeStruct.tempScalingAmps = spikeStruct.tempScalingAmps[spikeStruct.st >
                                                              0]  # Mine
    spikeStruct.st = spikeStruct.st[spikeStruct.st > 0]
    spikeStruct.rec_len = spikeStruct.rec_len - 2  # rec also stops 1 sec after video

    # Sort from deepest to most superficial cluster
    if any(spikeStruct.depth < 0):
        spikeStruct.depth = spikeStruct.depth + 3840

    sortedDepth = np.sort(spikeStruct.depth)
    sortid = np.argsort(spikeStruct.depth)
    spikeStruct.cidsSorted = spikeStruct.cids.iloc[sortid].reset_index(
        drop=True)  # sorted cids
    spikeStruct.cgsSorted = spikeStruct.cgs.iloc[sortid].reset_index(
        drop=True)  # sorted cgs
    spikeStruct.sortedDepth = sortedDepth

    return spikeStruct
