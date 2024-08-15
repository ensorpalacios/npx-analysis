#!/usr/bin/env python
"""
Helper module to load whisker and neuropixels data, based on info about
condition, cortex depth, CNO time drop, animal ID.
"""

# Ensure /data_analysis is in path
import os
import sys
if not ('data_analysis' in os.getcwd()):
    sys.path.append(os.path.join(os.getcwd(), 'data_analysis'))

# Import modules
import pdb
import numpy as np
import numpy as np
import whisk_analysis as wa
import load_npx as lnpx
import pdb
import pandas as pd



# def load_meta(rec_idx, t_wind=[-2, 3], var='angle', cgs=2, surr=False, discard=False):
def load_meta(mus=False):
    """ Load and align wsk and nxp data; tw_wind defins time window centered
    on whisking bout start; var defines whisking variable to consider;
    if cgs != 0 take only mua or good units, if cgs == 0 take both mua and good
    units; surr defines whether to use surrogate data (either False, shuffle'
    or 'ditter); discard allows to discart clusters with fr < x (e.g. 0.1) --
    not really used here (used in spatiotemporal analysis instead).
    """
    # Path to Muscimol data
    npxpath_mus = ['/media/bunaken/Ensor/npx/Muscimol/EP_PCP_220204_Muscimol_noC\
am_g0/catgt_EP_PCP_220204_Muscimol_noCam_g0/kilosort3/',
                   '/media/bunaken/Ensor/npx/Muscimol/EP_PCP_220209_Muscimol_noC\
am_g0/catgt_EP_PCP_220209_Muscimol_noCam_g0//kilosort3/',
                   '/media/bunaken/Ensor/npx/Muscimol/EP_PCP_220210_Muscimol_noC\
am_g0/catgt_EP_PCP_220210_Muscimol_noCam_g0/kilosort3/']

    # Whisking data: glyt2 CNO (gCNO, 19), wild type CNO (wCNO, 5), glyt2 and wilt
    # type PBS (aPBS, 9)
    wskpath_gCNO = ['/media/bunaken/Ensor/videos/CNO/Drop/EP_GLYT2_220316_CNO_\
Drop_A/',
                    '/media/bunaken/Ensor/videos/CNO/Drop/EP_GLYT2_220316_CNO_\
Drop_B/',
                    '/media/bunaken/Ensor/videos/CNO/Drop/EP_GLYT2_220318_CNO_\
Drop_A/',
                    '/media/bunaken/Ensor/videos/CNO/Drop/EP_GLYT2_220318_CNO_\
Drop_B/',
                    '/media/bunaken/Ensor/videos/CNO/Drop/EP_GLYT2_220325_CNO_\
Drop_A/',
                    '/media/bunaken/Ensor/videos/CNO/Drop/EP_GLYT2_220325_CNO_\
Drop_B/',
                    '/media/bunaken/Ensor/videos/CNO/Drop/EP_GLYT2_220429_CNO_\
Drop_B/',
                    '/media/bunaken/Ensor/videos/CNO/Drop/EP_GLYT2_220502_CNO_\
Drop_A/',
                    '/media/bunaken/Ensor/videos/CNO/Drop/EP_GLYT2_220506_CNO_\
Drop_A/',
                    '/media/bunaken/Ensor/videos/CNO/Drop/EP_GLYT2_220510_CNO_\
Drop_A/',
                    '/media/bunaken/Ensor/videos/CNO/Drop/EP_GLYT2_220510_CNO_\
Drop_B/',
                    '/media/bunaken/Ensor/videos/CNO/Drop/EP_GLYT2_220512_CNO_\
Drop_A/',
                    '/media/bunaken/Ensor/videos/CNO/Drop/EP_GLYT2_220512_CNO_\
Drop_B/',
                    '/media/bunaken/Ensor/videos/CNO/Drop/EP_GLYT2_220517_CNO_\
Drop_A/',
                    '/media/bunaken/Ensor/videos/CNO/Drop/EP_GLYT2_220517_CNO_\
Drop_B/',
                    '/media/bunaken/Ensor/videos/CNO/Drop/EP_GLYT2_220519_CNO_\
Drop_A/',
                    '/media/bunaken/Ensor/videos/CNO/Drop/EP_GLYT2_220519_CNO_\
Drop_B/',
                    '/media/bunaken/Ensor/videos/CNO/Drop/EP_GLYT2_220527_CNO_\
Drop_A/',
                    '/media/bunaken/Ensor/videos/CNO/Drop/EP_GLYT2_220527_CNO_\
Drop_B/']
    wskpath_wCNO = ['/media/bunaken/Ensor/videos/CNO/Drop/EP_WT_220303_CNO_\
Drop/',
                    '/media/bunaken/Ensor/videos/CNO/Drop/EP_WT_220422_CNO_\
Drop_A/',
                    '/media/bunaken/Ensor/videos/CNO/Drop/EP_WT_220422_CNO_\
Drop_B/',
                    '/media/bunaken/Ensor/videos/CNO/Drop/EP_WT_220425_CNO_\
Drop_A/',
                    '/media/bunaken/Ensor/videos/CNO/Drop/EP_WT_220425_CNO_\
Drop_B/']
    wskpath_aPBS = ['/media/bunaken/Ensor/videos/CNO/Drop/EP_GLYT2_220429_PBS_\
Drop_A/',
                    '/media/bunaken/Ensor/videos/CNO/Drop/EP_GLYT2_220502_PBS_\
Drop_B/',
                    '/media/bunaken/Ensor/videos/CNO/Drop/EP_GLYT2_220506_PBS_\
Drop_B/',
                    '/media/bunaken/Ensor/videos/WT/EP_WT_220602_A/',
                    '/media/bunaken/Ensor/videos/WT/EP_WT_220602_B/',
                    '/media/bunaken/Ensor/videos/WT/EP_WT_220607_A/',
                    '/media/bunaken/Ensor/videos/WT/EP_WT_220607_B/',
                    '/media/bunaken/Ensor/videos/WT/EP_WT_220609_A/',
                    '/media/bunaken/Ensor/videos/WT/EP_WT_220609_B/']

    # Neuropixels data: glyt2 CNO (gCNO, 19), wild type CNO (wCNO, 5), glyt2 and wilt
    # type PBS (aPBS, 9)
    npxpath_gCNO = ['/media/bunaken/Ensor/npx/CNO/Drop/EP_GLYT2_220316_CNO_Drop_A_g0/\
catgt_EP_GLYT2_220316_CNO_Drop_A_g0',
                    '/media/bunaken/Ensor/npx/CNO/Drop/EP_GLYT2_220316_CNO_Drop_B_g0/\
catgt_EP_GLYT2_220316_CNO_Drop_B_g0',
                    '/media/bunaken/Ensor/npx/CNO/Drop/EP_GLYT2_220318_CNO_Drop_A_g0/\
catgt_EP_GLYT2_220318_CNO_Drop_A_g0',
                    '/media/bunaken/Ensor/npx/CNO/Drop/EP_GLYT2_220318_CNO_Drop_B_g0/\
catgt_EP_GLYT2_220318_CNO_Drop_B_g0',
                    '/media/bunaken/Ensor/npx/CNO/Drop/EP_GLYT2_220325_CNO_Drop_A_g0/\
catgt_EP_GLYT2_220325_CNO_Drop_A_g0',
                    '/media/bunaken/Ensor/npx/CNO/Drop/EP_GLYT2_220325_CNO_Drop_B_g0/\
catgt_EP_GLYT2_220325_CNO_Drop_B_g0',
                    '/media/bunaken/Ensor/npx/CNO/Drop/EP_GLYT2_220429_CNO_Drop_B_g0/\
catgt_EP_GLYT2_220429_CNO_Drop_B_g0',
                    '/media/bunaken/Ensor/npx/CNO/Drop/EP_GLYT2_220502_CNO_Drop_A_g0/\
catgt_EP_GLYT2_220502_CNO_Drop_A_g0',
                    '/media/bunaken/Ensor/npx/CNO/Drop/EP_GLYT2_220506_CNO_Drop_A_g0/\
catgt_EP_GLYT2_220506_CNO_Drop_A_g0',
                    '/media/bunaken/Ensor/npx/CNO/Drop/EP_GLYT2_220510_CNO_Drop_A_g0/\
catgt_EP_GLYT2_220510_CNO_Drop_A_g0',
                    '/media/bunaken/Ensor/npx/CNO/Drop/EP_GLYT2_220510_CNO_Drop_B_g0/\
catgt_EP_GLYT2_220510_CNO_Drop_B_g0',
                    '/media/bunaken/Ensor/npx/CNO/Drop/EP_GLYT2_220512_CNO_Drop_A_g0/\
catgt_EP_GLYT2_220512_CNO_Drop_A_g0',
                    '/media/bunaken/Ensor/npx/CNO/Drop/EP_GLYT2_220512_CNO_Drop_B_g0/\
catgt_EP_GLYT2_220512_CNO_Drop_B_g0',
                    '/media/bunaken/Ensor/npx/CNO/Drop/EP_GLYT2_220517_CNO_Drop_A_g0/\
catgt_EP_GLYT2_220517_CNO_Drop_A_g0',
                    '/media/bunaken/Ensor/npx/CNO/Drop/EP_GLYT2_220517_CNO_Drop_B_g0/\
catgt_EP_GLYT2_220517_CNO_Drop_B_g0',
                    '/media/bunaken/Ensor/npx/CNO/Drop/EP_GLYT2_220519_CNO_Drop_A_g0/\
catgt_EP_GLYT2_220519_CNO_Drop_A_g0',
                    '/media/bunaken/Ensor/npx/CNO/Drop/EP_GLYT2_220519_CNO_Drop_B_g0/\
catgt_EP_GLYT2_220519_CNO_Drop_B_g0',
                    '/media/bunaken/Ensor/npx/CNO/Drop/EP_GLYT2_220527_CNO_Drop_A_g0/\
catgt_EP_GLYT2_220527_CNO_Drop_A_g0',
                    '/media/bunaken/Ensor/npx/CNO/Drop/EP_GLYT2_220527_CNO_Drop_B_g0/\
catgt_EP_GLYT2_220527_CNO_Drop_B_g0']
    npxpath_wCNO = ['/media/bunaken/Ensor/npx/CNO/Drop/EP_WT_220303_CNO_Drop_g0/\
catgt_EP_WT_220303_CNO_Drop_g0',
                    '/media/bunaken/Ensor/npx/CNO/Drop/EP_WT_220422_CNO_Drop_A_g0/\
catgt_EP_WT_220422_CNO_Drop_A_g0',
                    '/media/bunaken/Ensor/npx/CNO/Drop/EP_WT_220422_CNO_Drop_B_g0/\
catgt_EP_WT_220422_CNO_Drop_B_g0',
                    '/media/bunaken/Ensor/npx/CNO/Drop/EP_WT_220425_CNO_Drop_A_g0/\
catgt_EP_WT_220425_CNO_Drop_A_g0',
                    '/media/bunaken/Ensor/npx/CNO/Drop/EP_WT_220425_CNO_Drop_B_g0/\
catgt_EP_WT_220425_CNO_Drop_B_g0']
    npxpath_aPBS = ['/media/bunaken/Ensor/npx/CNO/Drop/EP_GLYT2_220429_PBS_Drop_A_g0/\
catgt_EP_GLYT2_220429_PBS_Drop_A_g0',
                    '/media/bunaken/Ensor/npx/CNO/Drop/EP_GLYT2_220502_PBS_Drop_B_g0/\
catgt_EP_GLYT2_220502_PBS_Drop_B_g0',
                    '/media/bunaken/Ensor/npx/CNO/Drop/EP_GLYT2_220506_PBS_Drop_B_g0/\
catgt_EP_GLYT2_220506_PBS_Drop_B_g0',
                    '/media/bunaken/Ensor/npx/WT/EP_WT_220602_A_g0/\
catgt_EP_WT_220602_A_g0',
                    '/media/bunaken/Ensor/npx/WT/EP_WT_220602_B_g0/\
catgt_EP_WT_220602_B_g0',
                    '/media/bunaken/Ensor/npx/WT/EP_WT_220607_A_g0/\
catgt_EP_WT_220607_A_g0',
                    '/media/bunaken/Ensor/npx/WT/EP_WT_220607_B_g0/\
catgt_EP_WT_220607_B_g0',
                    '/media/bunaken/Ensor/npx/WT/EP_WT_220609_A_g0/\
catgt_EP_WT_220609_A_g0',
                    '/media/bunaken/Ensor/npx/WT/EP_WT_220609_B_g0/\
catgt_EP_WT_220609_B_g0']

    # Animal id
    aid_mus = [0, 1, 2]
    aid_gCNO = [0, 0, 1, 1, 2, 2, 3, 4, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10]
    aid_wCNO = [11, 12, 12, 13, 13]
    aid_aPBS = [3, 4, 5, 14, 14, 15, 15, 16, 16]

    # Depth from which to consider cortex of interest
    cortex_depth_mus = [1000, 1160, 2000]
    cortex_depth_gCNO = [580, 740, 620, 1040, 2600, 760, 1080, 880, 1000, 1000, 1000, 1000, 900, 1000, 1000, 1000, 1000, 840, 1000]
    cortex_depth_wCNO = [1680, 880, 1100, 1180, 1140]
    cortex_depth_aPBS = [1100, 1000, 1000, 700, 700, 500, 800, 1080, 1060]

    # Depth npx insertion (3840 is most superficial position possible)
    def ml(x, tip):
        ''' 3840 is length probe, x is depth of insertion; 0 starts from tip;
        consider ml with depth 200μm, tip length 175
        '''
        if not tip:
            end_of_ml = 3840 - (3840 - x) - 201
        else:
            end_of_ml = 3840 - (3840 - x) - 201 - 175
        return end_of_ml

    mli_depth_mus = [0, 0, 0]
    mli_depth_gCNO = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    mli_depth_wCNO = [ml(2950, True), ml(2500, False), ml(2500, False), ml(2450, False), ml(2500, True)]
    mli_depth_aPBS = [ml(2450, True), ml(2400, True), ml(2300, True), ml(2250, True), ml(2050, True), ml(2250, True), ml(2300, True), ml(2350, True), ml( 2450, True)]

    # Time of CNO/APBS dropping
    t_drop_mus = [10, 10, 10]
    t_drop_gCNO = [10, 10, 10, 10, 10, 10, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20]
    t_drop_wCNO = [10, 20, 20, 20, 20]
    t_drop_aPBS = [20, 20, 20, 20, 20, 20, 20, 20, 20]

    # Selclu=True, clu_idx:
    # clusters to consider for each recording (not for muscimol cond) 
    clu_idx_gCNO = [[0], [0], [0], [0], [0], [0], [0], [0], [72, 89, 720, 683, 692, 101, 654, 152, 154, 153, 155, 156, 159, 162, 169, 161, 636, 165, 167, 168, 173, 170, 172, 178, 177, 176, 633, 448, 618, 614, 239, 576, 574, 562, 279, 503], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0]]
    clu_idx_wCNO = [[0], [0], [0], [0], [0]]
    clu_idx_aPBS = [[0], [0], [0], [1047, 135, 565, 793, 811, 211, 758, 246, 255, 274, 287, 316, 334, 953, 350, 821, 360, 881, 364, 385, 779, 392, 410, 411, 426], [0], [0], [0], [0], [0]]

    # Peth_discard=True:
    # clusters to discard because peth is blank
    pethdiscard_gCNO = [[10], [2], [0], [0], [], [0, 2, 6, 8], [], [], [], [], [], [], [], [], [], [], [], [], []]
    pethdiscard_wCNO = [[8, 9, 10, 12, 25, 41, 61, 65], [], [], [], [49]]
    pethdiscard_aPBS = [[], [], [], [], [], [], [], [], []]

    # Index of exemple bouts
    bout_idxs_gCNO = [[], [], [], [], [], [], [], [], [8, 10, 12, 13], [], [], [], [], [], [], [], [], [], []]
    bout_idxs_wCNO = [[], [], [], [], []]
    bout_idxs_aPBS = [[], [], [], [30, 32, 33, 34], [], [], [], [], []]

    # Index recordings with whisking
    wmask_gCNO = [True, True, True, True, True, True, True, False, True, True, True, False, True, True, True, True, True, True, True]
    wmask_wCNO = [True, True, True, True, True]
    wmask_aPBS = [True, False, True, True, True, True, True, True, True]

    # Mask Refined!!!!!
    # wmask_gCNO = [False, True, True, True, True, True, True, False, True, True, True, False, False, True, True, True, True, False, False]
    # wmask_wCNO = [True, True, True, True, True]
    # wmask_aPBS = [True, False, True, True, False, True, True, True, True]

    # Select or concatenate paths, aid, cortex_depths and t_drops
    wskpath = wskpath_gCNO + wskpath_wCNO + wskpath_aPBS
    npxpath = npxpath_gCNO + npxpath_wCNO + npxpath_aPBS
    aid = aid_gCNO + aid_wCNO + aid_aPBS
    cortex_depth = cortex_depth_gCNO + cortex_depth_wCNO + cortex_depth_aPBS
    t_drop = t_drop_gCNO + t_drop_wCNO + t_drop_aPBS
    clu_idx = clu_idx_gCNO + clu_idx_wCNO + clu_idx_aPBS
    pethdiscard_idx = pethdiscard_gCNO + pethdiscard_wCNO + pethdiscard_aPBS
    bout_idxs = bout_idxs_gCNO + bout_idxs_wCNO + bout_idxs_aPBS
    wmask = wmask_gCNO + wmask_wCNO + wmask_aPBS
    mli_depth = mli_depth_gCNO + mli_depth_wCNO + mli_depth_aPBS

    # Lable condition of each recording (compulsory)
    condition_mus = np.repeat('muscimol', len(npxpath_mus))
    conditions = np.concatenate(
        (np.repeat('gCNO', len(npxpath_gCNO)), np.repeat('wCNO', len(npxpath_wCNO)),
         np.repeat('aPBS', len(npxpath_aPBS))))

    # Check path existence
    for path in npxpath:
        print(os.path.exists(path))

    if mus:
        return npxpath_mus, aid_mus, cortex_depth_mus, t_drop_mus, clu_idx, condition_mus
    else:
        return wskpath, npxpath, aid, cortex_depth, t_drop, clu_idx, pethdiscard_idx, bout_idxs, wmask, conditions, mli_depth


def compute_metainfo():
    """ Function to compute general info about data, such as number of mice,
    recordings per conditions, units.
    """
    wskpath, npxpath, aid, cortex_depth, t_drop, clu_idx, pethdiscard_idx, bout_idxs, wmask, conditions, mli_depth = load_meta


if __name__ == '__main__':
    ''' Run script if spkcount_analysis.py module is main programme'''
    load_meta()

    wskpath, npxpath, aid, cortex_depth, t_drop, clu_idx, pethdiscard_idx, bout_idxs, wmask, conditions, mli_depth = load_meta()
    nn = 1
    for i in zip(wskpath, aid, wmask):
        print(i[0].split('/')[-2], i[1], i[2], nn)
        nn += 1
