""" Module for comparing whisker activity across datasets: load whisker
data from all selected recordings (now 25 after filtering for no/bad
whisking) - using get_wskvar and prepare_df; analyse slope of avg
whisking behaviour (pre and post drop) within t_slope (inspired by
Variance_pre_post_betweenr.svg window where variance across recording
pre and post differ maximally across conditions); then compute/get similarity
of trials based on pc dimensionality reduction; then plot.

Attention: last iteration of this module focused on for analise slope
(anal_slope), not for pc reductions.

Attention: computationally very expensive the correlation
analysis (pc reduction).

Consists of:

- load_wsk: used in get_wskvar to get wsk data

- align_wbouts: used in get_wskvar to get wsk data

- prepare_df: used in plot_wskvar to organise data for plotting

- corr_anal: used in plot_wskvar for correlation analysis of wsk data;
very expensive computationally, plus not used right now (but potentially)
again in future; output used also to sort trials in lowdim_visualisation_old.py

- setpoint_lowpass: another (better/straitforward?) way to get setpoint, by
low-pass filtering angle instead of avg points found with Hilbert Transform;
not used right now.
"""
import os
import sys
if not ('data_analysis' in os.getcwd()):
    sys.path.append(os.path.join(os.getcwd(), 'data_analysis'))

import pdb
import itertools
import pickle
import pandas as pd
import seaborn as sns
import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt
# import matplotlib.gridspec as gridspec
import whisk_analysis as wa
# from matplotlib.collections import LineCollection
from meta_data import load_meta
# from align_fun import align_spktrain
# from scipy.ndimage import gaussian_filter1d
# from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.linear_model import LinearRegression
from scipy import stats

def load_wsk(wskpath, t_drop=False, pre=None):
    """ Load whisker data
    """
    df, body, tot_wskdlen = wa.load_wdata(wskpath, t_drop=t_drop, pre=pre)
    angle_trace = wa.calc_angle(df, body)
    _, phase_trace, filtered_sig = wa.calc_phase(angle_trace)
    whiskd = wa.loadWhiskerData(df, angle_trace, phase_trace, filtered_sig)

    return whiskd


def align_wbouts(whisk_data, t_wind=[-2, 3], wsk=2, var='angle'):
    """ Align whisker variable trace to whisking bout onset, around t_wind; 
    similart to align_to_whisking in align_fun (but just for whisker variable,
    no spiking activity)
    """
    wsk = int(wsk[-1])
    sr = whisk_data.sr  # sampling rate
    t_wind = np.array(t_wind) * sr  # sec to frame
    start_bout = np.array(whisk_data.event_times[wsk]) * sr  # sec to frame
    aligned_wvar = np.empty([len(start_bout), np.diff(t_wind)[0]])
    aligned_wvar[:] = np.nan
    whisk_var = getattr(whisk_data, var)[wsk]

    for idx_b, stb in enumerate(start_bout):  # for all whisking bouts
        isw_edge = t_wind + int(stb)

        # Within edges of time
        if (isw_edge[0] >= 0) & (isw_edge[1] <= len(whisk_var)):

            # Align whisker variable (only once)
            aligned_wvar[idx_b, :] = whisk_var[isw_edge[0]:isw_edge[1]]

        # Account for neg time
        elif isw_edge[0] < 0:

            # Align whisker variable
            aligned_wvar[idx_b,
                         abs(isw_edge[0]):] = whisk_var[:isw_edge[1]]

        # Account for excess time
        elif isw_edge[1] > len(whisk_var):

            # Align whisker variable
            aligned_wvar[idx_b, :(
                len(whisk_var) -
                isw_edge[1])] = whisk_var[isw_edge[0]:]

    aligned_wvar_m = np.nanmean(aligned_wvar, 0)

    return aligned_wvar, aligned_wvar_m


def prepare_df(a_wvar_m, a_wvar, t_drop, conditions, n_rec, bstart, t_wind):
    """ Prepare data for plotting; return:
    a_wvar_m: a_wvar_m in df format;
    df_wpre_sb: single bouts pre drop, from a_wvar
    df_wpost_sb: single bouts post drop, from a_wvar
    df_wpre: average a_wvar pre drop
    df_wpost: average a_wvar post drop
    df_wpre_a: average a_wvar pre drop, aligned to 0 frame
    df_wpost_a: average a_wvar post drop, aligned to 0 frame
    df_diff: difference post - pre of aligned and average bouts (from
             df_wpre_a and df_wpost_a)
    df_var: variance in pre and post aligned and average bouts; it's
    list of [df_var_pre, df_var_post]

    Attention: averages is across single bouts within recordings
    """
    # Organise a_wvar_m in df
    len_trial = a_wvar_m.shape[1]
    __ = a_wvar_m.T - a_wvar_m[:, 0]  # align to start start
    midx = pd.MultiIndex.from_arrays([conditions, np.arange(n_rec)], names=('condition', 'rec'))
    a_wvar_m = pd.DataFrame(__, columns=midx).melt(value_name=f'{var}')
    a_wvar_m['time'] = np.arange(len_trial).tolist() * n_rec
    
    # Single bouts data
    list_pre = []
    list_post = []
    for rec in range(n_rec):
        # Mask pre and post
        mpre = bstart[rec] < t_drop[rec] * 60
        mpost = (bstart[rec] > (t_drop[rec] + 5) * 60) & (bstart[rec] < (t_drop[rec] * 2 + 5) * 60)
        # Pre
        midx = pd.MultiIndex.from_product([[rec], np.arange(sum(mpre))], names=('recording', 'bout'))
        # midx = pd.MultiIndex.from_product([[rec], np.arange(sum(mpre))])
        __ = pd.DataFrame(a_wvar[rec].T[:, mpre], columns=midx)
        list_pre.append(__)

        # Post
        midx = pd.MultiIndex.from_product([[rec], np.arange(sum(mpost))], names=('recording', 'bout'))
        # midx = pd.MultiIndex.from_product([[rec], np.arange(sum(mpost))])
        __ = pd.DataFrame(a_wvar[rec].T[:, mpost], columns=midx)
        list_post.append(__)

    df_wpre_sb = pd.concat(list_pre, axis=1)
    df_wpost_sb = pd.concat(list_post, axis=1)

    # Take mean
    df_wpre = df_wpre_sb.groupby(axis=1, level=0).mean()
    df_wpost = df_wpost_sb.groupby(axis=1, level=0).mean()

    # Variance across bouts within recordings - put in df
    # len = time * rec * t_drop(pre/post)
    df_var_pre_sb = df_wpre_sb.groupby(axis=1, level=0).var()
    df_var_post_sb = df_wpost_sb.groupby(axis=1, level=0).var()
    df_var_pre_sb.columns = pd.MultiIndex.from_arrays([conditions, np.arange(n_rec)], names=('condition', 'rec'))
    df_var_post_sb.columns = pd.MultiIndex.from_arrays([conditions, np.arange(n_rec)], names=('condition', 'rec'))
    df_var_pre_sb = df_var_pre_sb.melt(value_name=f'{var}')
    df_var_post_sb = df_var_post_sb.melt(value_name=f'{var}')
    df_var_pre_sb['time'] = np.arange(len_trial).tolist() * n_rec
    df_var_post_sb['time'] = np.arange(len_trial).tolist() * n_rec
    df_var_pre_sb['t_drop'] = ['pre'] * len_trial * n_rec
    df_var_post_sb['t_drop'] = ['post'] * len_trial * n_rec
    df_var_sb = pd.concat([df_var_pre_sb, df_var_post_sb], axis=0, ignore_index=True)

    # Add condition level
    df_wpre.columns = pd.MultiIndex.from_arrays([conditions, df_wpre.columns.values], names=('condition', 'rec'))
    df_wpost.columns = pd.MultiIndex.from_arrays([conditions, df_wpost.columns.values], names=('condition', 'rec'))
    # Transform in long format
    df_wpre = df_wpre.melt(value_name=f'{var}')
    df_wpost = df_wpost.melt(value_name=f'{var}')
    # Add time
    df_wpre['time'] = np.arange(len_trial).tolist() * n_rec
    df_wpost['time'] = np.arange(len_trial).tolist() * n_rec

    # Correct for baseline (align at start or baseline mean)
    # df_wpre_a = df_wpre - df_wpre.iloc[0, :]
    # df_wpost_a = df_wpost - df_wpost.iloc[0, :]
    pdb.set_trace()
    # df_wpre_a = df_wpre - df_wpre.iloc[:np.abs(t_wind[0]*299), :].mean(axis=0)
    # df_wpost_a = df_wpost - df_wpost.iloc[:np.abs(t_wind[0]*299), :].mean(axis=0)

    df_wpre_a = df_wpre.copy()
    df_wpost_a = df_wpost.copy()

    df_wpre_a[f'{var}'] = df_wpre_a.groupby(by='rec', group_keys=False)[f'{var}'].apply(lambda x: x - x.iloc[:np.abs(t_wind[0]*299)].mean())
    df_wpost_a[f'{var}'] = df_wpost_a.groupby(by='rec', group_keys=False)[f'{var}'].apply(lambda x: x - x.iloc[:np.abs(t_wind[0]*299)].mean())

    # Pivot over condition and rec (multiindex)
    df_wpre_a = df_wpre_a.pivot(index='time', columns=['condition','rec'], values=f'{var}')
    df_wpost_a = df_wpost_a.pivot(index='time', columns=['condition','rec'], values=f'{var}')

    # Variance pre and post across recording of (aligned) average responses
    df_var_pre = df_wpre_a.groupby(by='condition', group_keys=False, axis=1).var()
    df_var_post = df_wpost_a.groupby(by='condition', group_keys=False, axis=1).var()

    df_var_pre = df_var_pre.melt()
    df_var_post = df_var_post.melt()

    df_var_pre['t_drop'] = ['pre'] * len_trial * np.unique(conditions).size
    df_var_post['t_drop'] = ['post'] * len_trial * np.unique(conditions).size
    df_var_pre['time'] = np.arange(len_trial).tolist() * np.unique(conditions).size
    df_var_post['time'] = np.arange(len_trial).tolist() * np.unique(conditions).size
    df_var = pd.concat([df_var_pre, df_var_post], axis=0, ignore_index=True)
    df_var.rename(columns={'value':f'{var}'}, inplace=True)

    # Transform in long format
    df_wpre_a = df_wpre_a.melt(value_name=f'{var}')
    df_wpost_a = df_wpost_a.melt(value_name=f'{var}')
    # Add time
    df_wpre_a['time'] = np.arange(len_trial).tolist() * n_rec
    df_wpost_a['time'] = np.arange(len_trial).tolist() * n_rec

    # DataFrame differences pre-post (aligned averages )
    df_diff = df_wpre_a.copy()
    df_diff[f'{var}'] = df_wpost_a[f'{var}'] - df_wpre_a[f'{var}']

    return a_wvar_m, df_wpre_sb, df_wpost_sb, df_wpre, df_wpost, df_wpre_a, df_wpost_a, df_diff, df_var_sb, df_var


def corr_anal(data, pred, postd, nth_egnvec=0):
    """ Compute similarity of each bout with 1st, 2nd ad 3rd eignvec;
    use this to group similar whisker bouts
    """
    # Save
    simil_1pc = []
    simil_2pc = []
    simil_3pc = []
    var_explained = pd.DataFrame(columns=['pc1', 'pc2', 'pc3'])
    egnvec_save = []

    for idx_rec, allbouts in enumerate(data):
        df_wvar = pd.DataFrame(allbouts, copy=True)

        # Get correlation matrix
        corr = df_wvar.corr(method='pearson')

        # Get eignval, eignvec and project
        egnval, egnvec = la.eig(corr)
        idxsort = np.argsort(egnval.real)[::-1]
        egnval = egnval.real[idxsort]
        egnvec = egnvec.real[:, idxsort]

        egnvec_save.append(egnvec[:, :3])

        # var_explained.append(egnval[0] / np.sum(egnval))
        __ = np.array([egnval[0], egnval[1], egnval[2]]) / np.sum(egnval)
        __ = pd.DataFrame({'pc1': __[0], 'pc2': __[1], 'pc3': __[2]}, index=[idx_rec])
        var_explained = pd.concat([var_explained, __])

        proj1 = []
        proj2 = []
        proj3 = []
        pdb.set_trace()
        for __ in [pred, postd]:
            proj1.append(np.dot(egnvec[:, 0].real, __.loc[:, idx_rec]))
            proj2.append(np.dot(egnvec[:, 1].real, __.loc[:, idx_rec]))
            proj3.append(np.dot(egnvec[:, 2].real, __.loc[:, idx_rec]))

        simil_1pc.append(proj1)
        simil_2pc.append(proj2)
        simil_3pc.append(proj3)

    return simil_1pc, simil_2pc, simil_3pc, var_explained, egnvec_save


def setpoint_lowpass(data, order, Wn, sr):
    ''' Compute setpoint by lowpass filtering angle instead of
    Mary's get_slow_var method
    '''
    from scipy.signal import buttord, butter, filtfilt
    b, a = butter(order, Wn, btype='lowpass', fs=sr)
    spt = filtfilt(b, a, data)
    return spt


def anal_slope(df_wpre_a, df_wpost_a, n_rec, conditions, t_slope=[580, 660]):
    """
    """
    # Compute slope
    coef_pre = []
    coef_post = []

    # for i in range(25):
    #     # plt.plot(df_wpre_a[df_wpre_a.rec==i].angle.values[t_slope[0]:t_slope[1]])
    #     # plt.plot(setpoint_lowpass(df_wpre_a[df_wpre_a.rec==i].angle, 2, 10, 299)[t_slope[0]:t_slope[1]])
    #     plt.title(f'pre{i}')
    #     plt.show()
    #     # plt.plot(df_wpost_a[df_wpost_a.rec==i].angle.values[t_slope[0]:t_slope[1]])
    #     # plt.plot(setpoint_lowpass(df_wpost_a[df_wpost_a.rec==i].angle, 2, 10, 299)[t_slope[0]:t_slope[1]])
    #     plt.title(f'pre{i}')
    #     plt.show()
    pdb.set_trace()
    for rec in range(n_rec):
        pre_data = df_wpre_a[df_wpre_a.rec==rec]
        # pre_data = setpoint_lowpass(pre_data.angle.values, 2, 20, sr)
        post_data = df_wpost_a[df_wpost_a.rec==rec] # convert rad to degrees
        coef_pre.append(
            np.polyfit(np.arange(np.diff(t_slope)),
                       pre_data.angle.iloc[t_slope[0]:t_slope[1]], 1))
        coef_post.append(
            np.polyfit(np.arange(np.diff(t_slope)),
                       post_data.angle.iloc[t_slope[0]:t_slope[1]], 1))

    # Organise in df
    coef_pre = pd.DataFrame(coef_pre)
    coef_post = pd.DataFrame(coef_post)
    pdb.set_trace()
    coef_pre.index = conditions
    coef_post.index = conditions
    intercept = pd.concat([coef_pre[1], coef_post[1]]).values
    coef_pre.drop(1, axis=1, inplace=True)
    coef_post.drop(1, axis=1, inplace=True)
    coef_pre.columns = ['coef']
    coef_post.columns = ['coef']
    

    # Difference in coefficients
    # coef_diff = coef_pre - coef_post
    coef_diff = coef_post - coef_pre

    # Draw slope...
    df_slope = np.arange(np.diff(t_slope))
    df_slope = np.repeat(np.expand_dims(df_slope, axis=1), n_rec * 2, axis=1)
    df_slope = df_slope * pd.concat([coef_pre, coef_post]).values[:,0]
    # df_slope = df_slope + pd.concat([df_wpre_a[df_wpre_a.time==t_slope[0]], df_wpost_a[df_wpost_a.time==t_slope[0]]]).angle.values
    df_slope = df_slope + intercept
    df_slope = pd.DataFrame(df_slope)
    # df_slope.columns = midx
    df_slope.columns = pd.MultiIndex.from_product([['pre', 'post'], coef_pre.index])
    df_slope = df_slope.T
    df_slope['rec'] = np.arange(25).tolist() * 2
    df_slope = df_slope.set_index('rec', append=True)
    df_slope = df_slope.T
    df_slope.index = df_wpre_a.time.unique()[t_slope[0]:t_slope[1]]
    # .. and melt
    df_slope = df_slope.melt(ignore_index=False)
    df_slope.reset_index(inplace=True)
    df_slope.rename(columns={'index': 'time', 'variable_0': 't_drop', 'variable_1': 'condition', 'variable_2': 'rec', 'value': 'angle'}, inplace=True)
    pdb.set_trace()

    # Make coef_prepost
    coef_pre.columns = ['coef_pre']
    coef_post.columns = ['coef_post']
    coef_prepost = pd.concat([coef_pre, coef_post], axis=1)
    coef_prepost.rename(index={'wCNO':'control', 'aPBS':'control'}, inplace=True)
    

    return coef_prepost, coef_diff, df_slope


    # for i in range(25):
    #     plt.plot(np.arange(np.diff(t_slope)), df_wpre_a[df_wpre_a.rec==i].angle.iloc[t_slope[0]:t_slope[1]])
    #     plt.plot(np.arange(np.diff(t_slope)), np.arange(np.diff(t_slope))*coef_pre[i][0] + coef_pre[i][1])
    #     plt.title(f'pre{i}')
    #     plt.show()
    #     plt.plot(np.arange(np.diff(t_slope)), df_wpost_a[df_wpost_a.rec==i].angle.iloc[t_slope[0]:t_slope[1]])
    #     plt.plot(np.arange(np.diff(t_slope)), np.arange(np.diff(t_slope))*coef_post[i][0] + coef_post[i][1])
    #     plt.title(f'post{i}')
    #     plt.show()


def wsk_corr(whisk_data_all_plot):
    ''' Compute correlation between whisking for each recording, for
    pre- and post-drop period
    '''
    df_wskcorr = pd.DataFrame()
    triu_mask = np.triu(np.ones([3, 3]), k=1).astype(bool)
    cc_combination = ['0-1', '0-2', '1-2']
    for period in ['pre', 'post']:
        if period == 'pre':
            idx_period = 5
        elif period == 'post':
            idx_period = 6
        df_wsk0 = whisk_data_all_plot[0][idx_period].pivot(index='time', columns=['condition', 'rec'], values='angle')
        df_wsk0 = pd.concat([df_wsk0], axis=1, keys=['whisk0'], names=['whisker'])
        df_wsk1 = whisk_data_all_plot[1][idx_period].pivot(index='time', columns=['condition', 'rec'], values='angle')
        df_wsk1 = pd.concat([df_wsk1], axis=1, keys=['whisk1'], names=['whisker'])
        df_wsk2 = whisk_data_all_plot[2][idx_period].pivot(index='time', columns=['condition', 'rec'], values='angle')
        df_wsk2 = pd.concat([df_wsk2], axis=1, keys=['whisk2'], names=['whisker'])
        df_wsk = pd.concat([df_wsk0, df_wsk1, df_wsk2], axis=1)
        df_wskcorr__ = df_wsk.groupby(by='rec', axis=1, group_keys=False).corr()
        df_wskcorr_ = pd.DataFrame()
        for rec in range(25):
            __ = df_wskcorr__.iloc[rec * 3: rec * 3 + 3, rec * 3: rec * 3 + 3]
            cond = __.columns.get_level_values(1).unique().tolist()
            midx = pd.MultiIndex.from_product([cond, [rec], cc_combination])
            __ = pd.DataFrame(np.stack(__.where(triu_mask).values)).stack()
            __.index = midx
            __ = pd.DataFrame(__, columns=[period])
            df_wskcorr_ = pd.concat([df_wskcorr_, __], axis=0)

        df_wskcorr = pd.concat([df_wskcorr, df_wskcorr_], axis=1)
    df_wskcorr.index.names = ['cond', 'rec', 'wsk_comb']
    df_wskcorr.columns.names = ['period']
    df_wskcorr = df_wskcorr.reset_index().melt(id_vars=['cond', 'rec', 'wsk_comb'], value_vars=['pre', 'post'])
    df_wskcorr['cond'] = df_wskcorr['cond'].replace({'wCNO':'control', 'aPBS':'control'})

    return df_wskcorr


def get_wskvar(whisker='whisk2', t_wind=[-2, 3], drop_rec=[0, 10, 15, 16, 25], var='angle', cgs=2, surr=False, pre=None, save=False, load=True, plot=False):
    """ Load or compute wskvar
    """
    # load metadata
    wskpath, __, aid, cortex_depth, t_drop, __, __, __, wmask, conditions, __ = load_meta()

    # Drop rec with no whisk
    wskpath = np.array(wskpath).astype(str)[wmask].tolist()
    t_drop = np.array(t_drop)[wmask]
    conditions = conditions[wmask]

    # Drop rec with poor whisking (equivalent of good_list in other modules)
    wmask2 = ~np.in1d(np.arange(len(wskpath)), np.array(drop_rec))
    wskpath = np.array(wskpath).astype(str)[wmask2].tolist()
    t_drop = t_drop[wmask2]
    conditions = conditions[wmask2]
    n_rec = sum(wmask2)

    # Load and preprocess whisker data
    # pdb.set_trace()
    if load:
        # import pickle
        if os.path.basename(os.getcwd()) != 'data_analysis':
            os.chdir('./data_analysis')

        with open('save_data/aligned_wvar_goodlist.pickle', 'rb') as f:
            a_wvar, a_wvar_m, bstart = pickle.load(f)
    else:
        # Get data
        whiskd = []
        a_wvar = []
        a_wvar_m = []
        # Get all rec
        for wrec in wskpath:
            # pdb.set_trace()
            df, body, __ = wa.load_wdata(wrec, t_drop=False)
            angle_trace = wa.calc_angle(df, body)
            __, phase_trace, filtered_sig = wa.calc_phase(angle_trace)
            __ = wa.loadWhiskerData(df, angle_trace, phase_trace, filtered_sig)
            aligned_wvar, aligned_wvar_m = align_wbouts(__, t_wind=t_wind, wsk=whisker, var=var)
            print(wrec)

            whiskd.append(__)
            a_wvar.append(aligned_wvar)
            a_wvar_m.append(aligned_wvar_m)
            # pdb.set_trace()

        a_wvar_m = np.array(a_wvar_m)

        bstart = []
        for data in whiskd:
            bstart.append(data.event_times[int(whisker[-1])])

    # If load false, save wsk data
    if save:
        if os.path.basename(os.getcwd()) != 'data_analysis':
            os.chdir('./data_analysis')
        aligned_wvar = []
        aligned_wvar.append(a_wvar)
        aligned_wvar.append(a_wvar_m)
        aligned_wvar.append(bstart)

        try:
            with open('save_data/aligned_wvar_goodlist.pickle', 'wb') as f:
                pickle.dump(aligned_wvar, f)
        except FileNotFoundError:
            print('created dir save_data')
            os.makedirs(os.getcwd() + '/save_data')
            with open('save_data/aligned_wvar_goodlist.pickle', 'wb') as f:
                pickle.dump(aligned_wvar, f)

    return a_wvar, a_wvar_m, bstart, t_drop, conditions, n_rec, wskpath


def plot_wskvar(whisker='whisk2', t_wind=[-2, 3], t_slope = [580, 660], drop_rec=[0, 10, 15, 16, 25], var='angle', cgs=2, surr=False, pre=None, save=False, load=True, plot=False):
    """ Plot whisker activity; if pre=None, load all whisking bouts;
    alternatively load only pre or post.
    """
    # Get whisker data
    __ = get_wskvar(whisker=whisker, t_wind=t_wind, drop_rec=drop_rec, var=var, cgs=cgs, surr=surr, pre=pre, save=save, load=load)
    a_wvar = __[0]
    a_wvar_m = __[1]
    bstart = __[2]
    t_drop = __[3]
    conditions = __[4]
    n_rec = __[5]
    id_wtrace = __[6]

    # Prepare data for plotting
    __ = prepare_df(a_wvar_m, a_wvar, t_drop, conditions, n_rec, bstart, t_wind)
    a_wvar_m = __[0]
    df_wpre_sb = __[1]
    df_wpost_sb = __[2]
    df_wpre = __[3]
    df_wpost = __[4]
    df_wpre_a = __[5]
    df_wpost_a = __[6]
    df_diff = __[7]
    df_var_sb = __[8]
    df_var = __[9]

    # Convert rad to degrees
    df_wpre_a.angle = df_wpre_a.angle * 180 / np.pi
    df_wpost_a.angle = df_wpost_a.angle * 180 / np.pi

    pdb.set_trace()
    # Get analysis of average pre/post whisking slope
    coef_prepost, coef_diff, df_slope = anal_slope(df_wpre_a, df_wpost_a, n_rec, conditions, t_slope=t_slope)
    coef_prepost = coef_prepost * 1000 / 299 # from deg/frame to deg/msec
    coef_diff = coef_diff * 1000 / 299 # from deg/frame to deg/msec


    if save:
        with open('save_data/aligned_wvar_goodlist.pickle', 'wb') as f:
            pickle.dump(aligned_wvar, f)

        

    # Pull Controls together
    coef_diff_p = coef_diff.copy()
    coef_diff_p.rename(index={'wCNO':'control', 'aPBS':'control'}, inplace=True)
    df_slope_p = df_slope.copy()
    df_slope_p.condition.where(df_slope_p.condition=='gCNO', other='control', inplace=True)

    # Test for equal variance (Levene's test with median)
    lev = stats.levene(coef_diff.loc['gCNO', 'coef'].values, coef_diff.loc['wCNO', 'coef'].values, coef_diff.loc['aPBS', 'coef'].values)
    lev_p = stats.levene(coef_diff_p.loc['gCNO', 'coef'].values, coef_diff_p.loc['control', 'coef'].values)

    # # To test for different choices of t_slope !!!
    # t_slope2 = [550, 680]
    # coef_diff2, df_slope2 = anal_slope(df_wpre_a, df_wpost_a, n_rec, conditions, t_slope=t_slope2)
    # coef_diff2_p = coef_diff2.copy()
    # coef_diff2_p.rename(index={'wCNO':'control', 'aPBS':'control'}, inplace=True)
    # df_slope2_p = df_slope2.copy()
    # df_slope2_p.condition.where(df_slope2_p.condition=='gCNO', other='control', inplace=True)

    # t_slope3 = [560, 680]
    # coef_diff3, df_slope3 = anal_slope(df_wpre_a, df_wpost_a, n_rec, conditions, t_slope=t_slope3)
    # coef_diff3_p = coef_diff3.copy()
    # coef_diff3_p.rename(index={'wCNO':'control', 'aPBS':'control'}, inplace=True)
    # df_slope3_p = df_slope.copy()
    # df_slope3_p.condition.where(df_slope3_p.condition=='gCNO', other='control', inplace=True)


    # for rec in range(n_rec):
    pdb.set_trace()
    if load:
        with open('save_data/save_whisk_corr.pickle', 'rb') as f:
            simil_1pc, simil_2pc, simil_3pc, var_explained, egnvec_save = pickle.load(f)
    else:
        simil_1pc, simil_2pc, simil_3pc, var_explained, egnvec_save = corr_anal(a_wvar, df_wpre_sb, df_wpost_sb)

        if save:
            save_whisk_corr = [simil_1pc, simil_2pc, simil_3pc, var_explained, egnvec_save]
            try:
                with open('save_data/save_whisk_corr.pickle', 'wb') as f:
                    pickle.dump(save_whisk_corr, f)
            except FileNotFoundError:
                print('created dir save_data')
                os.makedirs(os.getcwd() + '/save_data')
                with open('save_data/save_whisk_corr.pickle', 'wb') as f:
                    pickle.dump(save_whisk_corr, f)


    # Plot Figures ##################################
    # Figure parameters
    # lcolor = 'darkgoldenrod'
    # style = 'italic'
    figsize = (14, 12)
    sns.set_context("poster", font_scale=1)
    sns.color_palette("deep")
    plt.rcParams['axes.facecolor'] = 'lavenderblush'
    plt.rcParams["figure.edgecolor"] = 'black'

    xticks = np.arange(0, np.diff(t_wind) * 299, 299)
    xtick_labels = np.arange(t_wind[0], t_wind[1])
    n_col = np.round(np.sqrt(n_rec)).astype(int)
    n_row = n_rec // n_col if (n_rec // n_col *
                               n_col) == n_rec else n_rec // n_col + 1

    szfig_y = 4 * n_row
    szfig_x = 6 * n_col

    xtick_labels_slope = np.array(t_slope) / 299 + t_wind[0]
    xtick_labels_slope = np.round(np.linspace(xtick_labels_slope[0], xtick_labels_slope[1], 4), 2)
    # xticks_slope = (xtick_labels_slope - xtick_labels_slope[0]) * 299 + t_slope[0]
    xticks_slope = np.linspace(t_slope[0], t_slope[1], 4)


    # fig1, ax1 = plt.subplots(2, figsize=figsize, tight_layout=True, sharex=True, sharey=True)
    # # for idx, cond in enumerate(np.unique(conditions)):
    # for idx, wdata in enumerate([df_wpre_a, df_wpost_a]):
    #     sns.lineplot(x='time', y=f'{var}', data=wdata[wdata['condition'] == 'gCNO'], errorbar='sd', ax=ax1[idx], label='gCNO')
    #     sns.lineplot(x='time', y=f'{var}', data=wdata[wdata['condition'] == 'wCNO'], errorbar='sd', ax=ax1[idx], label='wCNO')
    #     sns.lineplot(x='time', y=f'{var}', data=wdata[wdata['condition'] == 'aPBS'], errorbar='sd', ax=ax1[idx], label='aPBS')
    #     # ax1[idx].set_xticks(ax1[idx].get_xticks())
    #     # ax1[idx].xaxis.set_major_locator(mticker.FixedLocator(xtick_labels))
    #     ax1[idx].set_xticks(xticks)
    #     ax1[idx].set_xticklabels(xtick_labels)

    # ax1[0].set_title('pre')
    # ax1[1].set_title('post')

    # fig1.suptitle('Compare conditions (aligned average pre)')

    # # Figure pre vs post drop
    # fig2, ax2 = plt.subplots(3, figsize=figsize, tight_layout=True)
    # # for idx, cond in enumerate(np.unique(conditions)):
    # for idx, cond in enumerate(['gCNO', 'wCNO', 'aPBS']):
    #     sns.lineplot(x='time', y=f'{var}', data=df_wpre[df_wpre['condition'] == cond], errorbar='sd', ax=ax2[idx], label='pre')
    #     sns.lineplot(x='time', y=f'{var}', data=df_wpost[df_wpost['condition'] == cond], errorbar='sd', ax=ax2[idx], label='post')
    #     ax2[idx].set_xticks(xticks)
    #     ax2[idx].set_xticklabels(xtick_labels)
    #     ax2[idx].set_title(f'{cond}')

    # fig2.suptitle('Compare pre and post (not aligned)')

    # # Figure pre vs post drop (aligned)
    # fig3, ax3 = plt.subplots(3, figsize=figsize, tight_layout=True, sharex=True, sharey=True)
    # # for idx, cond in enumerate(np.unique(conditions)):
    # for idx, cond in enumerate(['gCNO', 'wCNO', 'aPBS']):
    #     sns.lineplot(x='time', y=f'{var}', data=df_wpre_a[df_wpre['condition'] == cond], ax=ax3[idx], units='rec', estimator=None, label='pre', legend=False)
    #     sns.lineplot(x='time', y=f'{var}', data=df_wpost_a[df_wpost['condition'] == cond], ax=ax3[idx], units='rec', estimator=None, label='post', legend=False)
    #     ax3[idx].set_xticks(xticks)
    #     ax3[idx].set_xticklabels(xtick_labels)
    #     ax3[idx].set_title(f'{cond}')
    #     # Set y tick labels
    #     if idx == 1:
    #         ax3[idx].set_ylabel('whisker angle (deg)')
    #     else:
    #         ax3[idx].set_ylabel('')
    #     # Legend
    #     if idx == 0:
    #         yticks = ax3[idx].get_yticks()
    #         scale = 10 / (abs(yticks[0]) * 180 / np.pi)
    #         ax3[idx].set_yticks(yticks * scale)
    #         ax3[idx].set_yticklabels((yticks * scale * 180 / np.pi).astype(int))

    #         lines, labels = ax3[idx].get_legend_handles_labels()
    #         ax3[idx].legend([lines[0], lines[18]], [labels[0], labels[18]])

    # # fig3.suptitle('Compare conditions (aligned to average pre)')

    # # Compare post-pre difference in aligned average responses
    # fig4, ax4 = plt.subplots(3, figsize=figsize, tight_layout=True, sharex=True, sharey=True)
    # for idx, cond in enumerate(['gCNO', 'wCNO', 'aPBS']):
    #     sns.lineplot(x='time', y=f'{var}', data=df_diff[df_diff['condition'] == cond], ax=ax4[idx], estimator=None, units='rec')
    #     ax4[idx].set_xticks(xticks)
    #     ax4[idx].set_xticklabels(xtick_labels)
    #     ax4[idx].set_title(f'{cond}')

    # fig4.suptitle('Difference post-pre whisker activity across conditions')

    # # Compare pre and post variance within recordings
    # fig5, ax5 = plt.subplots(3, figsize=figsize, tight_layout=False, sharex=True, sharey=True)
    # for idx, cond in enumerate(['gCNO', 'wCNO', 'aPBS']):
    #     sns.lineplot(x='time', y=f'{var}', data=df_var_sb[df_var_sb['condition'] == cond], hue='t_drop', errorbar='se', ax=ax5[idx], legend=False)
    #     ax5[idx].set_xticks(xticks)
    #     ax5[idx].set_xticklabels(xtick_labels)
    #     ax5[idx].set_title(f'{cond}')

    # fig5.suptitle('Compare pre and post variace bouts within recording across conditions')

    # # Compare pre and post variance of average whisker trace across recordings
    # fig6, ax6 = plt.subplots(3, figsize=figsize, tight_layout=True, sharex=True, sharey=False)
    # for idx, cond in enumerate(['gCNO', 'wCNO', 'aPBS']):
    #     sns.lineplot(x='time', y=f'{var}', data=df_var[df_var['condition'] == cond], hue='t_drop', ax=ax6[idx], legend=False)
    #     ax6[idx].set_xticks(xticks)
    #     ax6[idx].set_xticklabels(xtick_labels)
    #     ax6[idx].set_title(f'{cond}')
    #     yticks = ax6[idx].get_yticks()
    #     ax6[idx].set_yticks(yticks)
    #     ax6[idx].set_yticklabels(np.round(yticks * 180 / np.pi, 2))

    #     if idx == 0:
    #         # lines, labels = ax6[idx].get_legend_handles_labels()
    #         ax6[idx].legend(['pre', 'post'])

    #     if idx == 1:
    #         ax6[idx].set_ylabel(r'whisker angle (rad$^2$)')
    #     else:
    #         ax6[idx].set_ylabel('')
    #         # ax6[idx].get_yaxis().set_visible(False)

    # hspace = 0.0
    # fig6.subplots_adjust(hspace=hspace)


    # fig6.suptitle('Compare pre and post variace of average wsk trace across conditions')


    # # Plot eigenvectors
    # fig7, axs7 = plt.subplots(n_row,
    #                           n_col,
    #                           figsize=(szfig_x, szfig_y),
    #                           sharex=False,
    #                           tight_layout=True)

    # # Turn off all axes and turn on one-by-on to avoid empty axes
    # for axis in axs7.flat:
    #     axis.set_axis_off()

    # for idx, rec in zip(
    #         itertools.product(np.arange(n_row), np.arange(n_col)),
    #         range(n_rec)):

    #     # Plot tuning curves
    #     axs7[idx].set_axis_on()
    #     sns.lineplot(data=egnvec_save[rec], legend=False, ax=axs7[idx])
    #     axs7[idx].set_xticks(xticks)
    #     axs7[idx].set_xticklabels(xtick_labels)
    #     if (idx[0] == 0) & (idx[1] == n_col - 1):
    #         leg = axs7[idx].legend(bbox_to_anchor=(1, 1.0), loc='upper right', labels=['egvec1', 'egvec2', 'egvec3'], prop={'size': 20})
    #         for lines in leg.get_lines():
    #             lines.set_linewidth(3)

    # fig7.suptitle('First three eigenvectors', size=30, y=1)
    # fig7.subplots_adjust(top=0.8)

    # # Plot var explained
    # fig8, axs8 = plt.subplots(n_row,
    #                           n_col,
    #                           figsize=(szfig_x, szfig_y),
    #                           sharex=False,
    #                           tight_layout=True)

    # # Turn off all axes and turn on one-by-on to avoid empty axes
    # for axis in axs8.flat:
    #     axis.set_axis_off()

    # for idx, rec in zip(
    #         itertools.product(np.arange(n_row), np.arange(n_col)),
    #         range(n_rec)):

    #     # Plot tuning curves
    #     axs8[idx].set_axis_on()
    #     __ = var_explained.iloc[rec, :]
    #     sns.barplot(y=__.values, x=__.index, ax=axs8[idx])

    # fig8.suptitle('Variance explained', size=30, y=1)
    # fig8.subplots_adjust(top=0.8)


    # # Plot mean activity
    # fig9, axs9 = plt.subplots(n_row,
    #                           n_col,
    #                           figsize=(szfig_x, szfig_y),
    #                           sharex=False,
    #                           tight_layout=True)

    # # Turn off all axes and turn on one-by-on to avoid empty axes
    # for axis in axs9.flat:
    #     axis.set_axis_off()

    # for idx, rec in zip(
    #         itertools.product(np.arange(n_row), np.arange(n_col)),
    #         range(n_rec)):

    #     # Plot tuning curves
    #     axs9[idx].set_axis_on()
    #     sns.lineplot(x='time', y=var, data=a_wvar_m[a_wvar_m['rec'] == rec], legend=False, ax=axs9[idx])
    #     axs9[idx].set_xticks(xticks)
    #     axs9[idx].set_xticklabels(xtick_labels)

    # fig9.suptitle('Mean activity for recording', size=30, y=1)
    # fig9.subplots_adjust(top=0.8)

    # # Control separated
    # fig10, ax10 = plt.subplots(3, figsize=figsize, tight_layout=False, sharex=True)
    # for idx, cond in enumerate(['gCNO', 'wCNO', 'aPBS']):
    #     sns.histplot(data=coef_diff[coef_diff.index==cond], binwidth=0.00005, ax=ax10[idx], legend=None)
    #     ax10[idx].set_title(cond)
    #     if idx == 1:
    #         ax10[idx].set_ylabel('count')
    #     if idx == 2:
    #         ax10[idx].set_xlabel('coef diff')

    # fig10.suptitle('Histogram difference pre/post slopes wsk linear fit')

    # # Control pulled
    # fig11, ax11 = plt.subplots(2, figsize=figsize, tight_layout=False, sharex=True)
    # for idx, cond in enumerate(['gCNO', 'control']):
    #     sns.histplot(data=coef_diff_p[coef_diff_p.index==cond], binwidth=0.00005, ax=ax11[idx], legend=None)
    #     ax11[idx].set_title(cond)
    #     ax11[idx].set_ylabel('count')
    #     if idx == 1:
    #         ax11[idx].set_xlabel('coef diff')

    # fig11.suptitle('Histogram difference pre/post slopes wsk linear fit')

    # # Control separated
    # fig12, axs12 = plt.subplots(3, 2, figsize=figsize, tight_layout=True, sharex=True)
    # for col, (row, cond) in itertools.product(range(2), enumerate(['gCNO', 'wCNO', 'aPBS'])):
    #     data = [df_wpre_a, df_wpost_a][col]
    #     data = data[data.condition==cond]
    #     sns.lineplot(y='angle', x='time', data=data, units='rec', estimator=None, errorbar=None, ax=axs12[row, col])
    #     axs12[row, col].fill_between(np.array(t_slope), axs12[row, col].get_ylim()[0], axs12[row, col].get_ylim()[1], color='orange')
    #     axs12[row, col].set_xticks(xticks)
    #     axs12[row, col].set_xticklabels(xtick_labels)
    #     if col == 0:
    #         axs12[row, col].set_title(f'pre {cond}')
    #     else:
    #         axs12[row, col].set_title(f'post {cond}')

    # fig12.suptitle('Time window for wsk  linear fit')

    # # Control pulled
    # fig13, axs13 = plt.subplots(2, 2, figsize=figsize, tight_layout=True, sharex=True, sharey='row')
    # for col, (row, cond) in itertools.product(range(2), enumerate(['gCNO', 'control'])):
    #     data = [df_wpre_a, df_wpost_a][col]
    #     if cond == 'gCNO':
    #         data = data[data.condition==cond]
    #     else:
    #         data = data[data.condition!='gCNO']
    #     sns.lineplot(y='angle', x='time', data=data, units='rec', estimator=None, errorbar=None, ax=axs13[row, col])
    #     # axs13[row, col].fill_between(np.array(t_slope), axs13[row, col].get_ylim()[0], axs13[row, col].get_ylim()[1], color='orange')
    #     if row == 0:
    #         axs13[row, col].fill_between(np.array(t_slope), -0.05, 0.33, color='orange')
    #     else:
    #         axs13[row, col].fill_between(np.array(t_slope), -0.05, 0.47, color='orange')

    #     axs13[row, col].set_xticks(xticks)
    #     axs13[row, col].set_xticklabels(xtick_labels)
    #     if col == 0:
    #         axs13[row, col].set_title(f'pre {cond}')
    #     else:
    #         axs13[row, col].set_title(f'post {cond}')

    # fig13.suptitle('Time window for wsk  linear fit')

    # Control pulled
    fig_linefit_p = {}
    for tdrop, (__, cond) in itertools.product(range(2), enumerate(['gCNO', 'control'])):
        fig_line, ax_line = plt.subplots()
        # Whisker data
        data = [df_wpre_a, df_wpost_a][tdrop]
        if cond == 'gCNO':
            data = data[data.condition==cond]
        else:
            data = data[data.condition!='gCNO']
        # data = data[data.condition == cond]
        data = data[(data.time >= t_slope[0]) & (data.time < t_slope[1])]
        # sns.lineplot(x='time', y=f'{var}', data=data, units='rec', estimator=None, ax=axs2[row, col])
        sns.lineplot(x='time', y=f'{var}', data=data, units='rec', estimator=None, ax=ax_line)
        # Slopes
        data = [df_slope[df_slope.t_drop=='pre'], df_slope[df_slope.t_drop=='post']][tdrop]
        if cond == 'gCNO':
            data = data[data.condition==cond]
        else:
            data = data[data.condition!='gCNO']
        # data = data[data.condition==cond]
        # sns.lineplot(x='time', y=f'{var}', data=data, units='rec', estimator=None, ax=axs2[row, col])
        sns.lineplot(x='time', y=f'{var}', data=data, units='rec', estimator=None, ax=ax_line)

        # X ticks
        ax_line.set_xticks(xticks_slope)
        ax_line.set_xticklabels(xtick_labels_slope)

        # X ticks
        ax_line.set_xticks(xticks_slope)
        ax_line.set_xticklabels(xtick_labels_slope)

        # Title
        title = ['pre', 'post'][tdrop]
        title = title + f'_{cond}'
        fig_line.suptitle(title)

        # Save fig
        fig_linefit_p[title] = fig_line

    # fig_linefit_slopes_compare_p = {}
    # for idx, cond in enumerate(['gCNO', 'control']):
    #     fig_line, ax_line = plt.subplots()
    #     # Slopes
    #     data = df_slope_cent
    #     if cond == 'gCNO':
    #         data = data[data.condition==cond]
    #     else:
    #         data = data[data.condition!='gCNO']
    #     sns.lineplot(x='time', y=f'{var}', hue='t_drop', data=data, units='rec', estimator=None, ax=ax_line)

    #     ax_line.set_xlim(t_slope[0] - 20, t_slope[1] + 20)
    #     ax_line.set_ylim(-0.18, 0.18)

    #     # X ticks
    #     ax_line.set_xticks(xticks_slope)
    #     ax_line.set_xticklabels(xtick_labels_slope)

    #     # Title
    #     title = f'Compare slopes ({cond})'
    #     fig_line.suptitle(title)

    #     # Save fig
    #     fig_linefit_slopes_compare_p[title] = fig_line
    #     # plt.plot()

    fig14, ax14 = plt.subplots(figsize=figsize)
    sns.boxplot(data=coef_diff_p.reset_index(), x='index', y='coef', ax=ax14)
    sns.swarmplot(data=coef_diff_p.reset_index(), x='index', y='coef', hue='index', size=10, ax=ax14)
    ax14.set_ylabel('deg/msec')
    # Aesthetics
    ax14.set_ylabel('slope difference')

    # Plot Pre vs Post coefficients
    fig15, ax15 = plt.subplots(1, 2, sharey=True, figsize=figsize)
    for idx_cond, cond in enumerate(['gCNO', 'control']):
        data15 = coef_prepost.loc[cond].reset_index().reset_index().set_index('index').melt(id_vars='level_0', value_vars=['coef_pre', 'coef_post'])
        sns.lineplot(data=data15, x='variable', y='value', units='level_0', estimator=None, ax=ax15[idx_cond])
        ax15[idx_cond].set_title(f'{cond}')

        ax15[idx_cond].set_ylabel('deg/msec')
        
        # ax15[idx_cond].set_ylim(0.0005, 0.0046)
        # ax15[idx_cond].set_xlim(-0.25, 1.25)

    # Compare average wsk strace pre vs post
    # Control pulled
    list_fig_compare = []
    for rec_ in df_wpre_a.rec.unique():
        fig16, ax16 = plt.subplots(figsize=figsize, tight_layout=True)
        data16_pre = df_wpre_a[df_wpre_a['rec'] == rec_].copy()
        data16_pre = data16_pre.assign(period=['pre']*data16_pre.shape[0])
        # scale_pre = data16_pre['angle'].iloc[580]
        # data16_pre['angle'] = data16_pre['angle'].values / scale_pre
        data16_pre = data16_pre[(data16_pre['time'] > 500) & (data16_pre['time'] < 700)]
        # data16_pre['angle'] = np.convolve(data16_pre['angle'].values, np.ones(5)/5, mode='same')

        data16_post = df_wpost_a[df_wpost_a['rec'] == rec_].copy()
        data16_post = data16_post.assign(period=['post']*data16_post.shape[0])
        # scale_post = data16_post['angle'].iloc[580]
        # data16_post['angle'] = data16_post['angle'].values / scale_post
        data16_post = data16_post[(data16_post['time'] > 500) & (data16_post['time'] < 700)]
        # data16_post['angle'] = np.convolve(data16_post['angle'].values, np.ones(5)/5, mode='same')
        data16 = pd.concat([data16_pre, data16_post])
        # data16 = data16[(data16['time'] > 299) & (data16['time'] < 897)]

        if data16['condition'].iloc[0] != 'gCNO':
            data16['condition'] = 'control'
            cond_name = 'control'
        else:
            cond_name = 'gCNO'
        data_slope = df_slope[df_slope.rec==rec_].copy()
        # data_slope[data_slope.t_drop=='pre']['angle'] = data_slope[data_slope.t_drop=='pre']['angle'] / scale_pre
        # data_slope[data_slope.t_drop=='post']['angle'] = data_slope[data_slope.t_drop=='post']['angle'] / scale_post

        data16 = data16.melt(value_vars='angle', id_vars=['condition', 'time', 'period'])
        sns.lineplot(y='value', x='time', hue='period', errorbar='se', data=data16, ax=ax16)
        sns.lineplot(y='angle', x='time', hue='t_drop', errorbar='se', data=data_slope, ax=ax16)
        ylim = ax16.get_ylim()
        ax16.fill_between(np.array(t_slope), ylim[0] -0.01, ylim[1] + 0.01, color='orange')
        
        fig16.suptitle(cond_name)
        ax16.axvline((-t_wind[0] - 0.2) * 299) # line at -200ms
        ax16.annotate('-200ms', xy=((-t_wind[0] - 0.2) * 299, 0))
        ax16.axvline((-t_wind[0] + 0.2) * 299) # line at +200ms
        ax16.annotate('+200ms', xy=((-t_wind[0] + 0.2) * 299, 0))
        ax16.set_xticks(xticks)
        ax16.set_xticklabels(xtick_labels)
        ax16.set_xlim(480, 750)
        fig16.suptitle(f'{id_wtrace[rec_][-30:]}')
        
        list_fig_compare.append(fig16)


    if plot:
        plt.show()
    else:
        # Save figures and data
        if os.path.basename(os.getcwd()) != 'data_analysis':
            os.chdir('./data_analysis')

        try:
            fig1.savefig(os.getcwd() + '/images/whisk_plots/Compare_conditions.svg', format='svg')
            fig2.savefig(os.getcwd() + '/images/whisk_plots/Compare_pre_post.svg', format='svg')
            fig3.savefig(os.getcwd() + '/images/whisk_plots/Compare_pre_post_aligned.svg', format='svg')
            fig4.savefig(os.getcwd() + '/images/whisk_plots/Difference_pre_post.svg', format='svg')
            fig5.savefig(os.getcwd() + '/images/whisk_plots/Variance_pre_post_withinr.svg', format='svg')
            fig6.savefig(os.getcwd() + '/images/whisk_plots/Variance_pre_post_betweenr.svg', format='svg')
            fig7.savefig(os.getcwd() + '/images/whisk_plots/Eigenvectors.svg', format='svg')
            fig8.savefig(os.getcwd() + '/images/whisk_plots/Var_explained.svg', format='svg')
            fig9.savefig(os.getcwd() + '/images/whisk_plots/Mean_activity.svg', format='svg')
            fig10.savefig(os.getcwd() + '/images/whisk_plots/slope_lfit_diff.svg', format='svg')
            fig11.savefig(os.getcwd() + '/images/whisk_plots/slope_lfit_diff_pulledcontrol.svg', format='svg')
            fig12.savefig(os.getcwd() + '/images/whisk_plots/twind_lfit.svg', format='svg')
            fig13.savefig(os.getcwd() + '/images/whisk_plots/twind_lfit_pulledcontrol.svg', format='svg')
            for key in fig_linefit_p.keys():
                fig_linefit_p[key].savefig(os.getcwd() + f'/images/whisk_plots/linear_fit_pulledcontrol_{key}.svg', format='svg')
            # for key in fig_linefit_slopes_compare_p.keys():
            #     fig_linefit_slopes_compare_p[key].savefig(os.getcwd() + f'/images/whisk_plots/line
            #    ar_fit_pulledcontrol_{key}.svg', format='svg')
            fig14.savefig(os.getcwd() + '/images/whisk_plots/slope_lfit_diff_pulledcontrol_pointplot.svg', format='svg')
            fig15.savefig(os.getcwd() + '/images/whisk_plots/slope_lfit_diff_pulledcontrol_individual.svg', format='svg')
            for figs_idx, figs in enumerate(list_fig_compare):
                figs.savefig(os.getcwd() + f'/images/whisk_plots/fig_prepost_compare_slope_lfit{figs_idx}.svg', format='svg')

            # Save coef_diff for lowdim_visualisation.py + stats (Pulled control)
            save_data = {'coef_prepost':coef_prepost, 'coef_p':coef_diff_p, 'Levene_test':lev_p}
            with open(os.getcwd() + '/save_data/diff_slope_whisker_linfit_p.pickle', 'wb') as f:
                pickle.dump(save_data, f)

        except FileNotFoundError:
            print('created dir /images/whisk_plots')
            os.makedirs(os.getcwd() + '/images/whisk_plots')

            fig1.savefig(os.getcwd() + '/images/whisk_plots/Compare_conditions.svg', format='svg')
            fig2.savefig(os.getcwd() + '/images/whisk_plots/Compare_pre_post.svg', format='svg')
            fig3.savefig(os.getcwd() + '/images/whisk_plots/Compare_pre_post_aligned.svg', format='svg')
            fig4.savefig(os.getcwd() + '/images/whisk_plots/Difference_pre_post.svg', format='svg')
            fig5.savefig(os.getcwd() + '/images/whisk_plots/Variance_pre_post_withinr.svg', format='svg')
            fig6.savefig(os.getcwd() + '/images/whisk_plots/Variance_pre_post_betweenr.svg', format='svg')
            fig7.savefig(os.getcwd() + '/images/whisk_plots/Eigenvectors.svg', format='svg')
            fig8.savefig(os.getcwd() + '/images/whisk_plots/Var_explained.svg', format='svg')
            fig9.savefig(os.getcwd() + '/images/whisk_plots/Mean_activity.svg', format='svg')
            fig10.savefig(os.getcwd() + '/images/whisk_plots/slope_lfit_diff.svg', format='svg')
            fig11.savefig(os.getcwd() + '/images/whisk_plots/slope_lfit_diff_pulledcontrol.svg', format='svg')
            fig12.savefig(os.getcwd() + '/images/whisk_plots/twind_lfit.svg', format='svg')
            fig13.savefig(os.getcwd() + '/images/whisk_plots/twind_lfit_pulledcontrol.svg', format='svg')
            for key in fig_linefit_p.keys():
                fig_linefit_p[pre_gCNO].savefig(os.getcwd() + f'/images/whisk_plots/linear_fit_pulledcontrol_{key}.svg', format='svg')
            # for key in fig_linefit_slopes_compare_p.keys():
            #     fig_linefit_slopes_compare_p[key].savefig(os.getcwd() + f'/images/whisk_plots/linear_fit_pulledcontrol_{key}.svg', format='svg')
            fig14.savefig(os.getcwd() + '/images/whisk_plots/slope_lfit_diff_pulledcontrol_pointplot.svg', format='svg')
            fig15.savefig(os.getcwd() + '/images/whisk_plots/slope_lfit_diff_pulledcontrol_individual.svg', format='svg')
            for figs_idx, figs in enumerate(list_fig_compare):
                figs.savefig(os.getcwd() + f'/images/whisk_plots/fig_prepost_compare_slope_lfit{figs_idx}.svg', format='svg')

            # Save coef_diff for lowdim_visualisation.py + stats (Pulled control)
            save_data = {'coef_prepost':coef_prepost, 'coef_p':coef_diff_p, 'Levene_test':lev_p}
            with open(os.getcwd() + '/save_data/diff_slope_whisker_linfit_p.pickle', 'wb') as f:
                pickle.dump(save_data, f)


def plot_wskvar_compare(t_wind=[-2, 3], drop_rec=[0, 10, 15, 16, 25], var='angle', cgs=2, surr=False, pre=None, save=False, load=True, plot=False):
    ''' Compare correlation whiskers, for pre- and post-drop period'''
    # Get whisker data
    pdb.set_trace()
    if save:
        whisk_data_all = []
        for whisker in ['whisk0', 'whisk1', 'whisk2']:
            __ = get_wskvar(whisker=whisker, t_wind=t_wind, drop_rec=drop_rec, var=var, cgs=cgs, surr=surr, pre=pre, save=save, load=load)
            a_wvar = __[0]
            a_wvar_m = __[1]
            bstart = __[2]
            t_drop = __[3]
            conditions = __[4]
            n_rec = __[5]
            id_wtrace = __[6]

            whisk_data_all.append([a_wvar, a_wvar_m, bstart, t_drop, conditions, n_rec])
            pdb.set_trace()
        # Save
        if os.path.basename(os.getcwd()) != 'data_analysis':
            os.chdir('./data_analysis')
        with open(os.getcwd()+ '/save_data/whisk_data_all.pickle', 'wb') as f:
            pickle.dump(whisk_data_all, f)
    else:
        if os.path.basename(os.getcwd()) != 'data_analysis':
            os.chdir('./data_analysis')
        with open(os.getcwd()+ '/save_data/whisk_data_all.pickle', 'rb') as f:
            whisk_data_all = pickle.load(f)

    # Prepare data for plotting
    whisk_data_all_plot = []
    for whisker in range(3):
        a_wvar = whisk_data_all[whisker][0]
        a_wvar_m = whisk_data_all[whisker][1]
        bstart = whisk_data_all[whisker][2]
        t_drop = whisk_data_all[whisker][3]
        conditions = whisk_data_all[whisker][4]
        n_rec = whisk_data_all[whisker][5]
        
        __ = prepare_df(a_wvar_m, a_wvar, t_drop, conditions, n_rec, bstart, t_wind)

        a_wvar_m = __[0]
        df_wpre_sb = __[1]
        df_wpost_sb = __[2]
        df_wpre = __[3]
        df_wpost = __[4]
        df_wpre_a = __[5]
        df_wpost_a = __[6]
        df_diff = __[7]
        df_var_sb = __[8]
        df_var = __[9]

        whisk_data_all_plot.append([a_wvar_m, df_wpre_sb, df_wpost_sb, df_wpre, df_wpost, df_wpre_a, df_wpost_a, df_diff, df_var_sb, df_var])

    df_wskcorr = wsk_corr(whisk_data_all_plot)
    pdb.set_trace()

    from scipy.stats import wilcoxon, ttest_rel
    wilc = []
    ttest = []
    for cond in ['gCNO', 'control']:
        wilc.append(wilcoxon(df_wskcorr[(df_wskcorr['period']=='pre') & (df_wskcorr['cond']==cond)].value, df_wskcorr[(df_wskcorr['period']=='post') & (df_wskcorr['cond']==cond)].value))
        ttest.append(ttest_rel(df_wskcorr[(df_wskcorr['period']=='pre') & (df_wskcorr['cond']==cond)].value, df_wskcorr[(df_wskcorr['period']=='post') & (df_wskcorr['cond']==cond)].value))

    stats = [{'wilcoxon':wilc, 'ttest':ttest}]

    if save:
        if os.path.basename(os.getcwd()) != 'data_analysis':
            os.chdir('./data_analysis')

        with open(os.getcwd()+ '/save_data/whisk_data_all_stats.pickle', 'wb') as f:
            pickle.dump(stats, f)

    # Plot
    figsize = (14, 12)
    sns.set_context("poster", font_scale=1)
    sns.color_palette("deep")
    plt.rcParams['axes.facecolor'] = 'lavenderblush'
    plt.rcParams["figure.edgecolor"] = 'black'

    fig, ax = plt.subplots()
    sns.boxplot(data=df_wskcorr, x='period', y='value', hue='cond')
    sns.stripplot(data=df_wskcorr, x='period', y='value', hue='cond', dodge=True)

    if save:
        fig.savefig(os.getcwd() + '/images/whisk_plots/wsk_corr_compare.svg', format='svg')


if __name__ == '__main__':
    """ Run script if whisk_plot.py module is main programme
    """
    # Set parameters/arguments
    whisker='whisk2'
    t_wind = [-2, 3]
    drop_rec = [0, 10, 15, 16, 25]
    var = 'angle'
    cgs = 2
    surr = False
    pre = None
    load = True
    save = False
    # compare_whisk=True
    compare_whisk=False
    # load = False
    t_slope = [580, 660]
    if not compare_whisk:
        plot_wskvar(t_slope=t_slope, drop_rec=drop_rec, load=load)
    else:
        plot_wskvar_compare(var=var, drop_rec=drop_rec, load=load, save=save)
