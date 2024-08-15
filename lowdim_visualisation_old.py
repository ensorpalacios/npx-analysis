""" Module to extract low-dimensional properties from npx dataset;
plot population activity in pc space, both average and selected trials;
plot cross-correlogram between avg whisking activity and each PC's;
plot distribution cc peaks fo each PC's;
plot loadings to compare distribution of information across neurons for
each PC's and compute Kurtosis.
plot cumulative distribution of egnvalues;
plot cross-correlogram between avg PC's and whisking velocity 

Attention: load output of whisk_plot.py module, which return similarity of
whisker bouts for each recording based to eigenvectors from PCA
on whisker variable; use this to sort neural activity trials.
"""
import os
import sys
if not ('data_analysis' in os.getcwd()):
    sys.path.append(os.path.join(os.getcwd(), 'data_analysis'))

# Import modules
import pdb
import pickle
import itertools
# import pandas as pd
import seaborn as sns
import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import scipy.cluster.hierarchy as sch
import pandas as pd

from matplotlib.collections import LineCollection
from meta_data import load_meta
from align_fun import align_spktrain
from scipy.ndimage import gaussian_filter1d
from scipy import stats
from scipy import signal
# from scipy.cluster.hierarchy import dendrogram, linkage


def corr_anal(data, whisker, std):
    """ Correlation analysis of data: prepare data for analysis by stacking
    whisker bout data while keeping clusters separated at column level; get
    correlation matrix; get sorted correlation matrix by element dinstance;
    compute eingenvalues and vectors to project data onto 3D space; return
    idx for sorting cluster position in corr matrix for pre and post.
    """
    # Prepare data
    df_m = data.a_spktr_m.loc[:, whisker].copy()
    df = data.a_spktr.loc[:, whisker].copy()

    # Drop units with little/no activity (use for pre=True)
    mask = df_m.columns[(df_m.sum() < 0.01)].values
    df_m.drop(columns=mask, inplace=True)
    df.drop(columns=mask, level=0, inplace=True)

    # df = df.dropna(axis=0)
    n_bout = df.columns.get_level_values(1).unique().size
    # Smooth spikes and stack data over bouts
    df.iloc[:, :] = gaussian_filter1d(df, sigma=std, axis=0)
    df = df.stack(dropna=False)
    # df = df.stack()
    df = df.swaplevel().sort_index(level=0)

    # Get correlation matrix + sorted one
    # pdb.set_trace()
    corr = []
    for idx, spkdat in enumerate([df_m, df]):
        __ = spkdat.corr(method='pearson')
        # pdb.set_trace()
        pdist = sch.distance.pdist(__)
        link = sch.linkage(pdist, method='complete')
        cdist_threshold = pdist.max() / 2
        idx_to_cluster = sch.fcluster(link, cdist_threshold,
                                      criterion='distance')
        idx = np.argsort(idx_to_cluster)
        # corr.append(__.iloc[idx, idx])
        corr.append(__)

    # Get eignval, eigvec and project
    egnval, egnvec = la.eig(corr[0])
    idxsort = np.argsort(egnval.real)[::-1]
    egnval = egnval.real[idxsort]
    egnvec = egnvec[:, idxsort]

    projd = []
    for spkdat in [df_m, df]:
        projd.append(np.dot(spkdat, egnvec[:, :3]))

    # projd_ = []
    # for spkdat in [df, df_m]:
    #     projd_.append(np.dot(spkdat, egnvec_[:, :3]))

    # projd = np.dot(df_m, egnvec[:, :3])
    # else:
    #     projd = np.dot(df, egnvec[:, :3])

    # Z-score projections
    projd_zs = projd[0].copy()
    projd_zs = (projd_zs - projd_zs.mean(axis=0)) / projd_zs.std(axis=0)

    color = df_m.index
    # pdb.set_trace()

    return df, corr, egnvec, egnval, projd, color, n_bout, idxsort, projd_zs


def crosscorr(projd_zs, wsknpx_data, whisker, t_wind=[-2, 5], t_subwind=[-1.5, 1.5]):
    ''' Crosscorrelogram between projections and wsk data.
    Important: wsk data need to be zscored otherwise results are nonsese
    (~ to crosscorr in lowdim_visualisation)
    Attention: here I use z-scored means; in figures I instead use sns.lineplot
    to take the mean across (already) z-scored trials.
    '''
    # Long format projd and wsk (zscored)
    projd_df = pd.DataFrame(projd_zs, columns=['pc1', 'pc2', 'pc3'])
    wsk_df = wsknpx_data.a_wvar_m[whisker].copy()
    wsk_df = wsk_df.subtract(wsk_df.mean()).div(wsk_df.std())
    # projd_l = projd_l.melt(id_vars=('trial', 'time'), value_vars=('pc1', 'pc2', 'pc3'))
    # wsk_l = wsk_l.melt(id_vars=('trial', 'time')).drop(labels='variable', axis=1)

    # Flip projd
    projd_df.iloc[:, :] = projd_df.apply(lambda x: x * (-1) if abs(x.iloc[400:1000].max()) < abs(x.iloc[400:1000].min()) else x)

    # Crosscorrelation
    # projd_mean = projd.groupby('time').mean()
    # wsk_mean = wsk.groupby('time').mean()
    # pdb.set_trace()
    df_cc = pd.DataFrame(columns=['pc1', 'pc2', 'pc3']).astype(float)
    subwind = ((np.array(t_subwind) - t_wind[0]) * 299).astype(int)
    # subwind = np.array(t_subwind) - 299
    for pc in df_cc.columns:
        df_cc.loc[:, pc] = signal.correlate(wsk_df[0], projd_df.loc[:, pc])
    # Center time crosscorr and scale (area pc1=1)
    __ = df_cc.shape[0] // 2
    df_cc = df_cc.loc[int(__ + t_subwind[0] * 299):int(__ + t_subwind[1] * 299)]
    df_cc.set_index(df_cc.index - __, inplace=True)
    df_cc = df_cc.div(abs(df_cc).max().max())
    
    # Long format for sns plots
    df_cc = df_cc.reset_index(names='time').melt(id_vars=('time'))

    return projd_df,  wsk_df, df_cc, subwind


# def prepost_fun(data, whisker, egnvec, t_drop, idx_corr):
#     ''' Select pre and post spk data as well as whisking data based on
#     similarity to whisker eigenvector; filter and take average, compute
#     correlation matrix, and project onto 3d space from all data (pre=None);
#     select subset whisking bouts as well.
#     '''
#     # Masks
#     pdb.set_trace()
#     wsk = int(whisker[-1])
#     mask_pre = np.array(data.whiskd.event_times[wsk]) < t_drop * 60
#     mask_pre = np.nonzero(mask_pre)[0]
#     mask_post = np.logical_and(np.array(data.whiskd.event_times[wsk]) > (t_drop + 5) * 60, np.array(data.whiskd.event_times[wsk]) < (t_drop * 2 + 5) * 60)
#     mask_post = np.nonzero(mask_post)[0]

#     # # Subset whisker bouts based on eigenvec similarity
#     # mask_pre = mask_pre[simil_pc[0] >= simil_pc[0].mean()]
#     # mask_post = mask_post[simil_pc[1] >= simil_pc[1].mean()]

#     # Pre/post spk and whisking data
#     dfspk_pre = data.a_spktr.loc[:, (whisker, slice(None), mask_pre)].copy()
#     dfspk_post = data.a_spktr.loc[:, (whisker, slice(None), mask_post)].copy()
#     dfwsk_pre = data.a_wvar.loc[:, ('whisk2', mask_pre)]
#     dfwsk_post = data.a_wvar.loc[:, ('whisk2', mask_post)]

#     # Average (smoothed) data
#     pdb.set_trace()
#     std = 5
#     dfspk_pre = dfspk_pre.apply(gaussian_filter1d, axis=1, result_type='broadcast', sigma=std)
#     dfspk_pre = dfspk_pre.groupby(axis=1, level=1).mean()
#     dfspk_post = dfspk_post.apply(gaussian_filter1d, axis=1, result_type='broadcast', sigma=std)
#     dfspk_post = dfspk_post.groupby(axis=1, level=1).mean()


#     # Correlation matrix
#     corr_pre = dfspk_pre.corr(method='pearson')
#     # corr_pre = corr_pre.iloc[idx_corr, idx_corr]
#     corr_post = dfspk_post.corr(method='pearson')
#     # corr_post = corr_post.iloc[idx_corr, idx_corr]

#     # # Average data
#     # df_pre_m = df_pre.groupby(axis=1, level=1).mean()
#     # df_post_m = df_post.groupby(axis=1, level=1).mean()

#     # Project
#     # proj_prepost_m = []
#     # for spkdat in [df_pre_m, df_post_m]:
#     #     proj_prepost_m.append(np.dot(spkdat, egnvec[:, :3]))

#     proj_prepost = []
#     for spkdat in [dfspk_pre, dfspk_post]:
#         proj_prepost.append(np.dot(spkdat, egnvec[:, :3]))

#     corr_prepost = [corr_pre, corr_post]
#     dfspk_prepost = [dfspk_pre, dfspk_post]
#     dfwsk_prepost = [dfwsk_pre, dfwsk_post]

#     return proj_prepost, corr_prepost, dfspk_prepost, dfwsk_prepost


def plot_lowd(wsknpx_data,
              wsk_l,
              projd,
              projd_l,
              df_cc,
              peak_cc,
              whisker,
              var,
              t_wind,
              color,
              wskpath,
              surr=False,
              plot_sbouts=False):
    """ Function for plotting low dimensional properties of cluster population;
    use spikes from selected clusters clu_idx in
    choosen recording rec_idx; additionally specify whisking variable of
    interest, time window around whisking bout start to consider, cluster group
    (default 2==good), whether to use surrogate data, and filter clusters by min
    fr in each bin (don't use here: binl too small).
    - Note 1: rec_idx can be 0-18 gCNO, 19-23 wCNO, 24-32
    - Note 2: wsknpx_data contains: whiskd, spktr_sorted, cids_sorted, a_wvar,
    a_wvar_m, a_spktr, a_spktr_m.

    """

    # Plots
    # Initialise plot parameters
    figsize = (14, 12)
    wvar_m = wsknpx_data.a_wvar_m[whisker, 0]
    xticks = np.arange(0, np.diff(t_wind) * 299, 299)
    xtick_labels = np.arange(t_wind[0], t_wind[1])
    deg = np.array([wvar_m.max() - 5 / (180 / np.pi), wvar_m.max() + 5 / (180 / np.pi)])
    sec = np.array([200, 200 + 0.5 * 299])
    fcolor = 'lavenderblush'
    lcolor = 'darkgoldenrod'
    # mcolor = 'copper'
    mcolor = 'gist_earth'
    mcolor_pre = 'autumn'
    mcolor_post = 'winter'
    linewidth = 2
    sns.set_context("poster", font_scale=1)
    sns.color_palette("deep")
    plt.rcParams["figure.edgecolor"] = 'black'
    plt.rcParams['axes.facecolor'] = 'lavenderblush'

    # Color for whiskvar
    len_bout = np.diff(t_wind)[0] * 299
    x = np.arange(len_bout)
    # points = np.array([x, wvar]).T.reshape(-1, 1, 2)
    points = np.array([x, wvar_m]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    norm = plt.Normalize(color[0], color[-1])
    lc = LineCollection(segments, cmap=mcolor, norm=norm)
    lc.set_array(color)
    lc.set_linewidth(linewidth)
    
    # Plot cross-correlation
    # wsk_l.value = (wsk_l.value - wsk_l.value.mean()) / wsk_l.value.std()
    # Convert rad to deg
    wsk_l.value = (wsk_l.value - wsk_l.value.mean()) * 180 / np.pi
    fig, axs = plt.subplots(3, figsize=figsize)
    sns.lineplot(y='value', x='time', data=wsk_l, errorbar='se', ax=axs[0], c='darkgoldenrod', linewidth=linewidth, label='whisker position')
    # sns.lineplot(y='value', x='time', data=aaa, errorbar='se', ax=axs[0], c='darkgoldenrod', linewidth=linewidth, label='whisker position')
    # axs[0].plot(wvar_m)
    # axs[0].plot(wsknpx_data.a_wvar[whisker].mean(1))
    sns.lineplot(y='value', x='time', data=projd_l, hue='pcs', errorbar='se', linewidth=linewidth, ax=axs[1])
    sns.lineplot(y='value', x='time', data=df_cc, hue='pcs', linewidth=linewidth, ax=axs[2])
    
    #Add cc max lines
    __ = axs[2].get_yticks()
    axs[2].vlines(peak_cc.values, __[0], __[-1], color=['blue', 'orange', 'green'])

    # Set xticklabels
    axs[0].set_xticks(xticks)
    axs[0].set_xticklabels(xtick_labels)
    axs[1].set_xticks(xticks)
    axs[1].set_xticklabels(xtick_labels)
    __ = axs[2].get_xticks()
    __ = np.array([__[1], 0, __[-2]])
    axs[2].set_xticks(__)
    axs[2].set_xticklabels(np.round(__ / 299, 1))

    axs[0].set_ylabel('whiker position')
    axs[1].set_ylabel('a.u.')
    axs[2].set_ylabel('normalised cc')

    # Save figure 1
    if os.path.basename(os.getcwd()) != 'data_analysis':
        os.chdir('./data_analysis')

    if not surr:
        surr = 'nosurr'

    try:
        fig.savefig(os.getcwd() + f'/images/lowd_old/{os.path.basename(wskpath[:-1])}/crosscorr_newmethod.svg', format='svg')
    except FileNotFoundError:
        print(f'created dir images/lowd_old/{os.path.basename(wskpath[:-1])}')
        os.makedirs(os.getcwd() + f'/images/lowd_old/{os.path.basename(wskpath[:-1])}/')
        fig.savefig(os.getcwd() + f'/images/lowd_old/{os.path.basename(wskpath[:-1])}/crosscorr_newmethod.svg', format='svg')
    plt.close()
    # pdb.set_trace()

    # Plot single bout projections
    if plot_sbouts:
        bout_idxs = [15, 16, 17, 18]
        bout_idxs = np.arange(20)
        for idx, bout_idx in enumerate(bout_idxs):
            # Prepare data
            __ = projd[1][len_bout * bout_idx: len_bout * (bout_idx + 1), :]
            wvar = wsknpx_data.a_wvar[whisker, bout_idx]
            # midpoint = (wvar_m.max() + wvar_m.min()) / 2
            # deg = np.array([wvar_m.max(), wvar_m.max() + 10 / (180 / np.pi)])
            deg = np.array([wvar.max() - 10 / (180 / np.pi), wvar.max() + 0 / (180 / np.pi)])

            # Prepare axes
            fig9 = plt.figure(tight_layout=False, figsize=figsize)
            gs = gridspec.GridSpec(2, 2, height_ratios=[4, 2], wspace=0, hspace=0)
            ax9_0 = fig9.add_subplot(gs[0, 0], projection='3d')
            ax9_1 = fig9.add_subplot(gs[0, 1], projection='3d')
            ax9_2 = fig9.add_subplot(gs[1, :])

            # Plot 3D
            ax9_0.scatter3D(projd[0][:, 0], projd[0][:, 1], projd[0][:, 2], c=color, cmap=mcolor)
            ax9_1.scatter3D(__[:, 0], __[:, 1], __[:, 2], c=color, cmap=mcolor)

            # 3D Axis
            ax9_0.set_xticklabels([])
            ax9_0.set_yticklabels([])
            ax9_0.set_zticklabels([])
            # ax9_0.set_xlabel('PC1')
            # ax9_0.set_ylabel('PC2')
            # ax9_0.set_zlabel('PC3')

            ax9_1.set_xticklabels([])
            ax9_1.set_yticklabels([])
            ax9_1.set_zticklabels([])
            ax9_1.set_xlabel('PC1')
            ax9_1.set_ylabel('PC2')
            ax9_1.set_zlabel('PC3')

            ax9_0.set_title('Average activity')
            ax9_1.set_title('Single trial activity')

            # Plot whiskvar;
            # Color whisker plot (following https://matplotlib.org/stable/gallery/lines_bars_and_markers/multicolored_line.html)
            x = np.arange(wvar.size)
            # points = np.array([np.arange(wvar.size), wvar]).T.reshape(-1, 1, 2)
            points = np.array([x, wvar]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            norm = plt.Normalize(color[0], color[-1])
            lc = LineCollection(segments, cmap=mcolor, norm=norm)
            lc.set_array(color)
            lc.set_linewidth(linewidth)
            line = ax9_2.add_collection(lc)
            plt.colorbar(line, ax=ax9_2, label='time', ticks=[0, 598, 1196])
            ax9_2.set_xlim(x.min(), x.max())
            # ax9_2.set_ylim(0, 1.4)
            ax9_2.set_ylim(wvar.min() - 0.1, wvar.max() + 0.1)

            # Other aesthetics
            # ax9_0.set_facecolor(fcolor)
            # ax9_1.set_facecolor(fcolor)
            ax9_2.set_facecolor(fcolor)
            # fig9.patch.set_facecolor(fcolor)

            ax9_2.set_xticks([])
            ax9_2.set_yticks([])
            ax9_2.spines['top'].set_visible(False)
            ax9_2.spines['right'].set_visible(False)
            ax9_2.spines['bottom'].set_visible(False)
            ax9_2.spines['left'].set_visible(False)

            ax9_2.set_ylabel(f'whisker {var}')
            ax9_2.vlines(sec[0], deg[0], deg[1], colors=lcolor)
            ax9_2.hlines(deg[0], sec[0], sec[1], colors=lcolor)
            ax9_2.text(sec[0], deg[1] + 0.01, '10 deg')
            ax9_2.text(sec.mean(), deg[0] + 0.01, '0.5 sec')

            fig9.suptitle('Projected population activity')
            plt.show()

            fig9.savefig(os.getcwd() + f'/images/lowd_old/{os.path.basename(wskpath[:-1])}/{os.path.basename(wskpath[:-1])}_ProjectSingle{bout_idx}_{var}_surr{surr}.svg', format='svg')

def plot_avg_pcs(proj_data, wsk_data, croscorr_data, t_wind, var='angle', surr=False, savefig=True):
    """ Plot mean pc's within recorded, averaged across recordings; sflip within
    recording pc averages to have positive absolute max value.
    """
    idxsli = pd.IndexSlice
    proj_data = pd.concat(proj_data)
    wsk_data = pd.concat(wsk_data)
    # croscorr_data = pd.concat(croscorr_data, keys=np.arange(12), names=['rec'])
    
    proj_data = proj_data.set_index(['rec', 'trial', 'time', 'pcs'])
    # croscorr_data = croscorr_data.droplevel(level=1, axis=0)
    wsk_data = wsk_data.set_index(['rec', 'trial', 'time'])
    # croscorr_data = croscorr_data.set_index(['time', 'pcs'], append=True)

    # Standardize wsk
    wsk_data_z = wsk_data.groupby(by=['rec', 'trial'], group_keys=False).apply(lambda x: (x - x.mean()) / x.std())
    
    proj_data_mean = proj_data.groupby(by=['rec', 'time', 'pcs']).mean()
    # wsk_data_mean = wsk_data.groupby(by=['rec', 'time']).mean()
    wsk_data_mean = wsk_data_z.groupby(by=['rec', 'time']).mean()

    # Flip pc' and align to 0 time
    proj_data_mean_flip = proj_data_mean.groupby(by=['rec', 'pcs'], group_keys=False).apply(lambda x: x * (-1) if abs(x['value'].iloc[400:1000].max()) < abs(x['value'].iloc[400:1000].min()) else x)
    # proj_data_mean_flip.loc[idxsli[:, :, 'pc2']] = (proj_data_mean_flip.loc[idxsli[:, :, 'pc2']] * -1).values
    proj_data_mean_flip =  proj_data_mean_flip.groupby(by=['rec', 'pcs'], group_keys=False).apply(lambda x: x - x['value'].iloc[0])

    # # Mask pc2 data basd on cc (more than 50% or pc1 value)
    # mask_cc = croscorr_data.loc[idxsli[:, :, 'pc2']].groupby(by='rec').apply(lambda x: True if abs(np.max(x, axis=0)[0]) >= 0.3 else False)
    # mask_cc_idx = np.argwhere(~mask_cc.values).flatten()
    # filtered2d = proj_data_mean_flip.drop(index=mask_cc_idx, level=0)
    # filtered2d = proj_data_mean_flip.loc[idxsli[np.argwhere(mask_cc.values).flatten(), :, :]].copy()    

    # Derivative wsk (+ smooth)
    wsk_data_mean_der = wsk_data_mean.groupby(by='rec', group_keys=False).diff()
    __ = wsk_data_mean_der.groupby(by='rec', group_keys=False).apply(lambda x: gaussian_filter1d(x, axis=0, sigma=10))
    __ = np.array([x.T for x in __])[:, 0, :]
    __ = pd.DataFrame(__)
    __ = __.stack(dropna=False)
    wsk_data_mean_der['value'] = __.values

    # Initialise plot parameters
    figsize = (14, 12)
    xticks = np.arange(0, np.diff(t_wind) * 299, 299)
    xtick_labels = np.arange(t_wind[0], t_wind[1])
    linewidth = 2
    sns.set_context("poster", font_scale=1)
    sns.color_palette("deep")
    plt.rcParams["figure.edgecolor"] = 'black'
    plt.rcParams['axes.facecolor'] = 'lavenderblush'

    

    # sns.lineplot(data=proj_data_mean_flip.reset_index(), x='time', y='value', hue='pcs', units='rec', errorbar='se', estimator=None)
    fig, ax = plt.subplots(2, 1, sharex=True, figsize=figsize)
    sns.lineplot(data=wsk_data_mean.reset_index(), x='time', y='value', errorbar='se', ax=ax[0], color='darkgoldenrod')
    sns.lineplot(data=proj_data_mean_flip.reset_index(), x='time', y='value', hue='pcs', errorbar='se', ax=ax[1])
    # sns.lineplot(data=filtered2d.reset_index(), x='time', y='value', hue='pcs', errorbar='se', ax=ax[1])
    # Aesthetics
    fig.suptitle(f'Average pcs and wsk {var}')
    ax[1].set_xticks(xticks)
    ax[1].set_xticklabels(xtick_labels)
    ax[0].set_ylabel(f'{var} (a.u.)')
    ax[1].set_xlabel('time (sec)')


    fig1, ax1 = plt.subplots(2, 1, sharex=True, figsize=figsize)
    ax1_ = ax1[0].twinx()
    sns.lineplot(data=wsk_data_mean.reset_index(), x='time', y='value', errorbar='se', ax=ax1[0], color='darkgoldenrod')
    sns.lineplot(data=wsk_data_mean_der.reset_index(), x='time', y='value', errorbar='se', ax=ax1_)
    sns.lineplot(data=proj_data_mean_flip.reset_index(), x='time', y='value', hue='pcs', errorbar='se', ax=ax1[1])
    fig1.suptitle(f'Average pcs and wsk {var} (+ derivative)')

    # Aesthetics
    ax1[1].set_xticks(xticks)
    ax1[1].set_xticklabels(xtick_labels)
    ax1[0].set_ylabel(f'{var} (a.u.)')
    ax1[1].set_xlabel('time (sec)')

    if savefig:
        if os.path.basename(os.getcwd()) != 'data_analysis':
            os.chdir('./data_analysis')

        if not surr:
            surr = 'nosurr'

        try:
            # fig.savefig(os.getcwd() + '/images/lowd_old/proj_avg.svg', format='svg')
            fig.savefig(os.getcwd() + f'/images/lowd_old/proj_avg_pc2inv.svg', format='svg')
            fig1.savefig(os.getcwd() + f'/images/lowd_old/proj_avg_der.svg', format='svg')
        except FileNotFoundError:
            print('created dir images/lowd_old}')
            os.makedirs(os.getcwd() + '/images/lowd_old/')
            # fig.savefig(os.getcwd() + '/images/lowd_old/proj_avg.svg', format='svg')
            fig.savefig(os.getcwd() + f'/images/lowd_old/proj_avg_pc2inv.svg', format='svg')
            fig1.savefig(os.getcwd() + f'/images/lowd_old/proj_avg_der.svg', format='svg')



def plot_ccpeaks(df_ccpeaks, croscorr_data, surr=False):
    """ Plot distribution of cross-correlogram peaks
    """
    # pdb.set_trace()
    # Initialise plot parameters
    figsize = (14, 12)
    sns.set_context("poster", font_scale=1)
    sns.color_palette("deep")
    plt.rcParams["figure.edgecolor"] = 'black'
    plt.rcParams['axes.facecolor'] = 'lavenderblush'

    # # Mask pc2 data basd on cc (more than 50% or pc1 value)
    # croscorr_data = pd.concat(croscorr_data, keys=np.arange(12))
    # croscorr_data.index.names = ['rec', '__']
    # croscorr_data = croscorr_data.set_index(['time', 'pcs'], append=True)
    # croscorr_data = croscorr_data.droplevel(level=1, axis=0)
    # mask_cc = croscorr_data.loc[idxsli[:, :, 'pc2']].groupby(by='rec').apply(lambda x: True if np.max(abs(x), axis=0)[0] >= 0.2 else False)
    # mask_cc_idx = np.argwhere(mask_cc.values).flatten()
    # filtered2d = proj_data_mean_flip.drop(index=mask_cc_idx, level=0)
    # filtered2d = proj_data_mean_flip.loc[idxsli[np.argwhere(mask_cc.values).flatten(), :, :]].copy()    


    # Plot
    fig2, ax2 = plt.subplots()
    sns.swarmplot(data=df_ccpeaks, x='pcs', y='value', hue='pcs', ax=ax2, dodge=True)
    sns.boxplot(data=df_ccpeaks, x='pcs', y='value', hue='pcs', ax=ax2)
    # Set y ticks
    y_ticks = ax2.get_yticks()
    ax2.set_yticks(y_ticks)
    ax2.set_yticklabels(np.round(y_ticks / 299, 1))
    ax2.set_ylabel('sec')

    # Save figure 2
    if os.path.basename(os.getcwd()) != 'data_analysis':
        os.chdir('./data_analysis')

    if not surr:
        surr = 'nosurr'

    try:
        fig2.savefig(os.getcwd() + f'/images/lowd_old/dist_ccpeaks_nc_newmethod.svg', format='svg')
    except FileNotFoundError:
        print('created dir images/lowd_old}')
        os.makedirs(os.getcwd() + '/images/lowd_old/')
        fig2.savefig(os.getcwd() + '/images/lowd_old/dist_ccpeaks_nc_newmethod.svg', format='svg')
    plt.close()


def plot_loadings(df_load_l, croscorr_data, surr=False):
    """ Plot distribution of pc loadings
    """
    # Initialise plot parameters
    figsize = (14, 12)
    sns.set_context("poster", font_scale=1)
    sns.color_palette("deep")
    plt.rcParams["figure.edgecolor"] = 'black'
    plt.rcParams['axes.facecolor'] = 'lavenderblush'

    # Plot pc loadings
    fig3, ax3 = plt.subplots(1, 3, sharey=True, figsize=figsize)
    for idx, pc in enumerate(['pc1', 'pc2', 'pc3']):
        __ = df_load_l[df_load_l.pcs==pc]
        sns.violinplot(data=__, y='value', color='antiquewhite', ax=ax3[idx])
        sns.swarmplot(data=__, y='value', ax=ax3[idx], size=3)
        ax3[idx].set_xlabel(f'{pc}')
        ax3[idx].set_ylabel('pc load')

    # Plot pc loadings together (mean-centred)
    df_load_l_meancentered = df_load_l.copy()
    df_load_l_meancentered['value'] = df_load_l.set_index('pcs').groupby(by='pcs').apply(lambda x:x['value'] - x['value'].mean()).values
    fig4, ax4 = plt.subplots(figsize=figsize)
    sns.histplot(data=df_load_l_meancentered, x='value', hue='pcs', element='step', kde=True, ax=ax4)
    ax4.set_xlim([-1, 1])
    fig4.suptitle('pc_loadings collapsed')

    # Save figure 2
    if os.path.basename(os.getcwd()) != 'data_analysis':
        os.chdir('./data_analysis')

    if not surr:
        surr = 'nosurr'

    try:
        fig3.savefig(os.getcwd() + f'/images/lowd_old/pc_loadings/loadings_all_nc.svg', format='svg')
        fig4.savefig(os.getcwd() + f'/images/lowd_old/pc_loadings/loadings_all_nc_collapse.svg', format='svg')
    except FileNotFoundError:
        print('created dir images/lowd_old/pc_loadings}')
        os.makedirs(os.getcwd() + '/images/lowd_old/pc_loadings')
        fig3.savefig(os.getcwd() + f'/images/lowd_old/pc_loadings/loadings_all_nc.svg', format='svg')
        fig4.savefig(os.getcwd() + f'/images/lowd_old/pc_loadings/loadings_all_nc_collapse.svg', format='svg')
    plt.close()


def plot_egnval(egnval_data):
    ''' Plot cumulative distribution of egnvalues; check how much variance is explained
    by the first 3 eigenvectors
    '''
    figsize = (14, 12)
    sns.set_context("poster", font_scale=1)
    sns.color_palette("deep")
    plt.rcParams["figure.edgecolor"] = 'black'
    plt.rcParams['axes.facecolor'] = 'lavenderblush'

    egnval_data_l = egnval_data.iloc[:3, :].copy()
    egnval_data_l['pc'] = [1, 2, 3]
    egnval_data_l = egnval_data_l.melt(id_vars='pc', value_name='variance explained')

    egnval_data_20 = egnval_data.iloc[:20, :].copy()
    egnval_data_20 = 1 - egnval_data_20
    egnval_data_20.index = np.arange(1, 21)
    egnval_data_20 = pd.concat([pd.DataFrame(np.ones(12)).T, egnval_data_20])
    egnval_data_20['pc'] = [i for i in range(21)]
    egnval_data_20 = egnval_data_20.melt(id_vars='pc', value_name='variance explained')

    fig, ax = plt.subplots(figsize=figsize)
    sns.boxplot(data=egnval_data_l, x='pc', y='variance explained', ax=ax)
    sns.stripplot(data=egnval_data_l, x='pc', y='variance explained', hue='pc', ax=ax)

    fig1, ax1 = plt.subplots(figsize=figsize)
    sns.lineplot(data=egnval_data_20, x='pc', y='variance explained', ax=ax1)
    # sns.lineplot(data=egnval_data_20, x='pc', y='variance explained', ax=ax1, units='variable', estimator=None)
    ax1.vlines(3, 0, 1)
    ax1.set_xticks(np.arange(21))
    ax1.set_xticklabels(np.arange(21).astype(str))

    if not ('data_analysis' in os.getcwd()):
        os.chdir(os.path.join(os.getcwd(), 'data_analysis'))

    try:
        fig.savefig(os.getcwd() + '/images/lowd_old/3egn_var_explained.svg', format='svg')
        fig1.savefig(os.getcwd() + '/images/lowd_old/20egn_var.svg', format='svg')
    except FileNotFoundError:
        os.makedirs(os.getcwd() + '/images/lowd_old')
        fig.savefig(os.getcwd() + '/images/lowd_old/3egnvar_explained.svg', format='svg')
        fig1.savefig(os.getcwd() + '/images/lowd_old/20egn_var.svg', format='svg')


def crosscorr_vel(projdata, wskdata, t_wind=[-2, 5], t_subwind=[-1.5, 1.5]):
    """ Compute cross-correlation between average pcs and whisker velocity;
    compute time of peak.
    """
    # pdb.set_trace()
    idxsli = pd.IndexSlice
    projd = []
    for rec_idx, rec_data in enumerate(projdata):
        __ = rec_data.groupby(by=['pcs', 'time'])['value'].mean()
        __ = __.groupby(by=['pcs'], group_keys=False).apply(lambda x: x * (-1) if abs(x.iloc[400:1000].max()) < abs(x.iloc[400:1000].min()) else x)
        __ = pd.concat([__], keys=[rec_idx], names=['rec'])
        projd.append(__)
    projd = pd.concat(projd)

    wskd_vel = []
    for rec_idx, rec_data in enumerate(wskdata):
        __ = rec_data.groupby(by=['time'], group_keys=False)['value'].mean()
        __ = __.diff()
        __ = __.fillna(0)
        __ = (__ - __.mean()) / __.std()
        __ = pd.concat([__], keys=[rec_idx], names=['rec'])
        wskd_vel.append(__)
    wskd_vel = pd.concat(wskd_vel)
    wskd_vel = wskd_vel.fillna(0)

    df_cc_vel = []
    subwind = ((np.array(t_subwind) - t_wind[0]) * 299).astype(int)
    # subwind = np.array(t_subwind) - 299
    for rec_idx in projd.index.get_level_values(0).unique():
        __ = pd.DataFrame(columns=['pc1', 'pc2', 'pc3']).astype(float)
        for pc in __.columns:
            __.loc[:, pc] = signal.correlate(wskd_vel.loc[idxsli[rec_idx, :]], projd.loc[idxsli[rec_idx, pc, :]])
        __.index.names = ['time']
        __ = pd.concat([__], keys=[rec_idx], names=['rec'])
        df_cc_vel.append(__)

    df_cc_vel = pd.concat(df_cc_vel)
    # Center time crosscorr and scale (area pc1=1)
    __ = df_cc_vel.loc[0].shape[0] // 2
    df_cc_vel = df_cc_vel.loc[idxsli[:, int(__ + t_subwind[0] * 299):int(__ + t_subwind[1] * 299)], :]
    df_cc_vel = df_cc_vel.reset_index(level=1)
    df_cc_vel['time'] = df_cc_vel['time'] - __
    df_cc_vel.set_index('time', append=True, inplace=True)
    df_cc_vel = df_cc_vel.div(abs(df_cc_vel).max().max())
    
    # Long format for sns plots
    df_cc_vel = df_cc_vel.reset_index().melt(id_vars=('rec', 'time'))
    df_cc_vel.columns = ['rec', 'time', 'pcs', 'value']

    # Compute time of cc peaks
    __ = df_cc_vel.set_index('time').copy()
    df_cc_vel_peaks = __.groupby(by=['rec','pcs'], group_keys=False)['value'].idxmax()

    __ = df_cc_vel.set_index('time').copy()
    __['value'] = abs(__['value'].values)    
    df_cc_vel_peaks_abs = __.groupby(by=['rec','pcs'], group_keys=False)['value'].idxmax()

    return df_cc_vel, df_cc_vel_peaks, df_cc_vel_peaks_abs


def plot_cc_val(cc_vel, cc_vel_peaks, cc_vel_peaks_abs):
    """ Plot cross-correlation between average pcs and whisker velocity;
    plot time of positive or max or absolute peak.
    """
    figsize = (14, 12)
    sns.set_context("poster", font_scale=1)
    sns.color_palette("deep")
    plt.rcParams["figure.edgecolor"] = 'black'
    plt.rcParams['axes.facecolor'] = 'lavenderblush'

    # Plot cross-correlogram
    fig, ax = plt.subplots(figsize=figsize)
    sns.lineplot(data=cc_vel, x='time', y='value', hue='pcs', errorbar='sd', ax=ax)
    x_ticks = np.arange(-2, 3)
    ax.set_xticks(x_ticks * 299)
    ax.set_xticklabels(x_ticks)
    ax.set_xlabel('sec')
    fig.suptitle('cross-correlation velocity')

    # Plot max peak time
    fig1, ax1 = plt.subplots(figsize=figsize)
    sns.boxplot(data=cc_vel_peaks.reset_index(), x='pcs', y='value', ax=ax1)
    sns.stripplot(data=cc_vel_peaks.reset_index(), x='pcs', y='value', ax=ax1)
    y_ticks = ax1.get_yticks()
    ax1.set_yticks(y_ticks)
    ax1.set_yticklabels(np.round(y_ticks / 299, 1))
    ax1.set_ylabel('sec')
    fig1.suptitle('max peak time')


    # Plot abs peak time
    fig2, ax2 = plt.subplots(figsize=figsize)
    sns.boxplot(data=cc_vel_peaks_abs.reset_index(), x='pcs', y='value', ax=ax2)
    sns.stripplot(data=cc_vel_peaks_abs.reset_index(), x='pcs', y='value', ax=ax2)
    y_ticks = ax2.get_yticks()
    ax2.set_yticks(y_ticks)
    ax2.set_yticklabels(np.round(y_ticks / 299, 1))
    ax2.set_ylabel('sec')
    fig2.suptitle('max peak time')

    if not ('data_analysis' in os.getcwd()):
        os.chdir(os.path.join(os.getcwd(), 'data_analysis'))

    try:
        fig.savefig(os.getcwd() + '/images/lowd_old/velocity_crosscorr.svg', format='svg')
        fig1.savefig(os.getcwd() + '/images/lowd_old/velocity_max_peak_time.svg', format='svg')
        fig2.savefig(os.getcwd() + '/images/lowd_old/velocity_abs_peak_time.svg', format='svg')
    except FileNotFoundError:
        os.makedirs(os.getcwd() + '/images/lowd_old')
        fig.savefig(os.getcwd() + '/images/lowd_old/velocity_crosscorr.svg', format='svg')
        fig1.savefig(os.getcwd() + '/images/lowd_old/velocity_max_peak_time.svg', format='svg')
        fig2.savefig(os.getcwd() + '/images/lowd_old/velocity_abs_peak_time.svg', format='svg')



def run_lowd(drop_rec=[0, 10, 15, 16, 25], whisker='whisk2', t_wind=[-2, 3], var='angle', cgs=2, std=2, surr=False, pre=None, discard=False, pethdiscard = True, plot=True, load=True):
    ''' Project population activity in 3d eigenspace; do this for all
    recordings in good_list (filtered twice based on whisking); plot and save
    low dimensional visualisation for each recording; gather (and save) all
    data together; compute crosscorrelogram.
    Attention: keep pre=None; still returns pre and post

    Attention***: data saved in thesis and further use var='angle;
    var='setpoint' used only in decoding module to allow single-trial
    reconstruction; otherwise, var='angle' works well on averages.
    '''
    # load metadata
    __ = load_meta()
    wskpath_list = __[0]
    npxpath_list = __[1]
    aid_list = __[2]
    cortex_depth_list = __[3]
    t_drop_list = __[4]
    clu_idx_list = __[5]
    pethdiscard_idx_list = __[6]
    bout_idxs_list = __[7]
    wmask_list = __[8]
    conditions_list = __[9]

    good_list = np.arange(len(wskpath_list))
    good_list = good_list[wmask_list]
    wmask2 = ~np.in1d(np.arange(len(good_list)), np.array(drop_rec))
    good_list = good_list[wmask2]  # len=25

    #### NEED TO RERUN THESE DATA WITH GOOD MASK!!!
    # Load whisker pca data (all recordings)
    # if os.path.basename(os.getcwd()) != 'data_analysis':
    #     os.chdir('./data_analysis')
    # with open('save_data/save_whisk_corr.pickle', 'rb') as f:
    #     simil_1pc, simil_2pc, simil_3pc, var_explained, egnvec_save = pickle.load(f)

    # # ATTENTION: no cno (nc) here
    good_list = good_list[13:]
    # Save together projections  and wsk from single rec
    save_allproj = []
    save_allwsk = []
    rec_idx_fromzero = 0

    # Save corr matrix, egnvec, egnval, crosscorrelation peaks
    save_corr = []
    save_egnvec = []
    save_egnval = []
    save_ccpeaks = []
    save_croscorr = []
    # pdb.set_trace()
    # Run for each in good list
    if not load:
        for rec_idx in good_list:
            # rec_idx = 30            # EP_WT_220607_B
            # Load Data specific for rec_idx
            wskpath = wskpath_list[rec_idx]
            npxpath = npxpath_list[rec_idx]
            aid = aid_list[rec_idx]
            cortex_depth = cortex_depth_list[rec_idx]
            t_drop = t_drop_list[rec_idx]
            conditions = conditions_list[rec_idx]
            if pethdiscard:
                pethdiscard_idx = pethdiscard_idx_list[rec_idx]
            else:
                pethdiscard_idx = None
                # clu_idx = clu_idx[rec_idx]
            clu_idx = []
            bout_idxs = bout_idxs_list[rec_idx]

            # wsknpx_data contains:
            # - whiskd
            # - spktr_sorted
            # - cids_sorted
            # - a_wvar
            # - a_wvar_m
            # - a_spktr
            # - a_spktr_m
            wsknpx_data = align_spktrain(wskpath, npxpath, cortex_depth, cgs=cgs, t_wind=t_wind, var=var, surr=surr, t_drop=t_drop, pre=pre, clu_idx=clu_idx, discard=discard, pethdiscard_idx=pethdiscard_idx)

            #### NEED TO RERUN THESE DATA WITH GOOD MASK!!!
            # Get recording specific whisker pca data
            # pc_idx = np.argwhere(__ == rec_idx)[0][0]
            # simil_1pc = xsimil_1pc[pc_idx]
            # simil_2pc = simil_2pc[pc_idx]
            # simil_3pc = simil_3pc[pc_idx]
            # var_explained = var_explained.iloc[pc_idx, :]
            # egnvec_save = egnvec_save[pc_idx]

            # Correlation analysis
            __ = corr_anal(wsknpx_data, whisker, std)
            df = __[0]
            corr = __[1]
            egnvec = __[2]
            egnval = __[3]
            projd = __[4]
            color = __[5]
            n_bout = __[6]
            idx_corr = __[7]
            projd_zs = __[8]
            # pdb.set_trace()

            # Crosscorrelation neural-wsk data
            projd_df,  wsk_df, df_cc, subwind = crosscorr(projd_zs, wsknpx_data, whisker, t_wind=t_wind, t_subwind=t_subwind)

            # pdb.set_trace()
            # Arrange data in df, zscore projd
            # wsk_l = wsknpx_data.a_wvar[whisker].dropna()
            wsk_l = wsknpx_data.a_wvar[whisker]
            wsk_l.reset_index(names='time', inplace=True)
            wsk_l = wsk_l.melt(id_vars='time', var_name='trial')
            projd_l = wsk_l[['time', 'trial']].copy()
            projd_l = pd.concat([projd_l, pd.DataFrame(projd[1], columns=['pc1', 'pc2', 'pc3'])], axis=1)
            projd_l = projd_l.melt(id_vars=['time', 'trial'], var_name='pcs')
            projd_l['value'] = projd_l.groupby('pcs', group_keys=False)['value'].apply(lambda x: (x - x.mean()) / x.std()).values
            df_cc.rename(columns={'variable':'pcs'}, inplace=True)
            # max_cc = df_cc.groupby(by=['pcs'],group_keys=True)['time', 'value'].idxmax()
            __ = df_cc.copy()
            __['value'] = abs(__['value'])
            peak_cc = __.set_index('time').groupby(by=['pcs'],group_keys=True)['value'].idxmax()

            # Add rec to proj and wsk data
            projd_l['rec'] = rec_idx_fromzero
            wsk_l['rec'] = rec_idx_fromzero
            rec_idx_fromzero += 1

            # Save data
            save_corr.append(corr)
            save_egnvec.append(egnvec)
            save_egnval.append(egnval)
            save_ccpeaks.append(peak_cc)
            save_allproj.append(projd_l)
            save_allwsk.append(wsk_l)
            save_croscorr.append(df_cc)

            # pdb.set_trace()
            # Plot Data
            if plot:
                plot_lowd(wsknpx_data, wsk_l, projd, projd_l, df_cc, peak_cc, whisker, var, t_wind, color, wskpath)
            
        # Save data (no CNO - nc)
        save_all_nc = [save_corr, save_egnvec, save_egnval, save_ccpeaks, save_allproj, save_allwsk, save_croscorr]
        # # This is data saved up to thesis (should be the same as next)
        # with open(f'/home/yq18021/Documents/github_gcl/data_analysis/save_data/lowd_old_data_nc.pickle', 'wb') as f:
        #     pickle.dump(save_all_nc, f)
        # This is data saved up after adding savve_allproj and save_allwsk
        with open(f'/home/yq18021/Documents/github_gcl/data_analysis/save_data/lowd_old_data_nc_allprojwsk_ccnewmethod.pickle', 'wb') as f:
            pickle.dump(save_all_nc, f)
    else:
        # with open(f'/home/yq18021/Documents/github_gcl/data_analysis/save_data/lowd_old_data_nc.pickle', 'rb') as f:
        with open(f'/home/yq18021/Documents/github_gcl/data_analysis/save_data/lowd_old_data_nc_allprojwsk_ccnewmethod.pickle', 'rb') as f:
            save_all_nc = pickle.load(f)

        save_corr = save_all_nc[0]
        save_egnvec = save_all_nc[1]
        save_egnval = save_all_nc[2]
        save_ccpeaks = save_all_nc[3]
        save_allproj = save_all_nc[4]
        save_allwsk = save_all_nc[5]
        save_croscorr = save_all_nc[6]
    # pdb.set_trace()
    # Plot average pcs
    plot_avg_pcs(save_allproj, save_allwsk, save_croscorr, t_wind, var=var)
    

    # Check variance explained by first 3 pc's
    df_egnval = pd.DataFrame(save_egnval).T.copy()
    cumsum_egnval = df_egnval.cumsum(axis=0)
    cumsum_egnval_scaled = cumsum_egnval.div(df_egnval.sum())
    # cumsum_egnval_scaled = pd.concat([pd.DataFrame(np.zeros(12)).T, cumsum_egnval_scaled], axis=0)
    cumsum_egnval_scaled.reset_index(drop=True, inplace=True)

    if plot:
        plot_egnval(cumsum_egnval_scaled)


    # Plot cross-correlation peaks (pulled controls)
    # save_ccpeaks = save_all[3]
    df_ccpeaks = pd.DataFrame(save_ccpeaks, index=conditions_list[wmask_list][wmask2][13:])
    df_ccpeaks.rename({'wCNO':'control', 'aPBS':'control'}, inplace=True)
    df_ccpeaks.index.names = ['index']
    df_ccpeaks = df_ccpeaks.reset_index(names='cond').melt(id_vars='cond')
    if plot:
        plot_ccpeaks(df_ccpeaks, save_croscorr)

    # T-test for mean different from 0
    if not load:
        ttest_ccpeaks = []
        for pc in ['pc1', 'pc2', 'pc3']:
            value_pc = df_ccpeaks[df_ccpeaks.pcs == pc].value
            ttest_ccpeaks.append(list((stats.ttest_1samp(value_pc, popmean=0), {f'mean_{pc}': np.round(value_pc.mean())}, {f'sem_{pc}': np.round(value_pc.std()/np.sqrt(len(value_pc)))})))
            
            
        with open(f'/home/yq18021/Documents/github_gcl/data_analysis/save_data/lowd_old_ttest_ccpeaks_ccnewmethod.pickle', 'wb') as f:
            pickle.dump(ttest_ccpeaks, f)
    else:
        with open(f'/home/yq18021/Documents/github_gcl/data_analysis/save_data/lowd_old_ttest_ccpeaks_ccnewmethod.pickle', 'rb') as f:
            ttest_ccpeaks__ = pickle.load(f)

    

    # PC loadings
    df_load = pd.DataFrame()
    for idx, egnvec_data in enumerate(save_egnvec):
        __ = pd.DataFrame(egnvec_data[:, :3], columns=['pc1', 'pc2', 'pc3'])
        __ = pd.concat([__], keys=[idx])
        df_load = pd.concat([df_load, __], axis=0)

    df_load.index.names = ['rec', 'unit']
    df_load_l = df_load.reset_index().melt(id_vars=['rec', 'unit'], var_name='pcs')

    # Excess kurtosis (kurtosis - 3) is a measure of outlier frequency
    exc_kurt = df_load_l.groupby(by='pcs')['value'].apply(pd.DataFrame.kurtosis) - 3

    if not load:
        with open(f'/home/yq18021/Documents/github_gcl/data_analysis/save_data/lowd_old_data_exckurtosis_nc.pickle', 'wb') as f:
            pickle.dump(exc_kurt, f)
    else:
        with open(f'/home/yq18021/Documents/github_gcl/data_analysis/save_data/lowd_old_data_exckurtosis_nc.pickle', 'rb') as f:
            exc_kurt = pickle.load(f)

    if plot:
        plot_loadings(df_load_l, save_croscorr)

    # Encoding of whisking velocity
    df_cc_vel, df_cc_vel_peaks, df_cc_vel_peaks_abs = crosscorr_vel(save_allproj, save_allwsk, t_wind=t_wind, t_subwind=t_wind)

    if plot:
        plot_cc_val(df_cc_vel, df_cc_vel_peaks, df_cc_vel_peaks_abs)
    

if __name__ == '__main__':
    """ Run script if spkcount_analysis.py module is main programme

    Attention***: data saved in thesis and further use var='angle;
    var='setpoint' used only in decoding module to allow single-trial
    reconstruction; otherwise, var='angle' works well on averages.
    """

    whisker = 'whisk2'                 # which whisker "" ""
    t_wind = [-2, 3]            # t_wind around whisking bout start
    # t_wind = [-10, 10]            # t_wind around whisking bout start
    # t_subwind = [-1, 1]
    t_subwind = [-1.5, 1.5]
    var = 'angle'
    # var = 'setpoint'
    # var = 'amplitude'           # which whisker var "" ""
    cgs = 2
    std =  5                    # std for smoothing
    surr = False
    # surr = 'shuffle'
    pre = None
    # pre = True
    discard = False             # discard clu with fr < 0.1 Hz
    # bout_idxs = [15, 16, 17, 18]  # single wbout to display
    drop_rec=[0, 10, 15, 16, 25]
    pethdiscard = True

    plot = True
    # plot = False
    # load = True
    load = False

    # plot_lowd(rec_idx, whisker=whisker, t_wind=t_wind, var=var, cgs=cgs, surr=surr, pre=pre, discard=discard)
    run_lowd(drop_rec=drop_rec, whisker=whisker, t_wind=t_wind, var=var, cgs=cgs, std=std, surr=surr, pre=pre, discard=discard, pethdiscard=pethdiscard, plot=plot, load=load)
