""" Module to extract low-dimensional properties from npx dataset;
plot population activity in pc space, both average and selected trials;
used get_data from glm_module.py module to load data id df divided by pre
and post drop;

Attention: load output of whisk_plot.py module, which return similarity of
whisker bouts for each recording based to eigenvectors from PCA
on whisker variable; use this to sort neural activity trials.

Attention2: 'newmethod' figures use scipy.signal.correlate instead of np.correlate!
"""
import os
import sys
if not ('data_analysis' in os.getcwd()):
    sys.path.append(os.path.join(os.getcwd(), 'data_analysis'))

# Import modules
import pdb
import pickle
import itertools
import pandas as pd
import seaborn as sns
import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import scipy.cluster.hierarchy as sch
import pymc as pm
import arviz as az

from matplotlib.collections import LineCollection
from meta_data import load_meta
from glm_module_decoding import get_data
from glm_module_decoding import corr_anal
from align_fun import align_spktrain
from scipy.ndimage import gaussian_filter1d
from scipy.stats import wilcoxon, ttest_rel, ttest_ind
from scipy.special import logit
from scipy import signal, stats

from sklearn.linear_model import LinearRegression

# from scipy.cluster.hierarchy import dendrogram, linkage


# def corr_anal(data, whisker):
#     """ Correlation analysis of data: prepare data for analysis by stacking
#     whisker bout data while keeping clusters separated at column level; get
#     correlation matrix; get sorted correlation matrix by element dinstance;
#     compute eingenvalues and vectors to project data onto 3D space; return
#     idx for sorting cluster position in corr matrix for pre and post.
#     """
#     # Prepare data
#     df_m = data.a_spktr_m.loc[:, whisker].copy()
#     df = data.a_spktr.loc[:, whisker].copy()
#     n_bout = df.columns.get_level_values(1).unique().size
#     std = 5
#     # Smooth spikes and stack data over bouts
#     df.iloc[:, :] = gaussian_filter1d(df, sigma=std, axis=0)
#     df = df.stack()
#     df = df.swaplevel().sort_index(level=0)

#     # Get correlation matrix + sorted one
#     # pdb.set_trace()
#     corr = []
#     for idx, spkdat in enumerate([df_m, df]):
#         __ = spkdat.corr(method='pearson')
#         # pdb.set_trace()
#         pdist = sch.distance.pdist(__)
#         link = sch.linkage(pdist, method='complete')
#         cdist_threshold = pdist.max() / 2
#         idx_to_cluster = sch.fcluster(link, cdist_threshold,
#                                       criterion='distance')
#         idx = np.argsort(idx_to_cluster)
#         # corr.append(__.iloc[idx, idx])
#         corr.append(__)

#     # Get eignval, eigvec and project
#     egnval, egnvec = la.eig(corr[0])
#     idxsort = np.argsort(egnval.real)[::-1]
#     egnval = egnval.real[idxsort]
#     egnvec = egnvec[:, idxsort]

#     projd = []
#     for spkdat in [df_m, df]:
#         projd.append(np.dot(spkdat, egnvec[:, :3]))

#     # projd_ = []
#     # for spkdat in [df, df_m]:
#     #     projd_.append(np.dot(spkdat, egnvec_[:, :3]))

#     # projd = np.dot(df_m, egnvec[:, :3])
#     # else:
#     #     projd = np.dot(df, egnvec[:, :3])

#     color = df_m.index

#     return df, corr, egnvec, egnval, projd, color, n_bout, idxsort


# def prepost_fun(data, whisker, egnvec, t_drop, simil_pc, idx_corr):
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

#     # Subset whisker bouts based on eigenvec similarity
#     mask_pre = mask_pre[simil_pc[0] >= simil_pc[0].mean()]
#     mask_post = mask_post[simil_pc[1] >= simil_pc[1].mean()]

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


def crosscorr(df_projd_avg, df_wsk_avg, t_wind=[-2, 3], subwind=[-1.5, 1.5]):
    """ Compute the cross-correlation between projed and wsk data.
    """
    # pdb.set_trace()
    # subwind = ((np.array(subwind) - t_wind[0]) * 299).astype(int)
    # subwind = ((np.array(subwind) - t_wind[0]) * 299).astype(int)
    midx = df_projd_avg.columns
    # df_cc = pd.DataFrame(index=index, columns=midx).astype(float)
    df_cc = pd.DataFrame(columns=midx).astype(float)
    for rec, tdrop, pc in midx:
        # df_cc.loc[:, (rec, tdrop, pc)] = np.correlate(df_projd_avg.loc[:, (rec, tdrop, pc)], df_wsk_avg.loc[:, (rec, tdrop)], mode='same')
        df_cc.loc[:, (rec, tdrop, pc)] = signal.correlate(df_wsk_avg.loc[:, (rec, tdrop)], df_projd_avg.loc[:, (rec, tdrop, pc)])

    __ = df_cc.shape[0] // 2
    df_cc = df_cc.loc[int(__ + subwind[0] * 299):int(__ + subwind[1] * 299)]
    df_cc.set_index(df_cc.index - __, inplace=True)

    # df_cc.set_index(df_cc.index - df_cc.index[-1] // 2, inplace=True)

    return df_cc


def anal_cc(df_cc, conditions):
    """ Analysis cross-correlation: get absolute values, compute cumulative
    distribution, total cumulative value and peak (max of abs); return for
    plotting.
    """
    # pdb.set_trace()
    # Add condition level to df
    new_cond = conditions.to_dict()[0]
    new_df_cc = df_cc.copy()
    __ = new_df_cc.columns.to_frame()
    __['cond'] = __['rec']
    new_df_cc.columns = pd.MultiIndex.from_frame(__)
    new_df_cc = new_df_cc.reorder_levels(['cond', 'rec', 't_drop', 'pcs'], axis=1)
    new_df_cc.rename(columns=new_cond, level=0, inplace=True)

    # Take absolute value
    new_df_cc = new_df_cc.apply(np.abs, axis=0)
    # new_df_cc.sort_values(by='t_drop', ascending=False, axis=1, inplace=True)

    # Cumulative absolute cc
    cumsum_df_cc = new_df_cc.copy()
    cumsum_df_cc = new_df_cc.apply(np.cumsum, axis=0)
    lag = list(cumsum_df_cc.index.values)
    cumsum_df_cc = cumsum_df_cc.melt()
    cumsum_df_cc['lag'] = lag * cumsum_df_cc['rec'].unique().size * cumsum_df_cc['t_drop'].unique().size * cumsum_df_cc['pcs'].unique().size
    cumsum_df_cc.set_index(['cond', 'rec', 'pcs'], inplace=True)

    totcumsum_df_cc = cumsum_df_cc.groupby(by=['cond', 'rec', 't_drop', 'pcs']).sum()
    totcumsum_df_cc.drop(labels='lag', axis=1, inplace=True)
    totcumsum_df_cc.reset_index(2, inplace=True)

    # Peak
    max_df_cc = new_df_cc.melt()
    max_df_cc = max_df_cc.groupby(by=['cond', 'rec', 't_drop', 'pcs']).max()
    max_df_cc.reset_index(2, inplace=True)

    return cumsum_df_cc, totcumsum_df_cc, max_df_cc


def plot_lowd(df_projd_avg,
              df_wsk_avg,
              df_cc,
              cumsum_df_cc,
              totcumsum_df_cc,
              max_df_cc,
              df_ccpeak,
              coef_diff,
              cd_tertile,
              remap_cond,
              conditions,
              wskpaths,
              color,
              t_wind=[-2, 3],
              var='angle',
              save_plot=True):
    """ Generate one figure for each recording plotting together neural
    projections and whisking activity (averaged over trials); add results
    cross correlations projd-wsk.
    """
    # Organise data in long format
    df_projd_avg_l = df_projd_avg.reset_index(names='time')
    df_projd_avg_l = df_projd_avg_l.melt(id_vars=['time'])

    df_wsk_avg_l = df_wsk_avg.reset_index()
    df_wsk_avg_l = df_wsk_avg_l.melt(id_vars=['time'])

    df_cc_l = df_cc.reset_index(names='lag')
    df_cc_l = df_cc_l.melt(id_vars=['lag'])

    # Initialise plot parameters
    figsize = (14, 12)
    xticks = np.arange(0, np.diff(t_wind) * 299, 299)
    xtick_labels = np.arange(t_wind[0], t_wind[1])
    linewidth = 3
    sns.set_context("poster", font_scale=1)
    sns.color_palette("deep")
    plt.rcParams['axes.facecolor'] = 'lavenderblush'
    plt.rcParams["figure.edgecolor"] = 'black'

    pdb.set_trace()

    #         # Set x axes and lim
    #         axs[0, col].set_xticks(xticks)
    #         axs[0, col].set_xticklabels(xtick_labels)
    #         axs[1, col].set_xticks(xticks)
    #         axs[1, col].set_xticklabels(xtick_labels)
    #         # axs[2, col].set_xticks(xticks)
    #         # axs[2, col].set_xticklabels(xtick_labels)

    #         # Set subplot titles
    #         axs[0, col].set_title(f'{tdrop}')

    #         # Legends
    #         if col==0:
    #             axs[0, col].legend([], [], frameon=False)
    #             axs[1, col].legend([], [], frameon=False)
    #             axs[2, col].legend([], [], frameon=False)
    #         else:
    #             axs[0, col].legend(bbox_to_anchor=(.8, .6))
    #             axs[1, col].legend(bbox_to_anchor=(.8, .6))
    #             axs[2, col].legend([], [], frameon=False)
                

    #         # Labels and ticks outer
    #         axs[0, col].label_outer()

    #     # Set y axes
    #     axs[0, 0].set_ylabel(f'{var}')
    #     axs[1, 0].set_ylabel('amplitude')
    #     axs[2, 0].set_ylabel('cross-correlation')

    #     # Set titles
    #     fig.suptitle(f'{os.path.basename(wskpaths[rec_idx][:-1])}')

    #     fig_list.append(fig)
    #     plt.show()
    #     plt.close()
    pdb.set_trace()
    # Plot max cross correlation (wsk vs neural pcs)
    fig1, ax1 = plt.subplots(2, 3, tight_layout=True, sharex=True, sharey=True, figsize=figsize)
    for row, col in itertools.product(range(2), range(3)):
        # cond = ['gCNO', 'wCNO', 'aPBS'][row]
        cond = ['gCNO', 'control'][row]
        pcs = ['pc1', 'pc2', 'pc3'][col]
        data = max_df_cc.loc[(cond, slice(None), pcs)].copy()
        # data.value = data.value / data.value.max()
    
        sns.pointplot(x='t_drop', y='value', data=data, order=['pre', 'post'], errorbar='se', ax=ax1[row, col])
        # sns.boxplot(x='t_drop', y='value', data=data, order=['pre', 'post'], whis=2,  ax=ax1[row, col], palette='deep')

        sns.swarmplot(x='t_drop', y='value', data=data, order=['pre', 'post'], hue='t_drop', hue_order=['pre', 'post'], ax=ax1[row, col], palette='deep', size=7, edgecolor='black', linewidth=1)

        # Legend
        if (row == 0) & (col == 2):
            ax1[row, col].legend(bbox_to_anchor=(.8, .6))
        else:
            ax1[row, col].legend([], [], frameon=False)

        # Axis labels
        if row !=2:
            # ax1[row, col].set_xlabel()
            ax1[row, col].axes.xaxis.set_visible(False)
        if col != 0:
            ax1[row, col].axes.yaxis.set_visible(False)

        # y label
        if col==0:
            ax1[row, col].set_ylabel(f'{cond}')        

        # Title
        if row == 0:
            ax1[row, col].set_title(f'{pcs}')
    fig1.suptitle('Max cross correlation (wsk vs neural pcs)')
    
    # Compare cumulative cross-correlation
    fig3, ax3 = plt.subplots(2, 3, tight_layout=True, sharex=True, sharey=True, figsize=figsize)
    for row, col in itertools.product(range(2), range(3)):
        cond = ['gCNO', 'control'][row]
        pcs = ['pc1', 'pc2', 'pc3'][col]
        data = totcumsum_df_cc.loc[(cond, slice(None), pcs)].copy()
        # data.value = data.value / data.value.max()
        __ = np.meshgrid(['pre', 'post'],
                         [coef_diff['slope_change'][coef_diff.index==cond].values])[1]
        __ = np.array(__).reshape(-1)
        data['std'] = __.tolist()
        data.reset_index(inplace=True)

        # sns.barplot(x='t_drop', y='value', data=data, order=['pre', 'post'], errorbar='se', ax=ax3[row, col], palette='deep', alpha=0.5)
        sns.pointplot(x='t_drop', y='value', data=data, order=['pre', 'post'], errorbar='se', ax=ax3[row, col])
        # sns.boxplot(x='t_drop', y='value', data=data, order=['pre', 'post'], whis=2,  ax=ax3[row, col], palette='deep')
        # sns.boxplot(x='t_drop', y='value', data=data, ax=ax3[row, col])
        sns.swarmplot(x='t_drop', y='value', data=data, order=['pre', 'post'], hue='t_drop', hue_order=['pre', 'post'], ax=ax3[row, col], palette='deep', size=7, edgecolor='black', linewidth=1)

        # sns.pointplot(x='t_drop', y='value', data=data, ax=ax3[row, col])
        # sns.lineplot(x='t_drop', y='value', data=data, units=data.index.values, estimator=None, ax=ax3[row, col])

        # Legend
        if (row == 0) & (col == 2):
            ax3[row, col].legend(bbox_to_anchor=(.8, .6))
        else:
            ax3[row, col].legend([], [], frameon=False)

        # Axis labels
        if row !=1:
            # ax3[row, col].set_xlabel()
            ax3[row, col].axes.xaxis.set_visible(False)
        if col != 0:
            ax3[row, col].axes.yaxis.set_visible(False)

        # y label
        if col == 0:
            ax3[row, col].set_ylabel(f'{cond}')

        # Title
        if row == 0:
            ax3[row, col].set_title(f'{pcs}')

    fig3.suptitle('Cumulative cross correlation (wsk vs neural pcs)')

    # # Plot coefficient
    # fig5, ax5 = plt.subplots(2, 1, sharex=True, figsize=figsize)
    # # colors = {0:'b', 1:'r', 2:'g'}
    # for row in range(coef_diff.index.unique().size):
    #     cond = ['gCNO', 'control'][row]
    #     data = coef_diff[coef_diff.index==cond].copy()
    #     data['cat'] = cd_tertile[cd_tertile.index==cond].copy()
    #     sns.histplot(x='coef_diff', data=data, hue='cat', hue_order=cd_tertile.unique(), bins=100, palette='deep', ax=ax5[row])
    #     # Legend
    #     if row != 0:
    #         ax5[row].legend([], [], frameon=False)

    # fig5.suptitle('Whisking slope difference (pre-post)')

    fig6, ax6 = plt.subplots(2, 3, tight_layout=True, sharex=True, figsize=figsize)
    for row, col in itertools.product(range(2), range(3)):
        # Select data
        cond = ['gCNO', 'control'][row]
        pcs = ['pc1', 'pc2', 'pc3'][col]
        data = max_df_cc.copy()
        # data.rename(index=remap_cond, level=0, inplace=True)
        data = data.loc[(cond, slice(None), pcs)]
        # Normalise to 1
        # data.value = data.value / data.value.max()
        # Add category
        # __ = np.meshgrid(['pre', 'post'],
        #                  [cd_tertile[cd_tertile.index==cond].values])[1]
        __ = np.meshgrid(['pre', 'post'],
                         [coef_diff['slope_change'][coef_diff.index==cond].values])[1]
        __ = np.array(__).reshape(-1)
        data['std'] = __.tolist()

        # Plot
        # sns.pointplot(x='t_drop', y='value', order=['pre', 'post'], data=data, hue='std', hue_order=cd_tertile.unique(), palette='deep', ax=ax6[row, col])
        # sns.swarmplot(x='t_drop', y='value', order=['pre', 'post'], data=data, hue='std', hue_order=cd_tertile.unique(), ax=ax6[row, col], palette='deep', s=7, edgecolor='black', linewidth=1, legend=False)
        sns.pointplot(x='t_drop', y='value', order=['pre', 'post'], data=data, hue='std', palette='deep', ax=ax6[row, col])
        sns.swarmplot(x='t_drop', y='value', order=['pre', 'post'], data=data, hue='std', ax=ax6[row, col], palette='deep', s=7, edgecolor='black', linewidth=1, legend=False)
    
        # Legend
        if (row == 0) & (col == 2):
            ax6[row, col].legend(bbox_to_anchor=(.8, .6))
        else:
            ax6[row, col].legend([], [], frameon=False)


        # Axis labels
        # if row !=1:
        #     # ax1[row, col].set_xlabel()
        #     ax6[row, col].axes.xaxis.set_visible(False)
        if col != 0:
            ax6[row, col].axes.yaxis.set_visible(False)

        # y label
        if col==0:
            ax6[row, col].set_ylabel(f'{cond}')        
    # # Log scale yaxis
        # ax1[row, col].set_yscale('log')
    fig6.suptitle('Max cross correlation (stratified)')
    
    fig7, ax7 = plt.subplots(2, 3, tight_layout=True, sharex=True, sharey=True, figsize=figsize)
    for row, col in itertools.product(range(2), range(3)):
        # Select data
        cond = ['gCNO', 'control'][row]
        pcs = ['pc1', 'pc2', 'pc3'][col]
        data = totcumsum_df_cc.copy()
        data.rename(index=remap_cond, level=0, inplace=True)
        data = data.loc[(cond, slice(None), pcs)]
        __ = np.meshgrid(['pre', 'post'],
                         [coef_diff['slope_change'][coef_diff.index==cond].values])[1]
        __ = np.array(__).reshape(-1)
        data['std'] = __.tolist()
        
        sns.pointplot(x='t_drop', y='value', order=['pre', 'post'], data=data, hue='std', palette='deep', ax=ax7[row, col])
        sns.swarmplot(x='t_drop', y='value', order=['pre', 'post'], data=data, hue='std', ax=ax7[row, col], palette='deep', s=7, edgecolor='black', linewidth=1, legend=False)
    
        # Legend
        if (row == 0) & (col == 2):
            ax7[row, col].legend(bbox_to_anchor=(.8, .6))
        else:
            ax7[row, col].legend([], [], frameon=False)
        if col != 0:
            ax7[row, col].axes.yaxis.set_visible(False)

        # y label
        if col==0:
            ax7[row, col].set_ylabel(f'{cond}')        

    fig7.suptitle('Cumulative cross correlation (stratified)')


    fig8, ax8 = plt.subplots(2, 3, figsize=figsize, sharex=True, sharey=True)
    linregmodel = LinearRegression()
    totcumsum_df_cc_diff = totcumsum_df_cc[totcumsum_df_cc['t_drop'] == 'post']['value'] - totcumsum_df_cc[totcumsum_df_cc['t_drop'] == 'pre']['value']
    totcumsum_df_cc_diff = totcumsum_df_cc_diff.groupby(['cond', 'pcs'], group_keys=False).apply(lambda x: x / np.abs(x).max())
    # totcumsum_df_cc_diff = totcumsum_df_cc_diff.groupby(['cond', 'pcs'], group_keys=False).apply(lambda x: (x - x.mean()) / x.std())
    totcumsum_df_cc_diff = pd.DataFrame(totcumsum_df_cc_diff).rename(columns={'value':'max_cc_diff'})
    # coef_diff_ = coef_diff.groupby(by='index', group_keys=False)['coef_diff'].apply(lambda x: (x - x.mean()) / x.std())

    for row, col in itertools.product(range(2), range(3)):
        # Data for plot (in df)
        cond = ['gCNO', 'control'][row]
        pc = ['pc1', 'pc2', 'pc3'][col]
        df_plot = pd.concat([coef_diff.loc[cond]['coef_diff'].reset_index(drop=True), totcumsum_df_cc_diff.loc[(cond, slice(None), pc)].reset_index(drop=True)], axis=1)
        # df_plot = pd.concat([coef_diff_.loc[cond].reset_index(drop=True), totcumsum_df_cc_diff.loc[(cond, slice(None), pc)].reset_index(drop=True)], axis=1)
        # Linear regression
        linregmodel.fit(df_plot['coef_diff'].values[:, None], df_plot['max_cc_diff'].values[:, None])
        # Plot
        sns.scatterplot(data=df_plot, x='coef_diff', y='max_cc_diff', ax=ax8[row, col])
        # xlim = ax8[row, col].get_xlim()
        # ax8[row, col].plot(np.linspace(xlim[0], xlim[1], 20),  np.linspace(xlim[0], xlim[1], 20) * linregmodel.coef_[0][0] + linregmodel.intercept_[0])
        if row == 0:
            ax8[row, col].plot(np.linspace(-0.0025, 0.0025, 20),  np.linspace(-0.0025, 0.0025, 20) * linregmodel.coef_[0][0] + linregmodel.intercept_[0])
        else:
            ax8[row, col].plot(np.linspace(-0.001, 0.001, 20),  np.linspace(-0.001, 0.001, 20) * linregmodel.coef_[0][0] + linregmodel.intercept_[0])
        # Aesthetics
        # ax8[row, col].set_xlim(-0.003, 0.003)
        if col != 0:
            ax8[row, col].axes.yaxis.set_visible(False)
        # Aesthetics
        # y label
        if col==0:
            ax8[row, col].set_ylabel(f'{cond}')        
        # Add linear fit R and coef
        score = np.round(linregmodel.score(df_plot['coef_diff'].values[:, None], df_plot['max_cc_diff'].values[:, None]), 2)
        coef = np.round(linregmodel.coef_[0][0], 3)
        ax8[row, col].set_title(rf'$R^2$:{score}, coef:{coef}')

    fig8.suptitle('Pre-post wsk difference vs maximum cc difference')

    fig9, ax9 = plt.subplots(2, 3, figsize=figsize, sharex=True, sharey=True)
    max_df_cc_diff = max_df_cc[max_df_cc['t_drop'] == 'post']['value'] - max_df_cc[max_df_cc['t_drop'] == 'pre']['value']
    max_df_cc_diff = max_df_cc_diff.groupby(['cond', 'pcs'], group_keys=False).apply(lambda x: x / np.abs(x).max())
    max_df_cc_diff = pd.DataFrame(max_df_cc_diff).rename(columns={'value':'max_cc_diff'})
    linregmodel = LinearRegression()
    for row, col in itertools.product(range(2), range(3)):
        # Data for plot (in df)
        cond = ['gCNO', 'control'][row]
        pc = ['pc1', 'pc2', 'pc3'][col]
        df_plot = pd.concat([coef_diff.loc[cond]['coef_diff'].reset_index(drop=True), max_df_cc_diff.loc[(cond, slice(None), pc)].reset_index(drop=True)], axis=1)
        # df_plot = pd.concat([coef_diff_.loc[cond].reset_index(drop=True), max_df_cc_diff.loc[(cond, slice(None), pc)].reset_index(drop=True)], axis=1)
        # Linear regression
        linregmodel.fit(df_plot['coef_diff'].values[:, None], df_plot['max_cc_diff'].values[:, None])
        # Plot
        sns.scatterplot(data=df_plot, x='coef_diff', y='max_cc_diff', ax=ax9[row, col])
        xlim = ax9[row, col].get_xlim()
        # ax9[row, col].plot(np.linspace(xlim[0], xlim[1], 20),  np.linspace(xlim[0], xlim[1], 20) * linregmodel.coef_[0][0] + linregmodel.intercept_[0])
        if row == 0:
            ax9[row, col].plot(np.linspace(-0.002, 0.002, 20),  np.linspace(-0.002, 0.002, 20) * linregmodel.coef_[0][0] + linregmodel.intercept_[0])
        else:
            ax9[row, col].plot(np.linspace(-0.001, 0.001, 20),  np.linspace(-0.001, 0.001, 20) * linregmodel.coef_[0][0] + linregmodel.intercept_[0])
        # Aesthetics
        # ax9[row, col].set_xlim(-0.003, 0.003)
        if col != 0:
            ax9[row, col].axes.yaxis.set_visible(False)
        # y label
        if col==0:
            ax9[row, col].set_ylabel(f'{cond}')
        # Add linear fit R and coef
        score = np.round(linregmodel.score(df_plot['coef_diff'].values[:, None], df_plot['max_cc_diff'].values[:, None]), 2)
        coef = np.round(linregmodel.coef_[0][0], 3)
        print(coef)
        ax9[row, col].set_title(rf'$R^2$:{score}, coef:{coef}')

        # # log scale yaxis
        # ax1[row, col].set_yscale('log')
    fig9.suptitle('Pre-post wsk difference vs maximum cc difference')

    # Plot time of cc peaks
    idx = pd.IndexSlice
    df_ccpeak_short = df_ccpeak.reset_index()
    fig10, ax10 = plt.subplots(2, figsize=figsize, sharex=True, sharey=True)
    for idx, cond in enumerate(df_ccpeak_short['cond'].unique()):
        sns.boxplot(data=df_ccpeak_short[df_ccpeak_short.cond==cond], x='pcs', y='time', hue='t_drop', hue_order=['pre', 'post'], ax=ax10[idx])
        sns.stripplot(data=df_ccpeak_short[df_ccpeak_short.cond==cond], x='pcs', y='time', hue='t_drop', hue_order=['pre', 'post'], ax=ax10[idx], dodge=True, size=10)
        # Set y ticks and label
        y_ticks = ax10[idx].get_yticks()
        ax10[idx].set_yticks(y_ticks)
        ax10[idx].set_yticklabels(np.round(y_ticks / 299, 1))
        ax10[idx].set_ylabel('sec')
        # ax10[idx].get_legend().remove()
        ax10[idx].legend(loc='lower left', bbox_to_anchor=(0, -0.2))
        ax10[idx].set_ylabel(f'{cond}')

    fig10.suptitle('Time absolute cc peaks pre vs post')

    # ALTERNATIVE fig10 (similar to fig1/3)
    df_ccpeak_short = df_ccpeak.reset_index(level=2)
    fig11, ax11 = plt.subplots(2, 3, tight_layout=True, sharex=True, sharey=True, figsize=figsize)
    for row, col in itertools.product(range(2), range(3)):
        # cond = ['gCNO', 'wCNO', 'aPBS'][row]
        cond = ['gCNO', 'control'][row]
        pcs = ['pc1', 'pc2', 'pc3'][col]
        data = df_ccpeak_short.loc[(cond, slice(None), pcs)].copy()
        # data.value = data.value / data.value.max()
    
        sns.pointplot(x='t_drop', y='time', data=data, order=['pre', 'post'], errorbar='se', ax=ax11[row, col])
        # sns.boxplot(x='t_drop', y='value', data=data, order=['pre', 'post'], whis=2,  ax=ax1[row, col], palette='deep')

        sns.swarmplot(x='t_drop', y='time', data=data, order=['pre', 'post'], hue='t_drop', hue_order=['pre', 'post'], ax=ax11[row, col], palette='deep', size=7, edgecolor='black', linewidth=1)

        # Legend
        if (row == 0) & (col == 2):
            ax11[row, col].legend(bbox_to_anchor=(.8, .6))
        else:
            ax11[row, col].legend([], [], frameon=False)

        # Axis labels
        if row !=2:
            # ax1[row, col].set_xlabel()
            ax11[row, col].axes.xaxis.set_visible(False)
        if col != 0:
            ax11[row, col].axes.yaxis.set_visible(False)

        # y label
        if col==0:
            ax11[row, col].set_ylabel(f'{cond}')

        # Set y ticks and label
        y_ticks = ax11[row, col].get_yticks()
        ax11[row, col].set_yticks(y_ticks)
        ax11[row, col].set_yticklabels(np.round(y_ticks / 299, 1))
        # ax11[row, col].set_ylabel('sec')


        # Title
        if row == 0:
            ax11[row, col].set_title(f'{pcs}')
    fig11.suptitle('Time absolute cc peaks pre vs post (alternative visualisation)')


    # idx = pd.IndexSlice
    # df_contrasts_short = df_contrasts.reset_index()
    # fig11, ax11 = plt.subplots(2, figsize=figsize, sharex=True, sharey=True)
    # for idx, cond in enumerate(df_contrasts_short['cond'].unique()):
    #     sns.boxplot(data=df_contrasts_short[df_contrasts_short.cond==cond], x='pcs', y='time', hue='t_drop', hue_order=['pre', 'post'], ax=ax11[idx])
    #     sns.stripplot(data=df_contrasts_short[df_contrasts_short.cond==cond], x='pcs', y='time', hue='t_drop', hue_order=['pre', 'post'], ax=ax11[idx], dodge=True, size=11)
    #     # Set y ticks
    #     # y_ticks = ax11[idx].get_yticks()
    #     # ax11[idx].set_yticks(y_ticks)
    #     # ax11[idx].set_yticklabels(np.round(y_ticks / 299, 1))
    #     # ax11[idx].set_ylabel('sec')
    #     ax11[idx].get_legend().remove()

    # fig11.suptitle('Time absolute cc peaks pre vs post')



    plt.show()


    pdb.set_trace()
    if save_plot:
        # Check /data_analysis is in current directory 
        if os.path.basename(os.getcwd()) != 'data_analysis':
            os.chdir('./data_analysis')

        # Check if lowd folder exists
        if not os.path.isdir(os.getcwd() + '/images/lowd'):
            os.mkdir(os.getcwd() + '/images/lowd')

        for fig_idx, rec_idx in enumerate(df_wsk_avg_l.rec.unique()):
            # fig_idx and rec_idx different because of dropped data
            fig_list[fig_idx].savefig(os.getcwd() + f'/images/lowd/{os.path.basename(wskpaths[rec_idx][:-1])}_wsk_vs_pcs_vs_cc_newmethod.svg', format='svg')

        fig1.savefig(os.getcwd() + f'/images/lowd/boxplot_maxcc_newmethod.svg', format='svg')
        # fig2.savefig(os.getcwd() + f'/images/lowd/boxplot_maxcc_1controls.svg', format='svg')
        fig3.savefig(os.getcwd() + f'/images/lowd/boxplot_cumsumcc_newmethod.svg', format='svg')
        # fig4.savefig(os.getcwd() + f'/images/lowd/boxplot_cumsumcc_1controls.svg', format='svg')
        fig5.savefig(os.getcwd() + f'/images/lowd/coef_diff_stratified_newmethod.svg', format='svg')
        fig6.savefig(os.getcwd() + f'/images/lowd/maxcc_stratified_newmethod.svg', format='svg')
        fig7.savefig(os.getcwd() + f'/images/lowd/cumsumcc_stratified_newmethod.svg', format='svg')
        fig8.savefig(os.getcwd() + f'/images/lowd/wskdiff_vs_ccdiff_cumsumcc_newmethod.svg', format='svg')
        fig9.savefig(os.getcwd() + f'/images/lowd/wskdiff_vs_ccdiff_maxcc_newmethod.svg', format='svg')
        # fig8.savefig(os.getcwd() + f'/images/lowd/wskdiff_vs_ccdiff_cumsumcc_zscored.svg', format='svg')
        # fig9.savefig(os.getcwd() + f'/images/lowd/wskdiff_vs_ccdiff_maxcc_zscored.svg', format='svg')
        fig10.savefig(os.getcwd() + f'/images/lowd/ccpeaks_time_newmethod.svg', format='svg')
        fig11.savefig(os.getcwd() + f'/images/lowd/ccpeaks_time_alternative_newmethod.svg', format='svg')


def cc_regression(data):
    """ Use regression to assess effect of manipulation on changes in cc (either max or
    cumsum) between different pc's and wsk variable.
    """
    pdb.set_trace()
    # Add category combining condition and pc (used for β's)
    data.insert(loc=2, column='pc_cond', value=tuple(map(lambda x: x[::2], np.asarray(data.index))))
    data['pc_cond'] = data['pc_cond'].transform(lambda x: x[0] + '_' + x[1])

    # Logit transform data (avoid inf)
    data['logitvalue'] = logit(data.value * 0.99)

    # Dummy variable for time
    data['time_dummy'] = data.t_drop.map(lambda x: 0 if x=='pre' else 1)
    
    # Create indexes
    condpc_idx, condpc_codes = data.pc_cond.factorize()
    # time_idx, time_codes = data.t_drop.factorize()
    # coords = {'condpc':condpc_codes, 'time':time_codes}
    coords = {'condpc':condpc_codes}

    with pm.Model(coords=coords) as cc_reg:
        # Linear terms
        β_condpc = pm.Normal('β_condpc', mu=0, sigma=1, dims='condpc')
        # β_time = pm.Normal('β_time', mu=0, sigma=1, dims='time')
        β_time = pm.Normal('β_time', mu=0, sigma=1)
        α = pm.Normal('α', mu=0, sigma=1)
        # Deterministic mean
        mu = pm.Deterministic('mu', α + β_time * data.time_dummy + β_condpc[condpc_idx])
        # Prior std
        sigma_prior = pm.Exponential('sigma_prior', lam=2)
        # Likelihood
        likelihood = pm.Normal('likelihood', mu=mu, sigma=sigma_prior, observed=data.logitvalue)

    # Prior predictive checks
    with cc_reg:
        cc_reg_priorpc = pm.sample_prior_predictive(samples=1000)
    # Sampling
    with cc_reg:
        trace = pm.sample()
    # Posterior predictive checks
    with cc_reg:
        cc_reg_postpc = pm.sample_posterior_predictive(trace)


    az.plot_ppc(cc_reg_priorpc, group='prior')
    az.plot_ppc(cc_reg_postpc)
    az.summary(trace, var_names=['β'], filter_vars='like')
    az.plot_trace(trace)

    # Posterior predictive checks (mine)
    
    # Contrasts
    condpc_posteriors= pd.DataFrame()
    for condpc in trace.posterior['β_condpc'].condpc.values:
        __ = trace.posterior['β_condpc'].sel(condpc=[condpc]).copy()
        __ = __.stack(stack_dim=('chain', 'draw'))
        __ = __.values.copy()
        __ = pd.DataFrame(__.T, columns=[condpc])
        condpc_posteriors = pd.concat([condpc_posteriors, __], axis=1)

    df_contrasts = pd.DataFrame()
    for idx, contr in enumerate(['pc1', 'pc2', 'pc3']):
        __ = pd.DataFrame(condpc_posteriors.iloc[:, idx] - condpc_posteriors.iloc[:, idx + 3], columns=[contr])
        df_contrasts = pd.concat([df_contrasts, __], axis=1)
        
    sns.stripplot(data=df_contrasts.melt(), y='variable', x='value')
    
    # Visualise transformed data
    fig1, ax1 = plt.subplots(2, 3, tight_layout=True, sharex=True, sharey=True, figsize=figsize)
    for row, col in itertools.product(range(2), range(3)):
        # cond = ['gCNO', 'wCNO', 'aPBS'][row]
        cond = ['gCNO', 'control'][row]
        pcs = ['pc1', 'pc2', 'pc3'][col]
        __ = data.loc[(cond, slice(None), pcs)].copy()
        # data.value = data.value / data.value.max()
    
        sns.pointplot(x='t_drop', y='logitvalue', data=__, order=['pre', 'post'], errorbar='se', ax=ax1[row, col])
        # sns.boxplot(x='t_drop', y='value', data=data, order=['pre', 'post'], whis=2,  ax=ax1[row, col], palette='deep')

        sns.swarmplot(x='t_drop', y='logitvalue', data=__, order=['pre', 'post'], hue='t_drop', hue_order=['pre', 'post'], ax=ax1[row, col], palette='deep', size=7, edgecolor='black', linewidth=1)

        # Legend
        if (row == 0) & (col == 2):
            ax1[row, col].legend(bbox_to_anchor=(.8, .6))
        else:
            ax1[row, col].legend([], [], frameon=False)

        # Axis labels
        if row !=2:
            # ax1[row, col].set_xlabel()
            ax1[row, col].axes.xaxis.set_visible(False)
        if col != 0:
            ax1[row, col].axes.yaxis.set_visible(False)

        # y label
        if col==0:
            ax1[row, col].set_ylabel(f'{cond}')        

        # Title
        if row == 0:
            ax1[row, col].set_title(f'{pcs}')
    fig1.suptitle('Max cross correlation (wsk vs neural pcs)')


def time_peak(df_cc, conditions):
    """ Find time of absolute cc peak
    """
    new_cond = conditions[0].map({'gCNO':'gCNO', 'wCNO':'control', 'aPBS':'control'}).to_dict()
    new_df_cc = df_cc.copy()
    __ = new_df_cc.columns.to_frame()
    __['cond'] = __['rec']
    new_df_cc.columns = pd.MultiIndex.from_frame(__)
    new_df_cc = new_df_cc.reorder_levels(['cond', 'rec', 't_drop', 'pcs'], axis=1)
    new_df_cc.rename(columns=new_cond, level=0, inplace=True)

    # Take absolute value
    new_df_cc = new_df_cc.apply(np.abs, axis=0)

    # Series of time absolute cc peaks
    cc_peaks_time = pd.DataFrame(new_df_cc.idxmax(), columns=['time'])

    return cc_peaks_time


def fit_ccval_diff(df_ccval, cc_value):
    """ Fit the difference in post-pre absolute cc maximum or cumsum for each pc;
    compare (contrast) differences between conditions.
    """
    df_contrasts = df_ccval.copy()
    df_contrasts = df_contrasts.set_index('t_drop', append=True)
    df_contrasts = (df_contrasts - df_contrasts.mean()) / df_contrasts.std()
    df_contrasts = df_contrasts.unstack(level=3).diff(periods=-1, axis=1).iloc[:,0]
    df_contrasts = pd.DataFrame(df_contrasts).droplevel(level=0, axis=1)
    df_contrasts.columns = ['post-pre']

    condcodes, conduniques = pd.factorize(df_contrasts.index.droplevel(1))
    conduniques = pd.DataFrame(conduniques)[0].apply(lambda x: x[0] + '_' + x[1])

    with pm.Model(coords={'condition':conduniques}) as contrast_model:
        mu = pm.Normal('mu', mu=0, sigma=1, dims='condition')
        sigma = pm.Exponential('sigma', lam=3, dims='condition')
        likelihood = pm.Normal('likelihood', mu=mu[condcodes], sigma=sigma[condcodes], observed=df_contrasts.values[:, 0])

    with contrast_model:
        contrast_model_priorpc = pm.sample_prior_predictive()
    with contrast_model:
        contrast_model_trace = pm.sample()
    with contrast_model:
        contrast_model_postpc = pm.sample_posterior_predictive(contrast_model_trace)

    # DataFrame posterior predictive checks
    reduced_ppc = contrast_model_postpc.stack(stack_dim=('chain', 'draw'))
    df_ppc = reduced_ppc.posterior_predictive['likelihood'].values
    df_ppc = pd.DataFrame(df_ppc, index=df_contrasts.index)        

    # DataFrame contrasts of differences
    reduced_trace = contrast_model_trace.stack(stack_dim=('chain', 'draw'))
    df_postdiff = pd.DataFrame()
    for cond in conduniques:
        __ = reduced_trace.posterior['mu'].sel(condition=cond).copy()
        __ = pd.DataFrame(__, columns=[cond])
        df_postdiff = pd.concat([df_postdiff, __], axis=1)

    df_contrastdiff = pd.DataFrame()
    for pc in range(3):
        __ = df_postdiff.iloc[:, pc + 3] - df_postdiff.iloc[:, 0]
        __ = pd.DataFrame(__, columns=[pc])
        df_contrastdiff = pd.concat([df_contrastdiff, __], axis=1)

    # Hdi
    df_hdi = pd.DataFrame()
    for idx, pc in enumerate(['pc1', 'pc2', 'pc3']):
        __ = pd.DataFrame(az.hdi(df_contrastdiff[idx].values, hdi_prob=.94), columns=[pc])
        df_hdi = pd.concat([df_hdi, __], axis=1)

    pdb.set_trace()
    # Stats
    idxslice = pd.IndexSlice
    ttest_cc_values = []
    for pc in ['pc1', 'pc2', 'pc3']:
        __ = df_contrasts.loc[idxslice[:, :, pc]]
        ttest_cc_values.append(ttest_ind(__.loc['gCNO'], __ .loc['control']))
        
    # with open(os.getcwd() + '/save_data/lowd_tdiff_ttestInd', 'wb') as f:
    #     pickle.dump(ttest_cc_time, f)

    # Plot figures
    # Initialise plot parameters
    figsize = (14, 12)
    xticks = np.arange(0, np.diff(t_wind) * 299, 299)
    xtick_labels = np.arange(t_wind[0], t_wind[1])
    linewidth = 3
    sns.set_context("poster", font_scale=1)
    sns.color_palette("deep")
    plt.rcParams['axes.facecolor'] = 'lavenderblush'
    plt.rcParams["figure.edgecolor"] = 'black'
    plt.rcParams['patch.edgecolor'] = 'none'

    # Posterior predictive checks
    fig, ax = plt.subplots(1, 2, sharey=True, figsize=figsize)
    sns.boxplot(data=df_contrasts.reset_index(), y='post-pre', x='pcs', hue='cond', ax=ax[0], palette="Set3")
    sns.stripplot(data=df_contrasts.reset_index(), y='post-pre', x='pcs', hue='cond', ax=ax[0], palette="Set3", size=10, dodge=True)
    sns.boxplot(data=df_ppc.stack().reset_index(), y=0, x='pcs', hue='cond', ax=ax[1], palette="Set3", showfliers=False)
    fig.suptitle(f'(Post-pre) {cc_value} differences vs ppc')

    # Plot contrast of differences
    fig1, ax1 = plt.subplots(3, sharex=True, sharey=True, figsize=figsize)
    for idx, axis in enumerate(ax1):
        sns.histplot(data=df_contrastdiff[idx], ax=axis)
        axis.hlines(300, df_hdi.iloc[0, idx], df_hdi.iloc[1, idx], )
        axis.set_ylabel(f'pc{idx + 1}')
        axis.set_ylim([0, 380])
        axis.text(0, 310, 'hdi 94%')

    fig1.suptitle(f'(Condition) Contrasts of (post-pre) differeces in {cc_value}')

    # Save
    if not os.path.isdir(os.getcwd() + '/images/lowd'):
        os.mkdir(os.getcwd() + '/images/lowd')

    try:
        fig.savefig(os.getcwd() + f'/images/lowd/fittdiff_ppc_diff_{cc_value}_newmethod.svg', format='svg')
        fig1.savefig(os.getcwd() + f'/images/lowd/fittdiff_contrast_diff_{cc_value}_newmethod.svg', format='svg')
    except FileNotFoundError:
        print(f'create dir /images/lowd/')
        os.makedirs(os.getcwd() + '/images/lowd/')
        fig.savefig(os.getcwd() + f'/images/lowd/fittdiff_ppc_diff_{cc_value}_newmethod.svg', format='svg')
        fig1.savefig(os.getcwd() + f'/images/lowd/fittdiff_contrast_diff_{cc_value}_newmethod.svg', format='svg')    


def fit_cctime_diff(df_ccpeak):
    """ Fit the difference in post-pre absolute cc peak times for each pc;
    compare (contrast) difference between conditions.
    Note: newmethod uses scipy.signal.correlate instead of np.correlate!
    """
    # Compute difference of post-pre time cc absolute peaks (Z-scored by condition and pc)
    pdb.set_trace()
    df_contrasts = df_ccpeak.copy()
    df_contrasts = df_contrasts.groupby(by=['cond', 'pcs'], group_keys=False).apply(lambda x: (x - x.mean()) / x.std())
    df_contrasts = df_contrasts.unstack(level=2).diff(periods=-1, axis=1).iloc[:, 0]
    df_contrasts = pd.DataFrame(df_contrasts).droplevel(level=0, axis=1)
    df_contrasts.columns = ['post-pre']

    # Compute posterior distribution over differences
    condcodes, conduniques = pd.factorize(df_contrasts.index.droplevel(1))
    conduniques = pd.DataFrame(conduniques)[0].apply(lambda x: x[0] + '_' + x[1])

    with pm.Model(coords={'condition':conduniques}) as contrast_model:
        mu = pm.Normal('mu', mu=0, sigma=1, dims='condition')
        sigma = pm.Exponential('sigma', lam=3, dims='condition')
        likelihood = pm.Normal('likelihood', mu=mu[condcodes], sigma=sigma[condcodes], observed=df_contrasts.values[:, 0])

    # with contrast_model:
    #     contrast_model_priorpc = pm.sample_prior_predictive(samples=1000)
    with contrast_model:
        contrast_model_trace = pm.sample()
    with contrast_model:
        contrast_model_postpc = pm.sample_posterior_predictive(contrast_model_trace)

    # DataFrame posterior predictive checks
    reduced_ppc = contrast_model_postpc.stack(stack_dim=('chain', 'draw'))
    df_ppc = reduced_ppc.posterior_predictive['likelihood'].values
    df_ppc = pd.DataFrame(df_ppc, index=df_contrasts.index)        

    # DataFrame contrasts of differences
    reduced_trace = contrast_model_trace.stack(stack_dim=('chain', 'draw'))
    df_postdiff = pd.DataFrame()
    for cond in conduniques:
        __ = reduced_trace.posterior['mu'].sel(condition=cond).copy()
        __ = pd.DataFrame(__, columns=[cond])
        df_postdiff = pd.concat([df_postdiff, __], axis=1)

    df_contrastdiff = pd.DataFrame()
    for pc in range(3):
        __ = df_postdiff.iloc[:, pc + 3] - df_postdiff.iloc[:, 0]
        __ = pd.DataFrame(__, columns=[pc])
        df_contrastdiff = pd.concat([df_contrastdiff, __], axis=1)

    # Hdi
    df_hdi = pd.DataFrame()
    for idx, pc in enumerate(['pc1', 'pc2', 'pc3']):
        __ = pd.DataFrame(az.hdi(df_contrastdiff[idx].values, hdi_prob=.94), columns=[pc])
        df_hdi = pd.concat([df_hdi, __], axis=1)

    # Stats
    idxslice = pd.IndexSlice
    ttest_cc_time = []
    for pc in ['pc1', 'pc2', 'pc3']:
        __ = df_contrasts.loc[idxslice[:, :, pc]]
        ttest_cc_time.append(ttest_ind(__.loc['gCNO'], __ .loc['control']))
        
    with open(os.getcwd() + '/save_data/lowd_tdiff_ttestInd_newmethod', 'wb') as f:
        pickle.dump(ttest_cc_time, f)

    pdb.set_trace()
    # Plot figures
    # Initialise plot parameters
    figsize = (14, 12)
    xticks = np.arange(0, np.diff(t_wind) * 299, 299)
    xtick_labels = np.arange(t_wind[0], t_wind[1])
    linewidth = 3
    sns.set_context("poster", font_scale=1)
    sns.color_palette("deep")
    plt.rcParams['axes.facecolor'] = 'lavenderblush'
    plt.rcParams["figure.edgecolor"] = 'black'
    plt.rcParams['patch.edgecolor'] = 'none'

    # Posterior predictive checks
    fig, ax = plt.subplots(1, 2, sharey=True, figsize=figsize)
    sns.boxplot(data=df_contrasts.reset_index(), y='post-pre', x='pcs', hue='cond', ax=ax[0], palette="Set3")
    sns.boxplot(data=df_ppc.stack().reset_index(), y=0, x='pcs', hue='cond', ax=ax[1], palette="Set3", showfliers=False)
    fig.suptitle('(Post-pre) time differences vs ppc')

    # Plot contrast of differences
    fig1, ax1 = plt.subplots(3, sharex=True, sharey=True, figsize=figsize)
    for idx, axis in enumerate(ax1):
        sns.histplot(data=df_contrastdiff[idx], ax=axis)
        axis.hlines(300, df_hdi.iloc[0, idx], df_hdi.iloc[1, idx], )
        axis.set_ylabel(f'pc{idx + 1}')
        axis.set_ylim([0, 380])
        axis.text(0, 310, 'hdi 94%')

    fig1.suptitle('(Condition) Contrasts of (post-pre) time differeces')

    # Save
    if not os.path.isdir(os.getcwd() + '/images/lowd'):
        os.mkdir(os.getcwd() + '/images/lowd')

    try:
        fig.savefig(os.getcwd() + '/images/lowd/fittdiff_ppc_diff_time_newmethod.svg', format='svg')
        fig1.savefig(os.getcwd() + '/images/lowd/fittdiff_contrast_diff_time_newmethod.svg', format='svg')
    except FileNotFoundError:
        print(f'create dir /images/lowd/')
        os.makedirs(os.getcwd() + '/images/lowd/')
        fig.savefig(os.getcwd() + '/images/lowd/fittdiff_ppc_diff_time_newmethod.svg', format='svg')
        fig1.savefig(os.getcwd() + '/images/lowd/fittdiff_contrast_diff_time_newmethod.svg', format='svg')


def ttest_ccpeaks(ccpeak_data):
    """ T-test for difference pre vs post, and difference 0 mean
    """
    pdb.set_trace()
    idxslice = pd.IndexSlice
    ttest_tpeaks_prepost = []
    ttest_tpeaks_0 = []
    midx_prepost = []
    midx_0 = []
    for cond in ['gCNO', 'control']:
        for pc in ['pc1', 'pc2', 'pc3']:
            predata = ccpeak_data.loc[idxslice[cond, :, 'pre', pc]]['time']
            postdata = ccpeak_data.loc[idxslice[cond, :, 'post', pc]]['time']
            __ = stats.ttest_ind(predata, postdata)
            ttest_tpeaks_prepost.append(__)
            midx_prepost.append(np.array([cond, pc]))
            for time in ['pre', 'post']:
                __ = stats.ttest_1samp(ccpeak_data.loc[idxslice[cond, :, time, pc]]['time'], popmean=0)
                ttest_tpeaks_0.append(__)
                midx_0.append(np.array([cond, pc, time]))

    midx_prepost = pd.MultiIndex.from_arrays(np.array(midx_prepost).T)
    midx_0 = pd.MultiIndex.from_arrays(np.array(midx_0).T)

    ttest_tpeaks_prepost = pd.DataFrame(ttest_tpeaks_prepost, index=midx_prepost)
    ttest_tpeaks_0 = pd.DataFrame(ttest_tpeaks_0, index=midx_0)

    return ttest_tpeaks_prepost, ttest_tpeaks_0


def run_lowd(std=5,
             var='angle',
             whisker='whisk2',
             t_wind=[-2, 3],
             subwind=[-0.5, 1.5],
             drop_rec=[0, 10, 15, 16, 25],
             cgs=2,
             surr=False,
             pre=None,
             discard=False,
             load=True,
             save_data=False,
             save_plot=True):
    """ For each recording get data, perform correlation analysis,  Load data, perform correlation analysis, ......
    Attention: df_spk is smoothed in corr_anal function and is modified
    permanently.
    """
    projd_avg_list = []
    corr_list = []
    df_wsk_list = []
    df_spk_list = []
    df_fr_list = []
    fr_all_list = []
    conditions_list = []
    color_list = []
    wskpath_list = []

    pdb.set_trace()
    if not load:
        for rec_idx in range(25):   # 25 number of good recordings
            # Load data (get_data from glm_module.py)
            __ = get_data(rec_idx,
                          # std,
                          wvar=var,
                          whisk=whisker,
                          t_wind=t_wind,
                          drop_rec=drop_rec,
                          cgs=cgs)
            df_wsk = __[0]
            df_spk = __[1]
            df_fr = __[2]
            fr_all = __[3]
            conditions = __[4]
            wskpath = __[5]
            pdb.set_trace()

            # Correlation analysis (corr_anal from glm_module.py)
            __ = corr_anal(fr_all, df_spk, std)
            corr = __[0]
            egnvec = __[1]
            egnval = __[2]
            projd = __[3]
            color = __[4]

            projd_avg_list.append(projd.groupby(by=['time', 'pcs'], axis=1).mean())
            corr_list.append(corr)
            df_wsk_list.append(df_wsk)
            df_spk_list.append(df_spk)
            df_fr_list.append(df_fr)
            fr_all_list.append(fr_all)
            conditions_list.append(conditions)
            wskpath_list.append(wskpath)
            color_list.append(color)

        if save_data:
            all_data = [projd_avg_list, corr_list, df_wsk_list, df_spk_list, df_fr_list, fr_all_list, conditions_list, wskpath_list, color_list]
            with open('/home/yq18021/Documents/github_gcl/data_analysis/save_data/lowd_data.pickle', 'wb') as f:
                pickle.dump(all_data, f)

    else:                       # load data
        with open('/home/yq18021/Documents/github_gcl/data_analysis/save_data/lowd_data.pickle', 'rb') as f:
            all_data = pickle.load(f)

    # Organise in df
    pdb.set_trace()
    df_projd_avg = pd.concat(all_data[0], axis=1, keys=range(25), names=['rec', 't_drop', 'pcs'])
    df_wsk_avg = pd.concat(all_data[2], axis=1, keys=range(25), names=['rec', 't_drop', 'trial'])

    # Discard idx_rec=1
    # (220318 A): bad projd pre and post (should discard; adds noise)
    # df_projd_avg.drop(columns=[1], level='rec', inplace=True)
    # df_wsk_avg.drop(columns=[1], level='rec', inplace=True)

    # Get average whisker var per recording and period
    df_wsk_avg = df_wsk_avg.groupby(by=['rec', 't_drop'], axis=1).mean()

    # Other variables
    conditions = pd.DataFrame(all_data[6])
    wskpaths = all_data[7]
    color = all_data[8][0]

    # Compute cross-correlation, cumsum/peak and tertile
    df_cc = crosscorr(df_projd_avg, df_wsk_avg, t_wind=t_wind, subwind=subwind)
    cumsum_df_cc, totcumsum_df_cc, max_df_cc = anal_cc(df_cc, conditions)
    pdb.set_trace()

    # Load coeff_diff and divide in positive/negative values
    if os.path.basename(os.getcwd()) != 'data_analysis':
        os.chdir('./data_analysis')
    with open(os.getcwd() + '/save_data/diff_slope_whisker_linfit_p.pickle', 'rb') as f:
        coef_diff = pickle.load(f)

    coef_diff = coef_diff['coef_p']
    coef_diff.rename({'coef':'coef_diff'}, axis=1, inplace=True)
    coef_diff = coef_diff.reset_index().set_index('index')
    coef_diff['slope_change'] = np.where(coef_diff['coef_diff']>0, 'increase', 'decrease')

    # Remap controls
    remap_cond = {'gCNO':'gCNO', 'aPBS':'control', 'wCNO':'control'}
    totcumsum_df_cc.rename(index=remap_cond, level=0, inplace=True)
    max_df_cc.rename(index=remap_cond, level=0, inplace=True)

    # Scale by maximum
    __ = max_df_cc.set_index('t_drop', append=True).copy()
    max_df_cc['value'] = __.groupby(by=['cond', 'pcs'], group_keys=False).apply(lambda x: (x / x.max())).values
    __ = totcumsum_df_cc.set_index('t_drop', append=True).copy()
    totcumsum_df_cc['value'] = __.groupby(by=['cond', 'pcs'], group_keys=False).apply(lambda x: (x / x.max())).values


    # labels_cat = [r'$-$ slope', r'$+$ slope']
    labels_cat = [r'$+$ slope', r'$\sim$ slope', r'$-$ slope']
    cd_tertile = pd.qcut(coef_diff.iloc[:, 0], 3, labels=labels_cat)
    # labels_cat = [r'$+$ slope', r'$+\sim$ slope', r'$\sim$ slope', r'$-$ slope']
    # cd_tertile = coef_diff.groupby(by=
    #                                'index').apply(lambda x:
    #                                               pd.qcut(x.squeeze(),
    #                                                       2, labels=labels_cat))
    # cd_tertile = coef_diff.groupby(by=
    #                                'index').apply(lambda x:
    #                                               pd.qcut(x.squeeze(),
    #                                                       2, labels=[r'$-$ slope', r'$+$ slope']))
    # cd_tertile = cd_tertile.droplevel(level=1)
    pdb.set_trace()

    # Stats (ttest) cc peaks
    ttest_peak = pd.DataFrame()
    wilc_peak = pd.DataFrame()
    # for slope in ['increase', 'decrease']:
    for cond in ('gCNO', 'control'):
        for pc in ('pc1', 'pc2', 'pc3'):
            # Pre and post data
            __ = max_df_cc.loc[(cond, slice(None), pc)]
            pred = (__[__.t_drop == 'pre']).value.copy()
            postd = (__[__.t_drop == 'post']).value.copy()
            # Stats
            ttest_ = ttest_rel(postd, pred)
            wilc_ = wilcoxon(postd, pred)
            # Dataframe
            # midx = pd.MultiIndex.from_arrays([[slope], [cond], [pc]], names=('slope', 'condition', 'pcs'))
            midx = pd.MultiIndex.from_arrays([[cond], [pc]], names=('condition', 'pcs'))
            ttest_ = pd.DataFrame({'test':ttest_[0], 'pvalue':ttest_[1]}, index=midx)
            ttest_peak = pd.concat([ttest_peak, ttest_], axis=0)
            wilc_ = pd.DataFrame({'test':wilc_[0], 'pvalue':wilc_[1]}, index=midx)
            wilc_peak = pd.concat([wilc_peak, wilc_], axis=0)

    # Stats (ttest) cc cumsum
    ttest_cumsum = pd.DataFrame()
    wilc_cumsum = pd.DataFrame()
    # for slope in ['increase', 'decrease']:
    for cond in ('gCNO', 'control'):
        for pc in ('pc1', 'pc2', 'pc3'):
            # Pre and post data
            __ = totcumsum_df_cc.loc[(cond, slice(None), pc)]
            pred = (__[__.t_drop == 'pre']).value.copy()
            postd = (__[__.t_drop == 'post']).value.copy()
            # Stats
            ttest_ = ttest_rel(postd, pred)
            wilc_ = wilcoxon(postd, pred)
            # Dataframe
            # midx = pd.MultiIndex.from_arrays([[slope], [cond], [pc]], names=('slope', 'condition', 'pcs'))
            midx = pd.MultiIndex.from_arrays([[cond], [pc]], names=('condition', 'pcs'))
            ttest_ = pd.DataFrame({'test':ttest_[0], 'pvalue':ttest_[1]}, index=midx)
            ttest_cumsum = pd.concat([ttest_cumsum, ttest_], axis=0)
            wilc_ = pd.DataFrame({'test':wilc_[0], 'pvalue':wilc_[1]}, index=midx)
            wilc_cumsum = pd.concat([wilc_cumsum, wilc_], axis=0)

    # Save
    if os.path.basename(os.getcwd()) != 'data_analysis':
        os.chdir('./data_analysis')
    stats_cc_prepost = [ttest_peak, wilc_peak, ttest_cumsum, wilc_cumsum]
    with open(os.getcwd() + '/save_data/stats_cc_prepost_peak_cumsum_newmethod.pickle', 'wb') as f:
        pickle.dump(stats_cc_prepost, f)


    pdb.set_trace()

    # Time of absolute cc peak
    df_ccpeak = time_peak(df_cc, conditions)

    # Stats on cc peak times
    ttest_tpeaks_prepost, ttest_tpeaks_0 = ttest_ccpeaks(df_ccpeak)

    if os.path.basename(os.getcwd()) != 'data_analysis':
        os.chdir('./data_analysis')
    ttests_timepeaks = [ttest_tpeaks_prepost, ttest_tpeaks_0]
    with open(os.getcwd() + '/save_data/ttest_timepeaks_newmethod.pickle', 'wb') as f:
        pickle.dump(ttests_timepeaks, f)

    pdb.set_trace()

    # Compare time difference across conditions
    fit_cctime_diff(df_ccpeak)
    pdb.set_trace()

    # Compare cc maximum/cumsum
    for idx, val in enumerate(['max_cc', 'totcumsum_cc']):
        __ = [max_df_cc, totcumsum_df_cc][idx]
        fit_ccval_diff(__, val)

    pdb.set_trace()
    plot_lowd(df_projd_avg,
              df_wsk_avg,
              df_cc,
              cumsum_df_cc,
              totcumsum_df_cc,
              max_df_cc,
              df_ccpeak,
              coef_diff,
              cd_tertile,
              remap_cond,
              conditions,
              wskpaths,
              color,
              t_wind=t_wind,
              var=var,
              save_plot=save_plot)


if __name__ == '__main__':
    """ Run script if spkcount_analysis.py module is main programme
    """
    # Parameters
    std = 5                     # std to smooth spikes
    whisker = 'whisk2'                 # which whisker "" ""
    t_wind = [-2, 3]            # t_wind around whisking bout start
    # t_wind = [-10, 10]            # t_wind around whisking bout start
    subwind = [-1.5, 1.5]
    # subwind = [-1, 1]
    var = 'angle'
    # var = 'setpoint'
    # var = 'amplitude'           # which whisker var "" ""
    cgs = 2
    surr = False
    # surr = 'shuffle'
    pre = None
    # pre = True
    discard = False             # discard clu with fr < 0.1 Hz
    # bout_idxs = [15, 16, 17, 18]  # single wbout to display
    drop_rec = [0, 10, 15, 16, 25]
    load = True
    save_data = False
    save_plot = True

    # plot_lowd(rec_idx, whisker=whisker, t_wind=t_wind, var=var, cgs=cgs, surr=surr, pre=pre, discard=discard)
    run_lowd(std=std, drop_rec=drop_rec, whisker=whisker, t_wind=t_wind, subwind=subwind, var=var, cgs=cgs, surr=surr, pre=pre, discard=discard, load=load, save_data=save_data, save_plot=save_plot)
