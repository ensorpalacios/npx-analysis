"""
Script to compare properties of peri-event time histogram between pre- and post
drug drop, across conditions: leverages peth_module.py to compute PETH of all
good recordings (25, filtered twice); compute std max peak/trough within
subwind; compute std peaks for each recording and compare (Wilcoxon signed-rank
test) pre vs post, stratified by experimental condition.

Contains:

- helper_plot: function to plot results.

- run_peth_compare: compute PETH; leverages peth_module.py; calculate peak
activity for each cluster within subwind (smaller that trial length).

- std_compare: compute peak std for each recording, compare pre vs post
stratified by condition using Wilcoxon signed-rank test; use helper_plot to
generate plots and save them.

ATTENTION: loading whisk_plot.py output; run that before 

"""
# Ensure /data_analysis is in path
import os
import sys
if not ('data_analysis' in os.getcwd()):
    sys.path.append(os.path.join(os.getcwd(), 'data_analysis'))

import pdb
import pickle
from peth_module import peth
from meta_data import load_meta
from scipy.stats import wilcoxon, ttest_rel

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import pandas as pd
import pymc as pm
import arviz as az
from sklearn.linear_model import LinearRegression



def helper_plot(df_wvar_pre,
                df_wvar_post,
                spktr_data_pre,
                spktr_data_post,
                peak_pre,
                peak_post,
                n_clu,
                wskpath,
                windowlength,
                surr=False,
                peth_discard=True):
    """ Helper function to plot and save PETH
    """
    # Sort cluster by peak
    peak_idx_pre = np.argsort(peak_pre)
    peak_idx_post = np.argsort(peak_post)
    spktr_data_pre = spktr_data_pre[peak_idx_pre, :]
    spktr_data_post = spktr_data_post[peak_idx_post, :]
    peak_pre = peak_pre[peak_idx_pre]
    peak_post = peak_post[peak_idx_post]

    # Initialise figure
    lcolor = 'darkgoldenrod'
    style = 'italic'
    deg = np.array([df_wvar_pre.value.mean(),
                    df_wvar_pre.value.mean() + 5 / (180 / np.pi)])
    # deg = np.array([0.8, 0.8 + 10 / (180 / np.pi)])
    sec = np.array([200, 200 + 0.5 * 299])
    # figsize = (12, 12)
    figsize = (14, 12)
    sns.set_context("poster", font_scale=1)
    sns.color_palette("deep")
    plt.rcParams['axes.facecolor'] = 'lavenderblush'
    plt.rcParams["figure.edgecolor"] = 'black'

    # Plot
    fig1 = plt.figure(figsize=figsize)
    subfigs = fig1.subfigures(2, 1)

    # Subfigure PRE ###################
    ((ax1, __), (ax2, cbar_ax)) = subfigs[0].subplots(2, 2, sharex='col', gridspec_kw={'height_ratios': [1, 2], 'width_ratios': [10, 1]})

    # Add subplot title
    ax1.set_title('Pre drop')
    # Mean whisker activity
    sns.lineplot(x='idx', y='value', data=df_wvar_pre, color=lcolor, errorbar='se', ax=ax1)

    # Rescale ylim if necessary (need to fit unit bar in figure)
    ylim = ax1.get_ylim()
    ax1.set_ylim(ylim[0], max([deg[1] + 0.1, ylim[1]]))

    # Add scale bars
    ax1.vlines(sec[0], deg[0], deg[1], colors=lcolor)
    ax1.hlines(deg[0], sec[0], sec[1], colors=lcolor)
    ax1.text(sec[0], deg[1] + 0.01, '5 deg')
    ax1.text(sec.mean(), deg[0] + 0.01, '0.5 sec')

    # Clear axis except facecolor, add y label
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.spines['bottom'].set_visible(False)
    ax1.set_ylabel(f'whisker {var}')

    # Mean cluster activity
    hmap = ax2.matshow(spktr_data_pre, norm=colors.CenteredNorm(), cmap='coolwarm', aspect='auto')
    ax2.scatter(peak_pre, np.arange(n_clu), c='orangered')

    # Clear axis, add y label
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.spines['top'].set_visible(False)
    ax2.set_ylabel('Clusters')

    # Add colorbar
    cbar_ax.axis('off')

    plt.colorbar(hmap, ax=cbar_ax, label='activity (sd)')

    # Clear axis dummy subplot
    __.axis('off')
    for spine in __.spines.values():
        spine.set_visible(False)

    # Set subplots spacing
    left = 0.05
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

    # Subfigure POST ###################
    ((ax1, __), (ax2, cbar_ax)) = subfigs[1].subplots(2, 2, sharex='col', gridspec_kw={'height_ratios': [1, 2], 'width_ratios': [10, 1]})

    # Add subplot title
    ax1.set_title('Post drop')

    # Mean whisker activity
    sns.lineplot(x='idx', y='value', data=df_wvar_post, color=lcolor, errorbar='se', ax=ax1)

    # Rescale ylim if necessary (need to fit unit bar)
    ylim = ax1.get_ylim()
    ax1.set_ylim(ylim[0], max([deg[1] + 0.1, ylim[1]]))

    # Add scale bars
    ax1.vlines(sec[0], deg[0], deg[1], colors=lcolor)
    ax1.hlines(deg[0], sec[0], sec[1], colors=lcolor)
    ax1.text(sec[0], deg[1] + 0.01, '5 deg')
    ax1.text(sec.mean(), deg[0] + 0.01, '0.5 sec')

    # Clear axis except facecolor, add y label
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.spines['bottom'].set_visible(False)
    ax1.set_ylabel(f'whisker {var}')

    # Mean cluster activity
    hmap = ax2.matshow(spktr_data_post, norm=colors.CenteredNorm(), cmap='coolwarm', aspect='auto')
    ax2.scatter(peak_post, np.arange(n_clu), c='orangered')

    # Clear axis, add y label
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.spines['top'].set_visible(False)
    ax2.set_ylabel('Clusters')

    # Add colorbar
    cbar_ax.axis('off')

    plt.colorbar(hmap, ax=cbar_ax, label='activity (sd)')

    # Clear axis dummy subplot
    __.axis('off')
    for spine in __.spines.values():
        spine.set_visible(False)

    # Set subplots spacing
    left = 0.05
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

    fig1.suptitle('Peri-event time histogram', style=style)
    # pdb.set_trace()

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
        if not peth_discard:
            fig1.savefig(os.getcwd() + f'/images/peth/{os.path.basename(wskpath[:-1])}/{os.path.basename(wskpath[:-1])}_{var}_{surr}_pethdiscardFalse_{windowlength}.svg', format='svg')
        else:
            fig1.savefig(os.getcwd() + f'/images/peth/{os.path.basename(wskpath[:-1])}/{os.path.basename(wskpath[:-1])}_{var}_{surr}_pethdiscardTrue_{windowlength}.svg', format='svg')
    except FileNotFoundError:
        print(f'created dir images/peth/{os.path.basename(wskpath[:-1])}')
        os.makedirs(os.getcwd() + f'/images/peth/{os.path.basename(wskpath[:-1])}')
        if not selclu:
            fig1.savefig(os.getcwd() + f'/images/peth/{os.path.basename(wskpath[:-1])}/{os.path.basename(wskpath[:-1])}_{var}_{surr}_pethdiscardFalse_{windowlength}.svg', format='svg')
        else:
            fig1.savefig(os.getcwd() + f'/images/peth/{os.path.basename(wskpath[:-1])}/{os.path.basename(wskpath[:-1])}_{var}_{surr}_pethdiscardTrue_{windowlength}.svg', format='svg')

    fig1.clf()


def std_compare(peak_pre, peak_post, conditions, windowlength, plot_stdcomp=True, surr=False, save_data=False):
    """ Compute standard deviation of peaks for each recording pre and post;
    compare using Wilcoxon signed-rank test.
    """
    # Compute std and organise in df (pulled controls)
    df_std_p = pd.DataFrame(columns=['pre', 'post'])

    # Pool control conditions
    conditions_p = np.where(conditions=='gCNO', 'gCNO', 'control')
    for cond in ['gCNO', 'control']:
        # if cond == 'gCNO':
        #     mask = np.argwhere(conditions == cond)
        # else:
        #     mask = np.argwhere(conditions != 'gCNO')
        mask = np.argwhere(conditions_p == cond)
        data_pre = peak_pre[mask[0][0]: mask[-1][0] + 1]
        data_post = peak_post[mask[0][0]: mask[-1][0] + 1]
        std_pre = []
        std_post = []
        for rec in range(mask.size):
            std_pre.append(data_pre[rec].std())
            std_post.append(data_post[rec].std())

        __ = pd.DataFrame(np.array([std_pre, std_post]).T, index=[cond] * mask.size, columns=['pre', 'post'])

        df_std_p = pd.concat([df_std_p, __])

    # Compute Wilcoxon signed-rank test
    wilc_p = []
    for cond in np.unique(conditions_p):
        wilc_p.append(wilcoxon(df_std_p.loc[cond, 'pre'], df_std_p.loc[cond, 'post']))
    ttest_p = []
    for cond in np.unique(conditions_p):
        ttest_p.append(ttest_rel(df_std_p.loc[cond, 'pre'], df_std_p.loc[cond, 'post']))

    # Save stats
    stats = {'wilcoxon':wilc_p, 'ttest':ttest_p}
    if save_data:
        with open(f'/home/yq18021/Documents/github_gcl/data_analysis/save_data/stats_pethstd_compare_{windowlength}.pickle', 'wb') as f:
            pickle.dump(stats, f)

    # Compute difference of post-pre contrasts
    df_contrasts = df_std_p.diff(axis=1)['post'].reset_index()
    df_contrasts_z = df_contrasts.copy()
    df_contrasts_z['post'] = (df_contrasts['post'] - df_contrasts['post'].mean()) / df_contrasts['post'].std()
    # df_diff_cont = []
    # for cond in ['gCNO', 'control']:
    #     __ = df_contrasts.loc[cond].reset_index().pivot(columns='index')
    #     __ = __.droplevel(0, axis=1)
    #     df_diff_cont.append(__)
    # df_diff_cont = pd.concat(df_diff_cont, axis=1)
    codes, uniques = df_contrasts_z['index'].factorize()
    coords = dict(condition=uniques)

    RANDOM_SEED = 100
    rng = np.random.default_rng(RANDOM_SEED)

    with pm.Model(coords=coords) as fit_diff:
        mu = pm.Normal('mu', mu=0, sigma=1, dims='condition')
        sigma = pm.Exponential('sigma', lam=1, dims='condition')
        # sigma = pm.Exponential('sigma', lam=1)
        # alpha = pm.Normal('alpha', mu=0, sigma=1, dims='condition')
        likelihood = pm.Normal('likelihood', mu=mu[codes], sigma=sigma[codes], observed=df_contrasts_z['post'])        
        # likelihood = pm.Normal('likelihood', mu=mu[codes], sigma=sigma, observed=df_contrasts['post'])        
        # likelihood = pm.SkewNormal('likelihood', mu=mu[codes], sigma=sigma[codes], alpha=alpha[codes], observed=df_contrasts_z['post'])        
        
    with fit_diff:
        diff_priorpc = pm.sample_prior_predictive(random_seed=rng)
    with fit_diff:
        diff_trace = pm.sample(random_seed=rng)
    with fit_diff:
        diff_postpc = pm.sample_posterior_predictive(diff_trace, random_seed=rng)

    priorpc_plot = az.plot_ppc(diff_priorpc, group='prior')
    postpc_plot = az.plot_ppc(diff_postpc)
    trace_plot = az.plot_trace(diff_trace, legend=True)
    az.summary(diff_trace)

    # DataFrame contrasts of differences posteriors
    reduced_trace = diff_trace.stack(stack_dim=('chain', 'draw'))
    df_postdiff_mu = pd.DataFrame()
    for cond in ['gCNO', 'control']:
        __ = reduced_trace.posterior['mu'].sel(condition=cond).copy()
        __ = pd.DataFrame(__, columns=[cond])
        df_postdiff_mu = pd.concat([df_postdiff_mu, __], axis=1)

    df_postdiff_mu = df_postdiff_mu.diff(-1, axis=1)['gCNO']

    df_postdiff_sigma = pd.DataFrame()
    for cond in ['gCNO', 'control']:
        __ = reduced_trace.posterior['sigma'].sel(condition=cond).copy()
        __ = pd.DataFrame(__, columns=[cond])
        df_postdiff_sigma = pd.concat([df_postdiff_sigma, __], axis=1)

    df_postdiff_sigma = df_postdiff_sigma.diff(-1, axis=1)['gCNO']

    # Hdi
    hdi_mu = az.hdi(df_postdiff_mu.values)
    hdi_sigma = az.hdi(df_postdiff_sigma.values)
    # df_hdi = pd.DataFrame()
    #     __ = pd.DataFrame(az.hdi(df_contrastdiff[idx].values, hdi_prob=.94), columns=[pc])
    #     df_hdi = pd.concat([df_hdi, __], axis=1)

    save_to_dic = {'std_prepost': df_std_p, 'std_prepost_contrasts': df_contrasts}
    if save_data:
        with open(f'/home/yq18021/Documents/github_gcl/data_analysis/save_data/std_data_{windowlength}.pickle', 'wb') as f:
            pickle.dump(save_to_dic, f)

    # Plot distribution #######################
    # Initialise figure
    lcolor = 'darkgoldenrod'
    style = 'italic'
    figsize = (14, 12)
    sns.set_context("poster", font_scale=1)
    sns.color_palette("deep")
    plt.rcParams['axes.facecolor'] = 'lavenderblush'
    plt.rcParams["figure.edgecolor"] = 'black'

    # pdb.set_trace()
    # fig, axs = plt.subplots(2, figsize=figsize, sharex=True, sharey=True)
    # for idx, cond in enumerate(['gCNO', 'control']):
    #     sns.histplot(data=df_std_p.loc[cond], ax=axs[idx], binwidth=1)
    #     axs[idx].set_title(f'{cond}')

    #     # Axis label
    #     if idx == 1:
    #         axs[idx].set_xlabel('std')

    # fig.tight_layout()

    fig1, axs1 = plt.subplots(1, 2, figsize=figsize, sharey=True)
    for idx, cond in enumerate(['gCNO', 'control']):
        sns.pointplot(data=df_std_p.loc[cond], order=['pre', 'post'], errorbar='se', ax=axs1[idx], color='0.3')
        sns.swarmplot(data=df_std_p.loc[cond], order=['pre', 'post'], ax=axs1[idx], size=10)
        # Aesthetics
        axs1[idx].set_ylabel('std')
        axs1[idx].set_title(f'{cond}')


    fig2, axs2 = plt.subplots(figsize=figsize, sharey=True)
    for idx, cond in enumerate(['gCNO', 'control']):
        # sns.pointplot(data=df_std_p.loc[cond], order=['pre', 'post'], errorbar='se', ax=axs2[idx], color='0.3')
        sns.histplot(data=df_postdiff_mu, bins=100)
        axs2.hlines(y=260, xmin=hdi_mu[0], xmax=hdi_mu[1])
        # Aesthetics
        axs2.set_xlabel('mu difference in contrasts')

    fig3, axs3 = plt.subplots(figsize=figsize, sharey=True)
    for idx, cond in enumerate(['gCNO', 'control']):
        # sns.pointplot(data=df_std_p.loc[cond], order=['pre', 'post'], errorbar='se', ax=axs2[idx], color='0.3')
        sns.histplot(data=df_postdiff_sigma, bins=100)
        axs3.hlines(y=260, xmin=hdi_sigma[0], xmax=hdi_sigma[1])
        # Aesthetics
        axs3.set_xlabel('sigma difference in contrasts')

    # Save stdcompare plot
    if plot_stdcomp:
        plt.show()
    else:                       # save figure
        if os.path.basename(os.getcwd()) != 'data_analysis':
            os.chdir('./data_analysis')

        if not surr:
            surr = 'nosurr'
        try:
            if not peth_discard:
                fig1.savefig(os.getcwd() + f'/images/peth/{var}_{surr}_pethdiscardFalse_stdcompare_{windowlength}_pullcontrol_pointplot.svg', format='svg')
                fig2.savefig(os.getcwd() + f'/images/peth/{var}_{surr}_pethdiscardFalse_stdcompare_{windowlength}_pullcontrol_std_diff_contrast_mu.svg', format='svg')
                fig3.savefig(os.getcwd() + f'/images/peth/{var}_{surr}_pethdiscardFalse_stdcompare_{windowlength}_pullcontrol_std_diff_contrast_sigma.svg', format='svg')
                priorpc_plot.figure.savefig(os.getcwd() + f'/images/peth/priorpc_plot.svg', format='svg')
                postpc_plot.figure.savefig(os.getcwd() + f'/images/peth/postpc_plot.svg', format='svg')
                trace_plot.ravel()[0].figure.savefig(os.getcwd() + f'/images/peth/trace_plot.svg', format='svg')

            else:
                fig.savefig(os.getcwd() + f'/images/peth/{var}_{surr}_pethdiscardTrue_stdcompare_{windowlength}_pullcontrol.svg', format='svg')
                fig1.savefig(os.getcwd() + f'/images/peth/{var}_{surr}_pethdiscardTrue_stdcompare_{windowlength}_pullcontrol_pointplot.svg', format='svg')
                fig2.savefig(os.getcwd() + f'/images/peth/{var}_{surr}_pethdiscardFalse_stdcompare_{windowlength}_pullcontrol_std_diff_contrast_mu.svg', format='svg')
                fig3.savefig(os.getcwd() + f'/images/peth/{var}_{surr}_pethdiscardFalse_stdcompare_{windowlength}_pullcontrol_std_diff_contrast_sigma.svg', format='svg')
                priorpc_plot.figure.savefig(os.getcwd() + f'/images/peth/priorpc_plot.svg', format='svg')
                postpc_plot.figure.savefig(os.getcwd() + f'/images/peth/postpc_plot.svg', format='svg')
                trace_plot.ravel()[0].figure.savefig(os.getcwd() + f'/images/peth/trace_plot.svg', format='svg')
        except FileNotFoundError:
            print(f'created dir images/peth/{os.path.basename(wskpath[:-1])}')
            os.makedirs(os.getcwd() + f'/images/peth/{os.path.basename(wskpath[:-1])}')
            if not selclu:
                fig.savefig(os.getcwd() + f'/images/peth/{var}_{surr}_pethdiscardFalse_stdcompare_{windowlength}_pullcontrol.svg', format='svg')
                fig1.savefig(os.getcwd() + f'/images/peth/{var}_{surr}_pethdiscardFalse_stdcompare_{windowlength}_pullcontrol_pointplot.svg', format='svg')
                fig2.savefig(os.getcwd() + f'/images/peth/{var}_{surr}_pethdiscardFalse_stdcompare_{windowlength}_pullcontrol_std_diff_contrast_mu.svg', format='svg')
                fig3.savefig(os.getcwd() + f'/images/peth/{var}_{surr}_pethdiscardFalse_stdcompare_{windowlength}_pullcontrol_std_diff_contrast_sigma.svg', format='svg')
                priorpc_plot.figure.savefig(os.getcwd() + f'/images/peth/priorpc_plot.svg', format='svg')
                postpc_plot.figure.savefig(os.getcwd() + f'/images/peth/postpc_plot.svg', format='svg')
                trace_plot.ravel()[0].figure.savefig(os.getcwd() + f'/images/peth/trace_plot.svg', format='svg')
            else:
                fig.savefig(os.getcwd() + f'/images/peth/{var}_{surr}_pethdiscardTrue_stdcompare_{windowlength}_pullcontrol.svg', format='svg')
                fig1.savefig(os.getcwd() + f'/images/peth/{var}_{surr}_pethdiscardTrue_stdcompare_{windowlength}_pullcontrol_pointplot.svg', format='svg')
                fig2.savefig(os.getcwd() + f'/images/peth/{var}_{surr}_pethdiscardFalse_stdcompare_{windowlength}_pullcontrol_std_diff_contrast_mu.svg', format='svg')
                fig3.savefig(os.getcwd() + f'/images/peth/{var}_{surr}_pethdiscardFalse_stdcompare_{windowlength}_pullcontrol_std_diff_contrast_sigma.svg', format='svg')
                priorpc_plot.figure.savefig(os.getcwd() + f'/images/peth/priorpc_plot.svg', format='svg')
                postpc_plot.figure.savefig(os.getcwd() + f'/images/peth/postpc_plot.svg', format='svg')
                trace_plot.ravel()[0].figure.savefig(os.getcwd() + f'/images/peth/trace_plot.svg', format='svg')

        fig.clf()


def  time_compare(peak_pre, peak_post, conditions, windowlength, t_wind, plot_timecomp=False, surr=False):
    """ Compare the time of peth peaks: check if peaks occure at different times with respect
    to whisking onset pre-post in the two conditions (doesn't seem so; only std changes)
    """
    # pdb.set_trace()
    # Organise time in df, and set to seconds
    pre_data = pd.DataFrame(peak_pre, index=conditions).stack()
    post_data = pd.DataFrame(peak_post, index=conditions).stack()
    df_times = pd.concat([post_data, pre_data], keys=['post', 'pre'], axis=1)
    df_times = df_times / 299

    # Remap control conditions    
    df_times.rename(index={'gCNO':'gCNO', 'wCNO':'control', 'aPBS':'control'}, inplace=True)
    
    # Compute difference post-pre peak times
    df_tdiff = df_times.diff(periods=-1, axis=1).drop(columns='pre')
    df_tdiff.index.names = ['cond', 'cluster']
    df_tdiff.columns = ['post-pre (sec)']

    # Pull control conditions

    # Stats
    wilc_p = []
    for cond in np.unique(df_tdiff.index.get_level_values(0).unique()):
        wilc_p.append(wilcoxon(df_times.loc[(cond, slice(None)), 'pre'], df_times.loc[(cond, slice(None)), 'post']))
    ttest_p = []
    for cond in np.unique(df_tdiff.index.get_level_values(0).unique()):
        ttest_p.append(ttest_rel(df_times.loc[(cond, slice(None)), 'pre'], df_times.loc[(cond, slice(None)), 'post']))

    # Save stats
    stats = {'wilcoxon':wilc_p, 'ttest':ttest_p}
    with open(f'/home/yq18021/Documents/github_gcl/data_analysis/save_data/stats_pethtime_compare_{windowlength}.pickle', 'wb') as f:
        pickle.dump(stats, f)

    # Plot distribution #######################
    # Initialise figure
    lcolor = 'darkgoldenrod'
    style = 'italic'
    figsize = (14, 12)
    sns.set_context("poster", font_scale=1)
    sns.color_palette("deep")
    plt.rcParams['axes.facecolor'] = 'lavenderblush'
    plt.rcParams["figure.edgecolor"] = 'black'

    # Plot time differences
    fig, ax = plt.subplots(figsize=figsize)
    sns.boxplot(data=df_tdiff.reset_index(), y='post-pre (sec)', x='cond', ax=ax, palette='Set3')
    fig.suptitle('Post-pre peth time difference')

    fig1, ax1 = plt.subplots(2, figsize=figsize, sharex=True, sharey=True)
    for idx, cond in enumerate(['gCNO', 'control']):
        __ = df_times.loc[cond].melt(value_name='time (sec)')
        sns.boxplot(data=__, x='variable', y='time (sec)', order=['pre', 'post'], ax=ax1[idx])
        __ = ax1[idx].get_yticks()
        ax1[idx].set_yticks(__)
        ax1[idx].set_yticklabels(__ + t_wind[0])
    
    fig1.suptitle('Distribution pre and post peth times')

    # Save stdcompare plot
    if plot_stdcomp:
        plt.show()
    else:                       # save figure
        if os.path.basename(os.getcwd()) != 'data_analysis':
            os.chdir('./data_analysis')
        try:
            fig.savefig(os.getcwd() + f'/images/peth/{var}_peth_time_compare_{windowlength}.svg', format='svg')
            fig1.savefig(os.getcwd() + f'/images/peth/{var}_peth_times_prepost_{windowlength}.svg', format='svg')
        except FileNotFoundError:
            print(f'created dir images/peth/{os.path.basename(wskpath[:-1])}')
            os.makedirs(os.getcwd() + f'/images/peth/{os.path.basename(wskpath[:-1])}')
            fig.savefig(os.getcwd() + f'/images/peth/{var}_peth_time_compare_{windowlength}.svg', format='svg')
            fig1.savefig(os.getcwd() + f'/images/peth/{var}_peth_times_prepost_{windowlength}.svg', format='svg')

        fig.clf()


def compare_std_wskslope(windowlength):
    ''' Load data from std_compare function (above) and plot_wskvar from whisk_plot.py;
    plot std contrast against whisker slope contrasts
    ATTENTION: loading whisk_plot.py output; run that before 
    '''
    with open(f'/home/yq18021/Documents/github_gcl/data_analysis/save_data/std_data_{windowlength}.pickle', 'rb') as f:
        std_data = pickle.load(f)
    with open(f'/home/yq18021/Documents/github_gcl/data_analysis/save_data/diff_slope_whisker_linfit_p.pickle', 'rb') as f:
        wskslope_data = pickle.load(f)

    # Put together std and slope post-pre contrats; order of rec is the same and follows
    # order in meta_data.py on which same masks (mask1=wmask, mask2=drop_rec) are used.
    df_data = std_data['std_prepost_contrasts'].copy()
    df_data.rename(columns={'post':'std_contrasts'}, inplace=True)
    df_data['slope_contrasts'] = wskslope_data['coef_p'].values

    # Linear fit per condition
    x_ = {}
    y_ = {}
    linreg = {}
    score = {}
    for cond in ('gCNO', 'control'):
        data_cond = df_data[df_data['index'] == cond]
        y_[f'{cond}'] = data_cond['std_contrasts'].values
        x_[f'{cond}'] = data_cond['slope_contrasts'].values.reshape(-1, 1)
        linreg[f'{cond}'] = LinearRegression().fit(x_[f'{cond}'], y_[f'{cond}'])
        score[f'{cond}'] = linreg[f'{cond}'].score(x_[f'{cond}'], y_[f'{cond}'])

    # Plot std vs slopes
    figsize = (14, 12)
    sns.set_context("poster", font_scale=1)
    sns.color_palette("deep")
    plt.rcParams['axes.facecolor'] = 'lavenderblush'
    plt.rcParams["figure.edgecolor"] = 'black'

    fig, ax = plt.subplots(figsize=figsize)
    sns.scatterplot(data=df_data, x='slope_contrasts', y='std_contrasts', hue='index', ax=ax)
    # add fits
    for cond in ('gCNO', 'control'):
        xval = np.array([min(x_[f'{cond}'])[0], max(x_[f'{cond}'])[0]])
        yval = linreg[f'{cond}'].intercept_ + xval * linreg[f'{cond}'].coef_
        ax.plot(xval, yval)
        ax.text(xval[0], yval[0], np.round(score[f'{cond}'], 2))

    

    if os.path.basename(os.getcwd()) != 'data_analysis':
        os.chdir('./data_analysis')
        
    fig.savefig(os.getcwd() + f'/images/peth/stdContrast_vs_wskslopeContrast_{windowlength}.svg', format='svg')


def run_peth_compare(whisker='whisker2',
                     var='angle',
                     t_wind=[-2, 3],
                     subwind=[-0.5, 1],
                     drop_rec=[0, 10, 15, 16, 25],
                     cgs='good',
                     bmt=False,
                     surr=False,
                     discard=False,
                     selclu=False,
                     peth_discard=True,
                     plot_peth=True,
                     save_data=True,
                     plot_stdcomp=False,
                     plot_timecomp=False):
    """ Compute peth for each recording, both for pre- and post-drop time;
    for each calculate point within subwind where max occures.
    """
    # Load metadata, prepare second mask and final good list & get rec
    wskpath, npxpath, aid, cortex_depth, t_drop, clu_idx, pethdiscard_idx, bout_idxs, wmask, conditions, __ = load_meta()
    good_list = np.arange(len(wskpath))
    good_list = good_list[wmask]
    wmask2 = ~np.in1d(np.arange(len(good_list)), np.array(drop_rec))
    good_list = good_list[wmask2]
    conditions = conditions[good_list]


    spktr_data_pre = []
    df_wvar_pre = []
    peak_pre = []

    spktr_data_post = []
    df_wvar_post = []
    peak_post = []

    # Define windowlength
    if subwind == [-0.5, 1]:
            windowlength = 'SHORTWIND'
    elif subwind == [-0.7, 1.3]:
            windowlength = 'LONGWIND'

    n_clu = []
    # pdb.set_trace()
    if save_data:               # either compute and save data...
        for idx, rec_idx in enumerate(good_list):
            pre = True
            __ = peth(rec_idx,
                      whisker=whisker,
                      var=var,
                      t_wind=t_wind,
                      subwind=subwind,
                      cgs=cgs,
                      bmt=bmt,
                      pre=pre,
                      surr=surr,
                      discard=discard,
                      selclu=selclu,
                      peth_discard=peth_discard)

            # Save pre peth
            spktr_data_pre.append(__[0])
            df_wvar_pre.append(__[1])
            peak_pre.append(__[2])
            # n_clu_pre.append(__[3])

            pre = False
            __ = peth(rec_idx,
                      whisker=whisker,
                      var=var,
                      t_wind=t_wind,
                      subwind=subwind,
                      cgs=cgs,
                      bmt=bmt,
                      pre=pre,
                      surr=surr,
                      discard=discard,
                      selclu=selclu,
                      peth_discard=peth_discard)

            # Save post peth
            spktr_data_post.append(__[0])
            df_wvar_post.append(__[1])
            peak_post.append(__[2])
            n_clu.append(__[3])

            # Plot post peth (for single recordings)
            if plot_peth:
                helper_plot(df_wvar_pre[idx],
                            df_wvar_post[idx],
                            spktr_data_pre[idx],
                            spktr_data_post[idx],
                            peak_pre[idx],
                            peak_post[idx],
                            n_clu[idx],
                            wskpath[rec_idx],
                            windowlength,
                            surr=surr,
                            peth_discard=peth_discard)

        # pdb.set_trace()
        all_data = dict(spktr_data=[spktr_data_pre, spktr_data_post], df_wvar=[df_wvar_pre, df_wvar_post], peak=[peak_pre, peak_post], n_clu=n_clu)
        with open(f'/home/yq18021/Documents/github_gcl/data_analysis/save_data/peth_compare_{windowlength}.pickle', 'wb') as f:
            pickle.dump(all_data, f)

    else:                       # ... or load data
        with open(f'/home/yq18021/Documents/github_gcl/data_analysis/save_data/peth_compare_{windowlength}.pickle', 'rb') as f:
            all_data = pickle.load(f)
    # pdb.set_trace()

    # Compare standard deviation of peaks time across conditions
    std_compare(all_data['peak'][0], all_data['peak'][1], conditions, windowlength, plot_stdcomp=plot_stdcomp, surr=surr, save_data=save_data)

    # Compare timing of peaks across conditions
    time_compare(all_data['peak'][0], all_data['peak'][1], conditions, windowlength, t_wind, plot_timecomp=plot_timecomp, surr=surr)

    # Compare std peaks with whisker slopes from whisk_plot.py script
    compare_std_wskslope(windowlength)


if __name__ == '__main__':
    """ Run script if peth_compare.py module is main programme
    """
    drop_rec = [0, 10, 15, 16, 25]
    whisker = 'whisk2'
    var = 'angle'
    t_wind = [-2, 3]
    # subwind = [-0.5, 1]         # Short window
    subwind = [-0.7, 1.3]       # Long window
    cgs = 2
    bmt = False
    surr = False
    discard = False              # discard clu if fr<0.1Hz in all rec_idx
    selclu = False
    peth_discard = True          # discard clu if peth is blank
    plot_peth = True
    save_data = False
    plot_stdcomp = False
    plot_timecomp= False

    run_peth_compare(whisker=whisker,
                     var=var,
                     t_wind=t_wind,
                     subwind=subwind,
                     drop_rec=drop_rec,
                     cgs=cgs,
                     bmt=bmt,
                     surr=surr,
                     discard=discard,
                     selclu=selclu,
                     peth_discard=peth_discard,
                     plot_peth=plot_peth,
                     save_data=save_data)
