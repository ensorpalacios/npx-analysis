"""
Script to compute and collect properties of different clusters.
"""
import os
import sys
if not ('data_analysis' in os.getcwd()):
    sys.path.append(os.path.join(os.getcwd(), 'data_analysis'))
import itertools

import pdb
import align_fun
from meta_data import load_meta
from align_fun import align_spktrain, load_data, spike_train, spkcount_cortex, align_to_whisking, Data_container
from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd
import seaborn as sns
import pymc as pm
import aesara.tensor as at
import arviz as az
import matplotlib.pyplot as plt
import matplotlib
import pickle


def fr_count(spktr_array, iswhisking, tdrop, sr=299):
    """ Compute firing rate for each cluster during whisking and resting
    periods before and after drug application; original data have bin of
    len=frame (1/299 Hz), therefore every 299 bins = 1 sec length; use
    np.cumsum to sum each consecutive element, and ther retain each elements
    every 299 steps.
    - Attention: glue together separated recording periods!
    """
    time_drop = tdrop * 60 * sr  # in length of frames 
    iswhisking_mask = np.where(iswhisking==1, True, False)
    spktr_fr = pd.DataFrame()

    for iswhisk in ['whisk', 'nowhisk']:
        if iswhisk == 'whisk':
            mask_all = iswhisking_mask.copy()
        elif iswhisk == 'nowhisk':
            mask_all = (~iswhisking_mask).copy()
        for time in ['pre', 'post']:
            if time == 'pre':
                mask = mask_all.copy()
                mask[time_drop:] = False
            elif time == 'post':
                mask = mask_all.copy()
                mask[:time_drop] = False

            spktr_masked = spktr_array[:, mask]
            spktr_masked = np.cumsum(spktr_masked, axis=1)
            spktr_masked = spktr_masked[:, sr::sr] - spktr_masked[:, :-sr:sr]
            spktr_fr_masked = spktr_masked.mean(axis=1)

            # Save in df
            spktr_fr_masked = pd.DataFrame(spktr_fr_masked, columns=[f'{time}_{iswhisk}'])
            spktr_fr_masked.index.names = ['clu']
            spktr_fr_masked.columns.names = ['mask']
            spktr_fr = pd.concat([spktr_fr, spktr_fr_masked], axis=1)
    
    return spktr_fr

def fr_count_alltime(spktr_array, sr=299):
    """ Compute instantaneous fr for all recording (used later only for
    control data).
    """
    spktr_fr = np.cumsum(spktr_array, axis=1)
    spktr_fr = spktr_fr[:, sr::sr] - spktr_fr[:, :-sr:sr]
    spktr_fr = spktr_fr.mean(axis=1)
    spktr_fr = pd.DataFrame(spktr_fr)

    return spktr_fr


def clu_autocorr(spktr_array, window=200):
    """ Compute autocorrelation of spike counts for each cluster;
    window has units=1 frame (~3.3 msec); manually sec convolution=0
    for time lag=0.
    """
    df_autocorr = pd.DataFrame()
    for clu in range(spktr_array.shape[0]):
        __ = np.correlate(spktr_array[clu, :], spktr_array[clu, (window//2):-(window//2)])
        __[window//2] = 0
        __ = pd.DataFrame(__, index=np.arange(-window//2, window//2 + 1), columns=[clu]).T
        __.index.names = ['clu']
        __.columns.names = ['lag']
        df_autocorr = pd.concat([df_autocorr, __], axis=0)

    return df_autocorr

def fr_anal(fr_data, kmclusters_idx, saveplot=False):
    """ Compute mean relationship between pre and post firing rates
    """
    pdb.set_trace()
    idxsli = pd.IndexSlice
    # Add km cluster to fr data as index
    fr_data['kmclu'] = kmclusters_idx.values[:, 0]
    fr_data.set_index('kmclu', append=True, inplace=True)
    # Organise in df
    # col0: pre_whisk, col1: post_whisk, col2: pre_nowhisk, col3: post_nowhisk
    fr_data_pre = fr_data.loc[:, ('pre_whisk', 'pre_nowhisk')].stack().copy()
    fr_data_post = fr_data.loc[:, ('post_whisk', 'post_nowhisk')].stack().copy()
    df_data_stack = pd.concat([fr_data_pre, fr_data_post], keys=['pre','post'], names=['time'])

    # Prepare coordinates for pymc model
    newcoords = df_data_stack.index.get_level_values(5).values
    newcoords = np.array(tuple(map(lambda x: np.where('nowhisk' in x, 'nowhisk', 'whisk'), newcoords)))
    newcoords = np.char.add(newcoords, df_data_stack.index.get_level_values(1).values)
    
    df_data_stack = df_data_stack.reset_index('mask')
    df_data_stack['mask'] = newcoords
    df_data_stack = df_data_stack.set_index('mask', append=True)

    mask_idx, mask = pd.factorize(df_data_stack.loc[idxsli['pre']].index.get_level_values(4))
    coords = {'whisk_cond': mask.values.tolist()}
    
    with pm.Model(coords=coords) as lreg:
        # Priors
        # beta = pm.Normal('beta', mu=0, sigma=1, dims='whisk_cond')
        beta1 = pm.Normal('beta1', mu=0, sigma=1, dims='whisk_cond')
        sigma = pm.Exponential('sigma', lam=1, dims='whisk_cond')
        # Linear model
        # mu = pm.Deterministic('mu', beta[mask_idx] + beta1[mask_idx] * df_data_stack.loc[idxsli['pre']][0].values)
        mu = pm.Deterministic('mu', beta1[mask_idx] * df_data_stack.loc[idxsli['pre']][0].values)
        # Likelihood
        likelihood = pm.Normal('likelihood', mu=mu, sigma=sigma[mask_idx], observed=df_data_stack.loc[idxsli['post']][0])

    with lreg:
        prior_samples = pm.sample_prior_predictive()
        trace = pm.sample()
        post_samples = pm.sample_posterior_predictive(trace)

    # # Plot checks
    # az.plot_hdi(np.arange(11), post_samples.posterior_predictive.tc_val * sf, hdi_prob=0.68)
    # az.plot_ppc(prior_samples, group='prior')
    # az.plot_ppc(post_samples)
    # az.summary(trace, var_names=['~mu'])
    # az.plot_trace(trace, var_names=['~mu'], legend=True)

    # betas = trace.posterior['beta'].stack(z=('chain', 'draw'))
    betas1 = trace.posterior['beta1'].stack(z=('chain', 'draw'))
    sigmas = trace.posterior['sigma'].stack(z=('chain', 'draw'))
    trace_mean = trace.posterior.mean(dim=['chain', 'draw']).copy()

    # Max likelihood fit
    score = []
    coef = []
    intercept = []
    predict = []
    for cond in ['gCNO', 'control']:
        for whisking in range(2):
            __ = fr_data.loc[cond]
            x = __.iloc[:, 0 + (2 * whisking)].values.reshape(-1, 1)
            y = __.iloc[:, 1 + (2 * whisking)].values
            reg = LinearRegression(fit_intercept=False).fit(x, y)
            score.append(reg.score(x, y))
            coef.append(reg.coef_)
            intercept.append(reg.intercept_)
            predict.append(reg.predict(np.arange(200).reshape(-1, 1)))

    # Compute difference in fr
    fr_diff = df_data_stack.groupby(by=['cond', 'rec', 'clu', 'kmclu', 'mask']).diff().loc['post'].reset_index()
    fr_diff['whisking'] = ~(fr_diff['mask'].apply(lambda x: 'no' in x))

    pdb.set_trace()
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
    plt.rcParams['lines.markersize'] = 2

    # Plot
    fig, ax = plt.subplots(1, 2, figsize=(14, 12), sharex=True, sharey=True)
    for idx_cm, condmask in enumerate(coords['whisk_cond']):
        # Mean fit
        beta1_mean = betas1.sel(whisk_cond=condmask).values.mean()
        fit = np.arange(300) * beta1_mean
        # Fit with uncertainty
        beta1_ = betas1.sel(whisk_cond=condmask).values
        sigmas_ = sigmas.sel(whisk_cond=condmask).values
        mean = np.arange(300).reshape(-1, 1) * beta1_
        samp_fit = np.random.normal(loc=mean, scale=sigmas_)
        # Plot
        if idx_cm % 2 == 0:
            idx_wsk = 0
        else:
            idx_wsk = 1
        if idx_cm < 2:
            ax[idx_wsk].text(130, 10, '{}={:0.3}\n'.format(condmask, beta1_mean))
            loc_text = 0
            c = 'skyblue'
        else:
            ax[idx_wsk].text(130, 30, '{}={:0.3}\n'.format(condmask, beta1_mean))
            c = 'orange'
        az.plot_hdi(np.arange(300), np.expand_dims(samp_fit.T, 0), ax=ax[idx_wsk], color=c)
        ax[idx_wsk].scatter(df_data_stack.loc[idxsli['pre', :, :, :, :, condmask]], df_data_stack.loc[idxsli['post', :, :, :, :, condmask]])
        # ax[idx_wsk].plot(np.arange(300), fit, c=c)
        ax[idx_wsk].plot(np.arange(300), fit, label=condmask)
        # ax[idx_wsk].plot(np.arange(300), predict[idx_cm], c='orange')
        # Aesthetics
        ax[idx_wsk].set_title(f'{condmask}')
        ax[idx_wsk].set_xlabel('pre fr')
        ax[idx_wsk].set_ylabel('post fr')
        ax[idx_wsk].set_ylim(bottom=0)
        ax[idx_wsk].set_aspect('equal')
        if idx_wsk == 1:
            ax[idx_wsk].legend(loc='upper left')

    # fig, ax = plt.subplots(8, 2, sharex=True, sharey=True, figsize=figsize)
    fig1 = []
    for idx_wm, whiskmask in enumerate([True, False]):
        fig1_, ax1_ = plt.subplots(2, 4, sharex=True, sharey=True, figsize=figsize)
        for idx_kmclu in np.unique(kmclusters_idx.values):
            axes = ax1_.flatten()
            __ = df_data_stack.unstack(0).droplevel(0, axis=1)
            __ = __.reset_index()
            __['whisking'] = ~(__['mask'].apply(lambda x: 'no' in x))
            __ = __[(__['kmclu'] == idx_kmclu) & (__['whisking'] == whiskmask)]
            sns.scatterplot(data=__, x='pre', y='post', hue='cond', hue_order=['gCNO', 'control'], ax=axes[idx_kmclu], s=20)
            # ax[idx_kmclu, idx_wm].set_aspect('equal')
            axes[idx_kmclu].set_xlim([0, 250])
            axes[idx_kmclu].set_ylim([0, 250])
            # Fit line and add to plot
            x_gCNO = __[__['cond']=='gCNO']['pre'].values.reshape(-1, 1)
            x_control = __[__['cond']=='control']['pre'].values.reshape(-1, 1)
            y_gCNO = __[__['cond']=='gCNO']['post'].values.reshape(-1, 1)
            y_control = __[__['cond']=='control']['post'].values.reshape(-1, 1)
            reg_gCNO = LinearRegression(fit_intercept=False).fit(x_gCNO, y_gCNO)
            reg_control = LinearRegression(fit_intercept=False).fit(x_control, y_control)
            predict_gCNO = reg_gCNO.predict(x_gCNO)
            predict_control = reg_control.predict(x_control)
            axes[idx_kmclu].plot(x_gCNO, predict_gCNO, color='blue')
            axes[idx_kmclu].plot(x_control, predict_control, color='orange')
            axes[idx_kmclu].plot([0, 250], [0, 250], ls='--', c='black', ms=5)
            
            # Aesthetics
            if idx_kmclu < 7:
                axes[idx_kmclu].get_legend().remove()
            # axes[idx_kmclu].set_aspect('equal')
            axes[idx_kmclu].set_title(f'clu{idx_kmclu}')
        fig1_.suptitle(f'whisking{whiskmask}')
        fig1.append(fig1_)


    # Save figure
    pdb.set_trace()
    if saveplot:
        if os.path.basename(os.getcwd()) != 'data_analysis':
            os.chdir('./data_analysis')
        try:
            fig.savefig(os.getcwd() + '/images/clu_properties/compare_fr.svg', format='svg')
            fig1[0].savefig(os.getcwd() + '/images/clu_properties/compare_fr_kmclu_wsk.svg', format='svg')
            fig1[1].savefig(os.getcwd() + '/images/clu_properties/compare_fr_kmclu_nowsk.svg', format='svg')
        except FileNotFoundError:
            print('created dir /images/clu_properties')
            os.makedirs(os.getcwd() + '/images/clu_properties')
            fig.savefig(os.getcwd() + '/images/clu_properties/compare_fr.svg', format='svg')
            fig1[0].savefig(os.getcwd() + '/images/clu_properties/compare_fr_kmclu_wsk.svg', format='svg')
            fig1[1].savefig(os.getcwd() + '/images/clu_properties/compare_fr_kmclu_nowsk.svg', format='svg')
    else:
        plt.show()   

    # gcolors = {0:'blue', 1:'red', 2:'green', 3:'violet', 4:'yellow', 5:'pink', 6:'brown', 7:'black', 8:'cyan'}

    # fig, ax = plt.subplots(2, 2, figsize=(14, 12), sharex=True, sharey=True)
    # for idx_cm, condmask in enumerate(coords['whisk_cond']):
    #     ax = ax.flatten()
    #     colors = df_data_stack.loc[idxsli['pre', :, :, :, :, condmask]].index.get_level_values(3).map(gcolors)
    #     ax[idx_cm].scatter(df_data_stack.loc[idxsli['pre', :, :, :, :, condmask]], df_data_stack.loc[idxsli['post', :, :, :, :, condmask]], c=colors)
    #     # ax[idx_wsk].plot(np.arange(200), fit, c=c)
    #     ax[idx_wsk].plot(np.arange(200), fit, label=condmask)
    #     # ax[idx_wsk].plot(np.arange(200), predict[idx_cm], c='orange')
    #     # Aesthetics
    #     ax[idx_wsk].set_title(f'{condmask}')
    #     ax[idx_wsk].set_xlabel('pre fr')
    #     ax[idx_wsk].set_ylabel('post fr')
    #     ax[idx_wsk].set_ylim(bottom=0)
    #     ax[idx_wsk].set_aspect('equal')
    #     if idx_wsk == 1:
    #         ax[idx_wsk].legend(loc='upper left')    


    fr_diff['whisking'] = ~(fr_diff['mask'].apply(lambda x: 'no' in x))
    # fig, ax = plt.subplots(8, 2, sharex=True, sharey=True, figsize=figsize)
    for idx_wm, whiskmask in enumerate([True, False]):
        fig, ax = plt.subplots(2, 4, sharex=True, sharey=True, figsize=figsize)
        for idx_kmclu in np.unique(kmclusters_idx.values):
            axes = ax.flatten()
            __ = df_data_stack.unstack(0).droplevel(0, axis=1)
            __ = __.reset_index()
            __['whisking'] = ~(__['mask'].apply(lambda x: 'no' in x))
            __ = __[(__['kmclu'] == idx_kmclu) & (__['whisking'] == whiskmask)]
            sns.scatterplot(data=__, x='pre', y='post', hue='cond', hue_order=['gCNO', 'control'], ax=axes[idx_kmclu], s=20)
            # ax[idx_kmclu, idx_wm].set_aspect('equal')
            axes[idx_kmclu].set_xlim([0, 250])
            axes[idx_kmclu].set_ylim([0, 250])
            # Fit line and add to plot
            x_gCNO = __[__['cond']=='gCNO']['pre'].values.reshape(-1, 1)
            x_control = __[__['cond']=='control']['pre'].values.reshape(-1, 1)
            y_gCNO = __[__['cond']=='gCNO']['post'].values.reshape(-1, 1)
            y_control = __[__['cond']=='control']['post'].values.reshape(-1, 1)
            reg_gCNO = LinearRegression(fit_intercept=False).fit(x_gCNO, y_gCNO)
            reg_control = LinearRegression(fit_intercept=False).fit(x_control, y_control)
            predict_gCNO = reg_gCNO.predict(x_gCNO)
            predict_control = reg_control.predict(x_control)
            axes[idx_kmclu].plot(x_gCNO, predict_gCNO, color='blue')
            axes[idx_kmclu].plot(x_control, predict_control, color='orange')
            axes[idx_kmclu].plot([0, 250], [0, 250], ls='--', c='black', ms=5)
            
            # Aesthetics
            if idx_kmclu < 7:
                axes[idx_kmclu].get_legend().remove()
            # axes[idx_kmclu].set_aspect('equal')
            axes[idx_kmclu].set_title(f'clu{idx_kmclu}')
        fig.suptitle(f'whisking{whiskmask}')
            

            
    #         # gCNO_data = df_data_stack.loc[idxsli['pre', :, :, :, idx_kmclu, condmask]]
    #         # post_data = df_data_stack.loc[idxsli['post', :, :, :, idx_kmclu, condmask]]
    #         # ax[idx_kmclu, idx_cm].scatter(pre_data, pre_data)
    

    # fig, ax = plt.subplots(2, 2, figsize=(14, 12), sharex=True, sharey=True)
    # for idx_cm, condmask in enumerate([True, False, 'gCNO', 'control']):
    #     if idx_cm < 2:
    #         # Data
    #         __ = fr_diff[fr_diff.whisking==condmask]
    #         # Plot
    #         sns.histplot(data=__, x=0, hue='cond', ax=ax.flatten()[idx_cm])
    #         ax.flatten()[idx_cm].set_title(f'whisking {condmask}')
    #     if idx_cm >= 2:
    #         # Data
    #         __ = fr_diff[fr_diff.cond==condmask]
    #         # Plot
    #         sns.histplot(data=__, x=0, hue='whisking', ax=ax.flatten()[idx_cm])
    #         ax.flatten()[idx_cm].set_title(f'condition {condmask}')
    #     # Adornments
    #     ax.flatten()[idx_cm].set_xlabel('post-pre fr')


def fr_anal_oldcontr(fr_data, kmclusters_idx, wCNO_idx, aPBS_idx, saveplot=False):
    """ Compute mean relationship between pre and post firing rates using original
    controls: i.e., wCNO and aPBS instead of control.
    """
    pdb.set_trace()
    idxsli = pd.IndexSlice

    # Reset control to old
    oldmap = {}
    fr_data = fr_data.reset_index()
    fr_data['rec'] = fr_data['rec'].astype(int)
    fr_data['cond'][fr_data['rec'].isin(wCNO_idx)] = 'wCNO'
    fr_data['cond'][fr_data['rec'].isin(aPBS_idx)] = 'aPBS'
    fr_data = fr_data.set_index(['cond', 'rec', 'clu'])

    # Add km cluster to fr data as index
    fr_data['kmclu'] = kmclusters_idx.values[:, 0]
    fr_data.set_index('kmclu', append=True, inplace=True)
    # Organise in df
    # col0: pre_whisk, col1: post_whisk, col2: pre_nowhisk, col3: post_nowhisk
    fr_data_pre = fr_data.loc[:, ('pre_whisk', 'pre_nowhisk')].stack().copy()
    fr_data_post = fr_data.loc[:, ('post_whisk', 'post_nowhisk')].stack().copy()
    df_data_stack = pd.concat([fr_data_pre, fr_data_post], keys=['pre','post'], names=['time'])

    # Prepare coordinates for pymc model
    newcoords = df_data_stack.index.get_level_values(5).values
    newcoords = np.array(tuple(map(lambda x: np.where('nowhisk' in x, 'nowhisk', 'whisk'), newcoords)))
    newcoords = np.char.add(newcoords, df_data_stack.index.get_level_values(1).values)
    
    df_data_stack = df_data_stack.reset_index('mask')
    df_data_stack['mask'] = newcoords
    df_data_stack = df_data_stack.set_index('mask', append=True)

    mask_idx, mask = pd.factorize(df_data_stack.loc[idxsli['pre']].index.get_level_values(4))
    coords = {'whisk_cond': mask.values.tolist()}
    
    with pm.Model(coords=coords) as lreg:
        # Priors
        # beta = pm.Normal('beta', mu=0, sigma=1, dims='whisk_cond')
        beta1 = pm.Normal('beta1', mu=0, sigma=1, dims='whisk_cond')
        sigma = pm.Exponential('sigma', lam=1, dims='whisk_cond')
        # Linear model
        # mu = pm.Deterministic('mu', beta[mask_idx] + beta1[mask_idx] * df_data_stack.loc[idxsli['pre']][0].values)
        mu = pm.Deterministic('mu', beta1[mask_idx] * df_data_stack.loc[idxsli['pre']][0].values)
        # Likelihood
        likelihood = pm.Normal('likelihood', mu=mu, sigma=sigma[mask_idx], observed=df_data_stack.loc[idxsli['post']][0])

    with lreg:
        prior_samples = pm.sample_prior_predictive()
        trace = pm.sample()
        post_samples = pm.sample_posterior_predictive(trace)

    # # Plot checks
    # az.plot_hdi(np.arange(11), post_samples.posterior_predictive.tc_val * sf, hdi_prob=0.68)
    # az.plot_ppc(prior_samples, group='prior')
    # az.plot_ppc(post_samples)
    # az.summary(trace, var_names=['~mu'])
    # az.plot_trace(trace, var_names=['~mu'], legend=True)

    # betas = trace.posterior['beta'].stack(z=('chain', 'draw'))
    betas1 = trace.posterior['beta1'].stack(z=('chain', 'draw'))
    sigmas = trace.posterior['sigma'].stack(z=('chain', 'draw'))
    trace_mean = trace.posterior.mean(dim=['chain', 'draw']).copy()

    # Max likelihood fit
    score = []
    coef = []
    intercept = []
    predict = []
    for cond in ['gCNO', 'control']:
        for whisking in range(2):
            __ = fr_data.loc[cond]
            x = __.iloc[:, 0 + (2 * whisking)].values.reshape(-1, 1)
            y = __.iloc[:, 1 + (2 * whisking)].values
            reg = LinearRegression(fit_intercept=False).fit(x, y)
            score.append(reg.score(x, y))
            coef.append(reg.coef_)
            intercept.append(reg.intercept_)
            predict.append(reg.predict(np.arange(200).reshape(-1, 1)))

    # Compute difference in fr
    fr_diff = df_data_stack.groupby(by=['cond', 'rec', 'clu', 'kmclu', 'mask']).diff().loc['post'].reset_index()
    fr_diff['whisking'] = ~(fr_diff['mask'].apply(lambda x: 'no' in x))

    pdb.set_trace()
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
    plt.rcParams['lines.markersize'] = 2

    # Plot
    fig, ax = plt.subplots(1, 2, figsize=(14, 12), sharex=True, sharey=True)
    for idx_cm, condmask in enumerate(coords['whisk_cond']):
        # Mean fit
        beta1_mean = betas1.sel(whisk_cond=condmask).values.mean()
        fit = np.arange(300) * beta1_mean
        # Fit with uncertainty
        beta1_ = betas1.sel(whisk_cond=condmask).values
        sigmas_ = sigmas.sel(whisk_cond=condmask).values
        mean = np.arange(300).reshape(-1, 1) * beta1_
        samp_fit = np.random.normal(loc=mean, scale=sigmas_)
        # Plot
        if idx_cm % 2 == 0:
            idx_wsk = 0
        else:
            idx_wsk = 1
        if idx_cm < 2:
            ax[idx_wsk].text(130, 10, '{}={:0.3}\n'.format(condmask, beta1_mean))
            c = 'skyblue'
        elif idx_cm < 4:
            ax[idx_wsk].text(130, 30, '{}={:0.3}\n'.format(condmask, beta1_mean))
            c = 'orange'
        else:
            ax[idx_wsk].text(130, 50, '{}={:0.3}\n'.format(condmask, beta1_mean))
            c = 'green'
        az.plot_hdi(np.arange(300), np.expand_dims(samp_fit.T, 0), ax=ax[idx_wsk], color=c)
        ax[idx_wsk].scatter(df_data_stack.loc[idxsli['pre', :, :, :, :, condmask]], df_data_stack.loc[idxsli['post', :, :, :, :, condmask]])
        # ax[idx_wsk].plot(np.arange(200), fit, c=c)
        ax[idx_wsk].plot(np.arange(300), fit, label=condmask)
        # ax[idx_wsk].plot(np.arange(200), predict[idx_cm], c='orange')
        # Aesthetics
        ax[idx_wsk].set_title(f'{condmask}')
        ax[idx_wsk].set_xlabel('pre fr')
        ax[idx_wsk].set_ylabel('post fr')
        ax[idx_wsk].set_ylim(bottom=0)
        ax[idx_wsk].set_aspect('equal')
        if idx_wsk == 1:
            ax[idx_wsk].legend(loc='upper left')

    # Save figure
    pdb.set_trace()
    if saveplot:
        if os.path.basename(os.getcwd()) != 'data_analysis':
            os.chdir('./data_analysis')
        try:
            fig.savefig(os.getcwd() + '/images/clu_properties/compare_fr_cont_old.svg', format='svg')
        except FileNotFoundError:
            print('created dir /images/clu_properties')
            os.makedirs(os.getcwd() + '/images/clu_properties')
            fig.savefig(os.getcwd() + '/images/clu_properties/compare_fr_cont_old.svg', format='svg')
    else:
        plt.show()


def frpos_anal(all_data, kmclusters_idx, saveplot=False):
    """ Plot firing rate (all rec, control data) vs position
    """
    pdb.set_trace()
    fr_data = all_data['fr_alltime'].loc['control']
    fr_data.columns = ['fr']
    depth_data = all_data['depth'].loc['control']
    depth_data = depth_data.groupby(by='rec', group_keys=False).apply(lambda x: -(x - x.max()))
    depth_data.columns = ['depth']
    kmclusters_idx = kmclusters_idx.loc['control']

    # Plot attributes
    lcolor = 'darkgoldenrod'
    style = 'italic'
    figsize = (14, 12)          # for single unit tc plot! 
    sns.set_context("poster", font_scale=1)
    # sns.color_palette("deep")
    # sns.color_palette("flare")
    plt.rcParams['axes.facecolor'] = 'lavenderblush'
    plt.rcParams["figure.edgecolor"] = 'black'
    plt.rcParams['axes.linewidth'] = matplotlib.rcParamsDefault['axes.linewidth']
    plt.rcParams['lines.markersize'] = 2
    gcolors = {0:'blue', 1:'red', 2:'green', 3:'violet', 4:'yellow', 5:'pink', 6:'brown', 7:'black', 8:'cyan'}

    # Plot fr vs depth
    __ = fr_data.copy()
    __ = pd.concat([__, depth_data], axis=1)
    fig, ax = plt.subplots(figsize=figsize)
    sns.scatterplot(data=__, x='fr', y='depth', s=50, ax=ax)
    ax.invert_yaxis()

    # With fr vs depth (+ kmclusters)
    fig1, ax1 = plt.subplots(figsize=figsize)
    sns.scatterplot(data=__, x='fr', y='depth', c=kmclusters_idx[0].map(gcolors), s=50, ax=ax1)
    ax1.legend(gcolors)
    ax1.invert_yaxis()

    # Save figure
    if saveplot:
        if os.path.basename(os.getcwd()) != 'data_analysis':
            os.chdir('./data_analysis')
        try:
            fig.savefig(os.getcwd() + '/images/clu_properties/fr_vs_depth.svg', format='svg')
            fig1.savefig(os.getcwd() + '/images/clu_properties/fr_vs_depth_kmclu.svg', format='svg')
        except FileNotFoundError:
            print('created dir /images/clu_properties')
            os.makedirs(os.getcwd() + '/images/clu_properties')
            fig.savefig(os.getcwd() + '/images/clu_properties/fr_vs_depth.svg', format='svg')
            fig1.savefig(os.getcwd() + '/images/clu_properties/fr_vs_depth_kmclu.svg', format='svg')
    else:
        plt.show()   

    


def mli_anal(mli_data, all_data, kmclu_data, saveplot=False):
    """ Inspect properties of mli
    """
    idxsli = pd.IndexSlice
    pdb.set_trace()
    # Collect k-mean clusters and autocorrelogram for mli
    kmclu_mli = []
    ac_data = []
    for idx_recd, rec_data in enumerate(mli_data):
        if rec_data != ['no_clusters']:
            all_data['cids'].loc[idxsli[:, f'{idx_recd + 13}', :]]
            mask_mli = all_data['cids'].loc[idxsli[:, f'{idx_recd + 13}', :]].isin(rec_data.cids_sorted)
            kmclu_mli.append(kmclu_data.loc[idxsli[:, idx_recd + 13, :]][mask_mli[0].values])
            ac_data.append(all_data['autocorr'].loc[idxsli[:, f'{idx_recd + 13}', :]][mask_mli[0].values])
    kmclu_mli = pd.concat(kmclu_mli)
    ac_data = pd.concat(ac_data)
    ac_data_norm = ac_data.sub(ac_data.mean(axis=1), axis=0).div(ac_data.std(axis=1), axis=0)

    # Plot attributes
    lcolor = 'darkgoldenrod'
    style = 'italic'
    figsize = (14, 12)          # for single unit tc plot! 
    sns.set_context("poster", font_scale=1)
    # sns.color_palette("deep")
    # sns.color_palette("flare")
    plt.rcParams['axes.facecolor'] = 'lavenderblush'
    plt.rcParams["figure.edgecolor"] = 'black'
    plt.rcParams['axes.linewidth'] = matplotlib.rcParamsDefault['axes.linewidth']
    plt.rcParams['lines.markersize'] = 2

    # Plot
    # Plot km-clustering mli
    fig, ax= plt.subplots(figsize=figsize)
    sns.histplot(data=kmclu_mli, discrete=True, ax=ax)
    fig.suptitle('mli km-clusters')

    # Plot average autocorrelogram (from z-scored individual ac's)
    fig1, ax1 = plt.subplots(figsize=figsize)
    sns.lineplot(data=ac_data_norm.melt(), x='lag', y='value', errorbar='sd', ax=ax1)
    fig1.suptitle('mli average autocorrelogram (from z-scored ac\' s)')

    # Save figure
    if saveplot:
        if os.path.basename(os.getcwd()) != 'data_analysis':
            os.chdir('./data_analysis')
        try:
            fig.savefig(os.getcwd() + '/images/clu_properties/mli_kmsclu.svg', format='svg')
            fig1.savefig(os.getcwd() + '/images/clu_properties/mli_ac.svg', format='svg')
        except FileNotFoundError:
            print('created dir /images/clu_properties')
            os.makedirs(os.getcwd() + '/images/clu_properties')
            fig.savefig(os.getcwd() + '/images/clu_properties/mli_kmsclu.svg', format='svg')
            fig1.savefig(os.getcwd() + '/images/clu_properties/mli_ac.svg', format='svg')
    else:
        plt.show()


def km_autocorr_anal(autocorr, kmclu_data, saveplot=False):
    """ Plot average autocorrelations for units within km clusters; restrict to
    control (using whole trace data).
    """
    pdb.set_trace()
    idxsli = pd.IndexSlice    
    acorr_data = autocorr.loc['control'].copy()
    acorr_data['kmclu'] = kmclu_data.loc[idxsli['control', :, :,]].values
    acorr_data = acorr_data.set_index('kmclu', append=True)
    acorr_data_norm = []
    for kmclu in range(8):
        __ = acorr_data.loc[idxsli[:, :, kmclu]].copy()
        __ = __.sub(__.mean(axis=1), axis=0).div(__.std(axis=1), axis=0)
        __ = pd.concat([__], keys=[f'{kmclu}'], names=['kmclu'])
        acorr_data_norm.append(__)
    acorr_data_norm = pd.concat(acorr_data_norm)
    acorr_data_norm = acorr_data_norm.reset_index().melt(id_vars=['kmclu', 'rec', 'clu'])

    # Plot attributes
    lcolor = 'darkgoldenrod'
    style = 'italic'
    figsize = (14, 12)          # for single unit tc plot! 
    sns.set_context("poster", font_scale=1)
    # sns.color_palette("deep")
    # sns.color_palette("flare")
    plt.rcParams['axes.facecolor'] = 'lavenderblush'
    plt.rcParams["figure.edgecolor"] = 'black'
    plt.rcParams['axes.linewidth'] = matplotlib.rcParamsDefault['axes.linewidth']
    plt.rcParams['lines.markersize'] = 2

    fig, ax = plt.subplots(2, 4, sharey=True, sharex=True, figsize=figsize)
    axis = ax.flatten()
    for kmclu in range(8):
        __ = acorr_data_norm[acorr_data_norm.kmclu==f'{kmclu}']
        sns.lineplot(data=__, x='lag', y='value', errorbar='sd', ax=axis[kmclu])

    fig.suptitle('average autocorrelogram per km groups')

    # Save figure
    if saveplot:
        if os.path.basename(os.getcwd()) != 'data_analysis':
            os.chdir('./data_analysis')
        try:
            fig.savefig(os.getcwd() + '/images/clu_properties/autocorr_kmgrouped.svg', format='svg')
        except FileNotFoundError:
            print('created dir /images/clu_properties')
            os.makedirs(os.getcwd() + '/images/clu_properties')
            fig.savefig(os.getcwd() + '/images/clu_properties/autocorr_kmgrouped.svg', format='svg')
    else:
        plt.show()


def run_clu_properties(cgs=2,
                       var='angle',
                       whisker='whisk2',
                       surr=False,
                       discard=False,
                       pethdiscard=True,
                       window_ac=200,
                       save_data=False,
                       save_plot=False):
    """ Function to collect properties of all good clusters
    """
    # Select control datasets
    drop_rec = [0, 10, 15, 16, 25]
    wskpath, npxpath, aid, cortex_depth, t_drop, clu_idx, pethdiscard_idx, bout_idxs, wmask, conditions, mli_depth = load_meta()
    good_list = np.arange(len(wskpath))
    good_list = good_list[wmask]
    wmask2 = ~np.in1d(np.arange(len(good_list)), np.array(drop_rec))
    good_list = good_list[wmask2]

    # Length bin (in ms)
    frm_len = 1 / 299 * 1000  # frame length in ms
    pdb.set_trace()

    # Save all data in df
    if save_data:
        fr_alltime_all = pd.DataFrame()
        fr_prepost_whisk_all = pd.DataFrame()
        df_autocorr_all = pd.DataFrame()
        cids_sorted_all = pd.DataFrame()
        depth_sorted_all = pd.DataFrame()

        for idx, rec_idx in enumerate(good_list):
            whiskd, spk = load_data(wskpath[rec_idx], npxpath[rec_idx], t_drop=False)

            # Get spikes for all clusters
            # spktr_sorted bin=frm_len
            # spktr_sorted_msec bin=1msec
            spktr_sorted, endt, cids_sorted, _, spktr_sorted_msec = spike_train(spk,
                                                                                whiskd,
                                                                                binl=frm_len,
                                                                                surr=surr,
                                                                                cgs=cgs,
                                                                                msec=True)
            __ = cids_sorted.copy()
            # Retain only clusters in ml (~200μm depth??
            # Retain only clusters in cortex
            spktr_sorted, cids_sorted, depth_sorted = spkcount_cortex(spktr_sorted,
                                                                      spk.sortedDepth,
                                                                      cids_sorted,
                                                                      cortex_depth[rec_idx],
                                                                      binl=frm_len,
                                                                      discard=discard)
            # Same for spk train ins msec
            spktr_sorted_msec, __, __ = spkcount_cortex(spktr_sorted_msec,
                                                        spk.sortedDepth,
                                                        __,
                                                        cortex_depth[rec_idx],
                                                        binl=frm_len,
                                                        discard=discard)
            if pethdiscard:
                spktr_sorted = np.delete(spktr_sorted, pethdiscard_idx[rec_idx], axis=0)
                spktr_sorted_msec = np.delete(spktr_sorted_msec, pethdiscard_idx[rec_idx], axis=0)
                cids_sorted = np.delete(cids_sorted, pethdiscard_idx[rec_idx])
                depth_sorted = np.delete(depth_sorted, pethdiscard_idx[rec_idx])

            # Compute fr for all recording
            fr_alltime = fr_count_alltime(spktr_sorted,
                                          sr=299)

            # Compute fr sorted by whisking/no-whisking periods, pre/post drug application
            fr_prepost_whisk = fr_count(spktr_sorted,
                                        whiskd.iswhisking[int(whisker[-1])],
                                        t_drop[rec_idx],
                                        sr=299)


            df_autocorr = clu_autocorr(spktr_sorted_msec, window_ac)

            # Save in comprehensive df
            pdb.set_trace()
            fr_alltime = pd.concat([fr_alltime], keys=[f'{idx}'], names=['rec'], axis=0)
            fr_alltime = pd.concat([fr_alltime], keys=[f'{conditions[rec_idx]}'], names=['cond'], axis=0)
            fr_prepost_whisk = pd.concat([fr_prepost_whisk], keys=[f'{idx}'], names=['rec'], axis=0)
            fr_prepost_whisk = pd.concat([fr_prepost_whisk], keys=[f'{conditions[rec_idx]}'], names=['cond'], axis=0)
            df_autocorr = pd.concat([df_autocorr], keys=[f'{idx}'], names=['rec'], axis=0)
            df_autocorr = pd.concat([df_autocorr], keys=[f'{conditions[rec_idx]}'], names=['cond'], axis=0)
            cids_sorted = pd.DataFrame(cids_sorted)
            cids_sorted.index.names = ['clu']
            cids_sorted.columns.names = ['clu_id']
            cids_sorted = pd.concat([cids_sorted], keys=[f'{idx}'], names=['rec'], axis=0)
            cids_sorted = pd.concat([cids_sorted], keys=[f'{conditions[rec_idx]}'], names=['cond'], axis=0)
            depth_sorted = pd.DataFrame(depth_sorted)
            depth_sorted.index.names = ['clu']
            depth_sorted.columns.names = ['clu_id']
            depth_sorted = pd.concat([depth_sorted], keys=[f'{idx}'], names=['rec'], axis=0)
            depth_sorted = pd.concat([depth_sorted], keys=[f'{conditions[rec_idx]}'], names=['cond'], axis=0)
            fr_alltime_all = pd.concat([fr_alltime_all, fr_alltime], axis=0)
            fr_prepost_whisk_all = pd.concat([fr_prepost_whisk_all, fr_prepost_whisk], axis=0)
            df_autocorr_all = pd.concat([df_autocorr_all, df_autocorr], axis=0)
            cids_sorted_all = pd.concat([cids_sorted_all, cids_sorted], axis=0)
            depth_sorted_all = pd.concat([depth_sorted_all, depth_sorted], axis=0)

        pdb.set_trace()
        # Re-label control
        fr_alltime_all = fr_alltime_all.rename({'aPBS':'control', 'wCNO':'control'})
        fr_prepost_whisk_all = fr_prepost_whisk_all.rename({'aPBS':'control', 'wCNO':'control'})
        df_autocorr_all = df_autocorr_all.rename({'aPBS':'control', 'wCNO':'control'})
        cids_sorted_all = cids_sorted_all.rename({'aPBS':'control', 'wCNO':'control'})
        depth_sorted_all = depth_sorted_all.rename({'aPBS':'control', 'wCNO':'control'})
        
        clu_properties_data = {'fr_alltime': fr_alltime_all, 'fr': fr_prepost_whisk_all, 'autocorr': df_autocorr_all, 'cids': cids_sorted_all, 'depth': depth_sorted_all}

        # Save data
        with open('/home/yq18021/Documents/github_gcl/data_analysis/save_data/clu_properties_data.pickle', 'wb') as f:
            pickle.dump(clu_properties_data, f)
        

    else:                       # load data
        with open('/home/yq18021/Documents/github_gcl/data_analysis/save_data/clu_properties_data.pickle', 'rb') as f:
            clu_properties_data = pickle.load(f)

    pdb.set_trace()
    # Load kcluster data
    with open('/home/yq18021/Documents/github_gcl/data_analysis/save_data/km_clustered_data_pre.pickle', 'rb') as f:
        kmclusters = pickle.load(f)
    # Load MLI data (control only)
    with open('/home/yq18021/Documents/github_gcl/data_analysis/save_data/mli_data.pickle', 'rb') as f:
        all_data_mli = pickle.load(f)

    pdb.set_trace()

    idxsli = pd.IndexSlice
    kmclusters6 = kmclusters.loc[idxsli['6', :, :, :, :]].copy()
    kmclusters8 = kmclusters.loc[idxsli['8', :, :, :, :]].copy()

    mask_wCNO = conditions[good_list] == 'wCNO'
    mask_aPBS = conditions[good_list] == 'aPBS'
    idx_wCNO = np.arange(len(good_list))[mask_wCNO]
    idx_aPBS = np.arange(len(good_list))[mask_aPBS]

    # Firing rate analisy (+ plot and save figures)
    fr_anal_oldcontr(clu_properties_data['fr'], kmclusters8, idx_wCNO, idx_aPBS, saveplot=save_plot)
    pdb.set_trace()

    # Firing rate analisy (+ plot and save figures)
    fr_anal(clu_properties_data['fr'], kmclusters8, saveplot=save_plot)
    pdb.set_trace()

    # Km clusters of autocorrelation
    km_autocorr_anal(clu_properties_data['autocorr'], kmclusters8, saveplot=save_plot)

    
    # Fr-position analysis
    frpos_anal(clu_properties_data, kmclusters8, saveplot=save_plot)
    
    # MLI analysis
    mli_anal(all_data_mli, clu_properties_data, kmclusters8, saveplot=save_plot)





    
    # gcolors = kmclusters6[0].map({0:'blue', 1:'red', 2:'green', 3:'violet', 4:'yellow', 5:'pink', 6:'brown', 7:'black', 8:'cyan'})
    gcolors = kmclusters8[0].map({0:'blue', 1:'red', 2:'green', 3:'violet', 4:'yellow', 5:'pink', 6:'brown', 7:'black', 8:'cyan'})

    # Analyse autocorrelation
    df_ac = clu_properties_data['autocorr']
    df_ac['kmclu'] = kmclusters.loc[idxsli['8', :, :, :]].values
    df_ac.set_index('kmclu', append=True, inplace=True)

    __ = df_ac.copy()
    __ = __.reset_index()
    __ = __.melt(id_vars=['cond', 'rec', 'clu', 'kmclu'])

    fig, ax = plt.subplots(2, 4, figsize=(14, 12))
    for idx_kmsclu in range(8):
        ___ = __[(__.kmclu==idx_kmsclu) & (__.cond=='control')]
        sns.lineplot(data=___, x='lag', y='value', errorbar='sd', ax=ax.flatten()[idx_kmsclu])
        # sns.lineplot(data=___, x='lag', y='value', hue='cond', hue_order=['gCNO', 'control'], errorbar='sd', ax=ax.flatten()[idx_kmsclu])
        ax.flatten()[idx_kmsclu].get_legend().remove()
    anal_ac(df_ac)
    # fig = plt.figure()
    # subfig = fig.subfigures(2)
    # for idx_cond, cond in ['gCNO', 'control']:
    #     ax = subfig.subplots(2, 2)
    #     __ = clu_properties_data['fr'].loc[idxsli[cond, :, :]]
    #     for idx_ax, axis in enumerate(ax.flatten()):
    #         __ = __.iloc[:, idx_ax]

    pdb.set_trace()
        
    
    fig, ax = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(14, 12))
    for idx_cond, cond in enumerate(['gCNO', 'control']):
        # Data
        whisk_data = clu_properties_data['fr'].loc[idxsli[cond, :, :], ['pre_whisk', 'post_whisk']]
        nowhisk_data = clu_properties_data['fr'].loc[idxsli[cond, :, :], ['pre_nowhisk', 'post_nowhisk']]
        # Plot
        ax[idx_cond, 0].scatter(whisk_data.iloc[:, 0], whisk_data.iloc[:, 1], c=gcolors.loc[idxsli[cond, :, :]])
        ax[idx_cond, 0].plot([0, 200], [0, 200], '--', linewidth=3, c='gray')
        ax[idx_cond, 0].plot(np.arange(0, 200), predict[idx_cond][0], linewidth=3, c='gray')
        # ax[idx_cond, 0].plot(np.arange(0, 200), linefit.loc[:, idxsli[f'whisk{cond}']], linewidth=3, c='gray')
        ax[idx_cond, 0].set_aspect('equal', adjustable='box')
        ax[idx_cond, 1].scatter(nowhisk_data.iloc[:, 0], nowhisk_data.iloc[:, 1], c=gcolors.loc[idxsli[cond, :, :]])
        ax[idx_cond, 1].plot([0, 200], [0, 200], '--', linewidth=3, c='gray')
        ax[idx_cond, 1].plot(np.arange(0, 200), predict[idx_cond][1], linewidth=3, c='gray')
        # ax[idx_cond, 1].plot(np.arange(0, 200), linefit.loc[:, idxsli[f'nowhisk{cond}']], linewidth=3, c='gray')
        ax[idx_cond, 1].set_aspect('equal', adjustable='box')

        ax[idx_cond, 0].set_title('whisking')
        ax[idx_cond, 1].set_title('no-whisking')
        ax[idx_cond, 0].set_ylabel(f'{cond}_post')
        ax[idx_cond, 1].set_ylabel(f'{cond}_post')
        ax[idx_cond, 1].set_xlabel(f'{cond}_pre')
        ax[idx_cond, 1].set_xlabel(f'{cond}_pre')

        # ax[idx_cond, 0].set_aspect('equal', 'box')
        # ax[idx_cond, 1].set_aspect('equal', 'box')
    aa = clu_properties_data['autocorr'].loc['control'].copy()
    aa['kmclu'] = kmclusters.loc[idxsli['8', 'control', :, :,]].values
    aa = aa.set_index('kmclu', append=True)
    aa_norm = []
    for kmclu in range(8):
        __ = aa.loc[idxsli[:, :, kmclu]].copy()
        __ = __.sub(__.mean(axis=1), axis=0).div(__.std(axis=1), axis=0)
        __ = pd.concat([__], keys=[f'{kmclu}'], names=['kmclu'])
        aa_norm.append(__)
    aa_norm = pd.concat(aa_norm)
    aa_norm = aa_norm.reset_index().melt(id_vars=['kmclu', 'rec', 'clu'])

    fig, ax = plt.subplots(2, 4)
    axis = ax.flatten()
    for kmclu in range(8):
        __ = aa_norm[aa_norm.kmclu==f'{kmclu}']
        sns.lineplot(data=__, x='lag', y='value', errorbar='sd', ax=axis[kmclu])

    aa.reset_index().melt(id_vars=['rec', 'clu', 'kmclu'])


if __name__ == '__main__':
    """ Run script if clu_properties module is main programme
    """
    cgs = 2                     # 2: good clusters
    var = 'angle'
    whisker='whisk2'            # Focus on whisker 2 (anterior one)
    surr = False                # Real data (not shuffled)
    discard = False             # False: do not discard clusters based on fr (better not)
    pethdiscard = True          # True: discard based on bad peth (used)
    window_ac = 200             # Window autocorrelation (in ms)
    save_data = False           # Either compute and save data or load
    save_plot = False

    run_clu_properties(cgs=cgs,
                       var=var,
                       whisker=whisker,
                       surr=surr,
                       discard=discard,
                       pethdiscard=pethdiscard,
                       window_ac=window_ac,
                       save_data=save_data,
                       save_plot=save_plot)
