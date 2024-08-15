#!/usr/bin/env python
"""
Script for fitting model to population spike counts and individual cluster
spike counts; very similart to spatiotemporal_visualisation.py, but mainly
with the addition of Bayesian model fitting. Relies on some functions in
spatiotemporal_visualisation module, including spk_train_nw (computing
spike counts for each cluster, not based on whisker bins), and spkcount_cortex
(function to count total population spikes in cortex).
Contains the following:

- get_popdata: retrieve and organise spike count data for each recording;
input data are at cluster level, so compute total population level spike
counts, restricted to cerebellar cortex (using spkcount_cortex from
spatiotemporal_visualisation module); then align data with different (CNO and
PBS) t_drop times; organise in dataframe.

- fit_spkcount: use population data from get_popdata and run InverseGamma
model and plot results.

- clu_absdiff: analysis at single cluster level; compute spk_count diff
between time bins (5 min) and one baseline; organise in dataframe

- get_cludata: analysis at single cluster level; retrieve data, use
clu_absdiff, and organise in dataframe

- ml_fr_fit: analysis at single cluster level; compute maximum likelihood
for gamma parameters (on cluster data)

- fit_spkcount_clu: analysis at single cluster level; posterior model
fitting.

"""

# Ensure /data_analysis is in path
import os
import sys
if not ('data_analysis' in os.getcwd()):
    sys.path.append(os.path.join(os.getcwd(), 'data_analysis'))

# Import modules
import pdb
import itertools
import load_npx as lnpx
import pandas as pd
import seaborn as sns
import numpy as np
import pymc as pm
import arviz as az
import warnings
import itertools
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec 
from random import sample, randint
from scipy.stats import gamma
from scipy.stats import invgamma
from scipy.stats import multivariate_normal
from scipy.stats import norm
from scipy.stats import zscore
from scipy.stats import kruskal
from scipy.stats import f_oneway
from spatiotemporal_visualisation import spk_train_nw, spkcount_cortex
from meta_data import load_meta
from functools import partial

def get_popdata(npxpath,
                aid,
                cortex_depth,
                t_drop,
                conditions,
                tw_start,
                tw_width,
                t_scale,
                cgs=2):
    ''' Get spike counts of cerebellar cortical population across time in
    dataframe.

    Attention: same as 'plot_tanalysis in spatiotemporal_visualisation module.
    '''

    # Sanity check paths, depths and t_drops
    if (len(npxpath) != len(cortex_depth)) or (len(npxpath) != len(t_drop)):
        print('Attention: data length is different')
        exit()

    # Time interval at drop and reference tw (for scaling)
    drop_int = min(t_drop) // tw_width
    tw_scale = drop_int + (t_scale // tw_width)

    ctx_spkcnt = []
    for idx, dataset in enumerate(npxpath):
        # Get npx data
        spk = lnpx.loadKsDir(dataset)
        spk = lnpx.process_spiketimes(spk)
        print(dataset)
        # pdb.set_trace()
        # Compute spike trains for each cluster (sorted by depth)
        spktr_sorted = spk_train_nw(spk, binl=tw_width * 1000 * 60, surr=False, cgs=cgs)
        # Count cortical spikes for every time window
        ctx_d = cortex_depth[idx]
        tw_spkcnt = spkcount_cortex(spktr_sorted,
                                    spk.sortedDepth,
                                    cortex_depth=ctx_d,
                                    tw_start=tw_start,
                                    tw_width=tw_width,
                                    tw_scale=tw_scale,
                                    )

        # Expand data with nan if t_drop different across datasets
        if t_drop.count(t_drop[0]) != len(t_drop):
            if t_drop[idx] == 10:
                for i in range(int(10 // tw_width)):
                    tw_spkcnt.insert(0, np.nan)
            elif t_drop[idx] == 20:
                for i in range(int(10 // tw_width)):
                    tw_spkcnt.append(np.nan)

        ctx_spkcnt.append(tw_spkcnt)

    # Dataframe for cortex spike count (csc) - remove incomplete time intervals
    df_csc = pd.DataFrame(ctx_spkcnt)
    df_csc = df_csc.dropna(axis=1)

    # Rename colums with time
    df_csc.columns = np.arange(df_csc.columns.size) * tw_width - min(t_drop)

    # Scale by baseline (time interval just before drop)
    df_csc_scl = df_csc.div(df_csc.iloc[:, int(tw_scale)], axis=0)

    # Add conditions to each dataset (used for fit and hues)
    df_csc['condition'] = conditions
    df_csc['aid'] = aid
    df_csc['idx_rec'] = df_csc.index
    df_csc_scl['aid'] = aid
    df_csc_scl['condition'] = conditions
    df_csc_scl['idx_rec'] = df_csc.index

    # Melt datagrame
    df_csc_m = df_csc.melt(id_vars=['condition', 'aid','idx_rec'])
    df_csc_scl_m = df_csc_scl.melt(id_vars=['condition', 'aid', 'idx_rec'])

    # Rename variable to time and convert to int
    df_csc_m.rename(columns={'variable':'time'}, inplace=True)
    df_csc_scl_m.rename(columns={'variable':'time'}, inplace=True)

    df_csc_m.time = df_csc_m.time.astype(int)
    df_csc_scl_m.time = df_csc_scl_m.time.astype(int)

    return df_csc_m, df_csc_scl_m


def simple_stats(spk_data):
    """ Simple comparison between conditions (one way anova or Kruskal–Wallis test)
    """
    gCNO_array = spk_data[(spk_data.time>=0) & (spk_data.condition=='gCNO')]['value'].values
    wCNO_array = spk_data[(spk_data.time>=0) & (spk_data.condition=='wCNO')]['value'].values
    aPBS_array = spk_data[(spk_data.time>=0) & (spk_data.condition=='aPBS')]['value'].values
    F, p_f = f_oneway(gCNO_array, wCNO_array, aPBS_array)  # 10.91, 2.82e-05
    H, p_h = kruskal(gCNO_array, wCNO_array, aPBS_array)   # 46.60, 7.59e-11
    print(F, p_f, H, p_h)


def fit_spkcount(npxpath, aid, cortex_depth, t_drop, conditions, tw_start, tw_width, t_scale='-5', cgs=2):
    ''' Compute posterior for model of cortical (population) spikecounts.
    '''    
    t_scale = int(t_scale)
    # # Load dataframes...
    # with open(os.path.join(os.getcwd(), 'data_analysis/df_pop_-5')) as csv_file:
    #     df_l = pd.read_csv(csv_file)
    # df = df_l

    # ... or generate new (use only scaled data)
    __, df = get_popdata(npxpath, aid, cortex_depth, t_drop, conditions, tw_start, tw_width, t_scale, cgs=cgs)

    # Set Random seed
    RANDOM_SEED = 100
    rng = np.random.default_rng(RANDOM_SEED)

    # Drop -5 time, reset index
    df = df[df.time!=t_scale]
    df.reset_index(drop=True, inplace=True)

    # Create condtm from condition and time
    df['condtm'] = df.condition + df.loc[:, 'time'].apply(lambda x: 'post' if x>=0 else 'pre')

    # Before: test for equality spike counts across conditions (either mean or median)
    simple_stats(df)

    # Create model scaled spike counts INVERSE GAMMA
    def fxn():
        warnings.warn("deprecated", RuntimeWarning)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fxn()        
        with pm.Model(coords={'condition':df.condtm.factorize()[1], 'aid':df.aid.factorize()[1], 'time':df.time.factorize()[1]}) as catc_model:
            # Means and variances (set mu_a=1.2, mu_b=1.2, sigma_a=0.2, sigma_b=0.2)
            mu_a = 1.0
            mu_b = 1.2
            sigma_a = 0.2
            sigma_b = 0.2

            # Covariance for time MvNormal
            cov_a_t = []
            cov_b_t = []
            for t in df.time.factorize()[1]:
                dist = abs(df.time.factorize()[1] - t) / tw_width
                cov_a_t.append(np.around(np.exp(-dist * 0.7), decimals=3))
                cov_b_t.append(np.around(np.exp(-dist * 0.7), decimals=3))

            cov_a_t = np.array(cov_a_t) * sigma_a
            cov_b_t = np.array(cov_a_t) * sigma_b

            # Put in dictionary for later
            cov_prior = {'cov_a_t_prior':cov_a_t, 'cov_b_t_prior':cov_b_t}

            # Alpha parameter
            θ_a = pm.Normal('θ_a', mu=mu_a, sigma=sigma_a)
            θ_a_cond = pm.Normal('θ_a_cond', mu=mu_a, sigma=sigma_a, dims='condition')
            θ_a_aid = pm.Normal('θ_a_aid', mu=mu_a, sigma=sigma_a, dims='aid')
            θ_a_time = pm.MvNormal('θ_a_time', mu=mu_a, cov=cov_a_t, dims='time')

            alpha = pm.math.exp(θ_a + θ_a_cond[df.condtm.factorize()[0]] + θ_a_aid[df.aid.factorize()[0]] + θ_a_time[df.time.factorize()[0]])

            # Beta parameter
            θ_b = pm.Normal('θ_b', mu=mu_b, sigma=sigma_b)
            θ_b_cond = pm.Normal('θ_b_cond', mu=mu_b, sigma=sigma_b, dims='condition')
            θ_b_aid = pm.Normal('θ_b_aid', mu=mu_b, sigma=sigma_b, dims='aid')
            θ_b_time = pm.MvNormal('θ_b_time', mu=mu_b, cov=cov_b_t, dims='time')

            beta = pm.math.exp(θ_b + θ_b_cond[df.condtm.factorize()[0]] + θ_b_aid[df.aid.factorize()[0]] + θ_b_time[df.time.factorize()[0]])

            # Dependent variable
            y = pm.InverseGamma('counts', alpha=alpha, beta=beta, observed=df.value)

            # Sample
            catc_priorpc = pm.sample_prior_predictive(samples=1000, random_seed=rng)
            catc_trace = pm.sample(idata_kwargs={"log_likelihood": True}, random_seed=rng)
            catc_postpc = pm.sample_posterior_predictive(catc_trace, random_seed=rng)


    # Prepare data for plots
    # Extract effective sample size in df
    df_sum = az.summary(catc_trace)
    df_sum['cond'] = df_sum.index
    __ = lambda x: x[:] if x.find('[') == -1 else x[:x.find('[')]
    df_sum['cond'] = df_sum['cond'].apply(__).values

    # Extract posterior (condition) samples and get hdi
    post_cond = az.extract_dataset(catc_trace.posterior[['θ_a_cond', 'θ_b_cond']]).to_dataframe().droplevel(['chain', 'draw'])
    post_cond['condition'] = post_cond.index
    post_cond = post_cond.melt(id_vars='condition', var_name='parameter')
    midx = pd.MultiIndex.from_product([post_cond.condition.unique(), post_cond.parameter.unique()], names=['condition', 'parameter'])
    hdi_postcond = pd.DataFrame(index=midx, columns=['value'])
    for condpar in midx:
        __ = az.hdi(post_cond[(post_cond.condition==condpar[0]) & (post_cond.parameter==condpar[1])].value.values)
        hdi_postcond.loc[(condpar[0], condpar[1])] = [__]

    # Posterior of difference (contrast) for condition parameters
    df_diff = pd.DataFrame()

    # for c in itertools.combinations(df.cond.unique(), 2):
    for c_ in itertools.combinations(df.condtm.unique(), 2):
        if (c_[0].find('pre') != c_[1].find('pre')) & (c_[0][:2] == c_[1][:2]):
        # if c_[0].find('pre') ==  c_[1].find('pre'):
            post_diff_a = catc_trace.posterior.θ_a_cond.sel(condition=c_[0]).values.flatten() - catc_trace.posterior.θ_a_cond.sel(condition=c_[1]).values.flatten()
            post_diff_b = catc_trace.posterior.θ_b_cond.sel(condition=c_[0]).values.flatten() - catc_trace.posterior.θ_b_cond.sel(condition=c_[1]).values.flatten()

            post_diff_a = pd.DataFrame({'parameter':'alpha', f'{c_[0]} - {c_[1]}':post_diff_a})
            post_diff_b = pd.DataFrame({'parameter':'beta', f'{c_[0]} - {c_[1]}':post_diff_b})

            __ = pd.concat([post_diff_a, post_diff_b])

            df_diff = pd.concat([df_diff, __], axis=1)

    df_diff = df_diff.loc[:, ~df_diff.columns.duplicated()]
    df_diff = df_diff.melt(id_vars='parameter', var_name='condition')

    # Get hdi for contrasts (default stats.hdi_prob rcparams 0.94)
    midx = pd.MultiIndex.from_product([df_diff.condition.unique(), df_diff.parameter.unique()], names=['condition', 'parameter'])
    hdi_diff = pd.DataFrame(index=midx, columns=['value'])
    for condpar in midx:
        __ = az.hdi(df_diff[(df_diff.parameter == condpar[1]) & (df_diff.condition == condpar[0])].value.values)
        hdi_diff.loc[(condpar[0], condpar[1])] = [__]

    # Difference of differences
    df_diff_diff = df_diff[df_diff.condition == 'gCNOpre - gCNOpost'].copy()
    df_diff_diff['condition'] = 'gCNO - wCNO'
    df_diff_diff['value'] = df_diff[df_diff.condition == 'gCNOpre - gCNOpost']['value'].values - df_diff[df_diff.condition == 'wCNOpre - wCNOpost']['value'].values
    __  = df_diff[df_diff.condition == 'gCNOpre - gCNOpost'].copy()
    __['condition'] = 'gCNO - aPBS'
    __['value'] = df_diff[df_diff.condition == 'gCNOpre - gCNOpost']['value'].values - df_diff[df_diff.condition == 'aPBSpre - aPBSpost']['value'].values
    df_diff_diff = pd.concat([df_diff_diff, __])

    __  = df_diff[df_diff.condition == 'wCNOpre - wCNOpost'].copy()
    __['condition'] = 'wCNO - aPBS'
    __['value'] = df_diff[df_diff.condition == 'wCNOpre - wCNOpost']['value'].values - df_diff[df_diff.condition == 'aPBSpre - aPBSpost']['value'].values
    df_diff_diff = pd.concat([df_diff_diff, __])

    # Get hdi for contrasts (default stats.hdi_prob rcparams 0.94)
    midx_diff = pd.MultiIndex.from_product([df_diff_diff.condition.unique(), df_diff_diff.parameter.unique()], names=['condition', 'parameter'])
    hdi_diff_diff = pd.DataFrame(index=midx_diff, columns=['value'])
    for condpar in midx_diff:
        __ = az.hdi(df_diff_diff[(df_diff_diff.parameter == condpar[1]) & (df_diff_diff.condition == condpar[0])].value.values)
        hdi_diff_diff.loc[(condpar[0], condpar[1])] = [__]

    
    
    # Chek posterior covariance
    cov_post = {}
    for par in ['a', 'b']:
        __ = pd.DataFrame()
        for chain in range(4):
            __ = pd.concat([__, pd.DataFrame(catc_trace.posterior.θ_a_time[chain, :, :].values)])
        cov_post[f'cov_{par}_t_post'] = np.cov(__.T)

    # __ = pd.DataFrame()
    # for chain in range(4):
    #     __ = pd.concat([__, pd.DataFrame(catc_trace.posterior.θ_a_time[chain, :, :].values)])
    # cov_b_t_post = np.cov(__.T)

    # Simulate data
    # df_sim = pd.DataFrame(columns=df.columns[[0, 1, 3]])
    df_sim = pd.DataFrame(columns=df.columns[[0, 1, 3, 5]])

    def fxn():
        warnings.warn("deprecated", RuntimeWarning)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fxn()        
        for rowdf in range(df.shape[0]):
            # Get xarray subspace
            samp = catc_trace.posterior.sel(condition=df.condtm.iloc[rowdf], aid=df.aid.iloc[rowdf], time=df.time.iloc[rowdf])
            # Simulate
            alpha_sim = np.exp(samp.θ_a.values + samp.θ_a_cond.values + samp.θ_a_aid.values + samp.θ_a_time.values)
            beta_sim = np.exp(samp.θ_b.values + samp.θ_b_cond.values + samp.θ_b_aid.values + samp.θ_b_time.values)
            # sim = gamma.rvs(alpha_sim, scale=1/beta_sim)
            draw = partial(pm.draw, random_seed=RANDOM_SEED)
            sim = draw(pm.InverseGamma.dist(alpha=alpha_sim[:,:200].flatten(), beta=beta_sim[:,:200].flatten()))

            # Combine in dataframe
            __ = pd.DataFrame(dict(condition=df.condition.iloc[rowdf], aid=df.aid.iloc[rowdf], time=df.time.iloc[rowdf], value=sim, condtm=df.condtm.iloc[rowdf]))
            df_sim = pd.concat([df_sim, __])

    

    # Compare models
    # df_compare_loo = az.compare(dict(gamma=catc_trace, inv_gamma=catc_trace_invg))

    # Plots
    pdb.set_trace()
    # sns.set(style='darkgrid')
    sns.set(style='white')
    figsize = (14, 12)
    size = 'x-large'
    style = 'italic'
    # plt.rcParams['font.size'] = '40'
    # plt.rcParams['axes.labelsize'] = '30'
    # plt.rcParams['legend.fontsize'] = '25'
    # plt.rcParams['xtick.labelsize'] = '20'
    # plt.rcParams['ytick.labelsize'] = '20'
    # plt.rcParams['font.size'] = '1'
    # plt.rcParams['axes.labelsize'] = '1'
    # plt.rcParams['legend.fontsize'] = '1'
    # plt.rcParams['xtick.labelsize'] = '1'
    # plt.rcParams['ytick.labelsize'] = '1'
    sns.set_context("poster", font_scale=1)
    sns.color_palette("deep")
    # sns.set(rc={'axes.facecolor':'lavenderblush'})
    plt.rcParams['axes.facecolor'] = 'lavenderblush'
    # Plot data
    fig1, ax1 = plt.subplots(figsize=figsize)
    sns.boxplot(x=df.time,
                y=df.value,
                hue=df.condition,
                ax=ax1)
    sns.stripplot(x=df.time,
                  y=df.value,
                  hue=df.condition,
                  dodge=True,
                  palette='pastel',
                  ax=ax1)

    plt.ylabel('scaled spike count')
    plt.xlabel('Time from drop (min)')
    handles, labels = ax1.get_legend_handles_labels()
    labels = ['CNO GLYT2', 'CNO WT', 'PBS']
    ax1.legend(handles[:3], labels[:3], loc='upper right')
    # fig1.suptitle('Population spike count', size=size, style=style)
    fig1.suptitle('Population spike count', style=style)
    fig1.tight_layout()
    pdb.set_trace()
    # Save figure 1
    if os.path.basename(os.getcwd()) != 'data_analysis':
        os.chdir('./data_analysis')

    try:
        plt.savefig(os.getcwd() + f'/images/spkcnt/CNO_InvGamma_{t_scale}/data.svg', format='svg')
    except FileNotFoundError:
        print(f'created dir images/spkcnt/CNO/InvGamma_{t_scale}')
        os.makedirs(os.getcwd() + f'/images/spkcnt/CNO_InvGamma_{t_scale}')
        plt.savefig(os.getcwd() + f'/images/spkcnt/CNO_InvGamma_{t_scale}/data.svg', format='svg')

    # Plot data
    fig1_1, ax1_1 = plt.subplots(figsize=figsize)
    sns.boxplot(x=df.time,
                y=df.value,
                hue=df.condition,
                showfliers=False,
                ax=ax1_1)
    # sns.stripplot(x=df.time,
    #               y=df.value,
    #               hue=df.condition,
    #               dodge=True,
    #               palette='pastel',
    #               ax=ax1)

    plt.ylabel('scaled spike count')
    plt.xlabel('Time from drop (min)')
    handles, labels = ax1.get_legend_handles_labels()
    labels = ['CNO GLYT2', 'CNO WT', 'PBS']
    ax1_1.legend(handles[:3], labels[:3], loc='upper right')
    # fig1.suptitle('Population spike count', size=size, style=style)
    fig1_1.suptitle('Population spike count', style=style)
    fig1_1.tight_layout()

    # Save figure 1
    if os.path.basename(os.getcwd()) != 'data_analysis':
        os.chdir('./data_analysis')

    try:
        plt.savefig(os.getcwd() + f'/images/spkcnt/CNO_InvGamma_{t_scale}/data_nooutliers.svg', format='svg')
    except FileNotFoundError:
        print(f'created dir images/spkcnt/CNO/InvGamma_{t_scale}')
        os.makedirs(os.getcwd() + f'/images/spkcnt/CNO_InvGamma_{t_scale}')
        plt.savefig(os.getcwd() + f'/images/spkcnt/CNO_InvGamma_{t_scale}/data_nooutliers.svg', format='svg')

    # Plot ESS vs rhat
    fig2, ax2 = plt.subplots(2, 1, figsize=figsize, sharex=True)
    sns.swarmplot(x=df_sum['r_hat'], y=df_sum['ess_bulk'], hue=df_sum['cond'], ax=ax2[0])
    sns.swarmplot(x=df_sum['r_hat'], y=df_sum['ess_tail'], hue=df_sum['cond'], ax=ax2[1])
    fig2.suptitle('Sampling metrics', style=style)
    fig2.tight_layout()

    # Save figure 2
    plt.savefig(os.getcwd() + f'/images/spkcnt/CNO_InvGamma_{t_scale}/metric_samp.svg', format='svg')

    # Plot posterior samples all but condition parameter
    fig3, ax3 = plt.subplots(figsize=figsize)
    az.plot_forest(catc_trace, var_names=['~θ_a_cond', '~θ_b_cond'], combined=True, ax=ax3)
    fig3.suptitle('Posterior samples (excluded condition)', style=style)
    fig3.tight_layout()

    # Save figure 3
    plt.savefig(os.getcwd() + f'/images/spkcnt/CNO_InvGamma_{t_scale}/post_nocond.svg', format='svg')

    # Plot prior checks
    fig4, ax4 = plt.subplots(figsize=figsize)
    az.plot_ppc(catc_priorpc, group='prior', mean=True, ax=ax4)
    ax4.set_xlabel('scaled spike count')
    ax4.set_ylabel('density')
    fig4.suptitle('Prior predictive samples', style=style)
    fig4.tight_layout()

    # Save figure 4
    plt.savefig(os.getcwd() + f'/images/spkcnt/CNO_InvGamma_{t_scale}/priorcheck.svg', format='svg')

    # Plot posterior checks
    fig5, ax5 = plt.subplots(figsize=figsize)
    az.plot_ppc(catc_postpc, group='posterior', ax=ax5)
    ax5.set_xlabel('scaled spike count')
    ax5.set_ylabel('density')
    fig5.suptitle('Posterior predictive samples (aggregated)', style=style)
    fig5.tight_layout()

    # Save figure 5
    plt.savefig(os.getcwd() + f'/images/spkcnt/CNO_InvGamma_{t_scale}/postcheck.svg', format='svg')

    # Plot posterior checks stratified by condition
    fig5_1 = plt.figure(figsize=figsize)
    gs = GridSpec(2, 3, figure=fig5_1)
    ax5_11 = fig5_1.add_subplot(gs[0, :])
    sns.histplot(data=df, x='value', hue='condition', stat='count', ax=ax5_11, multiple='stack')
    ax5_12 = fig5_1.add_subplot(gs[1, 0])
    sns.histplot(data=df[df.condition == 'gCNO'], x='value', stat='density', ax=ax5_12, multiple='stack')
    sns.histplot(data=df_sim[df_sim.condition == 'gCNO'], x='value', stat='density', ax=ax5_12, multiple='stack')
    ax5_12.set_title('gCNO')
    ax5_13 = fig5_1.add_subplot(gs[1, 1])
    sns.histplot(data=df[df.condition == 'wCNO'], x='value', stat='density', ax=ax5_13, multiple='stack')
    sns.histplot(data=df_sim[df_sim.condition == 'wCNO'], x='value', stat='density', ax=ax5_13, multiple='stack')
    ax5_13.set_title('wCNO')
    ax5_14 = fig5_1.add_subplot(gs[1, 2])
    sns.histplot(data=df[df.condition == 'aPBS'], x='value', stat='density', ax=ax5_14, multiple='stack')
    sns.histplot(data=df_sim[df_sim.condition == 'aPBS'], x='value', stat='density', ax=ax5_14, multiple='stack')
    ax5_14.set_title('aPBS')
    for ax in [ax5_11, ax5_12, ax5_13, ax5_14]:
        ax.set_xlim([0, 4])
    fig5_1.tight_layout()
    plt.savefig(os.getcwd() + f'/images/spkcnt/CNO_InvGamma_{t_scale}/postcheck_stratified.svg', format='svg')
        

    # Plot posterior sample conditions
    color = ['tab:blue', 'tab:orange'] * 6
    fig6, ax6 = plt.subplots(figsize=figsize)
    sns.stripplot(x='value', y='condition', data=post_cond, hue='parameter', dodge=True, ax=ax6)
    for idx, hight in enumerate(np.array([[-0.3, -0.3], [0.1, 0.1], [0.7, 0.7], [1.1, 1.1], [1.7, 1.7], [2.1, 2.1], [2.7, 2.7], [3.1, 3.1], [3.7, 3.7], [4.1, 4.1], [4.7, 4.7], [5.1, 5.1]])):
        ax6.plot(hdi_postcond.value.values[idx], hight, color=color[idx])
    ax6.text(sum(hdi_postcond['value'][0]) / 2.15, -0.35, 'hdi 94%')
    ax6.set_yticklabels(ax6.get_yticklabels(), rotation=45)
    fig6.suptitle('Posterior samples (condition parameters)', style=style)
    fig6.tight_layout()

    # Save figure 6
    plt.savefig(os.getcwd() + f'/images/spkcnt/CNO_InvGamma_{t_scale}/post_cond.svg', format='svg')

    # Plot contrasts
    fig7, ax7 = plt.subplots(figsize=figsize)
    color = ['tab:blue', 'tab:orange'] * 3
    sns.stripplot(x='value', y='condition', data=df_diff.groupby(by=['parameter', 'condition'], group_keys=False, sort=False).apply(lambda x:x[::4]), hue='parameter', dodge=True, ax=ax7)
    for idx, hight in enumerate(np.array([[-0.3, -0.3], [0.1, 0.1], [0.7, 0.7], [1.1, 1.1], [1.7, 1.7], [2.1, 2.1]])):
        ax7.plot(hdi_diff.value.values[idx], hight, color=color[idx])

    # yticks, yticklabels = ax7.get_yticklabels()
    # # ax7.set_yticklabels(ax7.get_yticklabels(), rotation=45)
    ax7.set_yticklabels(['CNO GLYT2', 'CNO WT', 'PBS'], rotation=45)
    ax7.set_ylabel('')
    ax7.vlines(0, 2.2, -0.3)
    # ax7.text(sum(hdi_diff['value'][1]) / 2.15, -0.35, 'hdi 94%')
    ax7.text(np.diff(hdi_diff['value'][1]) / 2.3, 0.05, 'hdi 94%')
    fig7.suptitle('Contrasts', style=style)
    fig7.tight_layout()

    # Save figure 7
    plt.savefig(os.getcwd() + f'/images/spkcnt/CNO_InvGamma_{t_scale}/contrast.svg', format='svg')

    fig7_1, ax7_1 = plt.subplots(figsize=figsize)
    color = ['tab:blue', 'tab:orange'] * 3
    sns.stripplot(x='value', y='condition', data=df_diff_diff.groupby(by=['parameter', 'condition'], group_keys=False, sort=False).apply(lambda x:x[::4]), hue='parameter', dodge=True, ax=ax7_1)
    for idx, hight in enumerate(np.array([[-0.3, -0.3], [0.1, 0.1], [0.7, 0.7], [1.1, 1.1], [1.7, 1.7], [2.1, 2.1]])):
        ax7_1.plot(hdi_diff_diff.value.values[idx], hight, color=color[idx])

    # yticks, yticklabels = ax7.get_yticklabels()
    # # ax7.set_yticklabels(ax7.get_yticklabels(), rotation=45)
    # ax7_1.set_yticklabels(['CNO GLYT2', 'CNO WT', 'PBS'], rotation=45)
    ax7_1.set_ylabel('')
    ax7_1.vlines(0, 2.2, -0.3)
    # ax7.text(sum(hdi_diff['value'][1]) / 2.15, -0.35, 'hdi 94%')
    ax7_1.text(np.diff(hdi_diff['value'][1]) / 2.3, 0.05, 'hdi 94%')
    fig7_1.suptitle('Difference of Contrasts', style=style)
    fig7_1.tight_layout()

    # Save figure 7
    plt.savefig(os.getcwd() + f'/images/spkcnt/CNO_InvGamma_{t_scale}/difference_of_contrast.svg', format='svg')



    # Plot prior and post cov
    fig8, ax8 = plt.subplots(2, 2, figsize=figsize)
    sns.heatmap(cov_prior['cov_a_t_prior'], ax=ax8[0, 0])
    sns.heatmap(cov_post['cov_a_t_post'], ax=ax8[0, 1])
    sns.heatmap(cov_prior['cov_b_t_prior'], ax=ax8[1, 0])
    sns.heatmap(cov_post['cov_b_t_post'], ax=ax8[1, 1])

    ax8[0, 0].set_title('prior t_alpha')
    ax8[0, 1].set_title('posterior t_alpha')
    ax8[1, 0].set_title('prior t_beta')
    ax8[1, 1].set_title('posterior t_beta')

    fig8.suptitle('Time covariance matrix', style=style)
    fig8.tight_layout()

    # Save figure 8
    plt.savefig(os.getcwd() + f'/images/spkcnt/CNO_InvGamma_{t_scale}/cov.svg', format='svg')

    # Plot sim_data
    fig9, ax9 = plt.subplots(figsize=figsize)
    sns.boxplot(x=df_sim.time,
                y=df_sim.value,
                hue=df_sim.condition,
                ax=ax9)

    plt.ylabel('scaled spike count')
    plt.xlabel('Time from drop (min)')
    handles, labels = ax9.get_legend_handles_labels()
    ax9.legend(handles[:3], labels[:3], loc='upper right')
    fig9.suptitle('Population spike count - simulated', style=style)
    fig9.tight_layout()

    # Save figure 9
    plt.savefig(os.getcwd() + f'/images/spkcnt/CNO_InvGamma_{t_scale}/data_sim.svg', format='svg')

    # Plot posterior checks Simulation (and max likelihood estimate)
    fig10, ax10 = plt.subplots(figsize=figsize)
    ax10 = sns.histplot(data=df.value, stat='density', color='black', element='step', fill=False, label='observations', bins=100, ax=ax10)
    ax10 = sns.histplot(data=df_sim.value, stat='density', color='orange', element='step', fill=False, label='posterior predictive samples', ax=ax10)

    # fig10.suptitle('Posterior predictive (aggregated)', size=size, style=style)
    fig10.suptitle('Posterior predictive samples', style=style)
    fig10.tight_layout()
    ax10.legend()
    ax10.set_xlabel('scaled spike counts')
    ax10.set_xlim([0, 6])
    

    # Save figure 10
    plt.savefig(os.getcwd() + f'/images/spkcnt/CNO_InvGamma_{t_scale}/postchecks_mine_new.svg', format='svg')

    # Check sample correlation

    cond_a = catc_trace.posterior['θ_a_cond']
    cond_b = catc_trace.posterior['θ_b_cond']
    fig11, ax11 = plt.subplots(4, 6, figsize=figsize, sharex=True, sharey=True)
    for ch, cnd in itertools.product([0, 1, 2, 3], [0, 1, 2, 3, 4, 5]):
        sns.scatterplot(x=cond_a.values[ch, :, cnd], y=cond_b.values[ch, :, cnd], ax=ax11[ch, cnd])
        if cnd == 0:
            ax11[ch, cnd].set_ylabel(f'chain {ch}')
        if ch == 3:
            ax11[ch, cnd].set_xlabel(f'{cond_a.condition.values[cnd]}')

        ax11[ch, cnd].set_aspect('equal')

    fig11.suptitle('Samples θ_a_cond vs θ_b_cond', style=style)
    fig11.tight_layout()

    # Save figure 11
    plt.savefig(os.getcwd() + f'/images/spkcnt/CNO_InvGamma_{t_scale}/samp_corr.svg', format='svg')

    # Explain results !!!    
    df_mean = pd.DataFrame()
    df_var = pd.DataFrame()
    invGamma_samp = pd.DataFrame()
    list_idx = df[(df.time == -10) | (df.time == 15)].index
    def fxn():
        warnings.warn("deprecated", RuntimeWarning)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fxn()        
        for rowdf in list_idx:
            if rowdf <= 32:
                rowdf_true = rowdf
            else:
                rowdf_true = rowdf
                rowdf = rowdf - 132
            samp = catc_trace.posterior.sel(condition=df.condtm.iloc[rowdf], aid=df.aid.iloc[rowdf], time=df.time.iloc[rowdf])
            samp_cond = catc_trace.posterior.sel(condition=df.condtm.iloc[rowdf_true], aid=df.aid.iloc[rowdf_true], time=df.time.iloc[rowdf_true])

            # Simulate
            alpha_sim = np.exp(samp.θ_a.values + samp_cond.θ_a_cond.values + samp.θ_a_aid.values + samp.θ_a_time.values)[:, 500::20].flatten()
            beta_sim = np.exp(samp.θ_b.values + samp_cond.θ_b_cond.values + samp.θ_b_aid.values + samp.θ_b_time.values)[:,500::20].flatten()
            means_sim = (beta_sim / (alpha_sim - 1))
            var_sim = beta_sim**2 / ((alpha_sim - 1)**2 * (alpha_sim - 2))
            # Patching draw function for reproducibility
            draw = partial(pm.draw, random_seed=RANDOM_SEED)
            sim_dist = draw(pm.InverseGamma.dist(mu=means_sim, sigma=var_sim))

            # Combine in dataframe        
            __ = pd.DataFrame(dict(condition=df.condition.iloc[rowdf_true], aid=df.aid.iloc[rowdf_true], time=df.time.iloc[rowdf_true], value=means_sim, condtm=df.condtm.iloc[rowdf_true]))
            df_mean = pd.concat([df_mean, __])
            __ = pd.DataFrame(dict(condition=df.condition.iloc[rowdf_true], aid=df.aid.iloc[rowdf_true], time=df.time.iloc[rowdf_true], value=var_sim, condtm=df.condtm.iloc[rowdf_true]))
            df_var = pd.concat([df_var, __])
            __ = pd.DataFrame(dict(condition=df.condition.iloc[rowdf_true], aid=df.aid.iloc[rowdf_true], time=df.time.iloc[rowdf_true], value=sim_dist, condtm=df.condtm.iloc[rowdf_true]))
            invGamma_samp = pd.concat([invGamma_samp, __])
        

    # Add time variable
    df_mean['time'] = np.where(df_mean['time']<0, 'pre', 'post')
    df_var['time'] = np.where(df_var['time']<0, 'pre', 'post')
    
    # Compute contrasts
    df_mean_contrast = df_mean[df_mean['time']=='post']['value'] - df_mean[df_mean['time']=='pre']['value']
    df_mean_contrast = pd.DataFrame(df_mean_contrast)
    df_mean_contrast['condition'] = df_mean[df_mean['time']=='post']['condition']
    df_var_contrast = df_var[df_var['time']=='post']['value'] - df_var[df_var['time']=='pre']['value']
    df_var_contrast = pd.DataFrame(df_var_contrast)
    df_var_contrast['condition'] = df_var[df_var['time']=='post']['condition']

    # Compute difference of contrasts (mean)
    df_mean_contrast_diff = pd.DataFrame()
    __ = df_mean_contrast[df_mean_contrast.condition == 'gCNO'].value - df_mean_contrast[df_mean_contrast.condition == 'wCNO'].value
    __.index = ['gCNO - wCNO'] * __.shape[0]
    df_mean_contrast_diff = pd.concat([df_mean_contrast_diff, __])
    __ = df_mean_contrast[df_mean_contrast.condition == 'gCNO'].value - df_mean_contrast[df_mean_contrast.condition == 'aPBS'].value
    __.index = ['gCNO - aPBS'] * __.shape[0]
    df_mean_contrast_diff = pd.concat([df_mean_contrast_diff, __])
    __ = df_mean_contrast[df_mean_contrast.condition == 'wCNO'].value - df_mean_contrast[df_mean_contrast.condition == 'aPBS'].value
    __.index =['wCNO - aPBS'] * __.shape[0]
    df_mean_contrast_diff = pd.concat([df_mean_contrast_diff, __])
    df_mean_contrast_diff.reset_index(inplace=True)
    df_mean_contrast_diff.columns = ['condition', 'value']

    # Compute difference of contrasts (mean)
    df_var_contrast_diff = pd.DataFrame()
    __ = df_var_contrast[df_var_contrast.condition == 'gCNO'].value - df_var_contrast[df_var_contrast.condition == 'wCNO'].value
    __.index = ['gCNO - wCNO'] * __.shape[0]
    df_var_contrast_diff = pd.concat([df_var_contrast_diff, __])
    __ = df_var_contrast[df_var_contrast.condition == 'gCNO'].value - df_var_contrast[df_var_contrast.condition == 'aPBS'].value
    __.index = ['gCNO - aPBS'] * __.shape[0]
    df_var_contrast_diff = pd.concat([df_var_contrast_diff, __])
    __ = df_var_contrast[df_var_contrast.condition == 'wCNO'].value - df_var_contrast[df_var_contrast.condition == 'aPBS'].value
    __.index =['wCNO - aPBS'] * __.shape[0]
    df_var_contrast_diff = pd.concat([df_var_contrast_diff, __])
    df_var_contrast_diff.reset_index(inplace=True)
    df_var_contrast_diff.columns = ['condition', 'value']

    

    # Hdi mean/var difference
    hdi_mean_contrast = {}
    hdi_var_contrast = {}
    for cond in ['gCNO', 'wCNO', 'aPBS']:
        hdi_mean_contrast[cond] = az.hdi(df_mean_contrast[df_mean_contrast['condition']==cond]['value'].values)
        hdi_var_contrast[cond] = az.hdi(df_var_contrast[df_var_contrast['condition']==cond]['value'].values)

    hdi_mean_contrast_diff = {}
    hdi_var_contrast_diff = {}
    for cond in ['gCNO - wCNO', 'gCNO - aPBS', 'wCNO - aPBS']:
        hdi_mean_contrast_diff[cond] = az.hdi(df_mean_contrast_diff[df_mean_contrast_diff['condition']==cond].value.values)
        hdi_var_contrast_diff[cond] = az.hdi(df_var_contrast_diff[df_var_contrast_diff['condition']==cond].value.values)

    # Compare posterior for mean/var inverse Gamma
    fig18, ax18 = plt.subplots(figsize=figsize)
    sns.stripplot(data=df_mean, x='value', y='condition', hue='time', dodge=True, ax=ax18)
    fig18.suptitle('compare pre/post mean')
    # fig18, ax18 = plt.subplots(figsize=figsize)
    # sns.stripplot(data=df_mean, x='value', y='condtm', dodge=True, ax=ax18)
    # fig18.suptitle('compare pre/post mean')

    fig19, ax19 = plt.subplots(figsize=figsize)
    sns.stripplot(data=df_var, x='value', y='condition', hue='time', dodge=True, ax=ax19)
    fig19.suptitle('compare pre/post variance')
    # fig19, ax19 = plt.subplots(figsize=figsize)
    # sns.stripplot(data=df_var, x='value', y='condtm', dodge=True, ax=ax19)
    # fig19.suptitle('compare pre/post variance')

    # Pull contrasts and diff_contrasts
    hdi_contrast = {'mean_gCNO':hdi_mean_contrast['gCNO'],
                    'var_gCNO':hdi_var_contrast['gCNO'],
                    'mean_wCNO':hdi_mean_contrast['wCNO'],
                    'var_wCNO':hdi_var_contrast['wCNO'],
                    'mean_aPBS':hdi_mean_contrast['aPBS'],
                    'var_aPBS':hdi_var_contrast['aPBS']}
    df_hdi_contrast = pd.DataFrame(hdi_contrast).melt()

    hdi_contrast_diff = {'mean_gCNO-wCNO ':hdi_mean_contrast_diff['gCNO - wCNO'],
                         'var_gCNO-wCNO':hdi_var_contrast_diff['gCNO - wCNO'],
                         'mean_gCNO-aPBS':hdi_mean_contrast_diff['gCNO - aPBS'],
                         'var_gCNO-aPBS':hdi_var_contrast_diff['gCNO - aPBS'],
                         'mean_wCNO-aPBS':hdi_mean_contrast_diff['wCNO - aPBS'],
                         'var_wCNO-aPBS':hdi_var_contrast_diff['wCNO - aPBS']}
    df_hdi_contrast_diff = pd.DataFrame(hdi_contrast_diff).melt()

   
    # Contrasts for means/variance
    hight = [1, 1, 0.8, 0.8, 0, 0, -0.2, -0.2, -1, -1, -1.2, -1.2]
    fig20, ax20 = plt.subplots(figsize=figsize)
    sns.lineplot(data=df_hdi_contrast, x='value', y=hight, hue='variable', ax=ax20)
    ax20.vlines(0, -1.2, 1)
    fig20.suptitle('Contrasts for means/variance')

    # Difference in contrast mean/variance
    fig21, ax21 = plt.subplots(figsize=figsize)
    sns.lineplot(data=df_hdi_contrast_diff, x='value', y=hight, hue='variable', ax=ax21)
    ax21.vlines(0, -1.2, 1)
    fig21.suptitle('Difference contrasts for means/variance')

    fig18.savefig(os.getcwd() + f'/images/spkcnt/CNO_InvGamma_{t_scale}/understand/mean_pre_post.svg', format='svg')
    fig19.savefig(os.getcwd() + f'/images/spkcnt/CNO_InvGamma_{t_scale}/understand/var_pre_post.svg', format='svg')
    fig20.savefig(os.getcwd() + f'/images/spkcnt/CNO_InvGamma_{t_scale}/understand/contrasts_mean-var1.svg', format='svg')
    fig21.savefig(os.getcwd() + f'/images/spkcnt/CNO_InvGamma_{t_scale}/understand/diff_contrast_mean-var1.svg', format='svg')


    # Compare results with Gamma distribution ################
    def fxn():
        warnings.warn("deprecated", RuntimeWarning)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fxn()        
        with pm.Model(coords={'condition':df.condtm.factorize()[1], 'aid':df.aid.factorize()[1], 'time':df.time.factorize()[1]}) as catc_model_gamma:
            # Means and variances (set mu_a=1.2, mu_b=1.2, sigma_a=0.2, sigma_b=0.2)
            mu_a = 1.2
            mu_b = 1.2
            sigma_a = 0.2
            sigma_b = 0.2

            # Covariance for time MvNormal
            cov_a_t = []
            cov_b_t = []
            for t in df.time.factorize()[1]:
                dist = abs(df.time.factorize()[1] - t) / tw_width
                cov_a_t.append(np.around(np.exp(-dist * 0.7), decimals=3))
                cov_b_t.append(np.around(np.exp(-dist * 0.7), decimals=3))

            cov_a_t = np.array(cov_a_t) * sigma_a
            cov_b_t = np.array(cov_a_t) * sigma_b

            # Put in dictionary for later
            cov_prior = {'cov_a_t_prior':cov_a_t, 'cov_b_t_prior':cov_b_t}

            # Alpha parameter
            θ_a = pm.Normal('θ_a', mu=mu_a, sigma=sigma_a)
            θ_a_cond = pm.Normal('θ_a_cond', mu=mu_a, sigma=sigma_a, dims='condition')
            θ_a_aid = pm.Normal('θ_a_aid', mu=mu_a, sigma=sigma_a, dims='aid')
            θ_a_time = pm.MvNormal('θ_a_time', mu=mu_a, cov=cov_a_t, dims='time')

            alpha_gamma = pm.math.exp(θ_a + θ_a_cond[df.condtm.factorize()[0]] + θ_a_aid[df.aid.factorize()[0]] + θ_a_time[df.time.factorize()[0]])

            # Beta parameter
            θ_b = pm.Normal('θ_b', mu=mu_b, sigma=sigma_b)
            θ_b_cond = pm.Normal('θ_b_cond', mu=mu_b, sigma=sigma_b, dims='condition')
            θ_b_aid = pm.Normal('θ_b_aid', mu=mu_b, sigma=sigma_b, dims='aid')
            θ_b_time = pm.MvNormal('θ_b_time', mu=mu_b, cov=cov_b_t, dims='time')

            beta_gamma = pm.math.exp(θ_b + θ_b_cond[df.condtm.factorize()[0]] + θ_b_aid[df.aid.factorize()[0]] + θ_b_time[df.time.factorize()[0]])

            # Dependent variable
            y_gamma = pm.Gamma('counts', alpha=alpha_gamma, beta=beta_gamma, observed=df.value)

            # Sample
            catc_priorpc_gamma = pm.sample_prior_predictive(samples=1000, random_seed=rng)
            catc_trace_gamma = pm.sample(idata_kwargs={"log_likelihood": True}, random_seed=rng)
            catc_postpc_gamma = pm.sample_posterior_predictive(catc_trace_gamma, random_seed=rng)


    # az.plot_ppc(catc_priorpc_gamma, group='prior')
    # az.plot_ppc(catc_priorpc, group='prior')
    # az.plot_ppc(catc_postpc_gamma, group='posterior')
    # az.plot_ppc(catc_postpc, group='posterior')
    # az.plot_trace(catc_trace_gamma)
    # az.summary(catc_trace)
    # az.summary(catc_trace_gamma)

    df_sim_gamma = pd.DataFrame(columns=df.columns[[0, 1, 3, 5]])

    for rowdf in range(df.shape[0]):
        # Get xarray subspace
        samp_gamma = catc_trace_gamma.posterior.sel(condition=df.condtm.iloc[rowdf], aid=df.aid.iloc[rowdf], time=df.time.iloc[rowdf])
        # Simulate
        alpha_sim_gamma = np.exp(samp_gamma.θ_a.values + samp_gamma.θ_a_cond.values + samp_gamma.θ_a_aid.values + samp_gamma.θ_a_time.values)
        beta_sim_gamma = np.exp(samp_gamma.θ_b.values + samp_gamma.θ_b_cond.values + samp_gamma.θ_b_aid.values + samp_gamma.θ_b_time.values)
        # sim = gamma.rvs(alpha_sim, scale=1/beta_sim)
        draw = partial(pm.draw, random_seed=rng)
        sim_gamma = draw(pm.Gamma.dist(alpha=alpha_sim_gamma[:,:200].flatten(), beta=beta_sim_gamma[:,:200].flatten()))

        # Combine in dataframe
        __ = pd.DataFrame(dict(condition=df.condition.iloc[rowdf], aid=df.aid.iloc[rowdf], time=df.time.iloc[rowdf], value=sim_gamma, condtm=df.condtm.iloc[rowdf]))
        df_sim_gamma = pd.concat([df_sim_gamma, __])

    # Plot posterior checks Simulation (and max likelihood estimate)
    fig10_1, ax10_1 = plt.subplots(figsize=figsize)
    ax10_1 = sns.histplot(data=df.value, stat='density', color='black', element='step', fill=False, label='observations',bins=100, ax=ax10_1)
    ax10_1 = sns.histplot(data=df_sim_gamma.value, stat='density', color='orange', element='step', fill=False, label='posterior predictive samples', ax=ax10_1)

    # fig10.suptitle('Posterior predictive (aggregated)', size=size, style=style)
    fig10_1.suptitle('Posterior predictive samples', style=style)
    fig10_1.tight_layout()
    ax10_1.legend()
    ax10_1.set_xlabel('scaled spike counts')
    ax10_1.set_xlim([0, 6])
    
    # Save figure 10
    fig10_1.savefig(os.getcwd() + f'/images/spkcnt/CNO_InvGamma_{t_scale}/postchecks_GAMMAtoCompare.svg', format='svg')


    # Compare Gamma vs InvGamma models
    def fxn():
        warnings.warn("deprecated", RuntimeWarning)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fxn()        
        az.loo(catc_trace)          # InvGamma model
        az.loo(catc_trace_gamma)    # Gamma model
        df_comp_loo = az.compare({'InvGamma':catc_trace, 'Gamma':catc_trace_gamma}, ic='loo')
        df_comp_waic = az.compare({'InvGamma':catc_trace, 'Gamma':catc_trace_gamma}, ic='waic')

        comp_loo = az.plot_compare(df_comp_loo, insample_dev=False)
        comp_waic = az.plot_compare(df_comp_waic, insample_dev=False)

    comp_loo.figure.savefig(os.getcwd() + f'/images/spkcnt/CNO_InvGamma_{t_scale}/compare_loo.svg', format='svg', bbox_inches='tight')
    comp_waic.figure.savefig(os.getcwd() + f'/images/spkcnt/CNO_InvGamma_{t_scale}/compare_waic.svg', format='svg',bbox_inches='tight')


def clu_absdiff(spktr_sorted, sortedDepth, cidsSorted, cortex_depth, tw_int, tw_start):
    """
    Compute for each cluster within one dataset absolute difference
    of spike counts between two baselines and between baseline and after-drop
    time interval; tw_width is 5 min to allow two baselines pre-drop;
    time windows of interest (tw_int) are time intervals starting at -10, -5,
    15 and 30 min from CNO/PBS drop; drop clusters with <0.5 Hz fr during any
    tw_int; reference time interval is tw_int[1] (5 min before t_drop);
    also return firing rates (spike count/ 5 min is sec).
    """
    # Select clusters within cortex (e.g. >=1000μm from probe tip)
    ctx_start = np.sum(sortedDepth < cortex_depth)
    spktr_sorted = np.array(spktr_sorted[ctx_start:])
    cidsSorted = np.array(cidsSorted[ctx_start:])

    # Remove trailing time interval if necessary
    spktr_sorted = spktr_sorted[:, :len(tw_start)]

    # Discard clu with <0.5 Hz in tw_int (<150 5 min)
    mask = ~(spktr_sorted[:, tw_int] < 150).any(axis=1)
    spktr_sorted = spktr_sorted[mask, :]
    cidsS = cidsSorted[mask]
    print(len(spktr_sorted))

    # Dataframe clusters firing rate in each 5 min bin
    df_fr = pd.DataFrame(spktr_sorted).iloc[:, tw_int]
    # df_fr.columns = ['-10', '-5', '0', '15', '20', '25']
    df_fr.columns = ['-10', '-5', '5', '10', '15', '20', '25', '30', '35']

    # Scale spikecount by baseline (time of dropping) - Only for diff
    # spktr_sorted = spktr_sorted / np.expand_dims(spktr_sorted[:, tw_int[2]], axis=1)

    # Compute abs differences
    adiff_pre = abs(spktr_sorted[:, tw_int[1]] - spktr_sorted[:, tw_int[0]])
    adiff_post_0 = abs(spktr_sorted[:, tw_int[1]] - spktr_sorted[:, tw_int[2]])
    adiff_post_1 = abs(spktr_sorted[:, tw_int[1]] - spktr_sorted[:, tw_int[3]])
    adiff_post_2 = abs(spktr_sorted[:, tw_int[1]] - spktr_sorted[:, tw_int[4]])
    adiff_post_3 = abs(spktr_sorted[:, tw_int[1]] - spktr_sorted[:, tw_int[5]])
    adiff_post_4 = abs(spktr_sorted[:, tw_int[1]] - spktr_sorted[:, tw_int[6]])
    adiff_post_5 = abs(spktr_sorted[:, tw_int[1]] - spktr_sorted[:, tw_int[7]])
    adiff_post_6 = abs(spktr_sorted[:, tw_int[1]] - spktr_sorted[:, tw_int[8]])


    # Compute differences (No Abs) ATTENTION
    diff_pre = spktr_sorted[:, tw_int[0]] - spktr_sorted[:, tw_int[1]]
    diff_post_0 = spktr_sorted[:, tw_int[2]] - spktr_sorted[:, tw_int[1]]
    diff_post_1 = spktr_sorted[:, tw_int[3]] - spktr_sorted[:, tw_int[1]]
    diff_post_2 = spktr_sorted[:, tw_int[4]] - spktr_sorted[:, tw_int[1]]
    diff_post_3 = spktr_sorted[:, tw_int[5]] - spktr_sorted[:, tw_int[1]]
    diff_post_4 = spktr_sorted[:, tw_int[6]] - spktr_sorted[:, tw_int[1]]
    diff_post_5 = spktr_sorted[:, tw_int[7]] - spktr_sorted[:, tw_int[1]]
    diff_post_6 = spktr_sorted[:, tw_int[8]] - spktr_sorted[:, tw_int[1]]

    # Organise in dataframe
    # adiff = pd.DataFrame(np.array([adiff_pre, adiff_post_0, adiff_post_1, adiff_post_2, adiff_post_3]).T, columns=['-10', '0', '15', '20', '25'])
    adiff = pd.DataFrame(np.array([adiff_pre, adiff_post_0, adiff_post_1, adiff_post_2, adiff_post_3, adiff_post_4, adiff_post_5, adiff_post_6]).T, columns=['-10', '5', '10', '15', '20', '25', '30', '35'])
    # diff = pd.DataFrame(np.array([diff_pre, diff_post_0, diff_post_1, diff_post_2, diff_post_3]).T, columns=['-10', '0', '15', '20', '25'])
    diff = pd.DataFrame(np.array([diff_pre, diff_post_0, diff_post_1, diff_post_2, diff_post_3, diff_post_4, diff_post_5, diff_post_6]).T, columns=['-10', '5', '10', '15', '20', '25', '30', '35'])

    return adiff, diff, df_fr, cidsS


def get_cludata(npxpath,
                aid,
                cortex_depth,
                t_drop,
                conditions,
                tw_start,
                tw_width,
                t_scale='-5',
                cgs=2):
    """ Same as plot_cluanalysis in spatiotemporal_visualisation module
    """
    # Sanity check paths, depths and t_drops
    if (len(npxpath) != len(cortex_depth)) or (len(npxpath) != len(t_drop)):
        print('Attention: data length is different')
        exit()

    # Initialise abs difference and fr dataframes
    df_adiff = pd.DataFrame()
    df_diff = pd.DataFrame()
    df_fr = pd.DataFrame()
    df_fr_scl = pd.DataFrame()

    # Get npx data
    for idx, dataset in enumerate(npxpath):
        spk = lnpx.loadKsDir(dataset)
        spk = lnpx.process_spiketimes(spk)

        # Time window of interest
        # if t_drop[idx] == 10:
        #     tw_int = np.array([0, 1, 2, 5, 6, 7])
        # elif t_drop[idx] == 20:
        #     tw_int = np.array([2, 3, 4, 7, 8, 9])
        if t_drop[idx] == 10:
            tw_int = np.array([0, 1, 3, 4, 5, 6, 7, 8, 9])
        elif t_drop[idx] == 20:
            tw_int = np.array([2, 3, 5, 6, 7, 8, 9, 10, 11])


        # Compute spike trains for each cluster (sorted by depth)
        spktr_sorted = spk_train_nw(spk, binl=tw_width * 1000 * 60, surr=False, cgs=cgs)

        # Compute spkcnt abs difference for each cluster
        adiff, diff, fr, cidsS = clu_absdiff(spktr_sorted, spk.sortedDepth, spk.cidsSorted[spk.cgsSorted == cgs], cortex_depth[idx], tw_int, tw_start=tw_start)

        # Scale fr by baseline before drop
        # fr_scl = fr.div(fr['-5'], axis=0)
        fr_scl = fr.div(fr[t_scale], axis=0)

        # Add description
        adiff['condition'] = conditions[idx]
        adiff['aid'] = aid[idx]
        adiff['idx_rec'] = idx

        diff['condition'] = conditions[idx]
        diff['aid'] = aid[idx]
        diff['idx_rec'] = idx

        fr['condition'] = conditions[idx]
        fr['aid'] = aid[idx]
        fr['idx_rec'] = idx

        fr_scl['condition'] = conditions[idx]
        fr_scl['aid'] = aid[idx]
        fr_scl['idx_rec'] = idx

        # Concatenate recordings
        df_adiff = pd.concat([df_adiff, adiff])
        df_diff = pd.concat([df_diff, diff])
        df_fr = pd.concat([df_fr, fr])
        df_fr_scl = pd.concat([df_fr_scl, fr_scl])

    # Melt df (long format)
    df_adiff_m = df_adiff.melt(id_vars=['condition', 'aid', 'idx_rec'])
    df_diff_m = df_diff.melt(id_vars=['condition', 'aid', 'idx_rec'])
    df_fr_m = df_fr.melt(id_vars=['condition', 'aid', 'idx_rec'])
    df_fr_scl_m = df_fr_scl.melt(id_vars=['condition', 'aid', 'idx_rec'])

    # Rename variable to time and convert to int
    df_adiff_m.rename(columns={'variable':'time'}, inplace=True)
    df_diff_m.rename(columns={'variable':'time'}, inplace=True)
    df_fr_m.rename(columns={'variable':'time'}, inplace=True)
    df_fr_scl_m.rename(columns={'variable':'time'}, inplace=True)

    df_adiff_m.time = df_adiff_m.time.astype(int)
    df_diff_m.time = df_diff_m.time.astype(int)
    df_fr_m.time = df_fr_m.time.astype(int)
    df_fr_scl_m.time = df_fr_scl_m.time.astype(int)

    return df_fr_m, df_fr_scl_m, df_adiff_m, df_diff_m


def ml_fr_fit(df, tw_width):
    """ Plot firing rates and find maximum likelihood estimate of Gamma 
    parameters
    """
    # Get firing rates
    df['fr'] = df.loc[:, 'value'].div(tw_width * 60)

    # Fit Gamma
    # df_fr = pd.DataFrame()
    df_gml = pd.DataFrame()    
    for c in df.condition.unique():
        for t in df.time.unique():
            # shape, loc, scale = gamma.fit(df[(df.condition==c) & (df.time==t)].fr.values)
            shape, loc, scale = gamma.fit(df[(df.condition==c) & (df.time==t)].value.values)
            __ = pd.DataFrame(data={'condition':c, 'time':t, 'shape':[shape], 'loc':[loc], 'scale':[scale]})
            # df_fr = pd.concat([df_fr, __])
            df_gml = pd.concat([df_gml, __])

    df_gml_l = df_gml.melt(id_vars=['condition', 'time'])

    # Fit Gamma (maximum likelihood)
    fig1, ax1 = plt.subplots(3, 1, figsize=(12, 12))
    sns.lineplot(x='time', y='value', data=df_gml_l[df_gml_l.variable=='shape'], hue='condition', ax=ax1[0])
    sns.lineplot(x='time', y='value', data=df_gml_l[df_gml_l.variable=='loc'], hue='condition', ax=ax1[1])
    sns.lineplot(x='time', y='value', data=df_gml_l[df_gml_l.variable=='scale'], hue='condition', ax=ax1[2])

    fig2, ax2 = plt.subplots(figsize=(12, 12))
    sns.lineplot(x='time', y='value', data=df_gml_l[df_gml_l.variable=='shape'], hue='condition', ax=ax1[0])
    sns.lineplot(x='time', y='value', data=df_gml_l[df_gml_l.variable=='loc'], hue='condition', ax=ax1[1])
    sns.lineplot(x='time', y='value', data=df_gml_l[df_gml_l.variable=='scale'], hue='condition', ax=ax1[2])

    df_gml_sim = pd.DataFrame()
    for c in range(df_gml.shape[0]):
        __ = gamma.rvs(df_gml['shape'].iloc[c], loc=df_gml['loc'].iloc[c], scale=df_gml['scale'].iloc[c], size=100)
        df_gml = pd.concat([df_gml, __])


def fit_spkcount_clu(npxpath,
                     aid,
                     cortex_depth,
                     t_drop,
                     conditions,
                     tw_start,
                     tw_width,
                     t_scale='-5',
                     cgs=2):
    ''' Compute posterior for model of cluster spikecounts '''
    # Load dataframes...
    # with open(os.path.join(os.getcwd(), 'data_analysis/df_spkcnt_scl_clu_1hz')) as csv_file:
    #     df = pd.read_csv(csv_file)
    # with open(os.path.join(os.getcwd(), 'data_analysis/df_spkcnt_clu_1hz')) as csv_file:
    #     df = pd.read_csv(csv_file)

    # ... or calculate new
    t_scale='-5'
    df_uscl, df_scl, df_adiff_m, df_diff_m = get_cludata(npxpath,
                                                         aid,
                                                         cortex_depth,
                                                         t_drop,
                                                         conditions,
                                                         tw_start,
                                                         tw_width,
                                                         t_scale=t_scale,
                                                         cgs=2)

    # Choose scaled or unscaled
    df = df_scl

    # Set Random seed
    RANDOM_SEED = 100
    rng = np.random.default_rng(RANDOM_SEED)

    # Drop -5 time, reset index
    # df = df[df.time!=-5]
    # df.reset_index(drop=True, inplace=True)

    df = df[df.time!=-5]
    df.reset_index(drop=True, inplace=True)
    df = df[df.time!=35]        # leave 15, 20, 25
    df.reset_index(drop=True, inplace=True)

    
    # Create condtm from condition and time; reorder
    # df.loc[:, 'condtm'] = df.condition + df.time.apply(lambda x: 'post' if x>=0 else 'pre')
    # df.loc[:, 'cait'] = df.condition + '_' + df.aid.astype(str) + '_' + df.idx_rec.astype(str) + '_' + df.time.astype(str) + '_' + df.condtm
    df.loc[:, 'condtm'] = df.loc[:, 'condition'] + df.loc[:, 'time'].apply(lambda x: 'post' if x>=0 else 'pre')
    df.loc[:, 'cait'] = df.loc[:, 'condition'] + '_' + df.loc[:, 'aid'].astype(str) + '_' + df.loc[:, 'idx_rec'].astype(str) + '_' + df.loc[:, 'time'].astype(str) + '_' + df.loc[:, 'condtm']
    df = df.iloc[:, [0, 1, 2, 3, 5, 6, 4]]

    # Create df with unique combinations of conditions
    df_unique = pd.DataFrame({'all_':df.cait.unique()})
    df_unique['condition'] = df_unique.all_.str.split('_').str[0]
    df_unique['aid'] = df_unique.all_.str.split('_').str[1].astype(int)
    df_unique['idx_rec'] = df_unique.all_.str.split('_').str[2].astype(int)
    df_unique['time'] = df_unique.all_.str.split('_').str[3].astype(int)
    df_unique['condtm'] = df_unique.all_.str.split('_').str[4]

    # Log transform data
    df.value = np.log(df.value)
    
    # Z-score data by condition
    df.loc[:, 'value'] = df.loc[:, 'value'] / df.loc[:, 'value'].std()
    df.loc[:, 'value'] = zscore(df.loc[:, 'value'])
    for cond in df.condition.unique():
        df.loc[df.condition==cond, 'value'] = zscore(df.loc[df.condition==cond, 'value'])
        # df.value[df.condition==cond] = zscore(df.value[df.condition==cond])


    # Model on Scaled data - Gamma_Halfnormal_explinkfun_Normal
    # with pm.Model(coords={'condition':df_unique.condtm.factorize()[1], 'aid':df_unique.aid.factorize()[1], 'time':df_unique.time.factorize()[1], 'cait':df.cait.factorize()[1]}) as catc_model:
    # with pm.Model(coords={'condition':df.condtm.factorize()[1], 'aid':df.aid.factorize()[1], 'time':df.time.factorize()[1], 'cait':df.cait.factorize()[1]}) as catc_model:
    with pm.Model(coords={'condition':df.condtm.factorize()[1], 'time':df.time.factorize()[1], 'cait':df.cait.factorize()[1]}) as catc_model:
        # HALFNORMAL
        # Hyperparameters for θ_μ
        μ_θ_μ = 0.0
        σ_θ_μ = 0.5
        # Hyperparameters for θ_σ
        μ_θ_σ = 0.0
        σ_θ_σ = 1.5
        ## HALFCAUCHY
        # # Hyperparameters for θ_μ
        # μ_θ_μ = 0.0
        # σ_θ_μ = 0.5
        # # Hyperparameters for θ_σ
        # μ_θ_σ = 0.0
        # σ_θ_σ = 1

        # Covariance for time MvNormal
        cov_μ_t = []
        cov_σ_t = []
        for t in df.time.factorize()[1]:
            dist = abs(df.time.factorize()[1] - t) / tw_width
            cov_μ_t.append(np.around(np.exp(-dist * 0.7), decimals=3))
            cov_σ_t.append(np.around(np.exp(-dist * 0.7), decimals=3))

        cov_μ_t = np.array(cov_μ_t) * σ_θ_μ
        cov_σ_t = np.array(cov_σ_t) * σ_θ_σ

        # Put in dictionary for later
        cov_prior = {'cov_μ_t_prior':cov_μ_t, 'cov_σ_t_prior':cov_σ_t}


        # Prior linear model for mean parameter μ
        θ_μ = pm.Normal('θ_μ', mu=μ_θ_μ, sigma=σ_θ_μ)
        θ_μ_cond = pm.Normal('θ_μ_cond', mu=μ_θ_μ, sigma=σ_θ_μ, dims='condition')
        # θ_μ_aid = pm.Normal('θ_μ_aid', mu=μ_θ_μ, sigma=σ_θ_μ, dims='aid')
        # θ_μ_time = pm.Normal('θ_μ_time', mu=μ_θ_μ, sigma=σ_θ_μ, dims='time')
        θ_μ_time = pm.MvNormal('θ_μ_time', mu=μ_θ_μ, cov=cov_μ_t, dims='time')

        # Prior linear model for precision parameter τ
        θ_σ = pm.Normal('θ_σ', mu=μ_θ_σ, sigma=σ_θ_σ)
        θ_σ_cond = pm.Normal('θ_σ_cond', mu=μ_θ_σ, sigma=σ_θ_σ, dims='condition')
        # θ_σ_aid = pm.Normal('θ_σ_aid', mu=μ_θ_σ, sigma=σ_θ_σ, dims='aid')
        # θ_σ_time = pm.Normal('θ_σ_time', mu=μ_θ_σ, sigma=σ_θ_σ, dims='time')
        θ_σ_time = pm.MvNormal('θ_σ_time', mu=μ_θ_σ, cov=cov_σ_t, dims='time')

        # Log link function for mean parameter μ
        # τ_μ = pm.Deterministic('τ_μ', pm.math.exp(θ_μ + θ_μ_cond[df.condtm.factorize()[0]] + θ_μ_aid[df.aid.factorize()[0]] + θ_μ_time[df.time.factorize()[0]]))
        τ_μ = pm.Deterministic('τ_μ', pm.math.exp(θ_μ + θ_μ_cond[df.condtm.factorize()[0]] + θ_μ_time[df.time.factorize()[0]]))

        # Log link function for scale parameter σ
        # τ_σ = pm.Deterministic('τ_σ', pm.math.exp(θ_σ + θ_σ_cond[df.condtm.factorize()[0]] + θ_σ_aid[df.aid.factorize()[0]] + θ_σ_time[df.time.factorize()[0]]))
        τ_σ = pm.Deterministic('τ_σ', pm.math.exp(θ_σ + θ_σ_cond[df.condtm.factorize()[0]] + θ_σ_time[df.time.factorize()[0]]))

        μ = pm.HalfNormal('μ', tau=τ_μ)
        σ = pm.HalfNormal('σ', tau=τ_σ)
        # μ = pm.HalfCauchy('μ', beta=τ_μ)
        # σ = pm.HalfCauchy('σ', beta=τ_σ)

        # # Non-centered distribution
        # μ_std = pm.HalfNormal('μ_std', tau=1)
        # σ_std = pm.HalfNormal('σ_std', tau=1)
        
        # μ = pm.Deterministic('μ', μ_std / τ_μ)
        # σ = pm.Deterministic('σ', σ_std / τ_σ)
        

        # SkewNormal for log-trasformed/Z-scored data
        y = pm.Gamma('y', mu=μ[df.cait.factorize()[0]], sigma=σ[df.cait.factorize()[0]], observed=df.value)

        # Sample
        catc_priorpc = pm.sample_prior_predictive(samples=1000, random_seed=rng)
        catc_trace = pm.sample(tune=2000)

    with catc_model:
        catc_postpc = pm.sample_posterior_predictive(catc_trace, random_seed=rng)

    az.plot_ppc(catc_priorpc, group='prior', observed=True)
    # Show model
    # pm.model_to_graphviz(catc_model).render(view=True)
    # pm.model_to_graphviz(catc_model).render(directory=os.path.join(os.getcwd(), 'data_analysis/images/st_analysis/CNO/Fitting_spkCount/Net_SkewNormal_Zscored'))

    fig, ax = plt.subplots(6)
    ax[0].hist(τ_μ.eval(), bins=100, color='brown', label='τ_μ')
    ax[1].hist(τ_σ.eval(), bins=100, color='cyan', label='τ_σ')
    ax[2].hist(μ.eval(), bins=100, color='red', label='mean')
    ax[3].hist(σ.eval(), bins=100, color='yellow', label='sigma')
    # ax[4].hist(α.eval()/β.eval(), bins=100, color='cyan', label='mean Gamma')
    # ax[5].hist(α.eval()/(β.eval()**2), bins=100, color='cyan', label='var Gamma')
    ax[4].hist(pm.Gamma.dist(mu=μ.eval(), sigma=σ.eval()).eval(), bins=100, log=True, color='blue', label='sim_data')
    ax[5].hist(df.value, bins=100, log=True, color='brown', label='data')
    fig.legend()
    fig.tight_layout
    # sns.histplot(y.eval(), ax=ax[2])
    # sns.histplot(df.value.apply(lambda x: x/(60 * 5)), ax=ax[3])
    # sns.histplot(beta.eval()/(alpha.eval()-1), ax=ax[3])
    # plt.tight_layout()
    plt.show()












    # UNSCALED DATA - WITH POISSON -Gamma
    # with pm.Model(coords={'condition':df_unique.condtm.factorize()[1], 'aid':df_unique.aid.factorize()[1], 'time':df_unique.time.factorize()[1], 'cait':df.cait.factorize()[1]}) as catc_model:
    # with pm.Model(coords={'condition':df.condtm.factorize()[1], 'aid':df.aid.factorize()[1], 'time':df.time.factorize()[1], 'cait':df.cait.factorize()[1]}) as catc_model:
    with pm.Model(coords={'condition':df.condtm.factorize()[1], 'time':df.time.factorize()[1], 'cait':df.cait.factorize()[1]}) as catc_model:

        # Hyperpriors
        λ_α = .1
        λ_β = 1

        # Shape parameter α
        θ_α = pm.Exponential('θ_α', lam=λ_α)
        θ_α_cond = pm.Exponential('θ_α_cond', lam=λ_α, dims='condition')
        # θ_α_aid = pm.Exponential('θ_α_aid', lam=λ_α, dims='aid')
        θ_α_time = pm.Exponential('θ_α_time', lam=λ_α, dims='time')

        # α = pm.math.exp(θ_a + θ_a_cond[df_unique.condtm.factorize()[0]] + θ_a_aid[df_unique.aid.factorize()[0]] + θ_a_time[df_unique.time.factorize()[0]])
        # α = pm.Deterministic('α', θ_α + θ_α_cond[df.condtm.factorize()[0]] + θ_α_aid[df.aid.factorize()[0]] + θ_α_time[df.time.factorize()[0]])
        α = pm.Deterministic('α', θ_α + θ_α_cond[df.condtm.factorize()[0]] + θ_α_time[df.time.factorize()[0]])
            
        # Scale parameter β
        θ_β = pm.Exponential('θ_β', lam=λ_β)
        θ_β_cond = pm.Exponential('θ_β_cond', lam=λ_β, dims='condition')
        # θ_β_aid = pm.Exponential('θ_β_aid', lam=λ_β, dims='aid')
        θ_β_time = pm.Exponential('θ_β_time', lam=λ_β, dims='time')
        
        # β = pm.math.exp(θ_β + θ_β_cond[df_unique.condtm.factorize()[0]] + θ_β_aid[df_unique.aid.factorize()[0]] + θ_β_time[df_unique.time.factorize()[0]])
        # β = pm.Deterministic('β', θ_β + θ_β_cond[df.condtm.factorize()[0]] + θ_β_aid[df.aid.factorize()[0]] + θ_β_time[df.time.factorize()[0]])
        β = pm.Deterministic('β', θ_β + θ_β_cond[df.condtm.factorize()[0]] + θ_β_time[df.time.factorize()[0]])

        # Inverse gamma distribution of firing rates
        fr = pm.Gamma('fr', alpha=α, beta=β)

        # Poisson distribution of spike counts
        # y = pm.Poisson('counts', mu=fr[df.cait.factorize()[0]] * 60 * 5, observed=df.value)
        y = pm.Poisson('counts', mu=fr * 60 * 5, observed=df.value)

        # Sample
        catc_priorpc = pm.sample_prior_predictive(samples=1000, random_seed=rng)
        catc_trace = pm.sample()

    with catc_model:
        catc_postpc = pm.sample_posterior_predictive(catc_trace, random_seed=rng)

    az.plot_ppc(catc_priorpc, group='prior', observed=True)
    # Show model
    pm.model_to_graphviz(catc_model).render(view=True)
    pm.model_to_graphviz(catc_model).render(directory=os.path.join(os.getcwd(), 'data_analysis/images/st_analysis/CNO/Fitting_spkCount/spkcnt_GammaPoisson'))

    fig, ax = plt.subplots(6)
    ax[0].hist(α.eval(), bins=100, color='red', label='α')
    ax[1].hist(β.eval(), bins=100, color='yellow', label='β')
    ax[2].hist(fr.eval(), bins=100, color='blue', label='fr')
    ax[3].hist(fr.eval() * 60 * 5, color='green', label='fr')
    ax[4].hist(pm.Poisson.dist(mu=fr.eval() * 60 * 5).eval(), color='magenta', label='spkcnt', bins=100)
    ax[5].hist(df.value, color='cyan', label='data', bins=100)
    fig.legend()
    fig.tight_layout
    plt.show()



    # # SCALED MODEL - GAMMA/INVGAMMA
    # # with pm.Model(coords={'condition':df_unique.condtm.factorize()[1], 'aid':df_unique.aid.factorize()[1], 'time':df_unique.time.factorize()[1], 'cait':df.cait.factorize()[1]}) as catc_model:

    # with pm.Model() as trymodel:
    #     lam = 20

    #     a = pm.Exponential('a', lam=lam)
    #     b = pm.Exponential('b', lam=lam)
    #     c = pm.Exponential('c', lam=lam)
    #     d = pm.Exponential('d', lam=lam)

    #     β_α = pm.Deterministic('β_α', a + b + c + d, size=1000)
    #     β_β = pm.Deterministic('β_β', a + b + c + d, size=1000)

    #     alpha = pm.HalfCauchy('alpha', beta=β_α, size=1000)
    #     beta = pm.HalfCauchy('beta', beta=β_β, size=1000)

    #     y = pm.Gamma('y', alpha=alpha, beta=beta, observed=pm.Gamma.dist(alpha=3, beta=1, size=1000).eval())

    #     ppc_try = pm.sample_prior_predictive(samples=1000, random_seed=rng)


    # pm.model_to_graphviz(trymodel).render(view=True)
    # az.plot_ppc(ppc_try, group='prior', observed=True)

    # fig, ax = plt.subplots(4)
    # sns.histplot(alpha.eval(), bins=100, color='yellow', label='α', ax=ax[0])
    # sns.histplot(beta.eval(), bins=100, color='blue', label='β', ax=ax[1])
    # sns.histplot(pm.Gamma.dist(alpha=α.eval(), beta=β.eval()).eval(), bins=100, color='green', label='spkcnt_sim', ax=ax[2])
    # sns.histplot(pm.Gamma.dist(alpha=3, beta=1, size=1000).eval(), bins=100, color='green', label='spkcnt', ax=ax[3])
    # # sns.histplot(df.value.apply(lambda x: x/(60 * 5)), ax=ax[3])
    # # sns.histplot(beta.eval()/(alpha.eval()-1), ax=ax[3])
    # fig.legend()
    # plt.show()
                
    # with pm.Model(coords={'condition':df.condtm.factorize()[1], 'aid':df.aid.factorize()[1], 'time':df.time.factorize()[1], 'cait':df.cait.factorize()[1]}) as catc_model:

    #     λ_α = 20000
    #     λ_β = 20000

    #     # θ_α = pm.HalfNormal('θ_α', sigma=τ_α)
    #     # θ_α_cond = pm.HalfNormal('θ_α_cond', sigma=τ_α, dims='condition')
    #     # θ_α_aid = pm.HalfNormal('θ_α_aid', sigma=τ_α, dims='aid')
    #     # θ_α_time = pm.HalfNormal('θ_α_time',  sigma=τ_α, dims='time')
        
    #     # θ_β = pm.HalfNormal('θ_β', sigma=τ_β)
    #     # θ_β_cond = pm.HalfNormal('θ_β_cond', sigma=τ_β, dims='condition')
    #     # θ_β_aid = pm.HalfNormal('θ_β_aid', sigma=τ_β, dims='aid')
    #     # θ_β_time = pm.HalfNormal('θ_β_time', sigma=τ_β, dims='time')


        
    #     # θ_α = pm.HalfNormal('θ_α', tau=λ_α, dims='condition')
    #     # θ_β = pm.HalfNormal('θ_β', tau=λ_β, dims='condition')
    #     θ_α = pm.HalfNormal('θ_α', tau=λ_α)
    #     θ_β = pm.HalfNormal('θ_β', tau=λ_β)
                
        
    #     # β_α = pm.Deterministic('β_α', θ_α + θ_α_cond[df_unique.condtm.factorize()[0]] + θ_α_aid[df_unique.aid.factorize()[0]] + θ_α_time[df_unique.time.factorize()[0]])
    #     # β_β = pm.Deterministic('β_β', θ_β + θ_β_cond[df_unique.condtm.factorize()[0]] + θ_β_aid[df_unique.aid.factorize()[0]] + θ_β_time[df_unique.time.factorize()[0]])
    #     # β_α = pm.Deterministic('β_α', θ_α + θ_α_cond[df.condtm.factorize()[0]] + θ_α_aid[df.aid.factorize()[0]] + θ_α_time[df.time.factorize()[0]])
    #     # β_β = pm.Deterministic('β_β', θ_β + θ_β_cond[df.condtm.factorize()[0]] + θ_β_aid[df.aid.factorize()[0]] + θ_β_time[df.time.factorize()[0]])
    #     # τ_α = pm.Deterministic('τ_α', θ_α[df.condtm.factorize()[0]])
    #     # τ_β = pm.Deterministic('τ_β', θ_β[df.condtm.factorize()[0]])
    #     # τ_α = θ_α[df.condtm.factorize()[0]]
    #     # τ_β = θ_β[df.condtm.factorize()[0]]
    #     τ_α = θ_α
    #     τ_β = θ_β



    #     # InverseGamma priors
    #     # α = pm.HalfCauchy('α', beta=β_α[df.condtm.factorize()[0]])
    #     # β = pm.HalfCauchy('β', beta=β_β[df.condtm.factorize()[0]])
    #     α = pm.HalfNormal('α', tau=τ_α)
    #     β = pm.HalfNormal('β', tau=τ_β)

    #     y = pm.Gamma('y', alpha=α, beta=β, observed=df.value)
    
    #     # Sample
    #     catc_priorpc = pm.sample_prior_predictive(samples=1000, random_seed=rng)
    #     # catc_trace = pm.sample()

    # with catc_model:
    #     catc_postpc = pm.sample_posterior_predictive(catc_trace, random_seed=rng)


    # # Show model
    # pm.model_to_graphviz(catc_model).render(view=True)
    # # pm.model_to_graphviz(catc_model).render(directory=os.path.join(os.getcwd(), 'data_analysis/images/st_analysis/CNO/Fitting_spkCount/spkcnt_cluNet_Gamma'))
    

    # az.plot_ppc(catc_priorpc, group='prior', observed=True)
    
    # fig, ax = plt.subplots(4)
    # sns.histplot(α.eval(), bins=100, color='yellow', label='α', ax=ax[0])
    # sns.histplot(β.eval(), bins=100, color='blue', label='β', ax=ax[1])
    # sns.histplot(pm.Gamma.dist(alpha=α.eval(), beta=β.eval()).eval(), bins=100, color='green', label='spkcnt_sim', ax=ax[2])
    # sns.histplot(df.value, bins=100, color='green', label='spkcnt', ax=ax[3])
    # # sns.histplot(df.value.apply(lambda x: x/(60 * 5)), ax=ax[3])
    # # sns.histplot(beta.eval()/(alpha.eval()-1), ax=ax[3])
    # fig.legend()
    # plt.show()







    

    # Fit Difference data (z-scored)
    # with pm.Model(coords={'condition':df_unique.condtm.factorize()[1], 'aid':df_unique.aid.factorize()[1], 'time':df_unique.time.factorize()[1], 'cait':df.cait.factorize()[1]}) as catc_model:
    with pm.Model(coords={'condition':df.condtm.factorize()[1], 'aid':df.aid.factorize()[1], 'time':df.time.factorize()[1], 'cait':df.cait.factorize()[1]}) as catc_model:
    # with pm.Model(coords={'condition':df.condtm.factorize()[1], 'time':df.time.factorize()[1], 'cait':df.cait.factorize()[1]}) as catc_model:

        # Means and variances (set mu_a=1.2, mu_b=1.2, sigma_a=0.2, sigma_b=0.2)
        # mu_a = 0.5
        # mu_b = 0.2
        # sigma_a = 0.1
        # sigma_b = 0.1
        # mu_a = 0.9
        # mu_b = 0.2
        # sigma_a = 0.3
        # sigma_b = 0.3


        # Hyperparameters for θ's
        μ_θ_μ = 0
        σ_θ_μ = 0.5
        β_θ_σ = 0.5

        # Covariance for time MvNormal
        # cov_a_t = []
        # cov_b_t = []
        # for t in df.time.factorize()[1]:
        #     dist = abs(df.time.factorize()[1] - t) / tw_width
        #     cov_a_t.append(np.around(np.exp(-dist * 0.7), decimals=3))
        #     cov_b_t.append(np.around(np.exp(-dist * 0.7), decimals=3))

        # cov_a_t = np.array(cov_a_t) * sigma_a
        # cov_b_t = np.array(cov_b_t) * sigma_b

        # # Put in dictionary for later
        # cov_prior = {'cov_a_t_prior':cov_a_t, 'cov_b_t_prior':cov_b_t}

        θ_μ = pm.Normal('θ_μ', mu=μ_θ_μ, sigma=σ_θ_μ)
        θ_μ_cond = pm.Normal('θ_μ_cond', mu=μ_θ_μ, sigma=σ_θ_μ, dims='condition')
        θ_μ_aid = pm.Normal('θ_μ_aid', mu=μ_θ_μ, sigma=σ_θ_μ, dims='aid')
        θ_μ_time = pm.Normal('θ_μ_time', mu=μ_θ_μ, sigma=σ_θ_μ, dims='time')

        
        # Mean parameter μ
        μ = pm.Deterministic('μ', θ_μ + θ_μ_cond[df.condtm.factorize()[0]] + θ_μ_aid[df.aid.factorize()[0]] + θ_μ_time[df.time.factorize()[0]])
        # μ = pm.Deterministic('μ', θ_μ + θ_μ_cond[df.condtm.factorize()[0]] + θ_μ_time[df.time.factorize()[0]])

        # Hyperpriors σ
        θ_σ = pm.HalfCauchy('θ_σ', beta=β_θ_σ)
        θ_σ_cond = pm.HalfCauchy('θ_σ_cond', beta=β_θ_σ, dims='condition')
        θ_σ_aid = pm.HalfCauchy('θ_σ_aid', beta=β_θ_σ, dims='aid')
        θ_σ_time = pm.HalfCauchy('θ_σ_time', beta=β_θ_σ, dims='time')

        # Variance parameter σ
        σ = pm.Deterministic('σ', θ_σ + θ_σ_cond[df.condtm.factorize()[0]] + θ_σ_aid[df.aid.factorize()[0]] + θ_σ_time[df.time.factorize()[0]])
        # σ = pm.Deterministic('σ', θ_σ + θ_σ_cond[df.condtm.factorize()[0]] + θ_σ_time[df.time.factorize()[0]])

        # Inverse gamma distribution of firing rates
        y = pm.Normal('y', mu=μ, sigma=σ, observed=df.value)
    
        # Sample
        catc_priorpc = pm.sample_prior_predictive(samples=1000, random_seed=rng)
        catc_trace = pm.sample()

    with catc_model:
        catc_postpc = pm.sample_posterior_predictive(catc_trace, random_seed=rng)


    # Show model
    pm.model_to_graphviz(catc_model).render(view=True)
    # pm.model_to_graphviz(catc_model).render(directory=os.path.join(os.getcwd(), 'data_analysis/images/st_analysis/CNO/Fitting_spkCount/spkcnt_cluNet_Gamma'))
    

    az.plot_ppc(catc_priorpc, group='prior', observed=True)
    sns.histplot(df.value)

    fig, ax = plt.subplots(4)
    sns.histplot(μ.eval(), bins=50, color='yellow', label='μ', ax=ax[0])
    sns.histplot(σ.eval(), bins=50, color='blue', label='σ', ax=ax[1])
    sns.histplot(pm.Normal.dist(mu=μ.eval(), sigma=σ.eval()).eval(), bins=100, color='green', label='spkcnt_sim', ax=ax[2])
    sns.histplot(df.value, bins=100, color='green', label='spkcnt', ax=ax[3])
    # sns.histplot(df.value.apply(lambda x: x/(60 * 5)), ax=ax[3])
    # sns.histplot(beta.eval()/(alpha.eval()-1), ax=ax[3])
    fig.legend()
    plt.show()


        
    # Posterior of difference (contrast) for condition parameters
    df_diff = pd.DataFrame()
    for c_ in itertools.combinations(df.condtm.unique(), 2):
        if (c_[0].find('pre') != c_[1].find('pre')) & (c_[0][:2] == c_[1][:2]):
        # if (c_[0].find('pre') == c_[1].find('pre')):
            # post_diff_α = catc_trace.posterior.θ_α_cond.sel(condition=c_[0]).values.flatten() - catc_trace.posterior.θ_α_cond.sel(condition=c_[1]).values.flatten()
            # post_diff_β = catc_trace.posterior.θ_β_cond.sel(condition=c_[0]).values.flatten() - catc_trace.posterior.θ_β_cond.sel(condition=c_[1]).values.flatten()
            # post_diff_tau = catc_trace.posterior.θ_tau_cond.sel(condition=c_[0]).values.flatten() - catc_trace.posterior.θ_tau_cond.sel(condition=c_[1]).values.flatten()
            post_diff_μ = catc_trace.posterior.θ_μ_cond.sel(condition=c_[0]).values.flatten() - catc_trace.posterior.θ_μ_cond.sel(condition=c_[1]).values.flatten()
            post_diff_σ = catc_trace.posterior.θ_σ_cond.sel(condition=c_[0]).values.flatten() - catc_trace.posterior.θ_σ_cond.sel(condition=c_[1]).values.flatten()

            # post_diff_α = pd.DataFrame({'parameter':'θ_α_cond', f'{c_[0]} - {c_[1]}':post_diff_α})
            # post_diff_β = pd.DataFrame({'parameter':'θ_β_cond', f'{c_[0]} - {c_[1]}':post_diff_β})
            # post_diff_tau = pd.DataFrame({'parameter':'tau', f'{c_[0]} - {c_[1]}':post_diff_tau})
            post_diff_μ = pd.DataFrame({'parameter':'μ', f'{c_[0]} - {c_[1]}':post_diff_μ})
            post_diff_σ = pd.DataFrame({'parameter':'σ', f'{c_[0]} - {c_[1]}':post_diff_σ})

            # __ = pd.concat([post_diff_α, post_diff_β])
            # __ = pd.concat([post_diff_a, post_diff_tau, post_diff_m])
            __ = pd.concat([post_diff_μ, post_diff_σ])
            # __ = pd.concat([post_diff_l])

            df_diff = pd.concat([df_diff, __], axis=1)

    df_diff = df_diff.loc[:, ~df_diff.columns.duplicated()]
    df_diff = df_diff.melt(id_vars='parameter')

    # Get hdi for contrasts (default stats.hdi_prob rcparams 0.94)
    midx = pd.MultiIndex.from_product([df_diff.variable.unique(), df_diff.parameter.unique()], names=['parameter', 'variable'])
    hdi_diff = pd.DataFrame(index=midx, columns=['value'])
    for par in df_diff.parameter.unique():
        for var in df_diff.variable.unique():
            __ = az.hdi(df_diff[(df_diff.parameter==par) & (df_diff.variable==var)].value.values)
            # hdi_diff.loc[(par, var)].value1 = __[0]
            # hdi_diff.loc[(par, var)].value2 = __[1]
            hdi_diff.loc[(var, par)] = [__]

    # Chek posterior covariance
    cov_post = {}
    for par in ['μ', 'σ']:
        __ = pd.DataFrame()
        for chain in range(4):
            __ = pd.concat([__, pd.DataFrame(catc_trace.posterior.θ_σ_time[chain, :, :].values)])
        cov_post[f'cov_{par}_t_post'] = np.cov(__.T)


    # Simulate data
    # df_sim = pd.DataFrame(columns=df.columns[[0, 1, 3, 4]])
    df_sim = pd.DataFrame(columns=df.columns[[0, 3, 4]])
    for rowdf in range(df_unique.shape[0]):
        # Get xarray subspace
        # samp = catc_trace.posterior.sel(condition=df.condition.iloc[rowdf], aid=df.aid.iloc[rowdf], time=df.time.iloc[rowdf])
        # samp = catc_trace.posterior.sel(condition=df_unique.condtm.iloc[rowdf], aid=df_unique.aid.iloc[rowdf], time=df_unique.time.iloc[rowdf])
        samp = catc_trace.posterior.sel(condition=df_unique.condtm.iloc[rowdf], time=df_unique.time.iloc[rowdf])
        # Simulate
        # alpha_sim = np.exp(samp.θ_a.values + samp.θ_a_cond.values + samp.θ_a_aid.values + samp.θ_a_time.values)
        # β_μ_sim = pm.math.exp(samp.θ_μ.values + samp.θ_μ_cond.values + samp.θ_μ_aid.values + samp.θ_μ_time.values)
        β_μ_sim = pm.math.exp(samp.θ_μ.values + samp.θ_μ_cond.values + samp.θ_μ_time.values)
        # alpha_sim = samp.θ_a.values + samp.θ_a_cond.values + samp.θ_a_aid.values + samp.θ_a_time.values
        # beta_sim = np.exp(samp.θ_b.values + samp.θ_b_cond.values + samp.θ_b_aid.values + samp.θ_b_time.values)
        # β_σ_sim = pm.math.exp(samp.θ_σ.values + samp.θ_σ_cond.values + samp.θ_σ_aid.values + samp.θ_σ_time.values)
        β_σ_sim = pm.math.exp(samp.θ_σ.values + samp.θ_σ_cond.values + samp.θ_σ_time.values)
        # tau_sim = samp.θ_tau.values + samp.θ_tau_cond.values + samp.θ_tau_aid.values + samp.θ_tau_time.values
        # mu_sim = samp.θ_m.values + samp.θ_m_cond.values + samp.θ_m_aid.values + samp.θ_m_time.values
        μ_sim = pm.HalfNormal.dist(tau=β_μ_sim)
        σ_sim = pm.HalfNormal.dist(tau=β_σ_sim)
        # fr = pm.Gamma.dist(alpha=alpha_sim[:,:100].flatten(), beta=beta_sim[:,:100].flatten())
        # sim = pm.Gamma.dist(alpha=α_sim[:,:100].flatten(), beta=β_sim[:,:100].flatten())
        sim = pm.Gamma.dist(mu=μ_sim[:,:100].flatten(), sigma=σ_sim[:,:100].flatten())
        # sim = pm.SkewNormal.dist(alpha=alpha_sim[:,:100].flatten(), tau=tau_sim[:,:100].flatten(), mu=mu_sim[:,:100].flatten())
        # sim = pm.Poisson.dist(mu=fr * 60 * 5)
        # sim = pm.InverseGamma.dist(alpha=alpha_sim[:,:200].flatten(), beta=beta_sim[:,:200].flatten())

        # Combine in dataframe
        # __ = pd.DataFrame(dict(condition=df.condition.iloc[rowdf], aid=df.aid.iloc[rowdf], time=df.time.iloc[rowdf], value=sim.flatten(), condtm = df_unique.condtm.iloc[rowdf]))
        # __ = pd.DataFrame(dict(condition=df_unique.condition.iloc[rowdf], aid=df_unique.aid.iloc[r
        __ = pd.DataFrame(dict(condition=df_unique.condition.iloc[rowdf], time=df_unique.time.iloc[rowdf], value=sim.eval(), condtm = df_unique.condtm.iloc[rowdf]))
        df_sim = pd.concat([df_sim, __])

    # # Max_likelihood
    # ml_a, ml_loc, ml_scl = invgamma.fit(df.value)
    # ml_sim = invgamma.rvs(ml_a, scale=1/ml_scl, size=80000)

    # Compare models
    # df_compare_loo = az.compare(dict(gamma=catc_trace, inv_gamma=catc_trace_invg))

    # All plots
    sns.set(style='whitegrid')
    # Plot data
    fig1, ax1 = plt.subplots(figsize=(12, 12))
    sns.violinplot(x=df.time,
                   y=df.value,
                   hue=df.condition,
                   ax=ax1)
    sns.stripplot(x=df.time,
                  y=df.value,
                  hue=df.condition,
                  dodge=True,
                  palette='pastel',
                  ax=ax1)

    fig1.suptitle('Population spike count', size=size)
    plt.ylabel('spike count')
    plt.xlabel('Time from drop (min)')
    handles, labels = ax1.get_legend_handles_labels()
    ax1.legend(handles[:3], labels[:3], loc='upper right')


    fig11, ax11 = plt.subplots(3, 1, figsize=(12, 12))
    for c, axes in enumerate(ax11):
        sns.histplot(x='value', data=df[df.condition==df.condition.unique()[c]], element='step', kde=True, hue='time', log_scale=True, ax=axes)
        ax11[c].set_title(df.condition.unique()[c])
    plt.tight_layout
        # sns.histplot(x='value', data=df[df.condition==], hue='time', ax=axes)
        # sns.histplot(x='value', data=df[df.condition=='aPBS'], hue='time', ax=axes)


    fig11.suptitle('Population spike count', size=size)
    plt.ylabel('spike count')
    plt.xlabel('Time from drop (min)')
    handles, labels = ax1.get_legend_handles_labels()
    ax1.legend(handles[:3], labels[:3], loc='upper right')


    # Plot Model output
    az.summary(catc_trace, hdi_prob=0.9, var_names=['θ'], filter_vars='regex')
    az.plot_trace(catc_trace, legend=False, combined=False, var_names=['θ'], filter_vars='regex', figsize=(12, 12))
    az.plot_forest(catc_trace, combined=True, var_names=['θ'], filter_vars='regex', figsize=(12, 12))
    # az.plot_ppc(catc_priorpc, group='prior', mean=True, ax=ax2)
    # az.plot_ppc(catc_postpc, group='posterior', mean=False)
    priorpcfig = az.plot_ppc(catc_priorpc, group='prior', mean=True, figsize=(12, 12))
    priorpcfig.set_xlabel('cluster spike count')
    priorpcfig.set_xlim(0, 100000)
    postpcfig =az.plot_ppc(catc_postpc, group='posterior', observed=True, figsize=(12, 12))
    postpcfig.set_xlabel('cluster spike count')
    sns.histplot(df.value, stat='density')

    # Plot contrasts and hdi
    # fig3, ax3 = plt.subplots(figsize=(12, 12))
    # sns.pointplot(x='value', y='variable', data=df_diff, hue='parameter', dodge=True, join=False, ci='sd', ax=ax3)
    fig3, ax3 = plt.subplots(figsize=(12, 12))
    sns.stripplot(x='value', y='variable', data=df_diff, hue='parameter', dodge=True, ax=ax3)

    color = ['tab:blue', 'tab:orange'] * 3
    plot = sns.stripplot(x='value', y='variable', data=df_diff, hue='parameter', dodge=True, ax=ax3)
    for idx, hight in enumerate(np.array([[-0.3, -0.3], [0.1, 0.1], [0.7, 0.7], [1.1, 1.1], [1.7, 1.7], [2.1, 2.1]])):
        ax3.plot(hdi_diff.value.values[idx], hight, color=color[idx])
    plt.show()
 
    # Plot prior and post cov
    fig4, ax4 = plt.subplots(2, 2, figsize=(12, 12))
    sns.heatmap(cov_prior['cov_μ_t_prior'], ax=ax4[0, 0])
    sns.heatmap(cov_post['cov_μ_t_post'], ax=ax4[0, 1])
    sns.heatmap(cov_prior['cov_σ_t_prior'], ax=ax4[1, 0])
    sns.heatmap(cov_post['cov_σ_t_post'], ax=ax4[1, 1])

    ax4[0, 0].set_title('prior t_alpha')
    ax4[0, 1].set_title('posterior t_alpha')
    ax4[1, 0].set_title('prior t_beta')
    ax4[1, 1].set_title('posterior t_beta')

    # Plot sim_data
    fig5, ax5 = plt.subplots(figsize=(12, 12))
    # sns.violinplot(x=df_sim.time,
    #                y=np.log(df_sim.value),
    #                hue=df_sim.condition,
    #                ax=ax5)
    sns.stripplot(x=df_sim.time,
                  y=df_sim.value,
                  hue=df_sim.condition,
                  dodge=True,
                  palette='pastel',
                  ax=ax5)

    fig5.suptitle('Population spike count - simulated', size='xx-large')
    plt.ylabel('scaled spike count')
    plt.xlabel('Time from drop (min)')
    handles, labels = ax5.get_legend_handles_labels()
    ax5.legend(handles[:3], labels[:3], loc='upper right')

    # Plot posterior checks Simulation (and max likelihood estimate)
    fig6, ax6 = plt.subplots(figsize=(12, 12))
    axe = sns.histplot(data=df.value, stat='density', color='blue', element='step', fill=False, label='observations', ax=ax6)
    axe = sns.histplot(data=df_sim.value, stat='density', color='yellow', element='step', fill=False, label='posterior predictive samples', ax=ax6)
    # axe = sns.histplot(data=invgamma.rvs(ml_a, loc=ml_loc, scale=1/ml_scl, size=864000), stat='density', color='green', element='step', fill=False, label='maximum likelihood samples', ax=ax6)

    plt.show()


def plot_row(npxpath,
             aid,
             cortex_depth,
             t_drop,
             conditions,
             tw_start,
             tw_width,
             t_scale='-5',
             cgs=2):
    """ Plot row population spike count
    """

    t_scale = int(t_scale)
    __, df = get_popdata(npxpath, aid, cortex_depth, t_drop, conditions, tw_start, tw_width, t_scale, cgs=cgs)
    pdb.set_trace()
    sns.lineplot(data=df, x='time', y='value', hue='condition', estimator=None)



if __name__ == '__main__':
    """ Run script if spkcount_analysis.py module is main programme; if
    cgs != 0 take only mua or good units, if cgs == 0 take both mua and
    good units; select which analysis to run by defining data path: if path
    to one dataset use string for directory and plot st_analysis, otherwise
    use list and plot t_analysis.
    """
    # Info time windows
    tw_width = 5
    tw_start = np.arange(0, 60, tw_width)

    # Load metadata
    cgs = 2
    _, npxpath, aid, cortex_depth, t_drop, clu_idx, pethdiscard_idx, bout_idxs, wmask, conditions, _ = load_meta()
    # Choose to plot pop or single cluster data (do not change)
    plot_pop = 'yes'

    # Plot row data
    plot_raw = False
    
    # Check path existence
    for path in npxpath:
        print(os.path.exists(path))
        
    if plot_raw:
        tw_width = 1 / 6
        # tw_width = 1
        tw_start = np.arange(0, 60, tw_width)

        plot_row(npxpath,
                 aid,
                 cortex_depth,
                 t_drop,
                 conditions,
                 tw_start,
                 tw_width)
                 # t_scale=t_scale,
                 # cgs=cgs)
    else:
        if plot_pop == 'yes':
            t_scale = '-5'
            fit_spkcount(npxpath,
                         aid,
                         cortex_depth,
                         t_drop,
                         conditions,
                         tw_start,
                         tw_width,
                         t_scale=t_scale,
                         cgs=cgs)
        else: # Not used anymore
            t_scale = '-5' 
            fit_spkcount_clu(npxpath,
                             aid,
                             cortex_depth,
                             t_drop,
                             conditions,
                             tw_start=tw_start,
                             tw_width=tw_width,
                             t_scale=t_scale,
                             cgs=cgs)

    plt.show()

    
