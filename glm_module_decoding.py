"""
Script to decode whisker activity (wsk) from population activity projected
onto PCA space. Includes following functions:

- get_data: load and prepare spk and wsk data (subset of original, keeping
only rec with decent whiksing trials); standardise whisker data trials;
 divide in pre- and post-drop, organise in one df.

- cross_anal: perform PCA and project data on first 3 eingenvectors; Z-score
data trials for each dimension separately; attention: df_spk smoothed with
gaussian kernel and modified permanently.

- crosscorrelation: compute crosscorrelogram between predicted and real whisker
data

- prepare_fitdata: prepare data for whisker estimation by separating pre-
and post-wsk and spk data is different df, in long format; create lag
(specific to regression analysis) for dimension by shifting spk data from 0
to max_lag, and cut wks data earlier than max_lag.

- decode_fun: linear regression model with lag; get posterior distribution over
model parameters (one for each pc and lag); reconstruct or predict whisking
position, hdi, crosscorrelogram and average lppd

- trans_fun: compute linear transfer function (based on least squared error);
use tapering to reduce noise; predict post whisking position.

- plot_decoding: plot output of decode_fun

- plot_tsf: plot output of trans_fun

- run_glm: run either decode_fun (linear regression) or trans_fun (linear
transfer function); plot and save.

For Thesis: used to create tsf plots
"""

import os
import sys
if not ('data_analysis' in os.getcwd()):
    sys.path.append(os.path.join(os.getcwd(), 'data_analysis'))

import pickle
import pdb

from scipy import stats
from scipy import signal
from scipy.special import logsumexp
from meta_data import load_meta
from scipy.ndimage import gaussian_filter1d
from align_fun import align_spktrain

import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import pandas as pd
import arviz as az
import pymc as pm
import numpy as np
import scipy.linalg as la
import numpy.linalg as nlin
import aesara.tensor as at


def get_data(rec_idx,
             # std,
             wvar='setpoint',
             whisk='whisk2',
             t_wind=[-0.35, 1],
             drop_rec=[0, 10, 15, 16, 25],
             cgs=2,
             surr=False,
             pre=None,
             discard=False):
    """ Load and prepare spk and wsk data: loaded data exclude already rec with
    no/bad whisking; filter data a second time (with drop_rec) based on whisker
    activity aligned to whisking bout (selection done from all_spkcnt.py
    module); divide whisking bouts and spk data in pre and post; get fr from
    spk data; organise in df.
    """
    # Get data
    # Load metadata, prepare second mask and final good list & get rec
    wskpath, npxpath, aid, cortex_depth, t_drop, clu_idx, pethdiscard_idx, bout_idxs, wmask, conditions, __ = load_meta()
    good_list = np.arange(len(wskpath))
    good_list = good_list[wmask]
    wmask2 = ~np.in1d(np.arange(len(good_list)), np.array(drop_rec))
    good_list = good_list[wmask2]
    rec_idx = good_list[rec_idx]
    # pdb.set_trace()

    # Select rec
    wskpath = wskpath[rec_idx]
    npxpath = npxpath[rec_idx]
    aid = aid[rec_idx]
    cortex_depth = cortex_depth[rec_idx]
    t_drop = t_drop[rec_idx]
    conditions = conditions[rec_idx]
    # clu_idx = clu_idx[rec_idx]
    clu_idx = []
    pethdiscard_idx = pethdiscard_idx[rec_idx]
    bout_idxs = bout_idxs[rec_idx]

    # Load aligned data
    # wsknpx_data contains: whiskd, spktr_sorted, cids_sorted, a_wvar, a_wvar_m, a_spktr, a_spktr_m
    wsknpx_data = align_spktrain(wskpath, npxpath, cortex_depth, cgs=cgs, t_wind=t_wind, var=wvar, surr=surr, t_drop=t_drop, pre=pre, clu_idx=clu_idx, discard=discard, pethdiscard_idx=pethdiscard_idx)
    # Organise data
    # Separate spk and wsk pre and post data (wsk Z-scored)
    event_times = np.array(wsknpx_data.whiskd.event_times[int(whisk[-1])])
    mpre = np.arange(len(event_times))[event_times < t_drop * 60]
    mpost = np.arange(len(event_times))[(event_times > (t_drop + 5) * 60) & (event_times < (t_drop * 2 + 5) * 60)]

    __ = wsknpx_data.a_wvar.loc[:, whisk].stack()
    # __ = (__ - __.mean()) / __.std()
    __ = __.subtract(__.groupby(level=1).mean(),
                     level=1).divide(__.groupby(level=1).std(),
                                     level=1)
    __ = __.unstack()
    wsk_pre = __.loc[:, mpre]
    wsk_post = __.loc[:, mpost]
    # wsk_pre = wsknpx_data.a_wvar.loc[:, whisk].loc[:, mpre]
    # wsk_post = wsknpx_data.a_wvar.loc[:, whisk].loc[:, mpost]

    spk_pre = wsknpx_data.a_spktr.loc[:, whisk].loc[:, (slice(None), mpre)]
    spk_post = wsknpx_data.a_spktr.loc[:, whisk].loc[:, (slice(None), mpost)]

    # Drop trial with NaN
    wsk_pre.dropna(axis=1, inplace=True)
    wsk_post.dropna(axis=1, inplace=True)
    spk_pre.dropna(axis=1, inplace=True)
    spk_post.dropna(axis=1, inplace=True)

    # Firing rates
    __ = wsknpx_data.a_spktr.loc[:, whisk].dropna(axis=1)
    fr_all = __.groupby(axis=1, level=0).mean()
    fr_pre = spk_pre.groupby(axis=1, level=0).mean()
    fr_post = spk_post.groupby(axis=1, level=0).mean()
    
    # Discard clu with total fr < 0.1 in pre/post windows
    discard = fr_pre.columns.values[(fr_pre.sum() < 0.1) | (fr_post.sum() < 0.1)]
    spk_pre.drop(columns=discard, level=0, inplace=True)
    spk_post.drop(columns=discard, level=0, inplace=True)
    fr_pre.drop(columns=discard, inplace=True)
    fr_post.drop(columns=discard, inplace=True)
    fr_all.drop(columns=discard, inplace=True)

    # Organise in one df
    df_wsk = pd.concat([wsk_pre, wsk_post], axis=1, keys=['pre', 'post'])
    df_spk = pd.concat([spk_pre, spk_post], axis=1, keys=['pre', 'post'])
    df_fr = pd.concat([fr_pre, fr_post], axis=1, keys=['pre', 'post'])

    return df_wsk, df_spk, df_fr, fr_all, conditions, wskpath


def corr_anal(fr_all, df_spk, std):
    """ Correlation analysis: get correlation matrix from firing rates (length
    whole recording); compute eingenvalues and eigenvectors; smooth (gaussian
    filter) and project spk data onto 3D space (); df_spk is df of single
    trials pre and post.

    # Attention: df_spk modified permanently after smoothing!!!
    """
    # pdb.set_trace()
    # Get correlation matrix
    corr = fr_all.corr(method='pearson')
    # Get eignval, eigvec and project
    egnval, egnvec = la.eig(corr)
    idxsort = np.argsort(egnval.real)[::-1]
    egnval = egnval.real[idxsort]
    egnvec = egnvec[:, idxsort]

    # Smooth and project single trial spk data
    # ATTENTION df_spk modified permanently!!!
    df_spk.iloc[:, :] = gaussian_filter1d(df_spk, sigma=std, axis=0)

    projd = []
    for time in ['pre', 'post']:
        spk_data = df_spk[time]
        trials = spk_data.columns.get_level_values(1).unique()
        save_projd = pd.DataFrame()
        for tr in trials:
            # Dot product
            __ = np.dot(spk_data.loc[:, (slice(None), tr)], egnvec[:, :3])
            # Save trials
            levels = [[tr], ['pc1', 'pc2', 'pc3']]
            codes = [[0, 0, 0], [0, 1, 2]]
            midx = pd.MultiIndex(levels, codes=codes)
            __ = pd.DataFrame(__, columns=midx)
            save_projd = pd.concat([save_projd, __], axis=1)

        # Save in one df
        projd.append(save_projd)

    # Convert in df
    projd = pd.concat([projd[0], projd[1]], keys=['pre', 'post'], axis=1)
    projd.columns.names = ['time', 'trial', 'pcs']

    # Z-score projections
    __ = projd.stack(0).stack(0)
    # __ = (__ - __.mean()) / __.std()
    __ = __.subtract(__.groupby(by='trial').mean(),
                     level='trial').divide(__.groupby(by='trial').std(),
                                           level='trial')

    # __ = __ / __.std()
    projd = __.stack().unstack(0).T

    # Color for color coded time
    color = np.array(fr_all.index)

    return corr, egnvec, egnval, projd, color


def crosscorrelation(datax, datay, cclag=50):
    """ Compute crosscorrelogram; shift datay (fits or predictions);
    data are in long format: convert back to short format to shift
    single trials.
    """
    # datay = datay.unstack(level=0).droplevel(0, axis=1)
    __ = datay.unstack(level=0).droplevel(0, axis=1)

    # maxlag = lag - 1
    crosscorr = []
    # for lg in range(-maxlag, lag):
    for lg in range(-10, cclag):
        datay_shift = __.shift(lg, fill_value=0).stack().swaplevel().sort_index(level=0)
        corr = np.corrcoef([datax, datay_shift])[0, 1]
        crosscorr.append(corr)

    return crosscorr


def prepare_fitdata(projd, df_wsk, max_lag):
    """ Prepare projd and wsk data for regression models.
    Separate data in pre and post and organise in df; shift
    projd data for lag model and resize based on lag (wsk
    data as well).
    """
    # Pre drop data
    projd_pre = projd['pre']
    projd_post = projd['post']
    wsk_pre = df_wsk['pre']
    wsk_post = df_wsk['post']
    # pdb.set_trace()

    # Create lagged data for lag models
    projd_pre_lag = []
    projd_post_lag = []
    for lag in range(max_lag):
        projd_pre_lag.append(projd_pre.iloc[lag:(lag-max_lag), :])
        projd_post_lag.append(projd_post.iloc[lag:(lag-max_lag), :])

    # Reset trial idx
    for data in [projd_pre, projd_post]:
        midx = data.columns.remove_unused_levels()
        __ = midx.levels[0] - midx.levels[0][0]
        __ = midx.set_levels(__, level=0)
        data.columns = __

    for data in [wsk_pre, wsk_post]:
        __ = data.columns - data.columns[0]
        data.columns = __

    # Stack data
    projd_pre = projd_pre.stack(0)
    projd_post = projd_post.stack(0)
    wsk_pre = wsk_pre.stack(0)
    wsk_post = wsk_post.stack(0)

    # Swap index levels: 1 trial 2 time
    __ = projd_pre.index.swaplevel()  # index 1:trial, 2:time
    __.names = ['trial', 'time']
    projd_pre.index = __
    wsk_pre.index = __
    projd_pre.sort_index(inplace=True)
    wsk_pre.sort_index(inplace=True)

    __ = projd_post.index.swaplevel()  # index 1:trial, 2:time
    __.names = ['trial', 'time']
    projd_post.index = __
    wsk_post.index = __
    projd_post.sort_index(inplace=True)
    wsk_post.sort_index(inplace=True)

    # Same for lag data
    for lag in range(max_lag):
        # Stack data
        projd_pre_lag[lag] = projd_pre_lag[lag].stack(0)
        projd_post_lag[lag] = projd_post_lag[lag].stack(0)
        # Swap index levels
        __ = projd_pre_lag[lag].index.swaplevel()  # index 0:trial, 1:time
        projd_pre_lag[lag].index = __
        projd_pre_lag[lag].sort_index(inplace=True)
        __ = projd_post_lag[lag].index.swaplevel()  # index 0:trial, 1:time
        projd_post_lag[lag].index = __
        projd_post_lag[lag].sort_index(inplace=True)
        # Invert axes for tensor product
        projd_pre_lag[lag] = projd_pre_lag[lag].T
        projd_post_lag[lag] = projd_post_lag[lag].T

    projd_pre_lag = np.array(projd_pre_lag)
    projd_post_lag = np.array(projd_post_lag)
    # max_time_lag = projd['pre'].shape[0] - max_lag

    # Resize wsk data based on lag
    # pdb.set_trace()
    wsk_pre_lag = wsk_pre[wsk_pre.index.get_level_values(1) >= max_lag]
    wsk_post_lag = wsk_post[wsk_post.index.get_level_values(1) >= max_lag]
    # wsk_pre_lag = wsk_pre[wsk_pre.index.get_level_values(1) < max_lag]
    # wsk_post_lag = wsk_post[wsk_post.index.get_level_values(1) < 1495 -  max_lag]

    return projd_pre, projd_post, wsk_pre, wsk_post, projd_pre_lag, projd_post_lag, wsk_pre_lag, wsk_post_lag


def decode_fun(t_wind=[-0.35, 1],
               wvar="setpoint",
               whisk='whisk2',
               rec_idx=0,
               cgs=2,
               clu_idx=[],
               drop_rec=[0, 10, 15, 16, 25],
               std=20,
               cclag=50,
               max_lag=70,
               n_lag=5,
               rng=100,
               pre=None,
               discard=False,
               surr=False):
    """ Decode wsk activity (setpoint is best) from neural activity:
    - get data (function from all_spkcnt.py).
    - recover posterior distribution (use only pre-drop trials) for
    selected cluster; check chain divergence with az.plot_trace(); use model
    mean posteriors to predict second half of data; calculate lppd.
    Note: use **kwarg 'cgs' to select goo clusters (=2); use *args 'clu_id'
    to select specific clusters.
    """
    # pdb.set_trace()
    # Get and prepare data #####################
    df_wsk, df_spk, df_fr, fr_all, conditions, wskpath = get_data(rec_idx, wvar=wvar, whisk=whisk, t_wind=t_wind, cgs=cgs)
    # pdb.set_trace()

    # Correlation analysis (on pre-drop data)
    corr, egnvec, egnval, projd, color = corr_anal(fr_all, df_spk, std=std)

    # Prepare data for models
    projd_pre, projd_post, wsk_pre, wsk_post, projd_pre_lag, projd_post_lag, wsk_pre_lag, wsk_post_lag = prepare_fitdata(projd, df_wsk, max_lag)

    idx_trial = projd_pre.index.get_level_values(0)
    projd_pre_fft = projd_pre.groupby(by=idx_trial).apply(np.fft.fft)

    # Linear model No lag ##############################
    idx_pre = wsk_pre.index
    idx_post = wsk_post.index

    dims_lag = {'pc': np.arange(3), 'obs': pd.factorize(wsk_pre.index)[0]}

    with pm.Model() as llmodel:
        # Set mutable coordinates and data
        llmodel.add_coord('pc', dims_lag['pc'], mutable=False)
        llmodel.add_coord('obs', dims_lag['obs'], mutable=True)

        pc_data = pm.MutableData('pc_data', projd_pre, coords=['pc', 'obs'])
        wskd = pm.MutableData('wskd', wsk_pre, coords='obs')

        # PC hyperparameters and parameters
        μ_N = 0
        σ_N = 1
        μ_G = 1
        σ_G = 1
        θ_pc = pm.Normal('θ_pc', mu=μ_N, sigma=σ_N, size=(3), dims=['pc'])

        # Intercept
        θ = pm.Normal('θ', mu=μ_N, sigma=σ_N)

        # Mean and std
        μ = pm.Deterministic('μ', θ + at.dot(pc_data, θ_pc))
        σ = pm.Gamma('σ', mu=μ_G, sigma=σ_G)

        # Likelihood
        obs = pm.Normal('obs', mu=μ, sigma=σ, observed=wskd, size=pc_data.shape[0])

    # Run sampling on pre drop data
    with llmodel:
        pc_llm = pm.sample_prior_predictive(random_seed=rng)
        trace_llm = pm.sample(random_seed=rng)
        ppc_llm = pm.sample_posterior_predictive(trace_llm, random_seed=rng)

    # Run predictions of post drop data
    with llmodel:
        pm.set_data({'pc_data': projd_post})
        pred_wpost_llm = pm.sample_posterior_predictive(trace_llm, var_names=['μ', 'σ', 'obs'], random_seed=rng)
    # pdb.set_trace()

    # Generate pre-drop wsk fit and post-drop wsk predictions
    wsk_fit_llm = ppc_llm.posterior_predictive['obs'].mean(dim=['chain', 'draw']).values
    wsk_fit_llm = pd.DataFrame(wsk_fit_llm, index=idx_pre)
    wsk_predict_llm = pred_wpost_llm.posterior_predictive['obs'].mean(dim=['chain', 'draw']).values
    wsk_predict_llm = pd.DataFrame(wsk_predict_llm, index=idx_post)

    # Hdi's
    hdi_fit_llm = az.hdi(ppc_llm.posterior_predictive['obs'], hdi_prob=.9)
    hdi_fit_llm = pd.DataFrame(hdi_fit_llm['obs'], index=idx_pre)
    hdi_predict_llm = az.hdi(pred_wpost_llm.posterior_predictive['obs'], hdi_prob=.9)
    hdi_predict_llm = pd.DataFrame(hdi_predict_llm['obs'], index=idx_post)
    # pdb.set_trace()

    cc_pre_llm = []
    cc_post_llm = []

    θ_ols = []

    # Lppd
    # observed data...
    ll_pre = trace_llm.log_likelihood.to_stacked_array(new_dim='stacked_samp', sample_dims=['obs_dim_0'])
    # lppd_pre = sum(logsumexp(ll_pre, axis=1) - np.log(ll_pre.shape[1])) * -2
    avg_lppd_pre = np.mean(logsumexp(ll_pre, axis=1) - np.log(ll_pre.shape[1]))

    # ... & predicted data
    μ_post = pred_wpost_llm.posterior_predictive['μ'].stack(new_dim=['chain', 'draw'])
    σ_post = np.expand_dims(pred_wpost_llm.posterior_predictive['σ'].stack(new_dim=['chain', 'draw']), axis=0)
    ll_post = stats.norm.logpdf(x=np.expand_dims(wsk_post.values, axis=1), loc=μ_post, scale=σ_post)
    # lppd_post = sum(logsumexp(ll_post, axis=1) - np.log(ll_post.shape[1])) * -2
    avg_lppd_post = np.mean(logsumexp(ll_post, axis=1) - np.log(ll_post.shape[1]))

    return trace_llm, hdi_fit_llm, hdi_predict_llm, cc_pre_llm, cc_post_llm, θ_ols, avg_lppd_pre, avg_lppd_post, wsk_pre, wsk_post, wsk_fit_llm, wsk_predict_llm, wskpath


def plot_decoding(plot=False,
                  t_wind=[-0.35, 1],
                  wvar="setpoint",
                  rec_idx=0,
                  drop_rec=[0, 10, 15, 16, 25],
                  std=20,
                  cclag=50,
                  sr=299,
                  max_lag=70):

    # Call decode_fun
    all_data = decode_fun(t_wind=t_wind,
                          wvar=wvar,
                          rec_idx=rec_idx,
                          drop_rec=drop_rec,
                          std=std, cclag=cclag,
                          max_lag=max_lag)
    pdb.set_trace()

    trace_llm = all_data[0]
    hdi_fit_llm = all_data[1]
    hdi_predict_llm = all_data[2]
    cc_pre_llm = all_data[3]
    cc_post_llm = all_data[4]
    θ_ols = all_data[5]
    avg_lppd_pre = all_data[6]
    avg_lppd_post = all_data[7]
    wsk_pre = all_data[8]
    wsk_post = all_data[9]
    wsk_fit_llm = all_data[10]
    wsk_predict_llm = all_data[11]
    wskpath = all_data[12]

    # Generate plots ####
    # Initialise figure parameters
    wcolor = 'darkgoldenrod'
    # recfacecolor = 'darkorange'
    # reccolor = 'darkorange'
    recfacecolor = 'indianred'
    reccolor = 'indianred'
    style = 'italic'
    figsize = (14, 12)
    linewidth = 2
    # sns.set_context("poster", font_scale=1)
    # sns.color_palette("deep")
    plt.rcParams['axes.facecolor'] = 'lavenderblush'
    # plt.rcParams['axes.facecolor'] = 'white'
    plt.rcParams["figure.edgecolor"] = 'black'
    plt.rcParams["figure.edgecolor"] = 'black'
    plt.rcParams.update({'font.size': 22})

    len_test = wsk_pre.index.get_level_values(1).unique().shape[0]
    xticklabels = np.linspace(0, len_test / sr // 1, 5).astype(int)
    xticks = xticklabels * sr
    xticklabels = xticklabels + t_wind[0]


    # Plot forests
    forest_llm, ax_llm = plt.subplots(1, 3)
    az.plot_forest(trace_llm, var_names=['θ_pc'], filter_vars='like', coords={'pc':0}, ax=ax_llm[0])
    ax_llm[0].set_title('pc1')
    az.plot_forest(trace_llm, var_names=['θ_pc'], filter_vars='like', coords={'pc':1}, ax=ax_llm[1])
    ax_llm[1].set_title('pc2')
    az.plot_forest(trace_llm, var_names=['θ_pc'], filter_vars='like', coords={'pc':2}, ax=ax_llm[2])
    ax_llm[2].set_title('pc3')

    # Plot single traces (wsk vs fit/prediction)
    pre_trial_plots = {}
    __ = wsk_pre.index.get_level_values(1).unique().shape[0]
    __ = np.linspace(0, __, 6)
    # __ = np.arange(t_wind[0] + max_lag / 299, t_wind[1]).round(2)
    # __ = np.linspace(__[0], __[1], 5)
    # for nt in range(0, 75, 2):
    for nt in range(wsk_pre.index.get_level_values(0).unique().size):
        pre_trial_plots[f'trial{nt}'] = plt.figure()
        plt.plot(wsk_pre.loc[nt].values)
        # plt.plot(wsk_fit_lm.loc[nt].values)
        plt.plot(wsk_fit_llm.loc[nt].values[:, 0])
        az.plot_hdi(np.arange(wsk_pre.loc[nt].values.shape[0]), hdi_data=hdi_fit_llm.loc[nt].values, smooth_kwargs={'window_length':5, 'polyorder':1})
        plt.xticks(ticks=__, labels=np.round(__/299 + (t_wind[0] + max_lag / 299), 2))
        plt.xlabel('sec')
        plt.ylabel('amplitude')
        plt.show()

    post_trial_plots = {}
    # for nt in range(0, 171, 2):
    for nt in range(wsk_post.index.get_level_values(0).unique().size):
        post_trial_plots[f'trial{nt}'] = plt.figure()
        plt.plot(wsk_post.loc[nt].values, c=wcolor, linewidth=2, label='observed')
        # plt.plot(wsk_predict_lm.loc[nt].values)
        plt.plot(wsk_predict_llm.loc[nt].values[:, 0], c=reccolor, linewidth=2, label='predicted')
        az.plot_hdi(np.arange(wsk_post.loc[nt].values.shape[0]), hdi_data=hdi_predict_llm.loc[nt].values, smooth_kwargs={'window_length':5, 'polyorder':1}, color=recfacecolor)
        # plt.xticks(ticks=__, labels=np.round(__/299 + (t_wind[0] + max_lag / 299), 2))
        plt.xticks(ticks=xticks, labels=xticklabels)
        plt.xlabel('sec')
        plt.ylabel('amplitude')
        plt.legend(loc='upper right')
        plt.show()

    # # Plot crosscorrelograms & save
    # cc_llm, ax_ccllm = plt.subplots()
    # ax_ccllm.plot(cc_pre_llm, label='pre_lm')
    # __ = ax_ccllm.get_xticks()
    # ax_ccllm.set_xticks(__[1:-1])
    # ax_ccllm.set_xticklabels(__[:-2])

    # # Plot Ordinary Least Square regression coefficients
    # fig_ols, axs_ols = plt.subplots(1, 3)
    # axs_ols[0].scatter(θ_ols[0, :], np.arange(θ_ols[0, :].size, 0, -1))
    # axs_ols[1].scatter(θ_ols[1, :], np.arange(θ_ols[0, :].size, 0, -1))
    # axs_ols[2].scatter(θ_ols[2, :], np.arange(θ_ols[0, :].size, 0, -1))
    
    if plot:
        plt.show()

    else:
        # Set directories
        if os.path.basename(os.getcwd()) != 'data_analysis':
            os.chdir('./data_analysis')

        if os.path.isdir(os.getcwd() + f'/images/decoding/{os.path.basename(wskpath[:-1])}') is False:
            print(f'created dir /images/decoding/{os.path.basename(wskpath[:-1])}')
            os.makedirs(os.getcwd() + f'/images/decoding/{os.path.basename(wskpath[:-1])}')


        # Save forest
        # forest_lm.savefig(os.getcwd() + f'/images/decoding/{os.path.basename(wskpath[:-1])}/{os.path.basename(wskpath[:-1])}_forest_lm.svg', format='svg')
        forest_llm.savefig(os.getcwd() + f'/images/decoding/{os.path.basename(wskpath[:-1])}/{os.path.basename(wskpath[:-1])}_forest_llm.svg', format='svg')

        # Save single trials
        for __, plot in enumerate(pre_trial_plots.values()):
            plot.savefig(os.getcwd() + f'/images/decoding/{os.path.basename(wskpath[:-1])}/{os.path.basename(wskpath[:-1])}_fitlm{__}.svg', format='svg')

        for __, plot in enumerate(post_trial_plots.values()):
            plot.savefig(os.getcwd() + f'/images/decoding/{os.path.basename(wskpath[:-1])}/{os.path.basename(wskpath[:-1])}_predict_trace{__}.svg', format='svg')

    return trace_llm, hdi_fit_llm, hdi_predict_llm, avg_lppd_pre, avg_lppd_post


def run_glm(tsf=True,
            plot=False,
            t_wind=[-2, 3],
            t_subwind=[-0.35, 1],
            wvar="setpoint",
            drop_rec=[0, 10, 15, 16, 25],
            std=20,
            cclag=50,
            sr=299,
            max_lag=70,
            condition_tsf='convolution'):

    """ Run glm on all good datasets (25 in total, see good list in get_data
    function); good recordings are rec with decent whisking behaviour (filtered)
    twice: once get rid of rec with no whisking, then get rid of rec with bad
    whisking.
    
    Use plot=True to plot graphs for each recording, or plot=False to save
    """
    # Save results
    save_trace_llm = []
    save_hdi_fit_llm = []
    save_hdi_predict_llm = []
    save_cc_pre_llm = []
    save_cc_post_llm = []
    save_θ_ols = []
    save_avg_lppd_pre = []
    save_avg_lppd_post = []

    for rec_idx in range(25):       # all good recordings
        rec_idx = 22
        all_data = plot_decoding(plot=plot,
                                 t_wind=t_wind,
                                 wvar=wvar,
                                 rec_idx=rec_idx,
                                 drop_rec=drop_rec,
                                 std=std,
                                 cclag=cclag,
                                 sr=sr,
                                 max_lag=max_lag)

        save_trace_llm.append(all_data[0])
        save_hdi_fit_llm.append(all_data[1])
        save_hdi_predict_llm.append(all_data[2])
        save_avg_lppd_pre.append(all_data[3])
        save_avg_lppd_post.append(all_data[4])

    # Save
    save_glm_output = [save_trace_llm, save_hdi_fit_llm, save_hdi_predict_llm, save_cc_pre_llm, save_cc_post_llm, save_θ_ols, save_avg_lppd_pre, save_avg_lppd_post]
    with open('/home/yq18021/Documents/github_gcl/data_analysis/save_data/glm_output.pickle', 'wb') as f:
        pickle.dump(save_glm_output, f)


if __name__ == '__main__':
    """ Run script if whisk_plot.py module is main programme
    """
    # wvar='angle'
    wvar = 'setpoint'
    t_wind = [-2, 3]
    t_subwind = [-0.35, 1]
    drop_rec = [0, 10, 15, 16, 25]
    std = 5                     # st for smoothing spkcounts
    cclag = 50            # lag for crosscorrelogram
    max_lag = 50         # lag for lagged model (in frames)
    sr = 299             # assumed sampling rate

    plot = False
    # tsf = True                  # linear transfer function or regression
    tsf = False                  # linear transfer function or regression
    condition_tsf = 'convolution'
    # condition_tsf = 'tsf_taper'
    # condition_tsf = 'tsf'

    # Run glm for all datasets
    run_glm(tsf=tsf, plot=plot, t_wind=t_wind, t_subwind=t_subwind, wvar=wvar, drop_rec=drop_rec, std=std, cclag=cclag, sr=sr, max_lag=max_lag, condition_tsf=condition_tsf)
