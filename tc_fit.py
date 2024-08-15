"""
Script to analyse tuning curves for both controls and controls vs glyt2+cno.

Fit tuning curves and cluster putative neurons accordingly; used
b-spline to fit tuning curves (default angle).

Based on tc_fun from tc_module: function to calculate firing rate associated to each discrete
wvar value; uses load_data, spike_train (from align_fun module) and
tbin_wvar functions;

- tbin_wvar; compute average z-scored wvar value within time bins of length
fxb * frame length in ms.

- plot_tc; get firing rate per wvar bin with tc_fun; plot results.

Order: first compute spike count per time bin of length fxb (frames per bin)
in ms (spike_train function); discretise z-scored wvar (default angle) in
wv_nbin (default 12; tbin_wvar function); get index of each wvar bin along
wvar time series (used np.digitize) and use index to compute firing rate (spike
count / binw) associated to each wvar bin (tc_fun function).
"""

# Ensure /data_analysis is in path
import os
import sys
if not ('data_analysis' in os.getcwd()):
    sys.path.append(os.path.join(os.getcwd(), 'data_analysis'))


from scipy import stats
from patsy import dmatrix
from sklearn.cluster import KMeans

import matplotlib.pyplot as plt
import numpy as np
import arviz as az
import pymc as pm
import seaborn as sns

import itertools
import align_fun
import tc_module


def tc_fitting(whiskpath,
               npxpath,
               whisker=0,
               fxb=15,
               wvar="angle",
               whisking=True,
               cgs='good',
               w_bin=12,
               surr=False,
               deg=2):
    """ Fit clusters' firing rate tuning to discretised wvar; use b-spline
    model; fitted curve is the posterior over b-spline parameters given tc
    values and priors; tc values scaled up to 1 to facilitate definition
    of priors; default degree is 2; number of basis functions is 10/13th of
    number of wvar discrete bins; used exponential as prior of α and βs (non-
    -negative with higher probability around 0).
    Returns parameters (n_clu, n βs + 1 α), list of traces (posterior) for each
    cluster, scaled data by max value.
    """

    # # Initialise w_bin
    # w_bin = w_bin + 1

    # Compute firing rate per wvar bin
    _, wvar_tc_m, cids_sorted, _ = tc_module.tc_fun(whiskpath,
                                                    npxpath,
                                                    fxb=fxb,
                                                    whisking=whisking,
                                                    cgs=cgs,
                                                    wvar=wvar,
                                                    w_bin=w_bin,
                                                    surr=surr)

    # w_bin = w_bin - 1

    # Define design matrix for bayesian update of b-spline model
    n_bs = int(w_bin // 1.3)
    dmat = dmatrix(
        'bs(x,df={}, degree={}, include_intercept=True) - 1'.format(n_bs, deg),
        {'x': np.linspace(0, w_bin - 1, w_bin).astype(int)})

    # Bayesian update
    n_clu = len(wvar_tc_m[whisker])
    par = np.empty([n_clu, n_bs + 1])
    trace = []
    scl_data = []
    scl_factor = []
    # post_predict = []

    for clu in range(n_clu):
        data_s = np.asarray(wvar_tc_m[whisker][clu])
        sf = data_s.max()  # scaling factor
        data_s = data_s / sf

        with pm.Model() as m:
            α = pm.Exponential('α', lam=2.)
            β = pm.Exponential('β', lam=2., shape=dmat.shape[1])
            μ = pm.Deterministic('μ', α + pm.math.dot(dmat.base, β.T))
            σ = pm.Exponential('σ', 2.5)
            tc_val = pm.Normal('tc_val', μ, σ, observed=data_s)
            # prior_samples = pm.sample_prior_predictive(samples=1000)
            trace_m = pm.sample(1000)
            # ppc = pm.sample_posterior_predictive(trace_m, var_names=['tc_val'])

        trace.append(trace_m)
        scl_data.append(data_s)
        scl_factor.append(sf)
        #post_predict.append(ppc)
        par[clu, :] = np.insert(trace_m['β'].mean(0), 0, trace_m['α'].mean())

    # Cluster parameters using K-means
    kmeans = []
    for k in range(1, 3):
        kmeans.append(KMeans(n_clusters=k).fit(par))

    return par, trace, scl_data, scl_factor, cids_sorted, kmeans


def plot_tc_fit(whiskpath,
                npxpath,
                whisker=0,
                fxb=15,
                wvar="angle",
                whisking=True,
                cgs='good',
                w_bin=12,
                surr=False,
                deg=2):
    """ Plot tuning curves fit; default only good clusters; seaborn.barplot plots
    confidence intervals using bootstrapping; w_bin is initialised because the
    right edge is cancelled in tc_fun; variable cannot be phase!
    """
    # Compute posterior over b-spline parameters
    par, trace, scl_data, scl_factor, cids_sorted, kmeans = tc_fitting(
        whiskpath,
        npxpath,
        whisker=whisker,
        fxb=fxb,
        wvar=wvar,
        whisking=whisking,
        cgs=cgs,
        w_bin=w_bin,
        surr=surr,
        deg=deg)

    # Rescale parameters
    par = par * np.expand_dims(np.asarray(scl_factor), 1)

    # Initialise figure
    n_clu = par.shape[0]
    n_col = np.sqrt(n_clu).astype(int)
    n_row = n_clu // n_col if (n_clu // n_col *
                               n_col) == n_clu else n_clu // n_col + 1
    szfig_y = 12 * n_row
    szfig_x = 15 * n_col
    # plt_shape = [n_row, n_col]
    _, axs = plt.subplots(n_row,
                          n_col,
                          figsize=(szfig_x, szfig_y),
                          sharex=False)

    # Turn off all axes and turn on one-by-on to avoid empty axes
    for axis in axs.flat:
        axis.set_axis_off()

    # Plot b-spline model posterior
    for idx, clu in zip(itertools.product(np.arange(n_row), np.arange(n_col)),
                        range(n_clu)):

        if n_col == 1:
            idx = idx[0]

        # Plot posterior μs (model is α + dot(dmat, β))
        axs[idx].set_axis_on()
        az.plot_hdi(np.linspace(0, w_bin - 1, w_bin).astype(int),
                    trace[clu]['μ'] * scl_factor[clu],
                    ax=axs[idx],
                    hdi_prob=.89)
        sns.barplot(x=np.linspace(0, w_bin - 1, w_bin).astype(int),
                    y=scl_data[clu] * scl_factor[clu],
                    ax=axs[idx])

        axs[idx].set_ylabel('firing rate')
        axs[idx].set_xlabel(wvar)

        # Add text
        max_y = axs[idx].get_ylim()[1]
        axs[idx].text(w_bin - 1,
                      max_y * (5 / 6),
                      'id {}'.format(cids_sorted.iloc[clu]),
                      fontsize='large',
                      fontweight='semibold')

    plt.savefig(os.getcwd() + '/save_images/{}_tc_fit.png'.format(wvar))

    # Plot elbow
    elbow = np.empty(len(kmeans))
    for idx, kmean in enumerate(kmeans):
        elbow[idx] = kmean.inertia_

    _ = plt.figure()
    plt.plot(np.arange(1, len(kmeans) + 1), elbow)
    plt.xlabel('k clusters')
    plt.ylabel('SS')

    plt.savefig(os.getcwd() + '/save_images/{}_tc_fitElbow.png'.format(wvar))

    # Plot clustered parameters (k from 2 to 6)
    n_par = par.shape[1]
    n_comb = int((n_par * n_par) / 2)
    n_col = np.sqrt(n_comb).astype(int)
    n_row = n_comb // n_col if (n_comb // n_col *
                                n_col) == n_comb else n_comb // n_col + 1
    szfig_y = 5 * n_row
    szfig_x = 5 * n_col
    colormap = np.array(['r', 'g', 'b', 'y', 'c', 'm', 'k'])

    for k in range(1, 3):
        categories = kmeans[k - 1].labels_
        _, axs = plt.subplots(n_row,
                              n_col,
                              figsize=(szfig_x, szfig_y),
                              sharex=False)

        for idx, par_id in zip(
                itertools.product(np.arange(n_row), np.arange(n_col)),
                itertools.combinations(np.arange(n_par), r=2)):
            axs[idx].scatter(par[:, par_id[0]],
                             par[:, 1],
                             c=colormap[categories])
            max_y = axs[idx].get_ylim()[1]
            max_x = axs[idx].get_xlim()[1]
            axs[idx].text(max_x * 6 / 10,
                          max_y * 9 / 10,
                          '{}   {}'.format(par_id[0], par_id[1]),
                          fontsize='large',
                          fontweight='semibold')

        plt.savefig(os.getcwd() +
                    '/save_images/{}_tc_fitCatPar{}.png'.format(wvar, k))
