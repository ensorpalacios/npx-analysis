"""
Script to fit spline model to all pre and post drop tc on hpc. Load pre-computed
tc and fit model to each tc.

ATTENTION(1)1: run from high performace computer; run script from shell with argument
from 0-24, one for each recording.

ATTENTION(2): requires first generation of tc_all_data_nowhisk_compare.pickle file,
containing raw tuning curves (from tc_module.run_tc_compare)
"""
import sys
import os
import itertools
import pdb
import pickle
import matplotlib.pyplot as plt
import matplotlib
import warnings
import seaborn as sns
import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
import aesara.tensor as at
import scipy.linalg as la


from scipy import stats
from patsy import dmatrix
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans


def fit_bspline(tc_m, wbin, cidsSorted, deg=3):
    ''' Fit a basis spline function of order 3 to average tuning curve data; this
    function is a piecewise polynomial function; the B-splines are the basis
    functions that can be combined to form the function; noise model is positively
    truncated Gaussian.
    Important: scale data to 1
    '''
    # Create design matrix of B-splines
    n_bs = int(wbin // 1.5)
    dmat = dmatrix(f'bs(x, df=n_bs, degree={deg}, include_intercept=True) - 1', {'x':np.arange(wbin)})
    dmat = np.asarray(dmat)
    df_dmat = pd.DataFrame(dmat).reset_index(names='wbins').melt(id_vars='wbins', var_name='splines')

    # Define model (fake obs)
    coords_ = {'obs':np.arange(wbin), 'splines':df_dmat.splines.unique()} 
    with pm.Model(coords=coords_) as spline_model:
        tc_obs = pm.MutableData('tc_obs', np.arange(wbin, dtype=np.float64), dims='obs')
        # tc_obs = pm.MutableData('tc_obs', data_s, dims='obs')
        σ = pm.Exponential('σ', lam=.5)
        α = pm.Exponential('α', lam=.5)
        β = pm.Exponential('β', lam=.5, dims='splines')
        μ = pm.Deterministic('μ', α + at.dot(dmat, β))
        # tc_val = pm.Normal('tc_val', μ, σ, observed=tc_obs)
        tc_val = pm.TruncatedNormal('tc_val', μ, σ, lower=0, observed=tc_obs)
        # trace_spline = pm.sample()

    # Run for each cluster
    save_trace = []
    save_ppc = []
    save_sf = []
    for wsk in range(3):
        save_trace_wsk = []
        save_ppc_wsk = []
        save_sf_wsk = []
        for __, tc_data in enumerate(tc_m[wsk]):
            data_s = np.array(tc_data)
            sf = data_s.std()  # scaling factor
            data_s = data_s / sf

            with spline_model:
                pm.set_data({'tc_obs': data_s})
                # prior_samples = pm.sample_prior_predictive()
                trace_spline = pm.sample()
                post_samples = pm.sample_posterior_predictive(trace_spline)

            save_trace_wsk.append(trace_spline)
            save_ppc_wsk.append(post_samples)
            save_sf_wsk.append(sf)
        save_trace.append(save_trace_wsk)
        save_ppc.append(save_ppc_wsk)
        save_sf.append(save_sf_wsk)

    return save_trace, save_ppc, save_sf, df_dmat


def run_fit_tc(rec_fit, w_bin=11):
    """ Fit spline model to all tc (pre, post, 25 rec)
    """
    # Load pre-computed tc data
    # Organised as follow:
    # - wvar_tc, wvar_tc_m, tot_spkcount, cidsSorted, wvar_hist, path
    # - n_rec=25
    # - pre (0), post(1)
    # - n whisker (3)
    # - n clusters
    with open(os.getcwd() + '/tc_all_data_nowhisk_compare.pickle', 'rb') as f:
        save_all_tc_nowhisk = pickle.load(f)

    prepost_wvar_tc = save_all_tc_nowhisk[0]
    prepost_wvar_tc_m = save_all_tc_nowhisk[1]
    prepost_tot_spkcount = save_all_tc_nowhisk[2]
    prepost_cidsSorted = save_all_tc_nowhisk[3]
    prepost_wvar_hist = save_all_tc_nowhisk[4]
    prepost_path = save_all_tc_nowhisk[5]


    # Fit spline model
    prepost_data_fit = {}
    for idx, time in enumerate(['pre', 'post']):
        __ = fit_bspline(prepost_wvar_tc_m[rec_fit][idx], w_bin, prepost_cidsSorted[rec_fit][idx], deg=3)
        prepost_data_fit.updata({f'trace_{time}':__[0], f'ppc_{time}':__[1], f'sf_{time}':__[2], f'df_dmat_{time}':__[3]})

    # Save spline fits
    with open(os.getcwd() + f'tc_fit_nowhisk_compare_nrec{rec_fit}.pickle', 'wb') as f:
        pickle.dump(tc_fit, f)

    # # Fit spline model
    # tc_fit = []
    # for rec in range(25):
    #     prepost_data_fit = {}
    #     for idx, time in enumerate(['pre', 'post']):
    #         __ = fit_bspline(prepost_wvar_tc_m[rec][idx], w_bin, prepost_cidsSorted[rec][idx], deg=3)
    #         prepost_data_fit.updata({f'trace_{time}':__[0], f'ppc_{time}':__[1], f'sf_{time}':__[2], f'df_dmat_{time}':__[3]})
    #     tc_fit.append(prepost_data_fit)

    # # Save spline fits
    # with open(os.getcwd() + 'tc_fit_nowhisk_compare.pickle', 'wb') as f:
    #     pickle.dump(tc_fit, f)


if __name__ == '__main__':
    """ Run script from python shell; argument is 0-24, one
    for each recording
    """
    run_fit_tc(sys.argv[1])
