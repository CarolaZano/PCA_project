##########################################################################
import os

# set the environment variable to control the number of threads
# NEEDS TO BE DONE BEFORE CCL IS IMPORTED
original_omp_num_threads = os.environ.get('OMP_NUM_THREADS', None)
os.environ['OMP_NUM_THREADS'] = '1'

import pyccl as ccl

# Generic
import pandas as pd
import numpy as np
import scipy
from itertools import islice, cycle
import math
import sys
from scipy.integrate import odeint
from joblib import Parallel, delayed
import itertools
from importlib import reload
from functools import lru_cache
import scipy.integrate
from time import time

# cosmology
from astropy.io import fits
import yaml
import sacc
from multiprocessing import Pool, cpu_count

# Generate data sets
from sklearn.datasets import make_blobs

# PCA
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# SRD Binning
import srd_redshift_distributions as srd
import binning

# Data Visualization
import matplotlib.pyplot as plt
from tabulate import tabulate
import matplotlib.patches as mpatches
#import seaborn as sns

# MCMC
import emcee
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import corner
from chainconsumer import ChainConsumer, Chain, make_sample
from IPython.display import display, Math
from tqdm import tqdm

# nDGP NL and lin Pk
from nDGPemu import BoostPredictor

# MGCAMB
MODULE_PATH = "/home/c2042999/MGCAMB/camb/__init__.py"
MODULE_NAME = "MGCAMB"
import importlib
spec = importlib.util.spec_from_file_location(MODULE_NAME, MODULE_PATH)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module 
spec.loader.exec_module(module)
from MGCAMB import camb

# f(R) emu (eMANTIS)
from emantis import FofrBoost

# Load my functions
from Likelihood_PCADR import *
##########################################################################
###### Change all from here

def log_likelihood(theta, data, inv_cov, dndz_ph_bins, z_ph, ells):
    cosmo = ccl.Cosmology(
        Omega_c=theta[0],
        Omega_b=theta[1],
        h=theta[2],
        sigma8=theta[3],
        n_s=theta[4],
        matter_power_spectrum='halofit'
    )

    # Shift mean of photo-z bins based on theta
    # Can comment out if not using shifts
    shifted_z = [] 
    for i in range(len(dndz_ph_bins)):
        mean_z = np.average(z_ph, weights=dndz_ph_bins[i])
        shifted_z.append(z_ph + theta[5+i]*(1 + mean_z))
        # set any negative values to 0
        shifted_z[i][shifted_z[i] < 0] = 0

    model_vector = compute_spectra(cosmo, dndz_ph_bins, shifted_z, ells)
    model_vector = model_vector.flatten() # Flatten the array to make it a vector

    # Compute the log likelihood
    diff = data - model_vector
    log_like = -0.5 * np.dot(diff, np.dot(inv_cov, diff))

    return log_like, model_vector

def log_prior(theta, priors):
    for i, prior in enumerate(priors):
        if not prior[0] < theta[i] < prior[1]:
            return -np.inf
    return 0.0

def log_probability(theta, priors, data, inv_cov, dndz_ph_bins, z_ph, ells):
    lp = log_prior(theta, priors)
    if not np.isfinite(lp):
        return -np.inf, None
    ll, model_vector = log_likelihood(theta, data, inv_cov, dndz_ph_bins, z_ph, ells)
    if not np.isfinite(ll):
        return -np.inf, None
    log_prob = lp + ll
    return log_prob, model_vector

def main(args):

    # Create the output directory
    mcmc_dir = args.mcmc_dir
    if not os.path.exists(mcmc_dir):
        os.makedirs(mcmc_dir)

    cosmo_universe = ccl.Cosmology(Omega_c = 0.27, 
                          Omega_b = 0.046, 
                          h = 0.7, 
                          n_s = 0.974,
                          A_s = 2.01e-9)

    cosmo_universe_linear = ccl.Cosmology(Omega_c = 0.27, 
                          Omega_b = 0.046, 
                          h = 0.7, 
                          n_s = 0.974,
                          A_s = 2.01e-9,
                          matter_power_spectrum='linear')


    fR_universe = 1e-4
    H0rc_universe = 0.0
    MGParam_universe = [H0rc_universe,fR_universe,1,0,0]

    #command = 'python Get_Data_3x2pt_fsigma8_MG.py --OmgC {} --OmgB {} --h {} --ns {} --As {} --gravity_flag "f(R)" --MG_param {}'.format(cosmo_universe["Omega_c"],cosmo_universe["Omega_b"],cosmo_universe["h"], cosmo_universe["n_s"],cosmo_universe["A_s"],fR_universe)
    #os.system(command)
    # Define the cosmology
    cosmo = ccl.Cosmology(
        Omega_c=0.27,
        Omega_b=0.045,
        h=0.67,
        sigma8=0.83,
        n_s=0.96,
        matter_power_spectrum='halofit'
    )

    ells = np.geomspace(2, 2000, 30)

    # Compute the spectra
    zph_list = [z_ph for i in range(n_bins)]
    c_ells = compute_spectra(cosmo, dndz_ph_bins, zph_list, ells)

    data_vector = c_ells.flatten() # Flatten the array to make it a vector

    # Add noise
    noise = 0.1 * data_vector
    data_vector += np.random.normal(0, noise)

    # generate covariance matrix
    cov = compute_covariance(noise)
    inv_cov = np.linalg.inv(cov)
    print("Data vector and covariance matrix set up")
    
    # Set the random seed for reproducibility
    np.random.seed(args.seed)

    # test the likelihood function
    theta = [0.27, 0.045, 0.67, 0.83, 0.96]

    # Comment out if not using shifts
    for i in range(n_bins):
        theta.append(0.0)

    # Initialize the walkers
    nwalkers = args.n_walkers
    ndim = len(theta)
    chain_len = args.chain_len

    # Define the priors
    priors = [
        (0.1, 0.9), # Omega_c
        (0.01, 0.1), # Omega_b
        (0.5, 0.9), # h
        (0.7, 1.0), # sigma8
        (0.8, 1.1) # n_s
    ]

    # Comment out if not using shifts
    delta_z = 0.002 # LSST Y1 mean uncertainty
    for i in range(n_bins):
        priors.append((-delta_z, delta_z)) # Shifts for each redshift bin

    converged = False # Convergence flag

    # Check for existing backend file
    backend_file = os.path.join(mcmc_dir, 'chain_outputs.h5')
    if os.path.exists(backend_file):
        backend = emcee.backends.HDFBackend(backend_file)
        if backend.iteration == 0: # If the file is empty, start from scratch
            pos = theta + 1e-4 * np.random.randn(nwalkers, ndim)
        else:
            pos = None # backend will resume from previous position by default
        print("Resuming from existing backend file at iteration {}".format(backend.iteration))
    else:
        backend = emcee.backends.HDFBackend(backend_file)
        backend.reset(nwalkers, ndim) # Reset the backend
        pos = theta + 1e-4 * np.random.randn(nwalkers, ndim)
        print("No existing backend file found. Starting from scratch")

    # Initialise a pool 
    with Pool(args.n_threads) as pool:

        # set up the blob type
        btype = [('model_vector', float, len(data_vector))]

        # Set up the sampler
        sampler = emcee.EnsembleSampler(
            nwalkers,
            ndim,
            log_probability,
            args=(priors, data_vector, inv_cov, dndz_ph_bins, z_ph, ells),
            pool=pool,
            backend=backend,
            blobs_dtype=btype
        )
        
        while not converged:
            # Sample 
            start = time()
            sampler.run_mcmc(pos, chain_len, progress=True)
            end = time()

            # Update the initial positions
            pos = None # backend will resume from previous position by default

            # Check convergence
            try:
                tau = sampler.get_autocorr_time(tol=0)
                converged = np.all(tau * 50 < sampler.iteration)
                print("Current iteration: {}".format(sampler.iteration))
                print("Autocorrelation times: {}".format(tau * 50))
            except emcee.autocorr.AutocorrError:
                print("Autocorrelation time could not be estimated. Continuing...")

            # Estimate completion time
            time_per_iter = (end - start) / chain_len
            time_left = (
                np.max(tau) 
                * (chain_len - sampler.iteration)
            ) * time_per_iter if np.isfinite(tau).all() else np.inf
            print("Estimated time left: {}s".format(time_left))

            # Check data has saved correctly
            samples = backend.get_chain()
            log_prob = backend.get_log_prob()
            blobs = backend.get_blobs()
            model_vectors = blobs['model_vector']

            assert samples.shape == (sampler.iteration, nwalkers, ndim)
            assert log_prob.shape == (sampler.iteration, nwalkers)
            assert model_vectors.shape == (sampler.iteration, nwalkers, len(data_vector))

            if args.profile:
                break

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run MCMC to estimate cosmological parameters')
    parser.add_argument('--mcmc_dir', type=str, default='mcmc', help='Directory to save MCMC output')
    parser.add_argument('--z0', type=float, default=0.11, help='Smail parameter z0')
    parser.add_argument('--alpha', type=float, default=0.71, help='Smail parameter alpha')
    parser.add_argument('--n_bins', type=int, default=5, help='Number of redshift bins')
    parser.add_argument('--n_walkers', type=int, default=48, help='Number of walkers')
    parser.add_argument('--chain_len', type=int, default=1000, help='Length of chain to run between convergence checks')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--n_threads', type=int, default=cpu_count(), help='Number of threads to use')
    parser.add_argument('--profile', action='store_true', help='Run the profiler')
    args = parser.parse_args()

    if args.profile:
        pr = cProfile.Profile()
        pr.enable()
        main(args)
        pr.disable()
        ps = pstats.Stats(pr).sort_stats('cumulative')
        ps.print_stats(20)
    else:
        main(args)