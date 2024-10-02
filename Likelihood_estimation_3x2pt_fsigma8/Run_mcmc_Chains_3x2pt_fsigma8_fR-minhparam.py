##########################################################################
import os

# set the environment variable to control the number of threads
# NEEDS TO BE DONE BEFORE CCL IS IMPORTED
original_omp_num_threads = os.environ.get('OMP_NUM_THREADS', None)
os.environ['OMP_NUM_THREADS'] = '1'

""" import useful functions """
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

# cosmology
import pyccl as ccl
from astropy.io import fits
import yaml
import sacc
import time

# covariance - Charlie's version of TJPCov
MODULE_PATH = "/home/c2042999/TJPCov/tjpcov/__init__.py"
MODULE_NAME = "tjpcov"
import importlib
spec = importlib.util.spec_from_file_location(MODULE_NAME, MODULE_PATH)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module 
spec.loader.exec_module(module)
from tjpcov.covariance_calculator import CovarianceCalculator

# Generate data sets
from sklearn.datasets import make_blobs
import sklearn
#print(sklearn.__version__) # should be 1.3.2

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
import matplotlib
#import seaborn as sns

# MCMC
import emcee
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import corner
from chainconsumer import ChainConsumer, Chain, make_sample
from IPython.display import display, Math
from multiprocessing import Pool
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

"""Initialize some things (e.g. emulators and MGCAMB)"""
# Initialise and emulator instance.
emu_fR = FofrBoost()
# Load the nDGP emulator
model_nDGP = BoostPredictor()
# Initialize MGCAMB
pars = camb.CAMBparams()


# Load my functions
from Likelihood_PCADR import *

##########################################################################
#### Train the emulators ####
# Define the redshift interval and forecast years
redshift_range = np.linspace(0.01, 3.5, 500)
forecast_years = ["1", "10"]  # Assuming integers are appropriate

# Create a dictionary to store the redshift distributions
# for each forecast year and galaxy sample
redshift_distribution = {
    "sources": {},
    "lenses": {}
}

for year in forecast_years:
    source_dist = srd.SRDRedshiftDistributions(redshift_range, 
                                               galaxy_sample="source_sample",
                                               forecast_year=year)
    lens_dist = srd.SRDRedshiftDistributions(redshift_range, 
                                             galaxy_sample="lens_sample",
                                             forecast_year=year)

    redshift_distribution["sources"][year] = source_dist.get_redshift_distribution(normalised=True,
                                                                                   save_file=False)
    redshift_distribution["lenses"][year] = lens_dist.get_redshift_distribution(normalised=True,
                                                                                save_file=False)

# Uncomment to check if the dictionary is populated correctly
# print(redshift_distribution["sources"].keys())


bins = {
    "sources": {},
    "lenses": {}
}

# Perform the binning procedure
for year in forecast_years:
    bins["sources"][year] = binning.Binning(redshift_range, 
                                            redshift_distribution["sources"][year],
                                            year).source_bins(normalised=True,
                                                              save_file=False)
    bins["lenses"][year] = binning.Binning(redshift_range, 
                                           redshift_distribution["lenses"][year],
                                           year).lens_bins(normalised=True,
                                                           save_file=False)


#(5, 256)
Binned_distribution_lens = [list(bins["lenses"]["1"].items())[0][1]]
for i in range(4):
    Binned_distribution_lens = np.append(Binned_distribution_lens,\
               [list(bins["lenses"]["1"].items())[i+1][1]], axis=0)

Binned_distribution_source = [list(bins["sources"]["1"].items())[0][1]]
for i in range(4):
    Binned_distribution_source = np.append(Binned_distribution_source,\
               [list(bins["sources"]["1"].items())[i+1][1]], axis=0)

cosmo_test = ccl.Cosmology(Omega_c = 0.27, 
                          Omega_b = 0.046, 
                          h = 0.7, 
                          n_s = 0.974,
                          A_s = 2.01e-9)

Bias_distribution_fiducial = np.array([1.562362*np.ones(len(redshift_range)),
                             1.732963*np.ones(len(redshift_range)),
                             1.913252*np.ones(len(redshift_range)),
                             2.100644*np.ones(len(redshift_range)),
                             2.293210*np.ones(len(redshift_range))])

binned_ell_test = bin_ell_kk(20, 1478.5, 13, Binned_distribution_source)

mockdata_test = Cell(binned_ell_test, \
                cosmo_test, redshift_range , Binned_distribution_source,Binned_distribution_lens,Bias_distribution_fiducial,[0.2,1e-5,1,0,0],\
                Get_Pk2D_obj(cosmo_test, [0.2,1e-5,1,0,0], linear=False, gravity_model="f(R)"),tracer1_type="k", tracer2_type="k")

mockdata_test = Cell(binned_ell_test, \
                cosmo_test, redshift_range , Binned_distribution_source,Binned_distribution_lens,Bias_distribution_fiducial,[0.2,1e-5,1,0,0],\
                Get_Pk2D_obj(cosmo_test, [0.2,1e-5,1,0,0], linear=False, gravity_model="nDGP"),tracer1_type="k", tracer2_type="k")

#### Define log likelihood #####
def log_likelihood(theta, Data, L_ch_inv,Data_fsigma8):
    Omega_c, mu0,Sigma0, A_s1e9, h, n_s, wb, b1, b2, b3, b4, b5 = theta

    Bias_distribution = np.array([b1*np.ones(len(z)),
                             b2*np.ones(len(z)),
                             b3*np.ones(len(z)),
                             b4*np.ones(len(z)),
                             b5*np.ones(len(z))])
    A_s = A_s1e9*1e-9
    MGparams = [0.2,1e-4,1.0,mu0,Sigma0]

    cosmo = ccl.Cosmology(Omega_c = Omega_c, 
                      Omega_b = wb/h**2,
                      h = h,
                      n_s = n_s,
                      A_s = A_s)
    
    cosmo_linear = ccl.Cosmology(Omega_c = Omega_c, 
                      Omega_b = wb/h**2,
                      h = h,
                      n_s = n_s,
                      A_s = A_s,
                      matter_power_spectrum='linear')

    return loglikelihood(Data, cosmo,cosmo_linear, MGparams, L_ch_inv,Bias_distribution,Data_fsigma8)


#### Define log prior #####
def log_prior(theta):
    Omega_c, mu0,Sigma0, A_s1e9, h, n_s, wb, b1, b2, b3, b4, b5 = theta 

    #flat priors
    if not (0.28 < Omega_c + wb/h**2 < 0.36 and -2.0 < mu0 < 0.95 and -2.0 < Sigma0 < 0.95 and mu0 <= 2*Sigma0 + 1.0 \
            and 1.7 < A_s1e9 < 2.5 and 0.92 < n_s < 1 and 0.61 < h < 0.73 and 0.04 < wb/h**2 < 0.06 \
           and 0.8 < b1 < 3.0 and 0.8 < b2 < 3.0 and 0.8 < b3 < 3.0 and 0.8 < b4 < 3.0 and 0.8 < b5 < 3.0):
        return -np.inf
        
    gauss_funct = scipy.stats.multivariate_normal(mu_prior, cov_prior)
    
    return gauss_funct.logpdf([n_s, wb])

#### Define log probability #####
def log_probability(theta, Data, L_ch_inv,Data_fsigma8):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, Data, L_ch_inv,Data_fsigma8)

def main(args):

    # Define cosmology
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

    #### Get Planck priors #####
    global sampler_Planck_arr, mu_prior, cov_prior
    sampler_Planck_arr = np.load("/home/c2042999/PCA_project/Prior_Planck_arr.npy")
    mu_prior = [cosmo_universe['n_s'], cosmo_universe["Omega_b"]*cosmo_universe["h"]**2]
    cov_prior = np.cov(sampler_Planck_arr.T)

    # Create the output directory
    mcmc_dir = args.mcmc_dir
    if not os.path.exists(mcmc_dir):
        os.makedirs(mcmc_dir)

    ### Load and collect the data for likelihood
    
    ## Reload if needed
    #command = 'python Get_Data_3x2pt_fsigma8_MG.py --OmgC {} --OmgB {} --h {} --ns {} --As {} --gravity_flag "f(R)" --MG_param {}'.format(cosmo_universe["Omega_c"],cosmo_universe["Omega_b"],cosmo_universe["h"], cosmo_universe["n_s"],cosmo_universe["A_s"],fR_universe)
    #os.system(command)
    ## Collect the data
    npzfile = np.load("Data_storage.npz")

    C_ell_data_mock = [npzfile['C_ell_data'],npzfile['ell_data'],npzfile['z'],npzfile['Binned_distribution_source'],\
                      npzfile['Binned_distribution_lens'],20,1478.5,13]
    Data_fsigma8= [npzfile['z_eff_fsigma8'], npzfile['fsigma8_data'],np.matrix(npzfile['invcov_fsigma8'])]
    L_choleski_inv = np.matrix(npzfile['L_ch_inv'])
    gauss_invcov_rotated = np.matrix(npzfile['Inverse_cov'])
    global z
    z = npzfile['z']
    Bias_distribution_fiducial = np.array([1.229*np.ones(len(z)),
                             1.362*np.ones(len(z)),
                             1.502*np.ones(len(z)),
                             1.648*np.ones(len(z)),
                             1.799*np.ones(len(z))])
    
    # Set the random seed for reproducibility
    np.random.seed(args.seed)

    # Initialize the walkers
    Omega_c_est = 0.27
    h_est = 0.7
    A_s1e9_est = 2.1
    n_s_est = 0.94
    mu0_est = 0.1
    Sigma0_est = 0.05
    wb_est = 0.0223
    b1_est = Bias_distribution_fiducial[0][0]
    b2_est = Bias_distribution_fiducial[1][0]
    b3_est = Bias_distribution_fiducial[2][0]
    b4_est = Bias_distribution_fiducial[3][0]
    b5_est = Bias_distribution_fiducial[4][0]

    # Initialize the walkers
    pos_zero = [Omega_c_est, mu0_est,Sigma0_est, A_s1e9_est, h_est, n_s_est, wb_est,b1_est,b2_est,b3_est,b4_est,b5_est] \
+ np.append(np.append(1e-3 * np.random.randn(args.n_walkers, 5), 1e-5*np.random.randn(args.n_walkers, 2), axis = 1), \
            1e-3 * np.random.randn(args.n_walkers, 5), axis = 1)
    nwalkers, ndim = pos_zero.shape

    chain_len = args.chain_len

    converged = False # Convergence flag

    # Check for existing backend file
    backend_file = os.path.join(mcmc_dir, 'chain_outputs.h5')
    if os.path.exists(backend_file):
        backend = emcee.backends.HDFBackend(backend_file)
        if backend.iteration == 0: # If the file is empty, start from scratch
            pos = pos_zero
        else:
            pos = None # backend will resume from previous position by default
        print("Resuming from existing backend file at iteration {}".format(backend.iteration))
    else:
        backend = emcee.backends.HDFBackend(backend_file)
        backend.reset(nwalkers, ndim) # Reset the backend
        pos = pos_zero
        print("No existing backend file found. Starting from scratch")

    # Initialise a pool 
    with Pool(args.n_threads) as pool:

        # Set up the sampler
        sampler = emcee.EnsembleSampler(
            nwalkers,
            ndim,
            log_probability,
            args=(C_ell_data_mock, L_choleski_inv,Data_fsigma8),
            pool=pool,
            backend=backend
        )
        
        while not converged:
            # Sample 
            start = time.time()
            sampler.run_mcmc(pos, chain_len, progress=True)
            end = time.time()

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

            assert samples.shape == (sampler.iteration, nwalkers, ndim)
            assert log_prob.shape == (sampler.iteration, nwalkers)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run MCMC to estimate cosmological parameters')
    parser.add_argument('--mcmc_dir', type=str, default='mcmc', help='Directory to save MCMC output')
    parser.add_argument('--n_walkers', type=int, default=48, help='Number of walkers')
    parser.add_argument('--chain_len', type=int, default=1000, help='Length of chain to run between convergence checks')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--n_threads', type=int, default=cpu_count(), help='Number of threads to use')
    args = parser.parse_args()

    main(args)