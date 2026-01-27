##########################################################################
import os

# set the environment variable to control the number of threads
# NEEDS TO BE DONE BEFORE CCL IS IMPORTED
original_omp_num_threads = os.environ.get('OMP_NUM_THREADS', None)
os.environ['OMP_NUM_THREADS'] = '1'

# Generic
import numpy as np
import scipy
import sys
from scipy.integrate import odeint
import itertools
from functools import lru_cache
import scipy.integrate
import gc

# cosmology
import pyccl as ccl
import time

# SRD Binning
import srd_redshift_distributions as srd
import binning

# MCMC
import emcee

# nDGP NL and lin Pk
from nDGPemu import BoostPredictor

# f(R) emu (eMANTIS)
from emantis import FofrBoost

# Initialise and emulator instance.
emu_fR = FofrBoost()

from LikelihoodFuncts_PCADR_muSigma import *

from schwimmbad import MPIPool
from multiprocessing import cpu_count
from multiprocessing import Pool
import signal
import argparse

# Set up a flag for termination signal
terminate_flag = False

# Signal handler to catch termination signals (like SIGTERM from SLURM)
def handle_sigterm(signum, frame):
    global terminate_flag
    terminate_flag = True
    print("SIGTERM signal received. Preparing to exit...")

# Register the signal handler for SIGTERM (time limit on SLURM jobs)
signal.signal(signal.SIGTERM, handle_sigterm)

# globals (initialized later)
data_storage_file = None
ell_min_mockdata = None
ell_max_mockdata = None
ell_bin_num_mockdata = None
w_c_prior_min = None
w_c_prior_max = None
Omega_b_prior_min = None
Omega_b_prior_max = None
h_prior_min = None
h_prior_max = None
ns_prior_min = None
ns_prior_max = None
As1e9_prior_min = None
As1e9_prior_max = None
mu0_prior_min = None
mu0_prior_max = None
Sigma0_prior_min = None
Sigma0_prior_max = None
Planck_prior_true = None

#### Define log likelihood #####
def log_likelihood(theta, Data, L_ch_inv,Data_fsigma8):
    Omega_c, mu0,Sigma0, A_s1e9, h, n_s, wb, b1, b2, b3, b4, b5 = theta 
    Bias_distribution = np.array([b1*np.ones(len(z)),
                             b2*np.ones(len(z)),
                             b3*np.ones(len(z)),
                             b4*np.ones(len(z)),
                             b5*np.ones(len(z))])
    #h = cosmo_universe["h"]
    #A_s = cosmo_universe["A_s"]
    A_s = A_s1e9*1e-9
    #n_s = cosmo_universe["n_s"]
    MGparams = [0.2,1e-4,1.0,mu0, Sigma0]

    cosmo = ccl.Cosmology(Omega_c = Omega_c, 
                      Omega_b = wb/h**2,
                      h = h,
                      n_s = n_s,
                      A_s = A_s)

    
    return loglikelihood(Data, cosmo, MGparams, L_ch_inv,Bias_distribution,Data_fsigma8)

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

# Define cosmology
cosmo_universe = ccl.Cosmology(Omega_c = 0.269619, 
                          Omega_b = 0.050041,
                          h = 0.6688,
                          n_s = 0.9626,
                          A_s = 2.092e-9)

fR_universe = 0.0
H0rc_universe = 0.0
MGParam_universe = [H0rc_universe,fR_universe,0,0,0]

P_delta2D_GR_lin_universe = Get_Pk2D_obj_kk_GR_lin(cosmo_universe)
P_delta2D_GR_nl_universe = Get_Pk2D_obj_kk_GR_nl(cosmo_universe)
# Get fake interp objects (just to load emulator grids)
k_solver = np.logspace(-4, 1, 50)
a_solver = np.linspace(0.2, 1.0, 25)
interp_fR_Pk = scipy.interpolate.RegularGridInterpolator((k_solver,a_solver), np.zeros((len(k_solver),len(a_solver))),bounds_error=False, fill_value=1.0)

mockdata_test = Cell(binned_ell_test, \
                cosmo_universe, redshift_range , Binned_distribution_source,Binned_distribution_lens,Bias_distribution_fiducial,[0.2,1e-5,1,0,0],\
                Get_Pk2D_obj(P_delta2D_GR_nl_universe,interp_fR_Pk,cosmo_universe, [0.2,1e-5,1,0,0], linear=False, gravity_model="f(R)"),tracer1_type="k", tracer2_type="k")

mockdata_test = Cell(binned_ell_test, \
                cosmo_universe, redshift_range , Binned_distribution_source,Binned_distribution_lens,Bias_distribution_fiducial,[0.2,1e-5,1,0,0],\
                Get_Pk2D_obj(P_delta2D_GR_nl_universe,interp_fR_Pk,cosmo_universe, [0.2,1e-5,1,0,0], linear=False, gravity_model="nDGP"),tracer1_type="k", tracer2_type="k")


value1 = [cosmo_universe["Omega_c"], 0.0, 0.0,cosmo_universe["A_s"]*1e9, cosmo_universe["h"],\
          cosmo_universe["n_s"],cosmo_universe["Omega_b"]*cosmo_universe["h"]**2,\
         Bias_distribution_fiducial[0][0], Bias_distribution_fiducial[1][0],\
         Bias_distribution_fiducial[2][0],Bias_distribution_fiducial[3][0],\
         Bias_distribution_fiducial[4][0]]


# Create the output directory
mcmc_dir = "mcmc"

#### Get Planck priors #####
sampler_Planck_arr = np.load("Prior_Planck_arr.npy")
mu_prior = [cosmo_universe['n_s'], cosmo_universe["Omega_b"]*cosmo_universe["h"]**2]
cov_prior = np.cov(sampler_Planck_arr.T)

#### Define log prior #####
def log_prior(theta):
    Omega_c, mu0,Sigma0, A_s1e9, h, n_s, wb, b1, b2, b3, b4, b5 = theta
    
    #flat priors
    if not (w_c_prior_min < Omega_c + wb/h**2 < w_c_prior_max and\
            mu0_prior_min < mu0 < mu0_prior_max and Sigma0_prior_min < Sigma0 < Sigma0_prior_max and mu0 <= 2*Sigma0 + 1.0 \
            and As1e9_prior_min < A_s1e9 < As1e9_prior_max and ns_prior_min < n_s < ns_prior_max and h_prior_min < h < h_prior_max and Omega_b_prior_min < wb/h**2 < Omega_b_prior_max \
        and 0.8 < b1 < 3.0 and 0.8 < b2 < 3.0 and 0.8 < b3 < 3.0 and 0.8 < b4 < 3.0 and 0.8 < b5 < 3.0):
        return -np.inf
    elif not Planck_prior_true:
        return 0.0
    else:
        # Planck priors
        gauss_funct = scipy.stats.multivariate_normal(mu_prior, cov_prior)
        return gauss_funct.logpdf([n_s, wb])


##### Load and collect the data for likelihood ###

## Reload if needed
#command = 'python Get_Data_3x2pt_fsigma8_MG.py --OmgC {} --OmgB {} --h {} --ns {} --As {} --gravity_flag "f(R)" --MG_param {}'.format(cosmo_universe["Omega_c"],cosmo_universe["Omega_b"],cosmo_universe["h"], cosmo_universe["n_s"],cosmo_universe["A_s"],fR_universe)
#os.system(command)
## Collect the data

npzfile = np.load("Data_storage_nDGP_partial.npz")
npzfile_GR = np.load("../Parameter_inference_GR/Data_storage_GR.npz")

C_ell_data_mock = [npzfile['C_ell_data'],npzfile['ell_data'],npzfile['z'],npzfile['Binned_distribution_source'],\
                    npzfile['Binned_distribution_lens'],20,1478.5,13]

Data_fsigma8= [npzfile['z_eff_fsigma8'], npzfile['fsigma8_data'],np.matrix(npzfile['invcov_fsigma8'])]

L_choleski_inv = np.matrix(npzfile_GR['L_ch_inv'])

gauss_invcov_rotated = np.matrix(npzfile_GR['Inverse_cov'])

z = npzfile['z']

#### Define log probability #####
def log_probability(theta):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, C_ell_data_mock, L_choleski_inv,Data_fsigma8)

def main(args):
    ###############################################################################
    # Set up the input variables
    ###############################################################################
    # Data storage file
    global data_storage_file
    global ell_min_mockdata, ell_max_mockdata, ell_bin_num_mockdata
    global w_c_prior_min, w_c_prior_max
    global Omega_b_prior_min, Omega_b_prior_max
    global h_prior_min, h_prior_max
    global ns_prior_min, ns_prior_max
    global As1e9_prior_min, As1e9_prior_max
    global mu0_prior_min, mu0_prior_max
    global Sigma0_prior_min, Sigma0_prior_max
    global Planck_prior_true

    data_storage_file = args.datafile

    ell_min_mockdata = args.ellmin
    ell_max_mockdata = args.ellmax
    ell_bin_num_mockdata = args.ellbin
    w_c_prior_min = args.wC_min
    w_c_prior_max = args.wC_max
    Omega_b_prior_min = args.OmgB_min
    Omega_b_prior_max = args.OmgB_max
    h_prior_min = args.h_min
    h_prior_max = args.h_max
    ns_prior_min = args.ns_min
    ns_prior_max = args.ns_max
    As1e9_prior_min = args.As1e9_min
    As1e9_prior_max = args.As1e9_max
    mu0_prior_min = args.mu0_min
    mu0_prior_max = args.mu0_max
    Sigma0_prior_min = args.Sigma0_min
    Sigma0_prior_max = args.Sigma0_max
    Planck_prior_true = args.Planck_prior_true
        
    # Set the random seed for reproducibility
    np.random.seed(10)

    # Set up parameters
    n_steps = 20000  # Total steps you want to run
    nwalkers = cpu_count()  # Number of walkers (can be different if needed)
    chain_len = 100  # Number of steps after which to clear pool memory

    # Initialize walkers
    Omega_c_est = 0.27
    h_est = 0.7
    A_s1e9_est = 2.1
    n_s_est = 0.96
    mu0_est = 0.1
    Sigma0_est = 0.05
    wb_est = 0.0223
    b1_est = Bias_distribution_fiducial[0][0]
    b2_est = Bias_distribution_fiducial[1][0]
    b3_est = Bias_distribution_fiducial[2][0]
    b4_est = Bias_distribution_fiducial[3][0]
    b5_est = Bias_distribution_fiducial[4][0]

    # Random initial positions of walkers
    pos = [Omega_c_est, mu0_est, Sigma0_est, A_s1e9_est, h_est, n_s_est, wb_est, b1_est, b2_est, b3_est, b4_est, b5_est] + \
        np.append(np.append(1e-3 * np.random.randn(nwalkers, 5), 1e-5 * np.random.randn(nwalkers, 2), axis=1),
                    1e-3 * np.random.randn(nwalkers, 5), axis=1)

    nwalkers, ndim = pos.shape

    # Create the output directory and set up the HDF5 backend
    mcmc_dir = "mcmc"
    filename = mcmc_dir + "/mcmc_nDGP_muSigma_PCACuts.h5"
    backend = emcee.backends.HDFBackend(filename)

    # Optionally reset the backend if starting a new run
    # backend.reset(nwalkers, ndim)
    converged = False
    with Pool(nwalkers) as pool:
        # Initialize the sampler
        sampler_PCA = emcee.EnsembleSampler(nwalkers, ndim, log_probability, backend=backend, pool=pool)
        pos = sampler_PCA.get_last_sample() if backend.iteration > 0 else pos

        while not converged:
            gc.collect(generation=0)
            sampler_PCA.run_mcmc(pos, chain_len, progress=True, store=True)
            
            # Clear references in the worker pool
            pool.close()
            pool.join()
            del pool  # Ensure the pool object is removed
            gc.collect()  # Collect any lingering garbage
            pool = Pool(nwalkers)
            sampler_PCA.pool = pool
            pos = sampler_PCA.get_last_sample() if backend.iteration > 0 else pos
            
            # Check convergence
            try:
                tau = sampler_PCA.get_autocorr_time(tol=0)
                converged = np.all(tau * 100 < sampler_PCA.iteration)
            except emcee.autocorr.AutocorrError:
                pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Get simulated 3x2pt+fsigma8 data for GR universe.')
    # MCMC parameters
    parser.add_argument('--datafile', type=str, default="Data_storage_GR.npz", help='Data storage file path')
    parser.add_argument('--ellmin', type=float, default=20, help='Minimum ell value for mock data')
    parser.add_argument('--ellmax', type=float, default=1478.5, help='Maximum ell value for mock data')
    parser.add_argument('--ellbin', type=int, default=13, help='Number of ell bins for mock data')
    parser.add_argument('--wC_min', type=float, default=0.28, help='Minimum prior for Omega_c*h^2')
    parser.add_argument('--wC_max', type=float, default=0.36, help='Maximum prior for Omega_c*h^2')
    parser.add_argument('--OmgB_min', type=float, default=0.04, help='Minimum prior for Omega_b')
    parser.add_argument('--OmgB_max', type=float, default=0.06, help='Maximum prior for Omega_b')
    parser.add_argument('--h_min', type=float, default=0.61, help='Minimum prior for h')
    parser.add_argument('--h_max', type=float, default=0.73, help='Maximum prior for h')
    parser.add_argument('--ns_min', type=float, default=0.92, help='Minimum prior for n_s')
    parser.add_argument('--ns_max', type=float, default=1.0, help='Maximum prior for n_s')
    parser.add_argument('--As1e9_min', type=float, default=1.7, help='Minimum prior for A_s*10^9')
    parser.add_argument('--As1e9_max', type=float, default=2.5, help='Maximum prior for A_s*10^9')
    parser.add_argument('--mu0_min', type=float, default=-1.5, help='Minimum prior for mu0')
    parser.add_argument('--mu0_max', type=float, default=1.5, help='Maximum prior for mu0')
    parser.add_argument('--Sigma0_min', type=float, default=-1.5, help='Minimum prior for Sigma0')
    parser.add_argument('--Sigma0_max', type=float, default=1.5, help='Maximum prior for Sigma0')
    parser.add_argument('--Planck_prior_true', action='store_true', help='Use Planck priors on (n_s, Omega_b*h^2) if set')
    args = parser.parse_args()

    main(args)
