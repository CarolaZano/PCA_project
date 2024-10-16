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

# Initialise and emulator instance.
emu_fR = FofrBoost()

from Likelihood_PCADR import *

from schwimmbad import MPIPool

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


print("start script")

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

# Define cosmology
cosmo_universe = ccl.Cosmology(Omega_c = 0.269619, 
                          Omega_b = 0.050041,
                          h = 0.6688,
                          n_s = 0.9626,
                          A_s = 2.092e-9)

cosmo_universe_linear = ccl.Cosmology(Omega_c = 0.269619, 
                          Omega_b = 0.050041,
                          h = 0.6688,
                          n_s = 0.9626,
                          A_s = 2.092e-9,
                          matter_power_spectrum='linear')

fR_universe = 3e-5
H0rc_universe = 0.0
MGParam_universe = [H0rc_universe,fR_universe,1,0,0]


value1 = [cosmo_universe["Omega_c"], 0.0, 0.0,cosmo_universe["A_s"]*1e9, cosmo_universe["h"],\
          cosmo_universe["n_s"],cosmo_universe["Omega_b"]*cosmo_universe["h"]**2,\
         Bias_distribution_fiducial[0][0], Bias_distribution_fiducial[1][0],\
         Bias_distribution_fiducial[2][0],Bias_distribution_fiducial[3][0],\
         Bias_distribution_fiducial[4][0]]


# Create the output directory
mcmc_dir = "/home/c2042999/PCA_project/Likelihood_estimation_3x2pt_fsigma8/mcmc"

#### Get Planck priors #####
sampler_Planck_arr = np.load("/home/c2042999/PCA_project/Prior_Planck_arr.npy")
mu_prior = [cosmo_universe['n_s'], cosmo_universe["Omega_b"]*cosmo_universe["h"]**2]
cov_prior = np.cov(sampler_Planck_arr.T)

#### Define log prior #####
def log_prior(theta):
    Omega_c, mu0,Sigma0, A_s1e9, h, n_s, wb, b1, b2, b3, b4, b5 = theta 

    #flat priors
    if not (0.28 < Omega_c + wb/h**2 < 0.36 and 0.0 < mu0 < 1.0 and -0.7 < Sigma0 < 0.7 \
            and 1.7 < A_s1e9 < 2.5 and 0.92 < n_s < 1 and 0.61 < h < 0.73 and 0.04 < wb/h**2 < 0.06 \
           and 0.8 < b1 < 3.0 and 0.8 < b2 < 3.0 and 0.8 < b3 < 3.0 and 0.8 < b4 < 3.0 and 0.8 < b5 < 3.0):
        return -np.inf
        
    gauss_funct = scipy.stats.multivariate_normal(mu_prior, cov_prior)
    
    return gauss_funct.logpdf([n_s, wb])

##### Load and collect the data for likelihood ###

## Reload if needed
#command = 'python Get_Data_3x2pt_fsigma8_MG.py --OmgC {} --OmgB {} --h {} --ns {} --As {} --gravity_flag "f(R)" --MG_param {}'.format(cosmo_universe["Omega_c"],cosmo_universe["Omega_b"],cosmo_universe["h"], cosmo_universe["n_s"],cosmo_universe["A_s"],fR_universe)
#os.system(command)
## Collect the data
npzfile = np.load("/home/c2042999/PCA_project/Likelihood_estimation_3x2pt_fsigma8/Data_storage.npz")

C_ell_data_mock = [npzfile['C_ell_data'],npzfile['ell_data'],npzfile['z'],npzfile['Binned_distribution_source'],\
                    npzfile['Binned_distribution_lens'],20,1478.5,13]
Data_fsigma8= [npzfile['z_eff_fsigma8'], npzfile['fsigma8_data'],np.matrix(npzfile['invcov_fsigma8'])]
L_choleski_inv = np.matrix(npzfile['L_ch_inv'])

gauss_invcov_rotated = np.matrix(npzfile['Inverse_cov'])

z = npzfile['z']
Bias_distribution_fiducial = np.array([1.229*np.ones(len(z)),
                            1.362*np.ones(len(z)),
                            1.502*np.ones(len(z)),
                            1.648*np.ones(len(z)),
                            1.799*np.ones(len(z))])

## add C_ell_data_mock, L_choleski_inv, Data_fsigma8
#### Define log probability #####
def log_probability(theta):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, C_ell_data_mock, L_choleski_inv,Data_fsigma8)

# Set the random seed for reproducibility
np.random.seed(10)

# Initialize the walkers
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

n_steps = 5
nwalkers = cpu_count()
print(nwalkers)

# Initialize the walkers
pos = [Omega_c_est, mu0_est,Sigma0_est, A_s1e9_est, h_est, n_s_est, wb_est,b1_est,b2_est,b3_est,b4_est,b5_est] + np.append(np.append(1e-3 * np.random.randn(nwalkers, 5), 1e-5*np.random.randn(nwalkers, 2), axis = 1), 1e-3 * np.random.randn(nwalkers, 5), axis = 1)

nwalkers, ndim = pos.shape

with MPIPool() as pool:
    if not pool.is_master():
        pool.wait()
        sys.exit(0)
    sampler_PCA = emcee.EnsembleSampler(nwalkers, ndim, log_probability, pool=pool)
    start = time.time()
    sampler_PCA.run_mcmc(pos, n_steps, progress=True)
    end = time.time()
    multi_time = end - start
    print("Multiprocessing took {0:.1f} seconds".format(multi_time))
    #print("{0:.1f} times faster than serial".format(serial_time / multi_time))

############## PRINT AND SAVE THINGS ###############
tau = sampler_PCA.get_autocorr_time(tol=0)

max_tau_ratio = np.max(tau * 100 / sampler_PCA.iteration)
print("ratio convergence (want < 1) = ", max_tau_ratio)
print(sampler_PCA.chain.shape)
np.save(mcmc_dir + "/CG_fR_GammaParam_PCAcut.npy", sampler_PCA.chain)

print("script ran successfully")