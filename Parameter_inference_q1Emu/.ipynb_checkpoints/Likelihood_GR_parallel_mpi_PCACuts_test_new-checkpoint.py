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

# f(R) emu (eMANTIS)
from emantis import FofrBoost

# Initialise and emulator instance.
emu_fR = FofrBoost()

# screened gamma
import MGEmu as mgemu
emulator_MGEmu = mgemu.MG_boost(model='gamma')

from LikelihoodFuncts_PCADR_muSigma_4theories import *

from schwimmbad import MPIPool
from multiprocessing import cpu_count
import signal

# Set up a flag for termination signal
terminate_flag = False

# Signal handler to catch termination signals (like SIGTERM from SLURM)
def handle_sigterm(signum, frame):
    global terminate_flag
    terminate_flag = True
    print("SIGTERM signal received. Preparing to exit...")

# Register the signal handler for SIGTERM (time limit on SLURM jobs)
signal.signal(signal.SIGTERM, handle_sigterm)

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

# GET F(R) LINEAR POWER SPECTRUM, fiducial
##########################
idx_mdom = 0
a_solver = np.linspace(1/50,1.1,50)
k_solver = np.logspace(-5,4,200)

Soln = odeint(solverGrowth_nDGP, [a_solver[0], (E(cosmo_universe, a_solver[0])*a_solver[0]**3)], a_solver,\
              args=(cosmo_universe,[0,0,0,0,0]), mxstep=int(1e4))

Delta_GR = Soln.T[0]/Soln.T[0][idx_mdom]

Delta = []
for i in range(len(k_solver)):
    Soln = odeint(solverGrowth_fR, [a_solver[0], (E(cosmo_universe, a_solver[0])*a_solver[0]**3)], a_solver, \
                  args=(cosmo_universe,[0,1e-4,0,0,0], k_solver[i]), mxstep=int(1e4))
    
    Delta_i = Soln.T[0]
    Delta.append(Delta_i/Delta_i[idx_mdom])

Delta = np.array(Delta)

# Get Pk linear in GR

interp_fR_Pk = scipy.interpolate.RegularGridInterpolator((k_solver,a_solver), (Delta / Delta_GR)**2,bounds_error=False, fill_value=1.0)
##########################


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
    if not (0.2899 < Omega_c + wb/h**2 < 0.3392 and\
            -1.5 < mu0 < 1.5 and -1.5 < Sigma0 < 1.5 \
            and 1.7 < A_s1e9 < 2.5 and 0.9432 < n_s < 0.9862 and 0.629 < h < 0.73 and 0.04044 < wb/h**2 < 0.05686 \
           and 0.8 < b1 < 3.0 and 0.8 < b2 < 3.0 and 0.8 < b3 < 3.0 and 0.8 < b4 < 3.0 and 0.8 < b5 < 3.0):
        return -np.inf

    gauss_funct = scipy.stats.multivariate_normal(mu_prior, cov_prior)

    return gauss_funct.logpdf([n_s, wb])

##### Load and collect the data for likelihood ###

## Reload if needed
#command = 'python Get_Data_3x2pt_fsigma8_MG.py --OmgC {} --OmgB {} --h {} --ns {} --As {} --gravity_flag "f(R)" --MG_param {}'.format(cosmo_universe["Omega_c"],cosmo_universe["Omega_b"],cosmo_universe["h"], cosmo_universe["n_s"],cosmo_universe["A_s"],fR_universe)
#os.system(command)
## Collect the data
npzfile = np.load("Data_storage_GR.npz")

C_ell_data_mock = [npzfile['C_ell_data'],npzfile['ell_data'],npzfile['z'],npzfile['Binned_distribution_source'],\
                    npzfile['Binned_distribution_lens'],20,1478.5,13]
Data_fsigma8= [npzfile['z_eff_fsigma8'], npzfile['fsigma8_data'],np.matrix(npzfile['invcov_fsigma8'])]
L_choleski_inv = np.matrix(npzfile['L_ch_inv'])

gauss_invcov_rotated = np.matrix(npzfile['Inverse_cov'])

z = npzfile['z']

#### Define log probability #####
def log_probability(theta):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, C_ell_data_mock, L_choleski_inv,Data_fsigma8)

print("lp fid= ",log_probability([cosmo_universe["Omega_c"],0,0,cosmo_universe["A_s"]*1e9,cosmo_universe["h"],cosmo_universe["n_s"],\
                      cosmo_universe["Omega_b"]*cosmo_universe["h"]**2, Bias_distribution_fiducial[0][0],Bias_distribution_fiducial[1][0],\
                     Bias_distribution_fiducial[2][0],Bias_distribution_fiducial[3][0],Bias_distribution_fiducial[4][0]]))
print("lp random= ",log_probability([2.80767748e-01, -1.52992161e-01 , 2.11771587e-03 , 2.11379353e+00,\
  6.59369847e-01 , 9.62957982e-01 , 2.24256179e-02,  1.60484155e+00,\
  1.77209514e+00 , 1.94676198e+00 , 2.13164976e+00 , 2.32232300e+00]))