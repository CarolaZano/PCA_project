#######################################################################################
# This file runs an mcmc to constrain LSST-like 3x2pt data with some fsigma8 datapoints
# and Planck priors on wb and ns for an nDGP universe with standard conservative scale 
# cuts and saves a numpy array with the chains from the run in the current directory.
#######################################################################################

#######################################################################################
""" Section 0: import useful functions """
#######################################################################################

print("Importing functions and initializing")

# Generic
import pandas as pd
import numpy as np
import scipy
from itertools import islice, cycle
import math
import os
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

# MGCAMB - (!) do your own installation here (!)
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

# Initialise EMANTIS emulator.
emu_fR = FofrBoost()

# Load the nDGP emulator
model_nDGP = BoostPredictor()

# Initialize MGCAMB
pars = camb.CAMBparams()

#######################################################################################
'''Section 1: Create mock redshift and bias distributions from DESC SRD'''
#######################################################################################

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

Binned_distribution_lens = [list(bins["lenses"]["1"].items())[0][1]]
for i in range(4):
    Binned_distribution_lens = np.append(Binned_distribution_lens,\
               [list(bins["lenses"]["1"].items())[i+1][1]], axis=0)

Binned_distribution_source = [list(bins["sources"]["1"].items())[0][1]]
for i in range(4):
    Binned_distribution_source = np.append(Binned_distribution_source,\
               [list(bins["sources"]["1"].items())[i+1][1]], axis=0)

z = redshift_range

# Match SRD from Table 2 in https://arxiv.org/pdf/2212.09345
Bias_distribution_fiducial = np.array([1.229*np.ones(len(z)),
                             1.362*np.ones(len(z)),
                             1.502*np.ones(len(z)),
                             1.648*np.ones(len(z)),
                             1.799*np.ones(len(z))])

#######################################################################################
''' Section 2: Define functions to get various MG P(k)s from Emulators, Boltzmann solvers etc. '''
#######################################################################################

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~ Non-linear matter power spectra ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# NL matter power spectra in nDGP
def P_k_NL_nDGP(cosmo, MGparams, k, a):
    """
    input k (array) -> wavevector, units 1/Mpc
    input a (float) -> scale factor (1/(1+z))
    input cosmo (cosmology object) -> Cosmology object from CCL
    input MGparams (array) -> Modified gravity parameters ([Omega_rc, fR0, n, mu])
    
    output Pk_nDGP (array) -> Nonlinear matter power spectrum for nDGP gravity, units (Mpc)^3
    """
    # Turn k into units of h/Mpc
    k = k/cosmo["h"]

    H0rc, fR0, n, mu, Sigma = MGparams

    # nDGP emulator - get boost
    cosmo_params = {'Om':cosmo["Omega_m"],
                    'ns':cosmo["n_s"],
                    'As':cosmo["A_s"],
                    'h':cosmo["h"],
                    'Ob':cosmo["Omega_b"]}

    pkratio_nDGP = model_nDGP.predict(H0rc, 1/a -1 , cosmo_params, k_out=k)

    # Get GR power spectrum
    
    Pk_ccl = ccl.power.nonlin_power(cosmo, k*cosmo["h"], a=a) # units (Mpc)^3
    return pkratio_nDGP*Pk_ccl

# NL matter power spectra in fR
def P_k_NL_fR(cosmo, MGparams, k, a):
    """
    input k (array) -> wavevector, units 1/Mpc
    input a (float) -> scale factor (1/(1+z))
    input cosmo (cosmology object) -> Cosmology object from CCL
    input MGparams (array) -> Modified gravity parameters ([Omega_rc, fR0, n, mu])
    
    output Pk_fR (array) -> Nonlinear matter power spectrum for Hu-Sawicki fR gravity, units (Mpc)^3
    """
    H0rc, fR0, n, mu, Sigma = MGparams

    sigma8_VAL_lcdm = ccl.sigma8(cosmo)
    
    pkratio_fR = emu_fR.predict_boost(cosmo["Omega_m"], sigma8_VAL_lcdm, -np.log10(fR0), a, k = k/cosmo["h"]**2)
    # k is in units [h/Mpc]

    Pk_ccl = ccl.power.nonlin_power(cosmo, k, a=a) # units (Mpc)^3
    Pk = pkratio_fR*Pk_ccl

    return Pk

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~ linear matter power spectra ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# dimensionless hubble parameter in GR
def E(cosmoMCMCStep, a):
    Omg_r = cosmoMCMCStep["Omega_g"]*(1+ 3.046*7/8 * (4/11)**(7/8))
    return np.sqrt(cosmoMCMCStep["Omega_m"]/a**3 +Omg_r/a**4 + (1 - cosmoMCMCStep["Omega_m"] - Omg_r))

# deriv. of E wrt scale factor, GR
def dEda(cosmo, a):
    Omg_r = cosmo["Omega_g"]*(1+ 3.046*7/8 * (4/11)**(7/8))
    E_val = E(cosmo, a)
    
    return (-3*cosmo["Omega_m"]/a**4 -4*Omg_r/a**5)/2/E_val

# mu(k,a) = mu(a) in nDGP (modified gravity parametrization parameter)
def mu_nDGP(MGparams, cosmo, a):
    H0rc, fR0, n, mu, Sigma = MGparams
    if H0rc == 0: # just by convention, we want MGParams = [0,0,0,0] to be gr
        return 1
    elif 1/(4*H0rc**2) == 0:
        return 1
    else:
        Omg_rc = 1/(4*H0rc**2)
        E_val = E(cosmo, a)
        # from ReACT paper
        beta = 1 + E_val/np.sqrt(Omg_rc) * (1+ a*dEda(cosmo, a)/3/E_val)
        return 1 + 1/3/beta
    
def solverGrowth_nDGP(y,a,cosmo, MGparams):
    E_val = E(cosmo, a)
    D , a3EdDda = y
    
    mu = mu_nDGP(MGparams, cosmo, a)
    
    ydot = [a3EdDda / (E_val*a**3), 3*cosmo["Omega_m"]*D*(mu)/(2*E_val*a**2)]
    return ydot
    
#Linear matter power spectrum nDGP
def P_k_nDGP_lin(cosmo, MGparams, k, a):
    """
    input k (array) -> wavevector, units 1/Mpc
    input a (float) -> scale factor (1/(1+z))
    input cosmo (cosmology object) -> Cosmology object from CCL
    input MGparams (array) -> Modified gravity parameters ([Omega_rc, fR0, n,mu])
    
    output Pk_nDGP (array) -> linear matter power spectrum for nDGP gravity, units (Mpc)^3
    """
    
    # Get growth factor in nDGP and GR
    H0rc, fR0, n, mu, Sigma = MGparams
    
    Omega_rc = 1/(4*H0rc**2)
    
    a_solver = np.linspace(1/50,1,100)
    Soln = odeint(solverGrowth_nDGP, [a_solver[0], (E(cosmo, a_solver[0])*a_solver[0]**3)], a_solver, \
                  args=(cosmo,MGparams), mxstep=int(1e4))
    
    Delta = Soln.T[0]
    
    Soln = odeint(solverGrowth_nDGP, [a_solver[0], (E(cosmo, a_solver[0])*a_solver[0]**3)], a_solver,\
                  args=(cosmo,[0,0,0,0,0]), mxstep=int(1e4))
    
    Delta_GR = Soln.T[0]

    # Get Pk linear in GR
    
    Pk_GR = ccl.linear_matter_power(cosmo, k=k, a=a)

    # find the index for matter domination)
    #idx_mdom = np.argmax(a_solver**(-3) / E(cosmo, a_solver)**2)          
    idx_mdom = 0
    # get normalization at matter domination
    Delta_nDGP_49 = Delta[idx_mdom]
    Delta_GR_49 = Delta_GR[idx_mdom]
    return Pk_GR * np.interp(a, a_solver, (Delta / Delta_nDGP_49) **2 / (Delta_GR / Delta_GR_49)**2)  # units (Mpc)^3

# Function for linear matter power spectrum f(R)
@lru_cache(maxsize=128)  # You can adjust maxsize according to your memory constraints
def create_interpolator(cosmo_values, MGparams_tuple):
    
    H0rc, fR0, n, mu, Sigma = MGparams_tuple

    pars = camb.CAMBparams()
    pars.set_cosmology(H0=cosmo_values['h'] * 100, 
                       ombh2=cosmo_values['Omega_b'] * cosmo_values['h']**2, 
                       omch2=cosmo_values['Omega_c'] * cosmo_values['h']**2, 
                       omk=0, mnu=0.0)
    pars.InitPower.set_params(ns=cosmo_values['n_s'], As=cosmo_values['A_s'])
    pars.set_mgparams(MG_flag=3, GRtrans=0.0, QSA_flag=4, F_R0=fR0, FRn=1.0)
    pars.NonLinear = camb.model.NonLinear_none
    PK = camb.get_matter_power_interpolator(pars, nonlinear=False, hubble_units=False, k_hunit=False, zmax=100)
    return PK

# Linear matter power spectrum f(R)
def P_k_fR_lin(cosmo, MGparams, k, a):
    if MGparams[1] == 0:
        return ccl.linear_matter_power(cosmo, k=k, a=a)
    else:
        MGparams_tuple = tuple(MGparams)
        PK = create_interpolator(cosmo, MGparams_tuple)
        return PK.P(1/a-1, k)


# Linear matter power f(R) (function for mu(k,a))
def mu_fR(fR0, cosmo, k, a):
    # k is in units 1/Mpc
    # We want H0 in units 1/Mpc, so H0 = 100h/c
    if fR0 == 0:
        return np.ones(len(k))
    else:
        # from ReACT paper
        f0 = fR0 / (cosmo["h"]*100/3e5)**2
        Zi = (cosmo["Omega_m"] + 4*a**3*(1-cosmo["Omega_m"]))/a**3
        Pi = (k/a)**2 + Zi**3/2/f0/(3*cosmo["Omega_m"] - 4)**2
        return 1 + (k/a)**2/3/Pi

# ~~~~~~~~~~~~~~~~~~~~~~~~ linear matter power spectra - parametrizations ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# DE parametrization, function for mu (MG factor in the poisson equation)
def mu_lin_param(MGparams, cosmoMCMCStep, a):
    H0rc, fR0, n, mu0, Sigma0 = MGparams
    E_val = E(cosmoMCMCStep, a)
    return 1 + mu0/E_val**2

# DE parametrization, function for Sigma (MG factor affecting the Weyl potential)
def sigma_lin_param(MGparams, cosmoMCMCStep, a):
    H0rc, fR0, n, mu0, Sigma0 = MGparams
    E_val = E(cosmoMCMCStep, a)
    return 1 + Sigma0/E_val**2

# coupled ODE for growth with the DE parametrization of mu/Sigma
def solverGrowth_musigma(y,a,cosmoMCMCStep, MGparams):
    E_val = E(cosmoMCMCStep, a)
    D , a3EdDda = y

    mu = mu_lin_param(MGparams, cosmoMCMCStep, a)
    Sigma = sigma_lin_param(MGparams, cosmoMCMCStep, a)
    eta = 2*Sigma/mu - 1
    
    ydot = [a3EdDda / (E_val*a**3), 3*cosmoMCMCStep["Omega_m"]*D*(mu/eta)/(2*E_val*a**2)]
    return ydot

# Linear matter power spectrum for mu/Sigma in the DE parametrization
def P_k_musigma(cosmoMCMCStep, MGparams, k, a):
    
    """
    input k (array) -> wavevector, units 1/Mpc
    input a (float) -> scale factor (1/(1+z))
    input cosmo (cosmology object) -> Cosmology object from CCL
    input MGparams (array) -> Modified gravity parameters ([Omega_rc, fR0, n, mu])
    
    output P_k_musigma (array) -> linear matter power spectrum for mu sigma param, units (Mpc)^3
    """
    
    H0rc, fR0, n, mu, Sigma = MGparams
    
    # Get growth factor in nDGP and GR
    a_solver = np.linspace(1/50,1,100)
    Soln = odeint(solverGrowth_musigma, [a_solver[0], (E(cosmoMCMCStep, a_solver[0])*a_solver[0]**3)], a_solver, \
                  args=(cosmoMCMCStep,MGparams), mxstep=int(1e4))
    
    Delta = Soln.T[0]
    Soln = odeint(solverGrowth_musigma, [a_solver[0], (E(cosmoMCMCStep, a_solver[0])*a_solver[0]**3)], a_solver,\
                  args=(cosmoMCMCStep,[0,0,0,0,0]), mxstep=int(1e4))
    
    Delta_GR = Soln.T[0]

    # Get Pk linear in GR
    Pk_GR = ccl.linear_matter_power(cosmoMCMCStep, k=k, a=a)

    # find the index for matter domination)
    idx_mdom = np.argmax(a_solver**(-3) / E(cosmoMCMCStep, a_solver)**2)          
    # get normalization at matter domination
    Delta_49 = Delta[idx_mdom]
    Delta_GR_49 = Delta_GR[idx_mdom]
    
    return Pk_GR * np.interp(a, a_solver, (Delta / Delta_49) **2 / (Delta_GR / Delta_GR_49)**2)  # units (Mpc)^3

#######################################################################################
''' Section 3: Get f and sigma8 for MG theories '''
#######################################################################################

def sigma_8_musigma(cosmo, MGparams, a_array):
    k_val = np.logspace(-4, 3, 3000)
    sigma_8_vals = []

    for a in a_array:
        P_k_vals = P_k_musigma(cosmo, MGparams, k_val, a)
        j1_vals = 3 * scipy.special.spherical_jn(1, k_val * 8 / cosmo["h"], derivative=False) / (k_val * 8 / cosmo["h"])
        integrand = k_val**2 * P_k_vals * j1_vals**2
        integral_val = scipy.integrate.trapz(integrand, x=k_val)
        sigma_8_val = np.sqrt(integral_val / (2 * np.pi**2))
        sigma_8_vals.append(sigma_8_val)
    
    return np.array(sigma_8_vals)

def sigma_8_nDGP(cosmo, MGparams, a_array):
    k_val = np.logspace(-4, 3, 3000)
    sigma_8_vals = []

    for a in a_array:
        P_k_vals = P_k_nDGP_lin(cosmo, MGparams, k_val, a)
        j1_vals = 3 * scipy.special.spherical_jn(1, k_val * 8 / cosmo["h"], derivative=False) / (k_val * 8 / cosmo["h"])
        integrand = k_val**2 * P_k_vals * j1_vals**2
        integral_val = scipy.integrate.trapz(integrand, x=k_val)
        sigma_8_val = np.sqrt(integral_val / (2 * np.pi**2))
        sigma_8_vals.append(sigma_8_val)
    
    return np.array(sigma_8_vals)

def fsigma8_musigma(cosmoMCMCStep, MGparams, a):
    
    """
    input k (array) -> wavevector, units 1/Mpc
    input a (float) -> scale factor (1/(1+z))
    input cosmo (cosmology object) -> Cosmology object from CCL
    input MGparams (array) -> Modified gravity parameters ([Omega_rc, fR0, n, mu])
    
    output P_k_musigma (array) -> linear matter power spectrum for mu sigma param, units (Mpc)^3
    """
    
    H0rc, fR0, n, mu, Sigma = MGparams
    
    # Get growth factor in musigma
    a_solver = np.linspace(1e-3,1,int(1e3))
    Soln = odeint(solverGrowth_musigma, [a_solver[0], (E(cosmoMCMCStep, a_solver[0])*a_solver[0]**3)], a_solver, \
                  args=(cosmoMCMCStep,MGparams), mxstep=int(1e4))
    
    Delta = Soln.T[0]
    a3EdDda = Soln.T[1]
    
    f_musigma_interp = a3EdDda/a_solver**2 / Delta / E(cosmoMCMCStep, a_solver)
    
    f_musigma = np.interp(a, a_solver, f_musigma_interp)

    k_val = np.logspace(-4,3,3000)
    return f_musigma * sigma_8_musigma(cosmoMCMCStep, MGparams, a)

def fsigma8_nDGP(cosmoMCMCStep, MGparams, a):
    
    """
    input k (array) -> wavevector, units 1/Mpc
    input a (float) -> scale factor (1/(1+z))
    input cosmo (cosmology object) -> Cosmology object from CCL
    input MGparams (array) -> Modified gravity parameters ([Omega_rc, fR0, n, mu])
    
    output P_k_musigma (array) -> linear matter power spectrum for mu sigma param, units (Mpc)^3
    """
    
    H0rc, fR0, n, mu, Sigma = MGparams
    
    # Get growth factor in nDGP
    Omega_rc = 1/(4*H0rc**2)
    
    a_solver = np.linspace(1/50,1,100)
    Soln = odeint(solverGrowth_nDGP, [a_solver[0], (E(cosmoMCMCStep, a_solver[0])*a_solver[0]**3)], a_solver, \
                  args=(cosmoMCMCStep,MGparams), mxstep=int(1e4))
    
    Delta = Soln.T[0]
    a3EdDda = Soln.T[1]

    f_nDGP_interp = a3EdDda/a_solver**2 / Delta / E(cosmoMCMCStep, a_solver)
    
    f_nDGP = np.interp(a, a_solver, f_nDGP_interp)

    k_val = np.logspace(-4,3,3000)
    return f_nDGP * sigma_8_nDGP(cosmoMCMCStep, MGparams, a)

#######################################################################################
''' Section 4: Get C(ell) functions'''
#######################################################################################

"""Get n_zbins logarithmically spaced ell bins (total n of ell bins = ell_bin_num)"""
def bin_ell_kk(ell_min, ell_max, ell_bin_num, Binned_distribution):
    # define quantities for binning in ell
    n_zbins = int(((len(Binned_distribution)+1)*len(Binned_distribution))/2)
    ell_binned_limits = np.linspace(np.log10(ell_min),np.log10(ell_max),num=ell_bin_num + 1)
    bin_edge1 = ell_binned_limits[:-1]
    bin_edge2 = ell_binned_limits[1:]
    ell_binned = 10**((bin_edge1 + bin_edge2) / 2)

    # Repeat ell_binned over all redshift bins, so that len(ell_binned)=len(C_ell_array)
    ell_binned = np.repeat([ell_binned], repeats=n_zbins, axis=0)
    
    #ell_binned = list(islice(cycle(ell_binned), int(ell_bin_num*((len(Binned_distribution)+1)*len(Binned_distribution))/2)))
    return ell_binned
    
"""Get n_zbins logarithmically spaced ell bins (total n of ell bins = ell_bin_num)"""
def bin_ell_delk(ell_min, ell_max, ell_bin_num,Binned_distribution_s, Binned_distribution_l):
    # define quantities for binning in ell
    n_zbins = 0
    for j in range(len(Binned_distribution_l)):
        for k in range(len(Binned_distribution_s)):
            if k - 1 > j or (k == 4 and j == 3):
                n_zbins += 1
    
    ell_binned_limits = np.linspace(np.log10(ell_min),np.log10(ell_max),num=ell_bin_num + 1)
    bin_edge1 = ell_binned_limits[:-1]
    bin_edge2 = ell_binned_limits[1:]
    ell_binned = 10**((bin_edge1 + bin_edge2) / 2)

    # Repeat ell_binned over all redshift bins, so that len(ell_binned)=len(C_ell_array)
    ell_binned = np.repeat([ell_binned], repeats=n_zbins, axis=0)
    
    #ell_binned = list(islice(cycle(ell_binned), int(ell_bin_num*((len(Binned_distribution)+1)*len(Binned_distribution))/2)))
    return ell_binned

"""Get n_zbins logarithmically spaced ell bins (total n of ell bins = ell_bin_num)"""
def bin_ell_deldel(ell_min, ell_max, ell_bin_num, Binned_distribution):
    # define quantities for binning in ell
    n_zbins = len(Binned_distribution)
    ell_binned_limits = np.linspace(np.log10(ell_min),np.log10(ell_max),num=ell_bin_num + 1)
    bin_edge1 = ell_binned_limits[:-1]
    bin_edge2 = ell_binned_limits[1:]
    ell_binned = 10**((bin_edge1 + bin_edge2) / 2)

    # Repeat ell_binned over all redshift bins, so that len(ell_binned)=len(C_ell_array)
    ell_binned = np.repeat([ell_binned], repeats=n_zbins, axis=0)
    
    #ell_binned = list(islice(cycle(ell_binned), int(ell_bin_num*((len(Binned_distribution)+1)*len(Binned_distribution))/2)))
    return ell_binned

"""Functions to find Cell given a Pdelta_2D ccl object"""

# A: Function for cosmic shear angular power spectrum (lensing-lensing C_ell) from a given P_delta2D_S
def C_ell_arr_kk(P_delta2D_S_funct, ell_binned, cosmo, z, Binned_distribution_s, Binned_distribution_l,Bias_distribution):
    C_ell_array = []
    n_zbins = int(((len(Binned_distribution_s)+1)*len(Binned_distribution_s))/2)
    # how far along z binning we are
    idx = 0
    # at what z bin we start calculating Cell
    start_idx = n_zbins - len(ell_binned)

    for j in range(len(Binned_distribution_s)):
        tracer1 = ccl.WeakLensingTracer(cosmo, dndz=(z, Binned_distribution_s[j]))
        for k in range(len(Binned_distribution_s)):
            if k >= j:
                if start_idx <= idx:
                    tracer2 = ccl.WeakLensingTracer(cosmo, dndz=(z, Binned_distribution_s[k]))
                    C_ell = ccl.angular_cl(cosmo, tracer1, tracer2, ell_binned[idx - start_idx], p_of_k_a=P_delta2D_S_funct)
                    C_ell_array.append([C_ell])
                    idx += 1
                else:
                    idx += 1
    return C_ell_array

# B: Function for galaxy-galaxy lensing angular power spectrum (clustering-lensing C_ell) from a given P_delta2D_S
def C_ell_arr_delk(P_delta2D_S_funct, ell_binned, cosmo, z, Binned_distribution_s, Binned_distribution_l,Bias_distribution):
    C_ell_array = []
    
    n_zbins = 0
    for j in range(len(Binned_distribution_l)):
        for k in range(len(Binned_distribution_s)):
            if k - 1 > j or (k == 4 and j == 3):
                n_zbins += 1
                
    # how far along z binning we are
    idx = 0
    # at what z bin we start calculating Cell
    start_idx = n_zbins - len(ell_binned)

    for j in range(len(Binned_distribution_l)):
        tracer1 = ccl.NumberCountsTracer(cosmo, dndz=(z, Binned_distribution_l[j]), bias=(z, Bias_distribution[j]), has_rsd=False)
        for k in range(len(Binned_distribution_s)):
            if k - 1 > j or (k == 4 and j == 3):
                if start_idx <= idx:
                    tracer2 = ccl.WeakLensingTracer(cosmo, dndz=(z, Binned_distribution_s[k]))
                    C_ell = ccl.angular_cl(cosmo, tracer1, tracer2, ell_binned[idx - start_idx], p_of_k_a=P_delta2D_S_funct)
                    C_ell_array.append([C_ell])
                    idx += 1
                else:
                    idx += 1
    return C_ell_array

# C: Function for galaxy-galaxy clustering angular power spectrum (clustering-clustering C_ell) from a given P_delta2D_S
def C_ell_arr_deldel(P_delta2D_S_funct, ell_binned, cosmo, z, Binned_distribution_s, Binned_distribution_l,Bias_distribution):
    C_ell_array = []
    n_zbins = len(Binned_distribution_l)
    # how far along z binning we are
    idx = 0
    # at what z bin we start calculating Cell
    start_idx = n_zbins - len(ell_binned)

    for j in range(len(Binned_distribution_l)):
        tracer1 = ccl.NumberCountsTracer(cosmo, dndz=(z, Binned_distribution_l[j]), bias=(z, Bias_distribution[j]), has_rsd=False)
        for k in range(len(Binned_distribution_l)):
            if k == j:
                if start_idx <= idx:
                    tracer2 = ccl.NumberCountsTracer(cosmo, dndz=(z, Binned_distribution_l[k]), bias=(z, Bias_distribution[k]), has_rsd=False)
                    C_ell = ccl.angular_cl(cosmo, tracer1, tracer2, ell_binned[idx - start_idx], p_of_k_a=P_delta2D_S_funct)
                    C_ell_array.append([C_ell])
                    idx += 1
                else:
                    idx += 1
    return C_ell_array

def Get_Pk2D_obj(cosmo, MGParams,linear=False,gravity_model="GR"):
    """
    Finds Get_Pk2D object
    linear = True, False
    gravity theory = "GR", "nDGP", "f(R)", "muSigma"
    if linear=True, use linear matter power spectrum to compute the angular one, otherwise use the non-linear
    input:
        ell_binned: array of ell bins for the full C{ij}(ell) range (for all i and j), with scale cuts included
        cosmo: ccl cosmology object
        redshift z: numpy.array with dim:N
        Binned_distribution_s: numpy.array with dim:(N,M) (M = no. source z bins)
        Binned_distribution_l: numpy.array with dim:(N,L) (L = no. lens z bins)
        Bias_distribution: numpy.array with dim:(N,L) (galaxy bias)
        pk_F: function (cosmo, MGParams, k,a) for k (in 1/Mpc), returns matter power spectrum (in Mpc^3)
        MGParams: 
    returns:
        ell bins: numpy.array (dim = dim C_ell)
        C_ell: numpy.array
    """

    ########### Functions for non-linear matter power spectrum ###########
    def pk_func_GR_NL(k, a):
        return ccl.nonlin_matter_power(cosmo, k=k, a=a)

    def pk_func_nDGP_NL(k, a):
        # Assume Sigma is always 1 for now
        
        z = 1 / a - 1
        if z > 2:
            return ccl.nonlin_matter_power(cosmo, k=k, a=a)
        
        # Determine the index range for k
        k_min = 0.0156606 * cosmo["h"]
        k_max = 4.99465 * cosmo["h"]
        
        idx_min = np.searchsorted(k, k_min)
        idx_max = np.searchsorted(k, k_max, side='right')
        
        k_allowed = k[idx_min:idx_max]
        
        # Calculate power spectra for different k ranges
        pk_lin_start = P_k_nDGP_lin(cosmo, MGParams, k[:idx_min], a)
        pk_nl_mid = P_k_NL_nDGP(cosmo, MGParams, k_allowed, a)
        pk_nl_end = ccl.nonlin_matter_power(cosmo, k=k[idx_max:], a=a)
        
        pk = np.concatenate((pk_lin_start, pk_nl_mid, pk_nl_end), axis = 0)
        
        return pk
    

    def pk_func_fR_NL(k, a):
        # Assume Sigma is always 1 for now
        
        z = 1 / a - 1
        
        if a < 0.3333:
            return ccl.nonlin_matter_power(cosmo, k=k, a=a)
        
        # Determine the index range for k
        idx_min = np.argmin(np.abs(k - (emu_fR.kbins[0])*cosmo["h"]**2)) + 1
        idx_max = np.argmin(np.abs(k - (emu_fR.kbins[-1])*cosmo["h"]**2)) -1

        k_min = k[idx_min]
        k_max = k[idx_max]
        k_allowed = k[idx_min:idx_max]
        
        # Calculate power spectra for different k ranges
        pk_lin_start = P_k_fR_lin(cosmo, MGParams, k[:idx_min], a)
        pk_nl_mid = P_k_NL_fR(cosmo, MGParams, k_allowed, a)
        pk_nl_end = ccl.nonlin_matter_power(cosmo, k=k[idx_max:], a=a)
        
        pk = np.concatenate((pk_lin_start, pk_nl_mid, pk_nl_end), axis = 0)
        
        return pk
    
    def pk_func_muSigma_NL(k, a):
        raise Exception('there is no non-linear power spectrum available for muSigma parametrization.')
        
    ########### Functions for linear matter power spectrum multiplied by Sigma**2 ###########
    def pk_func_GR_lin(k, a):
        return ccl.linear_matter_power(cosmo, k=k, a=a)

    def pk_func_nDGP_lin(k, a):

        # condition on z
        return P_k_nDGP_lin(cosmo, MGParams, k, a)
        
    def pk_func_fR_lin(k, a):
        
        # condition on z
        return P_k_fR_lin(cosmo, MGParams, k, a)
        
            
    def pk_func_muSigma_lin(k, a):
        # condition on z
        return P_k_musigma(cosmo, MGParams, k, a)

    def invalid_op(k, a):
        raise Exception("Invalid gravity model entered or Linear must be True or False.")

    ops = {
        ("GR" , False): pk_func_GR_NL,
        ("nDGP" , False): pk_func_nDGP_NL, 
        ("f(R)" , False): pk_func_fR_NL, 
        ("muSigma" , False): pk_func_muSigma_NL,
        ("GR" , True): pk_func_GR_lin,
        ("nDGP" , True): pk_func_nDGP_lin, 
        ("f(R)" , True): pk_func_fR_lin, 
        ("muSigma" , True): pk_func_muSigma_lin
    }
    
    ########### Find matter power spectrum multiplied by Sigma**2 ###########
    pk_func = ops.get((gravity_model, linear), invalid_op)

    return ccl.pk2d.Pk2D.from_function(pkfunc=pk_func, is_logp=False)

########### Functions for NL P(k) multiplied by Sigma - only for Sigma diff 1, so MuSigma param only ###########

def Get_Pk2D_obj_delk_musigma(cosmo, MGParams):
   
    ########### Functions for linear matter power spectrum multiplied by Sigma**2 ###########        
    def pk_funcSigma_muSigma_lin(k, a):
        return sigma_lin_param(MGParams, cosmo,a)*P_k_musigma(cosmo, MGParams, k, a)

    return ccl.pk2d.Pk2D.from_function(pkfunc=pk_funcSigma_muSigma_lin, is_logp=False)


def Get_Pk2D_obj_deldel_musigma(cosmo, MGParams):
   
    ########### Functions for linear matter power spectrum multiplied by Sigma**2 ###########        
    def pk_funcSigma2_muSigma_lin(k, a):
        return sigma_lin_param(MGParams, cosmo, a)**2 * P_k_musigma(cosmo, MGParams, k, a)

    return ccl.pk2d.Pk2D.from_function(pkfunc=pk_funcSigma2_muSigma_lin, is_logp=False)

# ~~~~~~~~~~~~~~~~~~~~~~ ACTUAL FINAL FUNCTION TO GET C(ell) ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def Cell(ell_binned, cosmo, z, Binned_distribution_s, Binned_distribution_l,Bias_distribution, MGParams,P_delta2D_S,
         tracer1_type="k", 
         tracer2_type="k"):
    """
    Finds C^{i,j}(ell) for {i,j} redshift bins.
    tracer_type = "k", "g"
    linear = True, False
    gravity theory = "GR", "nDGP", "f(R)", "muSigma"
    if tracer1_type = "k" and tracer2_type = "k", shape-shape angular power spectrum
    if tracer1_type = "k" and tracer2_type = "g", galaxy-galaxy lensing angular power spectrum
    if tracer1_type = "g" and tracer2_type = "g", pos-pos angular power spectrum
    if linear=True, use linear matter power spectrum to compute the angular one, otherwise use the non-linear
    input:
        ell_binned: array of ell bins for the full C{ij}(ell) range (for all i and j), with scale cuts included
        cosmo: ccl cosmology object
        redshift z: numpy.array with dim:N
        Binned_distribution_s: numpy.array with dim:(N,M) (M = no. source z bins)
        Binned_distribution_l: numpy.array with dim:(N,L) (L = no. lens z bins)
        Bias_distribution: numpy.array with dim:(N,L) (galaxy bias)
        pk_F: function (cosmo, MGParams, k,a) for k (in 1/Mpc), returns matter power spectrum (in Mpc^3)
        MGParams: 
    returns:
        ell bins: numpy.array (dim = dim C_ell)
        C_ell: numpy.array
    """

    ops = {
        ("k" , "k"): C_ell_arr_kk,
        ("k" , "g"): C_ell_arr_delk, 
        ("g" , "k"): C_ell_arr_delk,
        ("g" , "g"): C_ell_arr_deldel
    }

    def invalid_op2():
        raise ValueError('invalid tracer selected.')
    ########## Find Cell ##########

    C_ell_array_funct = ops.get((tracer1_type, tracer2_type), invalid_op2)
    C_ell_array = C_ell_array_funct(P_delta2D_S, ell_binned, cosmo, z, Binned_distribution_s, Binned_distribution_l,Bias_distribution)

    return np.array(list(itertools.chain(*ell_binned))), C_ell_array

#######################################################################################
''' Section 5: Create Mock datavector'''
#######################################################################################
# Define cosmology -- our "universe cosmology"

cosmo_universe = ccl.Cosmology(Omega_c = 0.27, 
                          Omega_b = 0.046, 
                          h = 0.7, 
                          n_s = 0.974,
                          A_s = 2.01e-9)

fR_universe = 0
H0rc_universe = 1.0
MGParam_universe = [H0rc_universe,fR_universe,1,0,0]

# ~~~~~~~~~~~~~~~~~~~~~~~~~ 3x2pt ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""Define binning function"""

def bins(ell_min, ell_max, ell_bin_num):

    # define quantities for binning in ell
    ell_binned_limits = np.linspace(np.log10(ell_min),np.log10(ell_max),num=ell_bin_num + 1)
    bin_edge1 = ell_binned_limits[:-1]
    bin_edge2 = ell_binned_limits[1:]
    ell_binned = 10**((bin_edge1 + bin_edge2) / 2)
    # Repeat ell_binned over all redshift bins, so that len(ell_binned)=len(C_ell_array)
    return ell_binned

# Check we match SRD (ell binning for shear)
#print(np.loadtxt("./ell-values").shape)
#ells_SRD = np.loadtxt("./ell-values")[:13]
#print(ells_SRD)
#print(bins(20, 1478.5, 13))


# define ell and C_ell shapes -- will depend on the data

ell_min_mockdata = 20
ell_max_mockdata = 1478.5

# define quantities for binning of ell -- will depend on the data

ell_bin_num_mockdata = 13

"""Get mock C(ell) data"""

## LENSING - LENSING

binned_ell = bin_ell_kk(ell_min_mockdata, ell_max_mockdata, ell_bin_num_mockdata, Binned_distribution_source)

# find C_ell for non-linear matter power spectrum
mockdata = Cell(binned_ell, \
                cosmo_universe, z , Binned_distribution_source,Binned_distribution_lens,Bias_distribution_fiducial,MGParam_universe,\
                Get_Pk2D_obj(cosmo_universe, MGParam_universe, linear=False, gravity_model="nDGP"),tracer1_type="k", tracer2_type="k")

ell_kk_mockdata = mockdata[0]
D_kk_mockdata = mockdata[1]
D_kk_mockdata = (np.array(D_kk_mockdata)).flatten()

# For plot below, compare with linear
data_lin_plot = Cell(binned_ell, \
                cosmo_universe, z , Binned_distribution_source,Binned_distribution_lens,Bias_distribution_fiducial,MGParam_universe,\
                Get_Pk2D_obj(cosmo_universe, MGParam_universe, linear=True, gravity_model="nDGP"),tracer1_type="k", tracer2_type="k")

D_kk_data_lin_plot = data_lin_plot[1]
D_kk_data_lin_plot = (np.array(D_kk_data_lin_plot)).flatten()

## CLUSTERING - LENSING

binned_ell = bin_ell_delk(ell_min_mockdata, ell_max_mockdata, ell_bin_num_mockdata,Binned_distribution_source,Binned_distribution_lens)

# find C_ell for non-linear matter power spectrum
mockdata = Cell(binned_ell, \
                cosmo_universe, z , Binned_distribution_source,Binned_distribution_lens,Bias_distribution_fiducial,MGParam_universe,\
                Get_Pk2D_obj(cosmo_universe, MGParam_universe, linear=False, gravity_model="nDGP"), tracer1_type="k", tracer2_type="g")

ell_delk_mockdata = mockdata[0]
D_delk_mockdata = mockdata[1]
D_delk_mockdata = (np.array(D_delk_mockdata)).flatten()

# For plot below, compare with linear
data_lin_plot = Cell(binned_ell, \
                cosmo_universe, z , Binned_distribution_source,Binned_distribution_lens,Bias_distribution_fiducial,MGParam_universe,\
                Get_Pk2D_obj(cosmo_universe, MGParam_universe, linear=True, gravity_model="nDGP"), tracer1_type="k", tracer2_type="g")

D_delk_data_lin_plot = data_lin_plot[1]
D_delk_data_lin_plot = (np.array(D_delk_data_lin_plot)).flatten()

## CLUSTERING - CLUSTERING
binned_ell = bin_ell_deldel(ell_min_mockdata, ell_max_mockdata, ell_bin_num_mockdata,Binned_distribution_lens)

# find C_ell for non-linear matter power spectrum
mockdata = Cell(binned_ell, \
                cosmo_universe, z , Binned_distribution_source,Binned_distribution_lens,Bias_distribution_fiducial,MGParam_universe,\
                Get_Pk2D_obj(cosmo_universe, MGParam_universe, linear=False, gravity_model="nDGP"), tracer1_type="g", tracer2_type="g")

ell_deldel_mockdata = mockdata[0]
D_deldel_mockdata = mockdata[1]
D_deldel_mockdata = (np.array(D_deldel_mockdata)).flatten()

# For plot below, compare with linear
data_lin_plot = Cell(binned_ell, \
                cosmo_universe, z , Binned_distribution_source,Binned_distribution_lens,Bias_distribution_fiducial,MGParam_universe,\
                Get_Pk2D_obj(cosmo_universe, MGParam_universe, linear=True, gravity_model="nDGP"), tracer1_type="g", tracer2_type="g")

D_deldel_data_lin_plot = data_lin_plot[1]
D_deldel_data_lin_plot = (np.array(D_deldel_data_lin_plot)).flatten()


ell_mockdata = np.append(np.append(ell_kk_mockdata, ell_delk_mockdata), ell_deldel_mockdata)
D_mockdata = np.append(np.append(D_kk_mockdata, D_delk_mockdata), D_deldel_mockdata)
D_data_lin_plot = np.append(np.append(D_kk_data_lin_plot, D_delk_data_lin_plot), D_deldel_data_lin_plot)

del mockdata, data_lin_plot

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~mock fsigma8 data ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""Get simulated DESI LRG data using Danielle's code"""

def f_frac_err(cosmo, MGparams,b, zeff, nbar, V):
    """ Get the fractional error on the growth rate.
    params is a dictionary of required cosmological parameters.
    zeff is the effective redshift of the sample.
    nbar is the number density of the sample in units of (h/Mpc)^3.
    V is the volume of the survey in (Mpc/h)^3. """
    
    # Set up a k and mu vector at which to do the integrals
    # (Result depends on kmax chosen, see White et al. 2008)
	
    k = np.logspace(-3, -1, 400)
    mu = np.linspace(-1, 1., 200)
        
    # Get the inverse covariance value at each k and mu
    invcov = Pobs_covinv(cosmo, MGparams,b, k, mu, zeff, nbar)
    
    # Get the derivative of the observed z-space
    # P(k) with respect to b and f at each k and mu
    # (linear theory)
    dPdf = diff_P_f(cosmo, MGparams,b, k, mu, zeff)
    
    # Do the integration in k in each case
    int_in_k_ff = [scipy.integrate.simps(k**2 * dPdf[mi] * invcov[mi] * dPdf[mi], k) for mi in range(len(mu))]
	
    # And in mu.
    print("Doing mu integration.")
    int_in_mu_ff = scipy.integrate.simps(np.asarray(int_in_k_ff), mu)
	
    Fisher_ff = np.zeros((2,2)) # order is b then f
    
    # Add necessary factors of volume (Mpc/h)^3 and pi etc
    ff= V * int_in_mu_ff / (2. * np.pi**2)
    err_f = np.sqrt(1./ff)
    
    # Now use this to construct the error on f:
    f_sigma8_val = fsigma8_nDGP(cosmo, MGparams, [1./(1. + zeff),1./(1. + zeff)])[0]
    f_fid = f_sigma8_val/sigma_8_nDGP(cosmo, MGparams, [1./(1. + zeff),1./(1. + zeff)])[0]
    
    frac_err_f = err_f / f_fid
    
    return frac_err_f, f_sigma8_val
    
def diff_P_f(cosmo, MGparams,b, k, mu, zeff):
    """ Calculate the derivative of the redshift space power spectrum
    wrt linear growth rate f at each k and mu
    params: dictionary of cosmological parameters
    k: list or array of wavenumbers
    mu: list of array of angles
    lens: lens sample label
    """
		
    # Get the linear power spectrum at the effective z of the lens sample
    Pklin = P_k_nDGP_lin(cosmo, MGparams, k,  1./ (1. + zeff))
    
    # Get the derivative at each mu / k
    f = fsigma8_nDGP(cosmo, MGparams, [1./(1. + zeff),1./(1. + zeff)])[0]/sigma_8_nDGP(cosmo, MGparams, [1./(1. + zeff),1./(1. + zeff)])[0]
    dPdf = [2. * (b + mu[mi]**2*f) * mu[mi]**2 * Pklin  for mi in range(len(mu))]
	
    return dPdf
    
def Pobs_covinv(cosmo, MGparams,b, k, mu, zeff, nbar):
    """ Get the inverse covariance of the redshift space observed power 
    spectrum at a list of k and mu (cosine of angle to line of sight) vals.
    params: dictionary of cosmological parameters
    k: list or array of wavenumbers
    mu: list of array of angles
    lens: lens sample label """	
 
	
    # Get the linear power spectrum at the effective z of the lens sample
    Pklin = P_k_nDGP_lin(cosmo, MGparams, k,  1./ (1. + zeff))
	
    # Get the redshift space galaxy power spectrum in linear theory
    f = fsigma8_nDGP(cosmo, MGparams, [1./(1. + zeff),1./(1. + zeff)])[0]/sigma_8_nDGP(cosmo, MGparams, [1./(1. + zeff),1./(1. + zeff)])[0]
    Pgg = [(b + f * mu[mi]**2)**2 * Pklin for mi in range(len(mu))]
	
    # Get the covariance matrix at each k and mu
    cov = [ 2. * (Pgg[mi]**2 + 2 * Pgg[mi] / nbar + 1. / nbar**2) for mi in range(len(mu))]
	
    #Pgg_arr = np.zeros((len(k), len(mu)))
    #for mi in range(len(mu)):
    #	Pgg_arr[:, mi] = Pgg[mi]
	
    # Get the inverse at each k and mu
    invcov = [[1./ cov[mi][ki] for ki in range(len(k))] for mi in range(len(mu))]
			
    return invcov

zeff= 0.72
nbar = 5*10**(-4)
Vol = 3*10**9

f_fe, f_fid = f_frac_err(cosmo_universe,MGParam_universe,2.03, zeff, nbar, Vol)

'''Get remaining mock fsigma8 data based on existing datasets'''

# Using data described in Section 3.4 of https://arxiv.org/pdf/2201.07025 (Jaime's paper)
# Dataset 1 (x3): RSD BOSS DR12 data https://arxiv.org/pdf/1607.03155
# Dataset 2 (x1): BOSS DR16 quasar sample 𝑓𝜎8(𝑧eff) measurement https://arxiv.org/pdf/2007.08998
# Dataset 3 (x3): WiggleZ Dark Energy Survey data https://arxiv.org/pdf/1204.3674
# Dataset 4 (x1): 𝑓𝜎8(𝑧 = 0) from peculiar velocities of Democratic Samples of Supernovae https://arxiv.org/pdf/2105.05185

z_eff = np.array([0.38 , 0.51 , 0.61 , 1.48 , 0.44 , 0.6 , 0.73 , 0.0, zeff])

fsigma_8_realdata = np.array([0.497 , 0.458 , 0.436, 0.462 , 0.413 , 0.39 , 0.437 , 0.39])

fsigma_8_fracerror = np.array([0.045/0.497 , 0.038/0.458 , 0.034/0.436, 0.045/0.462 , \
                               0.08/0.413 , 0.063/0.39 , 0.072/0.437 , 0.022/0.39, f_fe])

reducedcov_fsigma_8 = np.array([[1 , 0.4773 , 0.1704 , 0 , 0 , 0 , 0 , 0,0],
                               [ 0.4773 , 1 , 0.5103 , 0 , 0 , 0 , 0 , 0,0],
                               [0.1704 , 0.5103 , 1 , 0 , 0 , 0 , 0 , 0,0],
                               [0 , 0 , 0 , 1 , 0 , 0 , 0 , 0,0],
                               [0 , 0 , 0 , 0 , 1 , 0.50992 , 0.0 , 0,0],
                               [0 , 0 , 0 , 0 , 0.50992 , 1 , 0.559965 , 0,0],
                               [0 , 0 , 0 , 0 , 0.0 , 0.559965 , 1 , 0,0],
                               [0 , 0 , 0 , 0 , 0 , 0 , 0 , 1,0],
                               [0 , 0 , 0 , 0 , 0 , 0 , 0 , 0,1]]
                       )

headers = z_eff.tolist()

## Create our data:

fsigma_8_data = fsigma8_nDGP(cosmo_universe, MGParam_universe, 1/(z_eff+1))

cov_fsigma8 = reducedcov_fsigma_8 * np.outer(fsigma_8_data*fsigma_8_fracerror, fsigma_8_data*fsigma_8_fracerror)
invcov_fsigma8 = np.linalg.inv(cov_fsigma8)

#######################################################################################
''' Section 6: Get Covariances'''
#######################################################################################

def cov2corr(cov):
    """
    Convert a covariance matrix into a correlation matrix
    input:
        cov: numpy.array with dim:(N,N)
    returns:
        corr: numpy.array with dim:(N,N)
    """
    sig = np.sqrt(cov.diagonal())
    return cov/np.outer(sig, sig)

"""Get SRD covariance matrix - eventually we want to use TJPCOV"""

# covariance for shear bin combinations, in order: z11, z12, z13,..., z15, z22, z23,...z55

########## Get full covariance (gauss only) ##########

covfile = np.genfromtxt("./Y1_3x2pt_clusterN_clusterWL_cov")

shear_SRD = np.zeros((705,705))
ell_test_SRD = np.zeros(705)

for i in range(0,covfile.shape[0]):
    shear_SRD[int(covfile[i,0]),int(covfile[i,1])] = covfile[i,8]+covfile[i,9] # with non-gauss term
    shear_SRD[int(covfile[i,1]),int(covfile[i,0])] = covfile[i,8]+covfile[i,9] # with non-gauss term
    if int(covfile[i,0]) == int(covfile[i,1]):
        ell_test_SRD[int(covfile[i,0])] = covfile[i,2]

del covfile

SRD_compare = shear_SRD[:540,:540].copy()

idx = 0

bins_SRD = int(len(SRD_compare)/(len(D_mockdata)/ell_bin_num_mockdata))


for j in range(int(len(D_mockdata)/ell_bin_num_mockdata)):
    for i in range(bins_SRD):
        if i >= ell_bin_num_mockdata:
            #print(int(len(SRD_compare)/(len(D_mockdata)/ell_bin_num_mockdata)))
            SRD_compare = np.delete(SRD_compare, j*bins_SRD + i - idx, 0)
            SRD_compare = np.delete(SRD_compare, j*bins_SRD + i - idx, 1)
            idx += 1

"""Get cholensky decomposition"""

L_choleski_uncut = np.linalg.cholesky(np.matrix(SRD_compare))
L_choleski_inv_uncut = np.linalg.inv(L_choleski_uncut)

#######################################################################################
''' Section 7: Apply Scale Cuts - Linear Conservative Cuts'''
#######################################################################################

"""Get list of lists rather than 1d array for non-uniform ell spacing"""

def ell_arrayfromlist(list):
    list_new = [[]]
    idx = 0
    for i in range(len(list)):
        if list[(i+1) % (len(list))] <= list[i]:
            list_new[idx].append(list[i])
            idx += 1
            list_new.append([])
        else:
            list_new[idx].append(list[i])
    del list_new[-1]
    return list_new

def linear_scale_cuts(ell, dvec_nl, dvec_lin, cov):
    """ Function from Danielle.
    Gets the scales (and vector indices) which are excluded if we
    are only keeping linear scales. We define linear scales such that 
    chi^2_{nl - lin) <=1.
    dvec_nl: data vector from nonlinear theory 
    dvec_lin: data vector from linear theory
    cov: data covariance """
    
    # Make a copy of these initial input things before they are changed,
    # so we can compare and get the indices
    dvec_nl_in = dvec_nl; dvec_lin_in = dvec_lin; cov_in = cov;
    
    # Check that data vector and covariance matrices have consistent dimensions.
    if ( (len(dvec_nl)!=len(dvec_lin)) or (len(dvec_nl)!=len(cov[:,0])) or (len(dvec_nl)!=len(cov[0,:])) or (len(dvec_nl)!=len(ell)) ):
        raise(ValueError, "in linear_scale_cuts: inconsistent shapes of data vectors and / or covariance matrix.")
    
    # Cut elements of the data vector / covariance matrix until chi^2 <=1
    inv_cov = np.linalg.pinv(cov)

    while(True):
        # Get an array of all the individual elements which would go into 
        # getting chi2
        sum_terms = np.zeros((len(dvec_nl), len(dvec_nl)))
        for i in range(len(dvec_nl)):
            for j in range(len(dvec_nl)):
                sum_terms[i,j] = (dvec_nl[i] - dvec_lin[i]) * inv_cov[i,j] * (dvec_nl[j] - dvec_lin[j])
    
            #print "sum_terms=", sum_terms
            #print "chi2=", np.sum(sum_terms)
            # Check if chi2<=1
        if (np.sum(sum_terms)<=1.0):
            break
        else:
            # Get the indices of the largest value in sum_terms.
            inds_max = np.unravel_index(np.argmax(sum_terms, axis=None), sum_terms.shape)
            #print "inds_max =", inds_max
            # Remove this / these from the data vectors and the covariance matrix
            
            if (inds_max[0] == inds_max[1]):
                
                inv_cov[inds_max[0]] = np.zeros(len(inv_cov[inds_max[0]]))
                inv_cov[:, inds_max[0]] = np.zeros(len(inv_cov[inds_max[0]]))
            else:
                
                inv_cov[inds_max] = np.zeros(len(inv_cov[inds_max]))
                inv_cov[:, inds_max] = np.zeros(len(inv_cov[inds_max]))
    # Now we should have the final data vector with the appropriate elements cut.
    # Use this to get the rp indices and scales we should cut.
    
    ex_inds = [i for i in range(len(dvec_nl_in)) if dvec_nl_in[i] not in dvec_nl]
    
    return ell, dvec_nl, dvec_lin, inv_cov #ex_inds

"""Get new Mock data with cuts in ell"""

newdat_test = linear_scale_cuts(ell_mockdata, D_mockdata, D_data_lin_plot, SRD_compare)
gauss_invcov_cut = newdat_test[3]

#######################################################################################
''' Section 8: Package Data '''
#######################################################################################

# WITHOUT NOISE
C_ell_data_mock = [D_mockdata, ell_mockdata, z,  Binned_distribution_source,\
                   Binned_distribution_lens, ell_min_mockdata, ell_max_mockdata, ell_bin_num_mockdata]

Data_fsigma8 = [z_eff, fsigma_8_data, invcov_fsigma8]


#######################################################################################
''' Section 9: Define Likelihood function '''
#######################################################################################

# log likelihood - standard scale cuts
def loglikelihood_noscalecut(Data, cosmo, MGparams, InvCovmat, Bias_distribution, data_fsigma8):
    #start = time.time()
    
    # Extract fsigma8 data vector
    z_fsigma8, fsigma_8_dataset, invcovariance_fsigma8 = data_fsigma8
    
    # Extract 3x2pt data vector
    D_data, ell_mockdata, z, Binned_distribution_s,Binned_distribution_l,\
                   ell_min_mockdata, ell_max_mockdata, ell_bin_num_mockdata = Data

    # shape-shape
    binned_ell_kk = bin_ell_kk(ell_min_mockdata, ell_max_mockdata, ell_bin_num_mockdata, Binned_distribution_s)

    # shape-pos
    binned_ell_delk = bin_ell_delk(ell_min_mockdata, ell_max_mockdata, ell_bin_num_mockdata, \
                              Binned_distribution_s,Binned_distribution_l)

    # pos-pos
    binned_ell_deldel = bin_ell_deldel(ell_min_mockdata, ell_max_mockdata, ell_bin_num_mockdata, Binned_distribution_l)

    # Precompute Pk2D objects
    P_delta2D_muSigma_kk = Get_Pk2D_obj(cosmo, MGparams, linear=True, gravity_model="muSigma")
    P_delta2D_muSigma_delk = Get_Pk2D_obj_delk_musigma(cosmo, MGparams)
    P_delta2D_muSigma_deldel = Get_Pk2D_obj_deldel_musigma(cosmo, MGparams)
    
    ########## Get theoretical data vector for single MCMC step - linear , muSigmaparam ##########
    # shape-shape
    D_theory_kk = np.array(Cell(binned_ell_kk, \
                cosmo, z , Binned_distribution_s,Binned_distribution_l,Bias_distribution,MGparams,\
                P_delta2D_muSigma_kk, tracer1_type="k", tracer2_type="k")[1]).flatten()
    # shape-pos
    
    D_theory_delk = np.array(Cell(binned_ell_delk, \
                cosmo, z , Binned_distribution_s,Binned_distribution_l,Bias_distribution,MGparams,\
                P_delta2D_muSigma_delk, tracer1_type="g", tracer2_type="k")[1]).flatten()
    # pos-pos

    D_theory_deldel = np.array(Cell(binned_ell_deldel, \
                cosmo, z , Binned_distribution_s,Binned_distribution_l,Bias_distribution,MGparams,\
                P_delta2D_muSigma_deldel, tracer1_type="g", tracer2_type="g")[1]).flatten()

    D_theory = np.append(np.append(D_theory_kk, D_theory_delk), D_theory_deldel)
    
    Diff = (D_data - D_theory)

    #print("time = ", time.time() - start)

    #### fsigma8 ####
    Diff_fsigma8 = fsigma_8_dataset - fsigma8_musigma(cosmo, MGparams, 1/(z_fsigma8+1))
    loglik_fsigma8 = -0.5*(np.matmul(np.matmul(Diff_fsigma8,invcovariance_fsigma8),Diff_fsigma8))

    return -0.5*(np.matmul(np.matmul(Diff,InvCovmat),Diff)) + loglik_fsigma8 

#######################################################################################
''' Section 10: Run the MCMC '''
#######################################################################################
def log_likelihood(theta, Data, fsigma8_Data, invcovmat):
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

    cosmoMCMCstep = ccl.Cosmology(Omega_c = Omega_c, 
                      Omega_b = wb/h**2,
                      h = h,
                      n_s = n_s,
                      A_s = A_s)
    return loglikelihood_noscalecut(Data, cosmoMCMCstep, MGparams, invcovmat, Bias_distribution,fsigma8_Data)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ LOAD PLANCK PRIORS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
sampler_Planck_arr = np.load("./Prior_Planck_arr.npy")
cov_prior = np.cov(sampler_Planck_arr.T)
mu_prior = [cosmo_universe['n_s'], cosmo_universe["Omega_b"]*cosmo_universe["h"]**2]

def log_prior(theta):
    Omega_c, mu0,Sigma0, A_s1e9, h, n_s, wb, b1, b2, b3, b4, b5 = theta 

    if not (0.06 < Omega_c*h**2 < 0.46 and -2.0 < mu0 < 0.95 and -2.0 < Sigma0 < 0.95 and 1.7 < A_s1e9 < 2.6 and 0.92 < n_s < 1 \
            and mu0 <= 2*Sigma0 + 1.0 and 0.45 < h < 0.8 and 0.04 < wb/h**2 < 0.06 and 0.8 < b1 < 3.0 and 0.8 < b2 < 3.0 \
           and 0.8 < b3 < 3.0 and 0.8 < b4 < 3.0 and 0.8 < b5 < 3.0):
        return -np.inf
        
    gauss_funct = scipy.stats.multivariate_normal(mu_prior, cov_prior)
    
    return gauss_funct.logpdf([n_s, wb])

def log_probability(theta, Data, fsigma8_Data, invcovmat):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, Data, fsigma8_Data, invcovmat)

print("Success! Starting the mcmc runs")

#MCMC method

# initializing the walkers in a tiny Gaussian ball around the maximum likelihood result
#and then run 3000 steps of MCMC
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

pos = [Omega_c_est, mu0_est,Sigma0_est, A_s1e9_est, h_est, n_s_est, wb_est,b1_est,b2_est,b3_est,b4_est,b5_est] \
+ np.append(np.append(1e-3 * np.random.randn(25, 5), 1e-5*np.random.randn(25, 2), axis = 1), \
            1e-3 * np.random.randn(25, 5), axis = 1)

nwalkers, ndim = pos.shape

sampler_scalecut = emcee.EnsembleSampler(
    nwalkers, ndim, log_probability, args=(C_ell_data_mock, Data_fsigma8, gauss_invcov_cut)
)

sampler_scalecut.run_mcmc(pos, 3000, progress=True);

# Save chains
print("Done, saving array of size = ", sampler_scalecut.chain.shape)
np.save("./nDGP_DEParam_scalecut.npy", sampler_scalecut.chain)
    