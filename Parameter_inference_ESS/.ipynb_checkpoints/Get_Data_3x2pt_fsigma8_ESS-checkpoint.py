""" import useful functions """

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
import argparse

# cosmology
import pyccl as ccl
from astropy.io import fits
import yaml
import sacc
import time

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
from multiprocessing import Pool
from tqdm import tqdm

# nDGP NL and lin Pk
from nDGPemu import BoostPredictor


# f(R) emu (eMANTIS)
from emantis import FofrBoost

# Initialise EMANTIS emulator.
emu_fR = FofrBoost()

from LikelihoodFuncts_PCADR_muSigma import *

def main(args):
    ###############################################################################
    # Set up the input variables
    ###############################################################################
    # MCMC parameters
    b1_var = args.b1
    b2_var = args.b2
    b3_var = args.b3
    b4_var = args.b4
    b5_var = args.b5
    Omega_c_var = args.OmgC
    Omega_b_var = args.OmgB
    h_var = args.h
    ns_var = args.ns
    As_var = args.As
    
    ###############################################################################
    #Create mock redshift distribution (define z and output Binned_distribution(z))
    ###############################################################################
    print("starting code")

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
    
    z = redshift_range
    
    # Match SRD
    # from Table 2 in https://arxiv.org/pdf/2212.09345
    Bias_distribution_fiducial = np.array([b1_var*np.ones(len(z)),
                                 b2_var*np.ones(len(z)),
                                 b3_var*np.ones(len(z)),
                                 b4_var*np.ones(len(z)),
                                 b5_var*np.ones(len(z))])

    ###############################################################################
    # Get ESS P(k) and other functions
    ###############################################################################
    
    """ESS, ~LCDM background (from HiCOLA)"""
    # Load the saved array from HiCOLA, function of z and k
    ESS_C_Boost_loaded = np.load('/home/c2042999/PCA_project/HiCOLA_files/Output/Boost.npy')
    
    z_k_arr_loaded = np.loadtxt('/home/c2042999/PCA_project/HiCOLA_files/Output/z_k.txt')
    ESS_C_z_loaded = z_k_arr_loaded.T[0][~np.isnan(z_k_arr_loaded.T[0])]
    ESS_C_k_loaded = z_k_arr_loaded.T[1]
    
    # NL matter power spectra
    interpolator_ESS_C_funct = RectBivariateSpline(ESS_C_z_loaded, ESS_C_k_loaded, ESS_C_Boost_loaded)
    
    def P_k_NL_ESS_C(cosmo, k, a):
        
        pkratio_ESS_C = interpolator_ESS_C_funct(1/a - 1, k/cosmo["h"]) # k is in units [h/Mpc]
    
        Pk_ccl = ccl.power.nonlin_power(cosmo, k, a=a) # units (Mpc)^3
        Pk = pkratio_ESS_C.reshape(Pk_ccl.shape)*Pk_ccl
    
        return Pk
    
    def P_k_ESS_C_lin(cosmo, k, a):
        
        pkratio_ESS_C = interpolator_ESS_C_funct(1/a - 1, 0.020046/cosmo["h"])[0] # k is in units [h/Mpc]
    
        Pk_ccl = ccl.linear_matter_power(cosmo, k, a=a) # units (Mpc)^3
        Pk = pkratio_ESS_C*Pk_ccl
    
        return Pk

    """ESS"""
    def mu_ESS_C(a):
    
        force_today = np.loadtxt("/home/c2042999/PCA_project/HiCOLA_files/Hi-COLA_Output/ESS_run_ESS_force.txt")
    
        a_today = force_today.T[0]
        coupling = force_today.T[2]
        return np.interp(a, a_today, coupling + 1)
            
    # No LCDM background
    def sigma_8_ESS_C(cosmo, MGparams, a_array):
        k_val = np.logspace(-4, 3, 3000)
        sigma_8_vals = []
    
        for a in a_array:
            P_k_vals = P_k_ESS_C_lin(cosmo, k_val, a)
            j1_vals = 3 * scipy.special.spherical_jn(1, k_val * 8 / cosmo["h"], derivative=False) / (k_val * 8 / cosmo["h"])
            integrand = k_val**2 * P_k_vals * j1_vals**2
            integral_val = scipy.integrate.trapz(integrand, x=k_val)
            sigma_8_val = np.sqrt(integral_val / (2 * np.pi**2))
            sigma_8_vals.append(sigma_8_val)
        
        return np.array(sigma_8_vals)
    
    def solverGrowth_ESS_C(y,a,cosmo, MGparams):
        E_val = E(cosmo, a)
        D , a3EdDda = y
        
        mu = mu_ESS_C(a)
        
        ydot = [a3EdDda / (E_val*a**3), 3*cosmo["Omega_m"]*D*(mu)/(2*E_val*a**2)]
        return ydot
        
    def fsigma8_ESS_C(cosmoMCMCStep, MGparams, a):
        
        H0rc, fR0, n, mu, Sigma = MGparams
        
        a_solver = np.linspace(1/50,1,100)
        Soln = odeint(solverGrowth_ESS_C, [a_solver[0], (E(cosmoMCMCStep, a_solver[0])*a_solver[0]**3)], a_solver, \
                      args=(cosmoMCMCStep,MGparams), mxstep=int(1e4))
        
        Delta = Soln.T[0]
        a3EdDda = Soln.T[1]
    
        f_ESS_C_interp = a3EdDda/a_solver**2 / Delta / E(cosmoMCMCStep, a_solver)
        
        f_ESS_C = np.interp(a, a_solver, f_ESS_C_interp)
    
        k_val = np.logspace(-4,3,3000)
        return f_ESS_C * sigma_8_ESS_C(cosmoMCMCStep, MGparams, a)

    ########### Functions for NL P(k) multiplied by Sigma - only for Sigma diff 1, so MuSigma param only ###########

    def Get_Pk2D_obj_deldel_ESS_C(cosmo, MGParams, linear=False):
       
        ########### Functions for matter power spectrum ###########    
        def pk_funcSigma_ESS_C(k, a):
            if linear == False:
                return P_k_NL_ESS_C(cosmo, k, a)
            elif linear == True:
                return P_k_ESS_C_lin(cosmo, k, a)
    
        return ccl.pk2d.Pk2D.from_function(pkfunc=pk_funcSigma_ESS_C, is_logp=False)
        
    def Get_Pk2D_obj_delk_ESS_C(cosmo, MGParams, linear=False):
       
        ########### Functions for matter power spectrum multiplied by Sigma ###########        
        def pk_funcSigma_ESS_C(k, a):
            if linear == False:
                return P_k_NL_ESS_C(cosmo, k, a) #*mu_ESS_C(a)
            elif linear == True:
                return P_k_ESS_C_lin(cosmo, k, a) #*mu_ESS_C(a)
    
        return ccl.pk2d.Pk2D.from_function(pkfunc=pk_funcSigma_ESS_C, is_logp=False)
    
    def Get_Pk2D_obj_kk_ESS_C(cosmo, MGParams, linear=False):
       
        ########### Functions for matter power spectrum multiplied by Sigma**2 ###########        
        def pk_funcSigma_ESS_C(k, a):
            if linear == False:
                return P_k_NL_ESS_C(cosmo, k, a)  #*mu_ESS_C(a)**2
            elif linear == True:
                return P_k_ESS_C_lin(cosmo, k, a) #*mu_ESS_C(a)**2
    
        return ccl.pk2d.Pk2D.from_function(pkfunc=pk_funcSigma_ESS_C, is_logp=False)
    
    ###############################################################################
    # Get mock data and Covariance
    ###############################################################################
    
    # Define cosmology -- our "universe cosmology"
    
    cosmo_universe = ccl.Cosmology(Omega_c = Omega_c_var, 
                              Omega_b = Omega_b_var, 
                              h = h_var, 
                              n_s = ns_var,
                              A_s = As_var)
    
    
    MGParam_universe = [0,0,1,0,0]
    # define ell and C_ell shapes -- will depend on the data

    ell_min_mockdata = 20
    ell_max_mockdata = 1478.5
    
    # define quantities for binning of ell -- will depend on the data
    
    ell_bin_num_mockdata = 13
    
    print("collecting data")
    
    # ~~~~~~~~~~~~~~~~~~~~~~~~ Get mock C(ell) data ~~~~~~~~~~~~~~~~~~~~~~~~
    binned_ell = bin_ell_kk(ell_min_mockdata, ell_max_mockdata, ell_bin_num_mockdata, Binned_distribution_source)

    P_delta2D_GR_lin_universe = Get_Pk2D_obj_kk_GR_lin(cosmo_universe)
    P_delta2D_GR_nl_universe = Get_Pk2D_obj_kk_GR_nl(cosmo_universe)

    P_delta2D_ESS_kk_lin_universe = Get_Pk2D_obj_kk_ESS_C(cosmo_universe, MGParam_universe, linear=True)
    P_delta2D_ESS_kk_nl_universe = Get_Pk2D_obj_kk_ESS_C(cosmo_universe, MGParam_universe, linear=False)
    P_delta2D_ESS_delk_lin_universe = Get_Pk2D_obj_delk_ESS_C(cosmo_universe, MGParam_universe, linear=True)
    P_delta2D_ESS_delk_nl_universe = Get_Pk2D_obj_delk_ESS_C(cosmo_universe, MGParam_universe, linear=False)
    P_delta2D_ESS_deldel_lin_universe = Get_Pk2D_obj_deldel_ESS_C(cosmo_universe, MGParam_universe, linear=True)
    P_delta2D_ESS_deldel_nl_universe = Get_Pk2D_obj_deldel_ESS_C(cosmo_universe, MGParam_universe, linear=False)
    
    # find C_ell for non-linear matter power spectrum
    mockdata = Cell(binned_ell, cosmo_universe, z , Binned_distribution_source,Binned_distribution_lens,Bias_distribution_fiducial,MGParam_universe,\
                P_delta2D_ESS_kk_nl_universe,tracer1_type="k", tracer2_type="k")
    
    ell_kk_mockdata = mockdata[0]
    D_kk_mockdata = mockdata[1]
    D_kk_mockdata = (np.array(D_kk_mockdata)).flatten()
    
    # For plot below, compare with linear
    data_lin_plot = Cell(binned_ell, cosmo_universe, z , Binned_distribution_source,Binned_distribution_lens,Bias_distribution_fiducial,MGParam_universe,\
                P_delta2D_ESS_kk_lin_universe,tracer1_type="k", tracer2_type="k")
    
    D_kk_data_lin_plot = data_lin_plot[1]
    D_kk_data_lin_plot = (np.array(D_kk_data_lin_plot)).flatten()
    
    ## CLUSTERING - LENSING
    
    binned_ell = bin_ell_delk(ell_min_mockdata, ell_max_mockdata, ell_bin_num_mockdata,Binned_distribution_source,Binned_distribution_lens)
    
    # find C_ell for non-linear matter power spectrum
    mockdata = Cell(binned_ell, cosmo_universe, z , Binned_distribution_source,Binned_distribution_lens,Bias_distribution_fiducial,MGParam_universe,\
                P_delta2D_ESS_delk_nl_universe,tracer1_type="k", tracer2_type="g")
    
    ell_delk_mockdata = mockdata[0]
    D_delk_mockdata = mockdata[1]
    D_delk_mockdata = (np.array(D_delk_mockdata)).flatten()
    
    # For plot below, compare with linear
    data_lin_plot =  Cell(binned_ell, cosmo_universe, z , Binned_distribution_source,Binned_distribution_lens,Bias_distribution_fiducial,MGParam_universe,\
                P_delta2D_ESS_delk_lin_universe,tracer1_type="k", tracer2_type="g")
    
    D_delk_data_lin_plot = data_lin_plot[1]
    D_delk_data_lin_plot = (np.array(D_delk_data_lin_plot)).flatten()
    
    ## CLUSTERING - CLUSTERING
    binned_ell = bin_ell_deldel(ell_min_mockdata, ell_max_mockdata, ell_bin_num_mockdata,Binned_distribution_lens)
    
    # find C_ell for non-linear matter power spectrum
    mockdata = Cell(binned_ell, cosmo_universe, z , Binned_distribution_source,Binned_distribution_lens,Bias_distribution_fiducial,MGParam_universe,\
                P_delta2D_ESS_deldel_nl_universe,tracer1_type="g", tracer2_type="g")
    
    ell_deldel_mockdata = mockdata[0]
    D_deldel_mockdata = mockdata[1]
    D_deldel_mockdata = (np.array(D_deldel_mockdata)).flatten()
    
    # For plot below, compare with linear
    data_lin_plot = Cell(binned_ell, cosmo_universe, z , Binned_distribution_source,Binned_distribution_lens,Bias_distribution_fiducial,MGParam_universe,\
                P_delta2D_ESS_deldel_lin_universe,tracer1_type="g", tracer2_type="g")
    
    D_deldel_data_lin_plot = data_lin_plot[1]
    D_deldel_data_lin_plot = (np.array(D_deldel_data_lin_plot)).flatten()
    
    
    ell_mockdata = np.append(np.append(ell_kk_mockdata, ell_delk_mockdata), ell_deldel_mockdata)
    D_mockdata = np.append(np.append(D_kk_mockdata, D_delk_mockdata), D_deldel_mockdata)
    D_data_lin_plot = np.append(np.append(D_kk_data_lin_plot, D_delk_data_lin_plot), D_deldel_data_lin_plot)
    
    del mockdata, data_lin_plot
    
    # ~~~~~~~~~~~~~~~~~~~~~~~~ Get fsigma8 data ~~~~~~~~~~~~~~~~~~~~~~~~
    
    zeff= 0.72
    nbar = 5*10**(-4)
    Vol = 3*10**9
    
    f_fe, f_fid = f_frac_err(P_delta2D_GR_lin_universe,cosmo_universe,MGParam_universe,2.333, zeff, nbar, Vol)
    
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
    
    fsigma_8_data = fsigma8_ESS_C(cosmo_universe, MGParam_universe, 1/(z_eff+1))
    #fsigma_8_data_GR = fsigma8_musigma(P_delta2D_GR_lin_universe, cosmo_universe, [0,0,0,0,0], 1/(z_eff+1))
    fsigma_8_realdata_full = np.append(fsigma_8_realdata,[fsigma8_musigma(P_delta2D_GR_lin_universe,cosmo_universe, [0,0,0,0,0], [1./(1. + zeff),1./(1. + zeff)])[0]])
    
    cov_fsigma8 = reducedcov_fsigma_8 * np.outer(fsigma_8_realdata_full*fsigma_8_fracerror, fsigma_8_realdata_full*fsigma_8_fracerror)
    invcov_fsigma8 = np.linalg.inv(cov_fsigma8)
    
    # ~~~~~~~~~~~~~~~~~~~~~~~~ Get Covariance - for now just using SRD one ~~~~~~~~~~~~~~~~~~~~~~~~
    
    covfile = np.genfromtxt("Y1_3x2pt_clusterN_clusterWL_cov")
    
    shear_SRD = np.zeros((705,705))
    ell_test_SRD = np.zeros(705)
    
    for i in range(0,covfile.shape[0]):
        shear_SRD[int(covfile[i,0]),int(covfile[i,1])] = covfile[i,8]+covfile[i,9] # non-gauss
        shear_SRD[int(covfile[i,1]),int(covfile[i,0])] = covfile[i,8]+covfile[i,9] # non-gauss
        if int(covfile[i,0]) == int(covfile[i,1]):
            ell_test_SRD[int(covfile[i,0])] = covfile[i,2]
    
    del covfile
    SRD_compare = shear_SRD[:540,:540].copy()
    
    idx = 0
    
    bins_SRD = int(len(SRD_compare)/(len(D_mockdata)/ell_bin_num_mockdata))
    
    for j in range(int(len(D_mockdata)/ell_bin_num_mockdata)):
        for i in range(bins_SRD):
            if i >= ell_bin_num_mockdata:
                SRD_compare = np.delete(SRD_compare, j*bins_SRD + i - idx, 0)
                SRD_compare = np.delete(SRD_compare, j*bins_SRD + i - idx, 1)
                idx += 1
    
    L_choleski_uncut = np.linalg.cholesky(np.matrix(SRD_compare))
    L_choleski_inv_uncut = np.linalg.inv(L_choleski_uncut)

    ###############################################################################
    # Apply Scale Cuts
    ###############################################################################
    print("No scale cuts - we are using GR cuts")

    """
    print("starting linear scale cuts")
    
    # ~~~~~~~~~~~~~~~~~~~~~~~~ linear scale cuts ~~~~~~~~~~~~~~~~~~~~~~~~
        
    newdat_test = linear_scale_cuts_v2(D_mockdata, D_data_lin_plot, SRD_compare)
    
    gauss_invcov_cut = np.linalg.pinv(SRD_compare.copy())
    D_mockdata_cut = D_mockdata.copy()
    
    for i in range(len(newdat_test)):
        gauss_invcov_cut[newdat_test[i]] = np.zeros(len(gauss_invcov_cut[0]))
        gauss_invcov_cut[:,newdat_test[i]] = np.zeros(len(gauss_invcov_cut[0]))
        D_mockdata_cut[newdat_test[i]] = 0
    
    # ~~~~~~~~~~~~~~~~~~~~~~~~ baryonic scale cuts ~~~~~~~~~~~~~~~~~~~~~~~~
    ####### Get mock C(ell) data
    
    ## LENSING - LENSING
    
    binned_ell = bin_ell_kk(ell_min_mockdata, ell_max_mockdata, ell_bin_num_mockdata, Binned_distribution_source)
    
    # find C_ell for non-linear matter power spectrum
    mockdata = Cell(binned_ell, cosmo_universe, z , Binned_distribution_source,Binned_distribution_lens,Bias_distribution_fiducial,MGParam_universe,\
                P_delta2D_GR_nl_universe,tracer1_type="k", tracer2_type="k")
    
    ell_kk_mockdata = mockdata[0]
    D_kk_mockdata_test = mockdata[1]
    D_kk_mockdata_test = (np.array(D_kk_mockdata_test)).flatten()
    
    ## CLUSTERING - LENSING
    
    binned_ell = bin_ell_delk(ell_min_mockdata, ell_max_mockdata, ell_bin_num_mockdata,Binned_distribution_source,Binned_distribution_lens)
    
    # find C_ell for non-linear matter power spectrum
    mockdata = Cell(binned_ell, cosmo_universe, z , Binned_distribution_source,Binned_distribution_lens,Bias_distribution_fiducial,MGParam_universe,\
                P_delta2D_GR_nl_universe,tracer1_type="k", tracer2_type="g")
    
    ell_delk_mockdata = mockdata[0]
    D_delk_mockdata = mockdata[1]
    D_delk_mockdata = (np.array(D_delk_mockdata)).flatten()
    
    ## CLUSTERING - CLUSTERING
    binned_ell = bin_ell_deldel(ell_min_mockdata, ell_max_mockdata, ell_bin_num_mockdata,Binned_distribution_lens)
    
    # find C_ell for non-linear matter power spectrum
    mockdata = Cell(binned_ell, cosmo_universe, z , Binned_distribution_source,Binned_distribution_lens,Bias_distribution_fiducial,MGParam_universe,\
                P_delta2D_GR_nl_universe,tracer1_type="g", tracer2_type="g")
    
    ell_deldel_mockdata = mockdata[0]
    D_deldel_mockdata = mockdata[1]
    D_deldel_mockdata = (np.array(D_deldel_mockdata)).flatten()
    
    ell_mockdata = np.append(np.append(ell_kk_mockdata, ell_delk_mockdata), ell_deldel_mockdata)
    D_mockdata_test = np.append(np.append(D_kk_mockdata, D_delk_mockdata), D_deldel_mockdata)
    
    del mockdata
    
    ######## Get mock C(ell, k_max) data
    
    ## LENSING - LENSING
    
    binned_ell = bin_ell_kk(ell_min_mockdata, ell_max_mockdata, ell_bin_num_mockdata, Binned_distribution_source)
    
    # find C_ell for non-linear matter power spectrum
    mockdata = Cell(binned_ell, cosmo_universe, z , Binned_distribution_source,Binned_distribution_lens,Bias_distribution_fiducial,MGParam_universe,\
                    Get_Pk2D_obj_OWLSAGN(cosmo_universe, MGParam_universe, linear=False, gravity_model="GR"),tracer1_type="k", tracer2_type="k")
    
    ell_kk_mockdata = mockdata[0]
    D_kk_mockdata_kmax = mockdata[1]
    D_kk_mockdata_kmax = (np.array(D_kk_mockdata_kmax)).flatten()
    
    del mockdata
    
    print("starting baryonic scale cuts")
    
    newdat = baryonic_scale_cuts_v2(cosmo_universe, ell_mockdata, D_mockdata_test, D_kk_mockdata_test, D_kk_mockdata_kmax, SRD_compare)
    
    L_choleski = L_choleski_uncut
    L_choleski_inv = L_choleski_inv_uncut
    gauss_invcov_rotated = np.linalg.pinv(SRD_compare)
    
    for i in range(len(newdat)):
        L_choleski[newdat[i]] = np.zeros(len(L_choleski[0]))
        L_choleski[:,newdat[i]] = np.zeros(len(L_choleski[0]))
        L_choleski_inv[newdat[i]] = np.zeros(len(L_choleski_inv[0]))
        L_choleski_inv[:,newdat[i]] = np.zeros(len(L_choleski_inv[0]))
        gauss_invcov_rotated[newdat[i]] = np.zeros(len(gauss_invcov_rotated[0]))
        gauss_invcov_rotated[:,newdat[i]] = np.zeros(len(gauss_invcov_rotated[0]))

    """
    np.savez("Data_storage_ESS_partial_Sigma1.npz",
             C_ell_data=D_mockdata,
             ell_data=ell_mockdata,
             z=z,
             Binned_distribution_source=Binned_distribution_source,
             Binned_distribution_lens=Binned_distribution_lens,
             fsigma8_data=fsigma_8_data,
             z_eff_fsigma8 = z_eff,
             invcov_fsigma8=invcov_fsigma8)
             #L_ch=L_choleski,
             #L_ch_inv = L_choleski_inv,
             #Inverse_cov=gauss_invcov_cut)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Get simulated 3x2pt+fsigma8 data for an ESS universe at fixed p cosmo.')
    # MCMC parameters
    parser.add_argument('--b1', type=float, default=1.562362, help='linear galaxy bias parameter for first bin')
    parser.add_argument('--b2', type=float, default=1.732963, help='linear galaxy bias parameter for second bin')
    parser.add_argument('--b3', type=float, default=1.913252, help='linear galaxy bias parameter for third bin')
    parser.add_argument('--b4', type=float, default=2.100644, help='linear galaxy bias parameter for fourth bin')
    parser.add_argument('--b5', type=float, default=2.293210, help='linear galaxy bias parameter for fifth bin')
    parser.add_argument('--OmgC', type=float, default=0.269619, help='Omega_c, Dark Matter dimensionless energy density')
    parser.add_argument('--OmgB', type=float, default=0.050041, help='Omega_b, Baryonic Matter dimensionless energy density')
    parser.add_argument('--h', type=float, default=0.6688, help='h, reduced hubble parameter')
    parser.add_argument('--ns', type=float, default=0.9626, help='n_s, scalar spectral index')
    parser.add_argument('--As', type=float, default=2.092e-9, help='A_s, primordial amplitude of matter fluctuations')
    
    # carry on from here
    args = parser.parse_args()

    main(args)
