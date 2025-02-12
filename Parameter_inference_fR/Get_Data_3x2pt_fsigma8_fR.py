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
    MG_param_var = args.MG_param
    gravity_flag = str(args.gravity_flag)
    
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
    # Get mock data and Covariance
    ###############################################################################
    
    # Define cosmology -- our "universe cosmology"
    
    cosmo_universe = ccl.Cosmology(Omega_c = Omega_c_var, 
                              Omega_b = Omega_b_var, 
                              h = h_var, 
                              n_s = ns_var,
                              A_s = As_var)
    
    
    if gravity_flag == "f(R)":
        fR_universe = MG_param_var
        H0rc_universe = 0.0
    elif gravity_flag == "nDGP":
        fR_universe = 0.0
        H0rc_universe = MG_param_var
    else:
        raise Exception('There is no {} gravity_flag.'.format(gravity_flag))

    MGParam_universe = [H0rc_universe,fR_universe,1,0,0]
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

    # GET F(R) LINEAR POWER SPECTRUM, universe
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
                      args=(cosmo_universe,MGParam_universe, k_solver[i]), mxstep=int(1e4))
        
        Delta_i = Soln.T[0]
        Delta.append(Delta_i/Delta_i[idx_mdom])
    
    Delta = np.array(Delta)
    
    # Get Pk linear in GR
    
    interp_fR_Pk_universe = scipy.interpolate.RegularGridInterpolator((k_solver,a_solver), (Delta / Delta_GR)**2,bounds_error=False, fill_value=1.0)
    ##########################
    

    P_delta2D_fR_lin_universe = Get_Pk2D_obj(P_delta2D_GR_lin_universe,interp_fR_Pk_universe,cosmo_universe, MGParam_universe, linear=True, gravity_model="f(R)")
    P_delta2D_fR_nl_universe = Get_Pk2D_obj(P_delta2D_GR_nl_universe,interp_fR_Pk_universe,cosmo_universe, MGParam_universe, linear=False, gravity_model="f(R)")
    
    # find C_ell for non-linear matter power spectrum
    mockdata = Cell(binned_ell, cosmo_universe, z , Binned_distribution_source,Binned_distribution_lens,Bias_distribution_fiducial,MGParam_universe,\
                P_delta2D_fR_nl_universe,tracer1_type="k", tracer2_type="k")
    
    ell_kk_mockdata = mockdata[0]
    D_kk_mockdata = mockdata[1]
    D_kk_mockdata = (np.array(D_kk_mockdata)).flatten()
    
    # For plot below, compare with linear
    data_lin_plot = Cell(binned_ell, cosmo_universe, z , Binned_distribution_source,Binned_distribution_lens,Bias_distribution_fiducial,MGParam_universe,\
                P_delta2D_fR_lin_universe,tracer1_type="k", tracer2_type="k")
    
    D_kk_data_lin_plot = data_lin_plot[1]
    D_kk_data_lin_plot = (np.array(D_kk_data_lin_plot)).flatten()
    
    ## CLUSTERING - LENSING
    
    binned_ell = bin_ell_delk(ell_min_mockdata, ell_max_mockdata, ell_bin_num_mockdata,Binned_distribution_source,Binned_distribution_lens)
    
    # find C_ell for non-linear matter power spectrum
    mockdata = Cell(binned_ell, cosmo_universe, z , Binned_distribution_source,Binned_distribution_lens,Bias_distribution_fiducial,MGParam_universe,\
                P_delta2D_fR_nl_universe,tracer1_type="k", tracer2_type="g")
    
    ell_delk_mockdata = mockdata[0]
    D_delk_mockdata = mockdata[1]
    D_delk_mockdata = (np.array(D_delk_mockdata)).flatten()
    
    # For plot below, compare with linear
    data_lin_plot =  Cell(binned_ell, cosmo_universe, z , Binned_distribution_source,Binned_distribution_lens,Bias_distribution_fiducial,MGParam_universe,\
                P_delta2D_fR_lin_universe,tracer1_type="k", tracer2_type="g")
    
    D_delk_data_lin_plot = data_lin_plot[1]
    D_delk_data_lin_plot = (np.array(D_delk_data_lin_plot)).flatten()
    
    ## CLUSTERING - CLUSTERING
    binned_ell = bin_ell_deldel(ell_min_mockdata, ell_max_mockdata, ell_bin_num_mockdata,Binned_distribution_lens)
    
    # find C_ell for non-linear matter power spectrum
    mockdata = Cell(binned_ell, cosmo_universe, z , Binned_distribution_source,Binned_distribution_lens,Bias_distribution_fiducial,MGParam_universe,\
                P_delta2D_fR_nl_universe,tracer1_type="g", tracer2_type="g")
    
    ell_deldel_mockdata = mockdata[0]
    D_deldel_mockdata = mockdata[1]
    D_deldel_mockdata = (np.array(D_deldel_mockdata)).flatten()
    
    # For plot below, compare with linear
    data_lin_plot = Cell(binned_ell, cosmo_universe, z , Binned_distribution_source,Binned_distribution_lens,Bias_distribution_fiducial,MGParam_universe,\
                P_delta2D_fR_lin_universe,tracer1_type="g", tracer2_type="g")
    
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
    
    fsigma_8_data = fsigma8_fR(P_delta2D_GR_lin_universe, interp_fR_Pk_universe, cosmo_universe, MGParam_universe, 1/(z_eff+1))
    
    #fsigma_8_data_GR = fsigma8_musigma(P_delta2D_GR_lin_universe, cosmo_universe, [0,0,0,0,0], 1/(z_eff+1))
    fsigma_8_realdata_full = np.append(fsigma_8_realdata,[fsigma8_musigma(P_delta2D_GR_lin_universe,cosmo_universe, [0,0,0,0,0], [1./(1. + zeff),1./(1. + zeff)])[0]])
    
    cov_fsigma8 = reducedcov_fsigma_8 * np.outer(fsigma_8_realdata_full*fsigma_8_fracerror, fsigma_8_realdata_full*fsigma_8_fracerror)
    invcov_fsigma8 = np.linalg.inv(cov_fsigma8)
    
    # ~~~~~~~~~~~~~~~~~~~~~~~~ Get Covariance - for now just using SRD one ~~~~~~~~~~~~~~~~~~~~~~~~
    print("No scale cuts - we are using GR cuts")

    """
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
    np.savez("Data_storage_fR_partial.npz",
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
    parser = argparse.ArgumentParser(description='Get simulated 3x2pt+fsigma8 data for an f(R) universe.')
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
    parser.add_argument('--gravity_flag', type=str, default='f(R)', help='gravity model, choice between f(R) and nDGP')
    parser.add_argument('--MG_param', type=float, default=1e-5, help='characteristic modified gravity parameter of the MG gravity (fR0 or H0r_c)')
    # carry on from here
    args = parser.parse_args()

    main(args)
