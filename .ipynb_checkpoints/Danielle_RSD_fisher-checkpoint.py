import numpy as np
import pyccl as ccl
import scipy.integrate

def f_frac_err(params, zeff, nbar, V):
    """ Get the fractional error on the growth rate.
    params is a dictionary of required cosmological parameters.
    zeff is the effective redshift of the sample.
    nbar is the number density of the sample in units of (h/Mpc)^3.
    V is the volume of the survey in (Mpc/h)^3. """
    
    # Set up a k and mu vector at which to do the integrals
    # (Result depends on kmax chosen, see White et al. 2008)
	
    k = np.logspace(-3, -1, 400)
    mu = np.linspace(-1, 1., 200)
    
    cosmo = ccl.Cosmology(Omega_c = params['OmM'] - params['OmB'], Omega_b = params['OmB'], h = params['h'], sigma8=params['sigma8'], n_s = params['n_s'], mu_0 = params['mu_0'], sigma_0 = params['sigma_0'], matter_power_spectrum='linear')
    
    # Get the inverse covariance value at each k and mu
    print("Getting inverse data covariance for beta variance calculation.")
    invcov = Pobs_covinv(params, k, mu, zeff, nbar)
    
    # Get the derivative of the observed z-space
    # P(k) with respect to b and f at each k and mu
    # (linear theory)
    print("Getting derivative.")
    dPdf = diff_P_f(params, k, mu, zeff)
    
    # Do the integration in k in each case
    print("Doing k integration.")
    int_in_k_ff = [ scipy.integrate.simps(k**2 * dPdf[mi] * invcov[mi] * dPdf[mi], k) for mi in range(len(mu))]
	
    # And in mu.
    print("Doing mu integration.")
    int_in_mu_ff = scipy.integrate.simps(np.asarray(int_in_k_ff), mu)
	
    Fisher_ff = np.zeros((2,2)) # order is b then f
    
    # Add necessary factors of volume (Mpc/h)^3 and pi etc
    ff= V * int_in_mu_ff / (2. * np.pi**2)
    err_f = np.sqrt(1./ff)
    
    # Now use this to construct the error on f:
    f_fid = ccl.growth_rate(cosmo, 1./(1. + zeff))

    frac_err_f = err_f / f_fid
    
    return frac_err_f, f_fid
    
def diff_P_f(params, k, mu, zeff):
    """ Calculate the derivative of the redshift space power spectrum
    wrt linear growth rate f at each k and mu
    params: dictionary of cosmological parameters
    k: list or array of wavenumbers
    mu: list of array of angles
    lens: lens sample label
    """
	
    # Set up a ccl cosmology
    cosmo = ccl.Cosmology(Omega_c = params['OmM'] - params['OmB'], Omega_b = params['OmB'], h = params['h'], sigma8=params['sigma8'], n_s = params['n_s'], mu_0 = params['mu_0'], sigma_0 = params['sigma_0'], matter_power_spectrum='linear')
	
    # Get the linear power spectrum at the effective z of the lens sample
    Pklin = ccl.linear_matter_power(cosmo, k * params['h'], 1./ (1. + zeff)) * params['h']**3

    # Get the derivative at each mu / k
    b = params['b']
    f = ccl.growth_rate(cosmo, 1./(1. + zeff))
    dPdf = [2. * (params['b'] + mu[mi]**2*f) * mu[mi]**2 * Pklin  for mi in range(len(mu))]
	
    return dPdf
    
def Pobs_covinv(params, k, mu, zeff, nbar):
    """ Get the inverse covariance of the redshift space observed power 
    spectrum at a list of k and mu (cosine of angle to line of sight) vals.
    params: dictionary of cosmological parameters
    k: list or array of wavenumbers
    mu: list of array of angles
    lens: lens sample label """	
 
    # Set up a ccl cosmology
    cosmo = ccl.Cosmology(Omega_c = params['OmM'] - params['OmB'], Omega_b = params['OmB'], h = params['h'], sigma8=params['sigma8'], n_s = params['n_s'], mu_0 = params['mu_0'], sigma_0 = params['sigma_0'], matter_power_spectrum='linear')
	
    # Get the linear power spectrum at the effective z of the lens sample
    Pklin = ccl.linear_matter_power(cosmo, k * params['h'], 1./ (1. + zeff)) * params['h']**3
	
    # Get the redshift space galaxy power spectrum in linear theory
    f = ccl.growth_rate(cosmo, 1./(1. + zeff))
    Pgg = [ (params['b'] + f * mu[mi]**2)**2 * Pklin for mi in range(len(mu))]
	
    # Get the covariance matrix at each k and mu
    cov = [ 2. * (Pgg[mi]**2 + 2 * Pgg[mi] / nbar + 1. / nbar**2) for mi in range(len(mu))]
	
    #Pgg_arr = np.zeros((len(k), len(mu)))
    #for mi in range(len(mu)):
    #	Pgg_arr[:, mi] = Pgg[mi]
	
    # Get the inverse at each k and mu
    invcov = [[1./ cov[mi][ki] for ki in range(len(k))] for mi in range(len(mu))]
			
    return invcov
    
params = h=0.69
OmB = 0.022/h**2
params = {'mu_0': 0., 'sigma_0':0., 'OmB':OmB, 'h':h, 'n_s':0.965, 'sigma8':0.82,'b':2.03, 'OmM': 0.292}

zeff= 0.72
nbar = 5*10**(-4)
Vol = 3*10**9

f_fe, f_fid = f_frac_err(params, zeff, nbar, Vol)

print('sigma(f) / f =',f_fe)
print('fid f=', f_fid)
print('sigma(f) =', f_fe*f_fid)

