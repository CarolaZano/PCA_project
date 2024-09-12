import numpy as np
import scipy.integrate
import pyccl as ccl


np.set_printoptions(linewidth=240)
	


def average_in_bins(F_, R_, Rp_):
    """ This function takes a 1D function F_ of projected radius evaluated at projected radial points R_ and outputs the averaged values in bins with edges Rp_
    F_: 1D function of R_
    R_: argument of F_
    Rp_: vector of bin edges in R_""" 

    # Interpolate in a more well-sampled vector in R_
    R_f_ = scipy.logspace(np.log10(R_[0]), np.log10(R_[-1]), 10000)
    interp_F = scipy.interpolate.interp1d(np.log10(R_), F_)
    F_f_ = interp_F(np.log10(R_f_))
    
    indlow= [next(j[0] for j in enumerate(R_f_) if j[1]>=(Rp_[iR])) for iR in range(0, len(Rp_)-1)]
    indhigh= [next(j[0] for j in enumerate(R_f_) if j[1]>=(Rp_[iR+1])) for iR in range(0, len(Rp_)-1)]
		
    F_binned= [2. * scipy.integrate.simps(F_f_[indlow[i]:indhigh[i]]* (R_f_[indlow[i]:indhigh[i]])**2, np.log(R_f_[indlow[i]:indhigh[i]])) / (R_f_[indhigh[i]]**2 - R_f_[indlow[i]]**2) for i in range(0, len(Rp_) - 1)]
		
    return F_binned
    
def average_in_bins_2D(F_, R_, Rp_):
    """ This function takes a 2D function F_ of projected radius evaluated at projected radial points R_ in both dimensions and outputs the averaged values in bins with edges Rp_
    F_: 2D function of R_, R_' (both given by R_)
    R_: argument of F_
    Rp_: vector of bin edges in R_""" 
    
    # Interpolate the first dimension in a more well-sampled vector in R_
    R_f_ = scipy.logspace(np.log10(R_[0]), np.log10(R_[-1]), 50000)
    print('Rf0=', R_f_[0], 'Rf-1=', R_f_[-1])
    F_f_1 = np.zeros((len(R_), len(R_f_)))
    for ri in range(len(R_)):
        interp_F = scipy.interpolate.interp1d(np.log10(R_), F_[ri, :])
        F_f_1[ri, :] = interp_F(np.log10(R_f_))
 
    # Get the indices of R_f_ corresponding to the bin edges
    indlow= [next(j[0] for j in enumerate(R_f_) if j[1]>=(Rp_[iR])) for iR in range(0, len(Rp_)-1)]
    indhigh= [next(j[0] for j in enumerate(R_f_) if j[1]>=(Rp_[iR+1])) for iR in range(0, len(Rp_)-1)]
    
    # Integrate in the first dimension	
    F_binned_1 = [[2. * scipy.integrate.simps(F_f_1[rpi, indlow[i]:indhigh[i]]* (R_f_[indlow[i]:indhigh[i]])**2, np.log(R_f_[indlow[i]:indhigh[i]])) / (R_f_[indhigh[i]]**2 - R_f_[indlow[i]]**2) for rpi in range(len(R_))] for i in range(0, len(Rp_) - 1)] 

    # Interpolate the second dimension 
    F_f_2 = np.zeros((len(R_f_), len(R_f_)))
    for ri in range(len(Rp_)-1):
        interp_F = scipy.interpolate.interp1d(np.log10(R_), F_binned_1[ri])
        F_f_2[:, ri] = interp_F(np.log10(R_f_))
		
    # Integrate in the first dimension	
    F_binned_2 = [[2. * scipy.integrate.simps(F_f_2[indlow[i]:indhigh[i], rpi]* (R_f_[indlow[i]:indhigh[i]])**2, np.log(R_f_[indlow[i]:indhigh[i]])) / (R_f_[indhigh[i]]**2 - R_f_[indlow[i]]**2) for i in range(0, len(Rp_) - 1)] for rpi in range(len(Rp_)-1)]
    
    F_binned = np.zeros((len(Rp_)-1, len(Rp_)-1))
    for rpi in range(len(Rp_)-1):
        F_binned[rpi, :] = F_binned_2[rpi]
		
    return F_binned

def rp_bins_mid(rp_edges):
	""" Gets the middle of each projected radius bin."""

	logedges=np.log10(rp_edges)
	bin_centers=np.zeros(len(rp_edges)-1)
	for ri in range(0,len(rp_edges)-1):
		bin_centers[ri]    =       10**((logedges[ri+1] - logedges[ri])/2. +logedges[ri])

	return bin_centers
	
def rp_bin_edges_log(rp_mid):
    """ Gets the appropriate middles of log-spaced bins given a set of bin edges."""

    #This assumes that the bin centres are log-spaced
    logrmid=np.log10(rp_mid)
    delta_logrmid = np.diff(logrmid)[0] # They will all be the same
    delta_logrmid_half = delta_logrmid/2.
    
    log_edges=np.zeros((len(rp_mid)+1))
    log_edges[0] = np.asarray([logrmid[0]-delta_logrmid_half])
    for i in range(0,len(rp_mid)):
        log_edges[i+1] = logrmid[i]+delta_logrmid_half
        
    edges = 10**log_edges

    return edges
    
def z_ofcom_func(params):
	""" Returns an interpolating function which can give z as a function of comoving distance. """

	z_vec = scipy.linspace(0., 10., 4000) # This hardcodes that we don't care about anything over z=2100
	
	cosmo = ccl.Cosmology(Omega_c = params['OmM'] - params['OmB'], Omega_b = params['OmB'], h = params['h'], sigma8=params['sigma8'], n_s = params['n_s'], mu_0 = params['mu_0'], sigma_0 = params['sigma_0'], matter_power_spectrum='linear')
	com_vec =  ccl.background.comoving_radial_distance(cosmo, 1./(1.+z_vec)) * params['h']
	print("maxchi=", np.amax(com_vec))

	z_of_com = scipy.interpolate.interp1d(com_vec, z_vec)

	return	z_of_com
	
def corr_mat(cov, endfilename):
	""" Takes the covariance matrix and gets the correlation matrix. """
	
	n = (cov.shape)[0] # It will be square
	
	corr = [ [ cov[i, j] / np.sqrt( cov[i,i] * cov[j,j]) for i in range(n)] for j in range(n)]
	
	# transform to array
	corr_arr = np.zeros((n,n))
	for i in range(n):
		corr_arr[i, :] = corr[i]
		
	np.savetxt('/home/danielle/Documents/CMU/Research/EG_comparison/txtfiles/joint_corrmat_'+endfilename+'.txt', corr_arr, fmt='%1.2f')
	
	return corr_arr
	
def linear_scale_cuts(dvec_nl, dvec_lin, cov, rpvec):
	""" Gets the scales (and vector indices) which are excluded if we
	are only keeping linear scales. We define linear scales such that 
	chi^2_{nl - lin) <=1.
	dvec_nl: data vector from nonlinear theory 
	dvec_lin: data vector from linear theory
	cov: data covariance
	rpvec: vector of projected radial bin centers. """
	
	# Make a copy of these initial input things before they are changed,
	# so we can compare and get the indices
	dvec_nl_in = dvec_nl; dvec_lin_in = dvec_lin; cov_in = cov;
	
	# Check that data vector and covariance matrices have consistent dimensions.
	if ( (len(dvec_nl)!=len(dvec_lin)) or (len(dvec_nl)!=len(cov[:,0])) 
	or (len(dvec_nl)!=len(cov[0,:])) ):
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
				
		print("sum_terms=", sum_terms)
		print("chi2=", np.sum(sum_terms))
		# Check if chi2<=1		
		if (np.sum(sum_terms)<=1.0):
			break
		else:
			# Get the indices of the largest value in sum_terms.
			inds_max = np.unravel_index(np.argmax(sum_terms, axis=None), sum_terms.shape)
			print("inds_max =", inds_max)
			# Remove this / these from the data vectors and the covariance matrix
			if (inds_max[0] == inds_max[1]):
				dvec_nl = np.delete(dvec_nl, inds_max[0])
				dvec_lin = np.delete(dvec_lin, inds_max[0])
				inv_cov = np.delete( np.delete(inv_cov, inds_max[0], axis=0), inds_max[0], axis=1)
			else:
				dvec_nl = np.delete(dvec_nl, inds_max)
				dvec_lin = np.delete(dvec_lin, inds_max)
				inv_cov = np.delete( np.delete( inv_cov, inds_max, axis=0 ), inds_max, axis=1)
				
	# Now we should have the final data vector with the appropriate elements cut.
	# Use this to get the rp indices and scales we should cut.

	ex_inds = [i for i in range(len(dvec_nl_in)) if dvec_nl_in[i] not in dvec_nl]
	print('ex_inds=', ex_inds)
	
	return ex_inds
	
def linear_scale_cuts_v2(dvec_nl, dvec_lin, cov, rpvec):
    """ Gets the scales (and vector indices) which are excluded if we
    are only keeping linear scales. We define linear scales such that 
    chi^2_{nl - lin) <=1.
	
    This is a version that is hopefully more reliable when data are highly correlated.
	
    dvec_nl: data vector from nonlinear theory 
    dvec_lin: data vector from linear theory
    cov: data covariance
    rpvec: vector of projected radial bin centers. """
	
    # Make a copy of these initial input things before they are changed,
    # so we can compare and get the indices
    dvec_nl_in = dvec_nl; dvec_lin_in = dvec_lin; cov_in = cov;
	
    # Check that data vector and covariance matrices have consistent dimensions.
    if ( (len(dvec_nl)!=len(dvec_lin)) or (len(dvec_nl)!=len(cov[:,0])) or (len(dvec_nl)!=len(cov[0,:])) ):
        raise(ValueError, "in linear_scale_cuts: inconsistent shapes of data vectors and / or covariance matrix.")
		
    # Cut elements of the data vector / covariance matrix until chi^2 <=1
    inv_cov = np.linalg.pinv(cov)

    while(True):
		
        # Get an array of all the individual elements which would go into 
        # getting chi2
        #sum_terms = np.zeros((len(dvec_nl), len(dvec_nl)))
        #for i in range(0,len(dvec_nl)):
        #    for j in range(0,len(dvec_nl)):
        #        sum_terms[i,j] = (dvec_nl[i] - dvec_lin[i]) * inv_cov[i,j] * (dvec_nl[j] - dvec_lin[j])
				
        #print("sum_terms=", sum_terms)
        #print("chi2=", np.sum(sum_terms))
        # Check if chi2<=1		
        
        # Get the chi2 in the case where you cut each data point
        # and then actually cut the one that reduces the chi2
        # the most
        chi2_temp = np.zeros(len(dvec_nl))
        for i in range(len(dvec_nl)):
            delta_dvec = np.delete(dvec_nl, i) - np.delete(dvec_lin, i)
            cov_cut = np.delete(np.delete(cov,i, axis=0), i, axis=1)
            inv_cov_cut = np.linalg.pinv(cov_cut)
            chi2_temp[i] = np.dot(delta_dvec, np.dot(inv_cov_cut, delta_dvec))
            #sum_temp[i] = np.sum(np.delete(np.delete(sum_terms, i, axis=0), i, axis=1))
        print('chi2_temp=', chi2_temp)
            
        #Find the index of data point that is cut to produce the smallest chi2:
        ind_min = np.argmin(chi2_temp)
        print('ind_min=', ind_min)
            
        # Cut that element
        dvec_nl = np.delete(dvec_nl, ind_min)
        dvec_lin = np.delete(dvec_lin, ind_min)
        cov = np.delete( np.delete(cov, ind_min, axis=0), ind_min, axis=1)
        #inv_cov = np.linalg.pinv(cov)
        print('cov=', cov)
            
        if (chi2_temp[ind_min]<=1.0):
            break
				
    # Now we should have the final data vector with the appropriate elements cut.
    # Use this to get the rp indices and scales we should cut.

    ex_inds = [i for i in range(len(dvec_nl_in)) if dvec_nl_in[i] not in dvec_nl]
    print('ex_inds=', ex_inds)
	
    return ex_inds
	
