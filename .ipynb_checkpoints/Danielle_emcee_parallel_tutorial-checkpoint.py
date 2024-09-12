import os

os.environ["OMP_NUM_THREADS"] = "1"

import time
import numpy as np


def log_prob(theta):
    t = time.time() + np.random.uniform(0.5, 0.9)
    while True:
        if time.time() >= t:
            break
    return -0.5 * np.sum(theta**2)


import emcee

np.random.seed(42)
initial = np.random.randn(32, 5)
nwalkers, ndim = initial.shape
print('nwalker=',nwalkers,' ndim=', ndim)
nsteps = 100

#sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob)
#start = time.time()
#sampler.run_mcmc(initial, nsteps, progress=True)
#end = time.time()
#serial_time = end - start
#print("Serial took {0:.1f} seconds".format(serial_time))

from multiprocessing import Pool

with Pool() as pool:
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob, pool=pool)
    start = time.time()
    sampler.run_mcmc(initial, nsteps, progress=True)
    end = time.time()
    multi_time = end - start
    print("Multiprocessing took {0:.1f} seconds".format(multi_time))
    #print("{0:.1f} times faster than serial".format(serial_time / multi_time))