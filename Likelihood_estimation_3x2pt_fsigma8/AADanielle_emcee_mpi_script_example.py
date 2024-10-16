import os
import mpi4py
#os.environ["OMP_NUM_THREADS"] = "1"
import cProfile

import time
import numpy as np
import sys

def log_prob(theta):
    t = time.time() + np.random.uniform(0.5, 0.9)
    while True:
        if time.time() >= t:
            break
    return -0.5 * np.sum(theta**2)


import emcee
from schwimmbad import MPIPool

def main():
    np.random.seed(42)
    initial = np.random.randn(64, 5)
    nwalkers, ndim = initial.shape
    #print('nwalker=',nwalkers,' ndim=', ndim)
    nsteps = 100

    with MPIPool() as pool:
        if not pool.is_master():
            pool.wait()
            sys.exit(0)
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob, pool=pool)
        start = time.time()
        sampler.run_mcmc(initial, nsteps, progress=True)
        end = time.time()
        multi_time = end - start
        print("Multiprocessing took {0:.1f} seconds".format(multi_time))
        #f = open("/scratch/dp339/dc-leon2/scaling_tests/times.dat", "a")
        #f.write(str(multi_time))
        #f.write('\n')
        #f.close()
        #print("{0:.1f} times faster than serial".format(serial_time / multi_time))

