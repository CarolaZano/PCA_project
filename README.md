This code allows one to run a simplified 3x2pt LSST Y1-like simulated analysis with either standard scale cuts or with a novel PCA-based data reduction method.

Running Likelihood_{THEORY}_parallel_mpi_{CUTS}.py (from the Parameter_Inference_{THEORY} directory) allows one to find the posterior chains for a given theory (THEORY = GR, f(R), nDGP or ESS) and a given data reduction method (CUTS = PCACuts or StandardCuts for the PCA-based reduction or for linear scale cuts respectively).

The survey specs and covariance are currently hard-coded.
