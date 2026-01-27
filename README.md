# PCA-based 3x2pt Parameter Inference Framework
[![arXiv](https://img.shields.io/badge/arXiv-2503.20951-b31b1b.svg)](https://arxiv.org/abs/2503.20951)

This repository provides a **Principal Components Analysis (PCA)–based data reduction framework** to perform parameter inference for modified gravity parametrizations with 3×2pt LSST Y1–like simulated data.

The code supports both:

* **Standard linear scale cuts**, and
* A **novel PCA-based data reduction method** designed to retain maximal cosmological information while reducing data dimensionality.

---

## Overview

The pipeline allows you to:

1. Generate or load a 3x2pt data vector and covariance matrix with scale cuts applied.
2. Optionally perform PCA-based data reduction.
3. Run likelihood-based parameter inference.

Multiple gravity theories are supported:

* **GR**
* **f(R)**
* **nDGP**
* **ESS** (jupyter notebooks only)

---

## Repository Structure

```
.
├── Parameter_Inference_GR/
    ├── Data_storage_*.npy
    ├── LikelihoodFuncts_*.py
    ├── Get_Data_3x2pt_fsigma8_*.py
    ├── Likelihood_*_parallel_mpi_*.py
    ├── Y1_3x2pt_clusterN_clusterWL_cov
    ├── Likelihood_*.ipynb
    ....
├── Parameter_Inference_fR/
├── Parameter_Inference_nDGP/
├── Parameter_Inference_ESS/
├── Parameter_inference_q1Emu/
├── Visualizing_PCAs/
    ├── manim_visualising_PCAs.ipynb
├── Emulator_Tutorials
├── README.md
```

Each `Parameter_Inference_{THEORY}` directory contains all scripts required to generate data products and run inference for a given gravity model.

Likelihood_*.ipynb show the parameter inference process in notebook form. They are a good place to start from to figure out how the code works. LikelihoodFuncts_*.py contain most of the underlying functions for the code.

manim_visualising_PCAs.ipynb within Visualizing_PCAs/ has a good visualization of how PCA data reduction for MG theories works. Alternatively, see the video below:

<video width="600" autoplay loop muted>
  <source src="PCA_elltodataspace.mp4" type="video/mp4">
</video>

---

## Environment Setup

We recommend running the code inside a dedicated Python virtual environment.

### Create virtual environment and Install dependencies

```bash
python3 -m venv MG-PCA_venv
source MG-PCA_venv/bin/activate
pip install -r requirements.txt
```

---

## Running the Code

### 1. Choose a gravity theory

Navigate to the directory corresponding to the gravity theory of interest:

```bash
cd Parameter_Inference_{THEORY}
```

where `{THEORY}` can be `GR`, `fR`, `nDGP`, or `ESS`.

---

### 2. Generate data vector and covariance (optional)

To compute the **data vector**, **covariance**, and **Cholesky decomposition** with scale cuts applied, run:

```bash
python Get_Data_3x2pt_fsigma8_GR.py
```

**Warning:** This step is **very time-consuming**. We strongly recommend using the precomputed files already included in the repository unless you explicitly need to regenerate them.

#### Notes:

* Cosmological and bias parameters can be passed as command-line arguments.
* Binning options are, in principle, configurable.
* The current setup uses a **hard-coded LSST Y1 covariance** from the DESC SRD:

  * [https://github.com/CosmoLike/DESC_SRD](https://github.com/CosmoLike/DESC_SRD)
* This implies a **fixed binning scheme**.

All necessary code to modify the binning is already in place. Once a covariance file is available:

* Load it from a text file via `covariance_file`
* Remove the hard-coded section:

```python
## NOTE: change/remove the code below when there is a better way to find the covariance
# rather than just taking the SRD one and cutting it down.
###############################################################################
...

# remove code until here when better covariance is available. Define new SRD_compare
```

---

### 3. Run parameter inference

To run the likelihood analysis and obtain posterior chains, execute:

```bash
python Likelihood_{THEORY}parallel_mpi{CUTS}.py
```

where:

* `{THEORY}` = `GR`, `fR` or `nDGP` (for `ESS`, only notebook is available)
* `{CUTS}` = `PCACuts` or `StandardCuts`

for now, using emcee package. Implementing the nautilus sampler is work in progress.
---

## Configuration Options

The likelihood script supports several command-line arguments, including:

* Prior choices for cosmological and nuisance parameters
* A flag to include **Planck priors** on `ω_b` and `n_s`
* Input data file (default: `Data_storage_{THEORY}.npy`)
* Binning configuration

Refer to the script header for a complete list of available options.

---

## Notes and Limitations

* Survey specifications and covariance matrices are currently **hard-coded**.
* The framework is designed for flexibility but assumes familiarity with 3×2pt analyses and likelihood-based inference.

## Reference / Citation

If you use this code in your work, please cite:

- **Principal Components for Model-Agnostic Modified Gravity with 3x2pt**  
  C. M. A. Zanoletti & C. D. Leonard, *arXiv:2503.20951*  
  https://arxiv.org/abs/2503.20951