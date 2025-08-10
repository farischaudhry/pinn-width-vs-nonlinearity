# Scaling Laws for Single-Layer PINNs: Network Width, PDE Nonlinearity, and Approximation Error

This repository contains the PyTorch implementation and experimental data for the paper, "Scaling Laws and Pathologies of Single-Layer PINNs: Network Width and PDE Nonlinearity".

## Abstract

While Physics-Informed Neural Networks (PINNs) are widely used, a quantitative understanding of how their approximation error scales with network architecture and intrinsic problem difficulty is lacking. We conduct a systematic empirical study of Single-Layer PINNs (SLNs), fitting scaling laws of the form `error ≈ A * N^-α * κ^γ` where `N` is network width and `κ` quantifies PDE nonlinearity. Across a suite of canonical nonlinear PDEs (KdV, Sine-Gordon, Allen-Cahn), we find a consistent trend: the width scaling exponent `α` systematically degrades as nonlinearity `κ` increases. Our results are validated against approximation theory on the linear Poisson equation, where our framework recovers the expected theoretical scaling rate. This work provides the first quantitative benchmarks for this trade-off, offering practical guidance on the limitations of scaling width for SLNs in highly nonlinear regimes and motivating the need for more complex architectures.

## Framework Overview

This project provides a highly modular and extensible framework for conducting large-scale empirical studies on Physics-Informed Neural Networks. The core components are:

**`PDEProblem` (Abstract Base Class):** A flexible interface for defining new PDE problems, including their governing equations, domain, boundary/initial conditions, ground truth solutions, and derivative requirements.
**Generalized Derivative Computation:** A `Trainer` class with a powerful derivative engine that can compute arbitrary non-mixed and mixed derivatives of the network's outputs with respect to its inputs using `torch.autograd`.
**Systematic Experiment Management:** An `ExperimentRunner` that takes declarative configuration objects (`ExperimentConfig`) to systematically sweep over parameters like PDE type, network architecture, and problem hardness (`kappa`).
**Comprehensive Logging:** Each experiment run automatically saves its configuration, a detailed epoch-wise training log (`training_log.csv`), and a final summary of metrics (`summary.json`).

## Experimental Results

Git Large File Storage (LFS) is used for the experimental results. These can be found in the 'experiment_data_test_v1.zip' file.
