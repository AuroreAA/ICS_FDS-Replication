# Invariant Coordinate Selection and Fisher discriminant subspace beyond the case of two groups

## About ICS

Invariant Coordinate Selection is  a powerful unsupervised multivariate method designed to identify the structure of multivariate datasets on a subspace. It relies on the joint diagonalization of two scatter matrices and is particularly relevant as a dimension reduction tool prior to clustering or outlier detection.

More information can be found in our article:

Becquart, C., Archimbaud, A., Ruiz-Gazen, A., Prilć, L., & Nordhausen, K. (2024).
Invariant Coordinate Selection and Fisher discriminant subspace beyond the case of two groups. 
*arXiv preprint* [arXiv:2409.17631](https://arxiv.org/abs/2409.17631).

## Reproduce results

This repository provides a collection of [python](https://www.python.org/) 
ersion 3.10.13  scripts to reproduce all computations in Sections 2, 3, 4. 
Furthermore, all simulations in Section 6 are performed with
[R](https://CRAN.R-project.org/) version 4.3.3. The easiest way to reproduce the 
results is to clone this repository.

Please note that this repository is rather large because it also contains 
data files with all simulation results.  This way, if you only want to quickly
reproduce the figures with simulation results, you do not actually need to run 
the simulations first.