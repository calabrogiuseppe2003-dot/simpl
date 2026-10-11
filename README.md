# SiMPL

## Introduction

This package implements the SiMPL algorithm in Firedrake (https://doi.org/10.1007/s00158-025-04008-9).

SiMPL is a structure-preserving proximal point method for density-based topology optimization problems. Mainly applied to compliance and multi-material problems, this package also extends the algorithm to fluid topology optimization constrained by Navier-Stokes.

Each descent step requires five solves:
- Forward solve (potentially nonlinear)
- Adjoint solve (can be skipped if the problem is self-adjoint like in compliance)
- Adjoint filter solve
- Projection (diagonal matrix when piecewise constants are used for the density)
- Filter solve

This package makes it simple to compose iterative linear solves for each solve. In particular, geometric multigrid may be used for all the solves but the mass projection which itself only requires one Jacobi iteration.

## Preconditioning for compliance

In [examples/compliance](https://github.com/calabrogiuseppe2003-dot/simpl/tree/main/examples/compliance), we include the implementation of a 2D cantilever. The problem is first solved on a base mesh where an LU factorization is used to solve the linear systems. We then prolong the solution and, on the finer mesh, we use geometric multigrid for the forward linear elasticity solve, filter solve, and adjoint filter solve. The adjoint solve is not required as the problem is self-adjoint. The projection is solved with one Jacobi iteration.


## Reynolds-robust preconditioning for Navier-Stokes

Using augmented Lagrangian preconditioning techniques with specialised vertex-star patch smoothers for the multigrid cycles, we include examples of robust linear solvers for Navier-Stokes flow with high Reynolds number (up to Re=5000 in 3D). These can be found in [examples/navierstokes](https://github.com/calabrogiuseppe2003-dot/simpl/tree/main/examples/navierstokes)

For details of robust geometric multigrid techniques for high Re Navier-Stokes flow see https://doi.org/10.1137/18M1219370.

This package implements a cheaper variant that only requires one multigrid cycle per outer FGMRES iteration, see https://www.firedrakeproject.org/demos/reynolds_robust_navier_stokes_hdiv.py.html.

### Installation

These scripts use Firedrake (finite element library), Netgen (to generate meshes) and VTK (to plot the solutions).

Follow the instructions here for installing Firedrake (https://www.firedrakeproject.org/install.html#installing-firedrake). To also install netgen and VTK, please run:

```
pip install --no-binary h5py 'firedrake[check,netgen,vtk]'
```

After Firedrake is installed,  clone the ```simpl``` package and pip install.

```
git clone git@github.com:calabrogiuseppe2003-dot/simpl.git
cd simpl
pip install .
```