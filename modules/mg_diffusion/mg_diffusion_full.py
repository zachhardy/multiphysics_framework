import numpy as np
import numpy.linalg as npla
from scipy.sparse import bmat
from scipy.sparse.linalg import spsolve

from physics.discrete_system import DiscreteSystem
from .mg_diffusion_base import MultiGroupDiffusionBase

class MultiGroupDiffusionFull(DiscreteSystem, MultiGroupDiffusionBase):

  def __init__(self, problem, discretization, bcs, ics=None):
    MultiGroupDiffusionBase.__init__(
      self, problem, discretization, bcs, ics
    )
    DiscreteSystem.__init__(self, self.n_dofs, self.bcs)

  def solve_system(self, method=None, time=None, dt=None, *args):
    if not self.problem.is_transient:
      self.u[:] = self.solve_steady_state()
    else:
      u_tmp = None if args==() else args[0][self.dofs]
      self.u[:] = self.solve_time_step(
        method, time, dt, self.u_old, u_tmp
      )

  def assemble_physics(self):
    if self.is_nonlinear or self.A is None:
      matrix = [[None]*self.n_grps for _ in range(self.n_grps)]
      for igrp in self.groups:
        i = igrp.group_num
        for jgrp in self.groups:
          j = jgrp.group_num
          matrix[i][j] = igrp.assemble_fission(jgrp)
          matrix[i][j] += igrp.assemble_scattering(jgrp)
          if i == j:
            igrp.assemble_physics()
            matrix[i][j] += igrp.A
      self.A = bmat(matrix, format='csr')

  def assemble_mass(self):
    if self.is_nonlinear or self.M is None:
      blocks = [[None]*self.n_grps for _ in range(self.n_grps)]
      for group in self.groups:
        g = group.group_num
        group.assemble_mass()
        blocks[g][g] = group.M
      self.M = bmat(blocks, format='csr')   

  def assemble_forcing(self, time=0):
    blocks = []
    for group in self.groups:
      group.assemble_forcing(time)
      blocks.append(group.rhs)
    self.rhs = np.block(blocks)

  def apply_bcs(self, matrix=None, vector=None):
    for group in self.groups:
      offset = group.group_num*group.n_dofs
      group.apply_bcs(matrix, vector, offset)

  def compute_old_physics_action(self):
    if self.A is None:
      self.assemble_physics()
    self.f_old = self.A @ self.u_old

  def assemble_transport_operator(self):
    L = [[None]*self.n_grps for _ in range(self.n_grps)]
    for group in self.groups:
      g = group.group_num
      group.assemble_physics()
      L[g][g] = group.A
    return bmat(L, format='csr')

  def assemble_source_operator(self):
    Q = [[None]*self.n_grps for _ in range(self.n_grps)]
    for igrp in self.groups:
      i = igrp.group_num
      for jgrp in self.groups:
        j = jgrp.group_num
        Q[i][j] = -igrp.assemble_fission(jgrp)
        Q[i][j] -= igrp.assemble_scattering(jgrp)
    return bmat(Q, format='csr')

  def compute_k_eigenvalue(self, tol=1e-8, maxit=100, verbosity=0):
    # Zero out source and set to steady state
    self.problem.is_transient = False
    for material in self.materials:
      if hasattr(material, 'q'):
        material.q = np.zeros(self.n_grps)

    # Initialize initial guesses and operators
    L = self.assemble_transport_operator()
    Q = self.assemble_source_operator()
    self.u_ell[:] = 1
    k_eff_ell = 1

    # Inverse power iteration
    converged = False
    for nit in range(maxit):
      # Form right hand side, apply cfe boundary values.
      # Note that this is simply to apply dirichlet bcs.
      rhs = Q @ self.u_ell
      if self.discretization.dtype == 'cfe':
        self.apply_bcs(vector=rhs)
      # Solve the system
      self.u[:] = spsolve(L, rhs)
      # Recompute k-eff
      k_eff = self.compute_fission_power()
      # Compute change and reset prev iteration params
      k_error = np.abs(k_eff-k_eff_ell) / np.abs(k_eff)
      k_eff_ell = k_eff
      self.u_ell[:] = self.u / k_eff
      # Check convergence
      if k_error < tol:
        converged = True
        break
      if verbosity > 1:
        self.print_k_iter_summary(nit, k_eff, k_error)
    self.print_k_calc_summary(converged, nit, k_eff, k_error)


      
    





 
  


