import sys
import numpy as np
import numpy.linalg as npla
from scipy.sparse import bmat, lil_matrix, block_diag
from scipy.sparse.linalg import spsolve

from physics.time_stepper import TimeStepperMixin
from .mg_diffusion_base import MultiGroupDiffusionBase

class MultiGroupDiffusionFull(TimeStepperMixin, MultiGroupDiffusionBase):

  def __init__(self, problem, discretization, bcs, ics=None, **kwargs):
    super().__init__(problem, discretization, bcs, ics, **kwargs)
    # Initialize discrete system
    self.A_ = None
    self.M_ = None
    if self.use_precursors and self.sub_precursors:
      self.b = np.zeros(self.n_nodes * self.n_grps)
      self.f = np.zeros(self.n_nodes * self.n_grps)
      self.f_old = np.zeros(self.n_nodes * self.n_grps)
    else:
      self.b = np.zeros(self.n_dofs)
      self.f = np.zeros(self.n_dofs)
      self.f_old = np.zeros(self.n_dofs)
      
  def solve_system(self, opt=0):
    if not self.problem.is_transient:
      self.u[:] = self.solve_steady_state()
    else:
      self.prepare_system(opt)
      self.u[:] = self.solve_time_step(opt)
      if self.use_precursors and self.sub_precursors:
        self.compute_fission_rate()
        for precursor in self.precursors:
          precursor.update_precursor(opt)

  def solve_steady_state(self):
    self.assemble_forcing()
    self.apply_bcs(self.A, self.b)
    return spsolve(self.A, self.b)

  def prepare_system(self, opt=0):
    if self.use_precursors and self.sub_precursors:
      if self.problem.method=='tbdf2':
        self.assemble_physics_matrix(opt)
      for group in self.groups:
        group.assemble_full_substitution_rhs(opt)
      self.f = np.block([g.f for g in self.groups])

  @property
  def A(self):
    if self.is_nonlinear or self.A_ is None:
      print('hree')
      return self.assemble_physics_matrix()
    return self.A_

  @property
  def M(self):
    if self.M_ is None:
      return self.assemble_mass_matrix()
    return self.M_

  def assemble_physics_matrix(self, opt=0):
    n_blks = self.n_grps
    if self.use_precursors and not self.sub_precursors:
      n_blks += self.n_precursors
    tmp = lil_matrix(tuple([self.n_nodes]*2))
    blocks = [[tmp]*n_blks for _ in range(n_blks)]

    # Assemble multi-group terms
    for igrp in self.groups:
      i = igrp.g
      blocks[i][i] += igrp.A
      for jgrp in self.groups:
        j = jgrp.g
        blocks[i][j] -= igrp.assemble_cross_group_matrix(jgrp)
        # Precursor substitution
        if self.use_precursors and self.sub_precursors:
          blocks[i][j] -= igrp.assemble_substitution_matrix(jgrp, opt)
  
      # Precursor source terms
      if self.use_precursors:
        if not self.sub_precursors:
          for precursor in self.precursors:
            j = self.n_grps + precursor.j
            blocks[i][j] -= igrp.assemble_precursor_matrix(precursor)
            
    # Assemble precursor terms
    if self.use_precursors and not self.sub_precursors:
      for precursor in self.precursors:
        i = self.n_grps + precursor.j
        blocks[i][i] += precursor.A
        for group in self.groups:
          j = group.g
          blocks[i][j] -= precursor.assemble_production_matrix(group)

    self.A_ = bmat(blocks, format='csr')
    return self.A_

  def assemble_mass_matrix(self):
    blocks = []
    # Assemble multi-group terms
    for group in self.groups:
      blocks += [group.M]

    # Assemble precursor terms
    if self.use_precursors and not self.sub_precursors:
      for precursor in self.precursors:
        blocks += [precursor.M]

    self.M_ = block_diag(blocks, format='csr')
    return self.M_   

  def assemble_forcing(self, time=0):
    blocks = []

    # Assemble multi-group forcing
    for group in self.groups:
      group.assemble_forcing(time)
      blocks.append(group.b)

    # Assemble precursor forcing
    if self.use_precursors and not self.sub_precursors:
      for precursor in self.precursors:
        precursor.assemble_forcing(time)
        blocks.append(precursor.b)

    self.b = np.block(blocks)

  def apply_bcs(self, matrix=None, vector=None):
    for group in self.groups:
      offset = group.g*group.n_dofs
      group.apply_bcs(matrix, vector, offset)

  def assemble_old_physics_action(self):
    self.compute_fission_rate(old=True)

    blocks = []
    for group in self.groups:
      group.f_old = -group.A @ group.u_old
      group.assemble_cross_group_rhs(old=True)
      if self.use_precursors:
        group.assemble_precursor_rhs(old=True)
      blocks += [group.f_old]
    if self.use_precursors and not self.sub_precursors:
      for precursor in self.precursors:
        precursor.f_old = -precursor.A @ precursor.u_old
        precursor.assemble_production_rhs(old=True)
        blocks += [precursor.f_old]
    self.f_old = np.block(blocks)
    
  def assemble_transport_operator(self):
    L = [[None]*self.n_grps for _ in range(self.n_grps)]
    for group in self.groups:
      g = group.g
      L[g][g] = group.A
    return bmat(L, format='csr')

  def assemble_sources(self):
    blocks = []
    for igrp in self.groups: 
      igrp.f[:] = 0.0
      for jgrp in self.groups:
        igrp.assemble_cross_group_rhs(jgrp)
      blocks.append(igrp.f)
    self.f = np.block(blocks)

  def compute_k_eigenvalue(self, tol=1e-8, maxit=100, verbosity=0):
    # Zero out source and set to steady state
    self.problem.is_transient = False
    self.use_precursors = False
    for source in self.sources:
      source.q = np.zeros(self.n_grps)

    # Initialize initial guesses and operators
    L = self.assemble_transport_operator()
    self.flux_ell[:] = 1
    k_eff_ell = 1

    # Inverse power iteration
    converged = False
    for nit in range(maxit):
      # Form right hand side, apply cfe boundary values.
      # Note that this is simply to apply dirichlet bcs.
      self.assemble_sources()
      # Solve the system
      self.flux[:] = spsolve(L, self.f)
      # Recompute k-eff
      k_eff = self.compute_fission_power()
      # Compute change and reset prev iteration params
      k_error = np.abs(k_eff-k_eff_ell) / np.abs(k_eff)
      k_eff_ell = k_eff
      self.flux_ell[:] = self.flux / k_eff
      # Check convergence
      if k_error < tol:
        converged = True
        break
      if verbosity > 1:
        self.print_k_iter_summary(nit, k_eff, k_error)
    self.print_k_calc_summary(converged, nit, k_eff, k_error)

  @property
  def u(self):
    if self.use_precursors and self.sub_precursors:
      start = min([g.field.dof_start for g in self.groups])
      end = max([g.field.dof_end for g in self.groups])
      return self.problem.u[start:end]
    else:
      return super().u

  @property
  def u_ell(self):
    if self.use_precursors and self.sub_precursors:
      start = min([g.field.dof_start for g in self.groups])
      end = max([g.field.dof_end for g in self.groups])
      return self.problem.u_ell[start:end]
    else:
      return super().u_ell

  @property
  def u_half(self):
    if self.use_precursors and self.sub_precursors:
        start = min([g.field.dof_start for g in self.groups])
        end = max([g.field.dof_end for g in self.groups])
        return self.problem.u_half[start:end]
    return super().u_half

  @property
  def u_old(self):
    if self.use_precursors and self.sub_precursors:
      start = min([g.field.dof_start for g in self.groups])
      end = max([g.field.dof_end for g in self.groups])
      return self.problem.u_old[start:end]
    else:
      return super().u_old




 
  


