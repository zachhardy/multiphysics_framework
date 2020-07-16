import sys
import numpy as np
import numpy.linalg as npla
from scipy.sparse.linalg import spsolve

from .mg_diffusion_base import MultiGroupDiffusionBase

class MultiGroupDiffusionGroupWise(MultiGroupDiffusionBase):

  def __init__(self, problem, discretization, bcs, ics=None, **kwargs):
    super().__init__(problem, discretization, bcs, ics, **kwargs)
    self.tol = kwargs['tol'] if 'tol' in kwargs else 1e-8
    self.maxit = kwargs['maxit'] if 'maxit' in kwargs else 500

  def solve_system(self, opt=0):
    converged = False
    for nit in range(self.maxit):        
      diff = 0
      if not self.problem.is_transient:
        self.solve_steady_state()
      else:
        self.solve_time_step(opt)

      for group in self.groups:
        diff += npla.norm(group.u-group.u_ell, ord=2)
        group.u_ell[:] = group.u
      for precursor in self.precursors:
        diff += npla.norm(precursor.u-precursor.u_ell, ord=2)
        precursor.u_ell[:] = precursor.u

      # Check convergence
      if diff < self.tol:
        converged = True
        break

    # Iteration summary
    if converged:
      if self.problem.verbosity > 0:
        print("\n*** Diffusion Solver Converged ***")
        print("Iterations: {}".format(nit))
        print("Final Error: {:.3e}".format(diff))
    else:
      if self.problem.verbosity > 0:
        print("\n*** WARNING: DID NOT CONVERGE. ***")

  def solve_steady_state(self):
    for group in self.groups:
      self.assemble_group_system(group)
      group.assemble_forcing()
      group.b += group.f
      group.u[:] = spsolve(group.A, group.b)
    
  def solve_time_step(self, opt=0):
    for group in self.groups:
      self.assemble_group_system(group, opt)
      group.u[:] = group.solve_time_step(opt)
    if self.use_precursors:
      self.compute_fission_rate()
      for precursor in self.precursors:
        if not self.sub_precursors:
          self.assemble_precursor_system(precursor, opt)
          precursor.u[:] = precursor.solve_time_step(opt)
        else:
          precursor.update_precursor(opt)
  
  def assemble_group_system(self, group, opt=0):
    group.f[:] = 0
    group.assemble_cross_group_rhs(old=False)
    if self.use_precursors:
      if not self.sub_precursors:
        group.assemble_precursor_rhs(old=False)
      elif self.sub_precursors:
        group.assemble_gw_substitution_rhs(opt)

  def assemble_precursor_system(self, precursor, opt=0):
    precursor.f[:] = 0
    precursor.assemble_production_rhs(old=False)

  def assemble_old_physics_action(self):
    self.compute_fission_rate(old=True)
    for group in self.groups:
      group.f_old = -group.A @ group.u_old
      group.assemble_cross_group_rhs(old=True)
      if self.use_precursors:
        group.assemble_precursor_rhs(old=True)
    if self.use_precursors: 
      if not self.sub_precursors:
        for precursor in self.precursors:
          precursor.f_old[:] = -precursor.A @ precursor.u_old
          precursor.assemble_production_rhs(old=True)
    
  def compute_k_eigenvalue(self, tol=1e-8, maxit=100, verbosity=0):
    # Zero out source and set to steady state
    self.problem.is_transient = False
    for source in self.sources:
      source.q = np.zeros(self.n_grps)

    # Initialize initial guesses and operators
    for group in self.groups:
      group.u_ell[:] = 1
    k_eff_old = 1
    
    # Inverse power iterations
    converged = False
    for nit in range(maxit):
      self.solve_steady_state()  
      k_eff = self.compute_fission_power()
  
      # Reinit and normalize group fluxes
      for group in self.groups:
        group.u_ell[:] = group.u / k_eff

      # Compute the change in k-eff and reinit
      k_error = np.abs(k_eff-k_eff_old) / np.abs(k_eff)
      k_eff_old = k_eff
      # Check convergence
      if k_error < tol:
        converged = True
        break
      
      # Iteration printouts
      if verbosity > 1:
        self.print_k_iter_summary(nit, k_eff, k_error)
    self.print_k_calc_summary(converged, nit, k_eff, k_error)