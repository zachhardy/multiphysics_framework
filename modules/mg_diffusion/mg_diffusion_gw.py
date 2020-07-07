import numpy as np
import numpy.linalg as npla

from .mg_diffusion_base import MultiGroupDiffusionBase

class MultiGroupDiffusionGroupWise(MultiGroupDiffusionBase):

  def __init__(self, problem, discretization, bcs, ics=None,
               tol=1e-8, maxit=500):
    super().__init__(problem, discretization, bcs, ics)
    self.tol = tol
    self.maxit = maxit

  def solve_system(self, method, time, dt, *args):
    converged = False
    for nit in range(self.maxit):  
      diff = 0
      for group in self.groups:
        if method != 'fwd_euler':
          self.assemble_lagged_sources(group)

        # Solve the group
        if not self.problem.is_transient:
          group.u[:] = group.solve_steady_state()
        else:
          u_tmp = None if args==() else args[0][group.field.dofs]
          group.u[:] = group.solve_time_step(
            method, time, dt, group.u_old, u_tmp
          )

        # Compute the change in solution and reset
        diff += npla.norm(group.u-group.u_ell, ord=2)
        group.u_ell[:] = group.u

      # Check convergence
      if diff < self.tol:
        converged = True
        break

    # Iteration summary
    if converged:
      if self.problem.verbosity > 0:
        print("\n*** Converged in {} iterations. ***".format(nit))
    else:
      if self.problem.verbosity > 0:
        print("\n*** WARNING: DID NOT CONVERGE. ***")

  def assemble_lagged_sources(self, group, old=False):
    f = group.f_old if old else group.f_ell
    f[:] = 0
    for group_ in self.groups:
      u_gprime = group_.u_old if old else group_.u_ell
      group.assemble_fission_source(group_, u_gprime, f)
      group.assemble_scattering_source(group_, u_gprime, f)
  
  def compute_old_physics_action(self):
    for group in self.groups:
      self.assemble_lagged_sources(group, old=True)
      group.f_old += group.compute_old_physics_action()
      
  def compute_k_eigenvalue(self, tol=1e-8, maxit=100, verbosity=0):
    # Zero out source and set to steady state
    self.problem.is_transient = False
    for material in self.materials:
      if hasattr(material, 'q'):
        material.q = np.zeros(self.n_grps)

    # Initialize initial guesses and operators
    for group in self.groups:
      group.assemble_physics()
      group.u_ell[:] = 1
    k_eff_old = 1
    
    # Inverse power iterations
    converged = False
    for nit in range(maxit):
      # Solve group-wise
      for group in self.groups:
        self.assemble_lagged_sources(group)
        group.u[:] = group.solve_steady_state()
      
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