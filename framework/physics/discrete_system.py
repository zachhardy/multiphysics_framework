import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve

class DiscreteSystem:

  def __init__(self, n_dofs, bcs):
    # Objects
    self.n_dofs = n_dofs
    self.bcs = bcs

    # System information
    self.A = None
    self.M = None
    self.f_ell = np.zeros(self.n_dofs)
    self.f_old = np.zeros(self.n_dofs)
    self.rhs = np.zeros(self.n_dofs)

  def assemble_physics(self):
    raise NotImplementedError

  def assemble_mass(self):
    raise NotImplementedError

  def assemble_forcing(self, time=0):
    raise NotImplementedError

  def assemble_lagged_sources(self, old=False):
    raise NotImplementedError

  def apply_bcs(self, matrix=None, vector=None):
    raise NotImplementedError

  def solve_steady_state(self):
    self.assemble_physics()
    self.assemble_forcing()
    self.assemble_lagged_sources()
    self.rhs -= self.f_ell
    self.apply_bcs(vector=self.rhs)
    return spsolve(self.A, self.rhs)

  def solve_time_step(self, method, time, dt, u_old, *args):
    # Assemble matrices
    self.assemble_physics()
    self.assemble_mass()

    # Shorthand
    A, M, rhs = self.A, self.M, self.rhs
    f_ell, f_old = self.f_ell, self.f_old

    # Forward Euler
    if method == 'fwd_euler':
      self.assemble_forcing(time)
      matrix = M/dt
      rhs += M/dt @ u_old - f_old
      
    # Backward Euler
    elif method == 'bwd_euler':
      self.assemble_forcing(time+dt)
      self.assemble_lagged_sources()
      matrix = M/dt + A
      rhs += M/dt @ u_old - f_ell
    # Crank Nicholson
    elif method == 'cn':
      self.assemble_forcing(time+dt/2)
      self.assemble_lagged_sources()
      matrix = M/dt + A/2
      rhs += M/dt @ u_old - (f_ell + f_old)/2
    # BDF2
    elif method == 'bdf2':
      assert isinstance(args[0], np.ndarray), (
        "Additional numpy array for u_half must be"
        " specified for BDF2."
      )
      assert len(args[0])==self.n_dofs, (
        "u_half is not the"
      )
      self.assemble_forcing(time+dt)
      self.assemble_lagged_sources()
      matrix = 1.5*M/dt + A
      rhs += 2*M/dt @ args[0] - 0.5*M/dt @ u_old - f_ell

    # Apply boundary conditions and solve
    self.apply_bcs(matrix, rhs)
    return spsolve(matrix, rhs)

  def compute_old_physics_action(self, u_old):
    if self.A is None:
      self.assemble_physics()
    self.f_old += self.A @ u_old
