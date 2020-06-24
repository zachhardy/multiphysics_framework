#!/usr/bin/env python3

import sys
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
from .physics_base import PhysicsBase

class PhysicsSystem(PhysicsBase):
  """ Class for physics that use a system of equation. """
  def __init__(self, problem, field, bcs, ics=None):
    super().__init__(problem, field, bcs, ics)

    # System storage
    self.A = None
    self.b = np.zeros(self.n_dofs)
    if ics is not None:
      self.M = None

  def SolveSystem(self, time=None, dt=None, 
                  method=None, u_half=None):
    """ Solve a time step of a transient problem. 
    
    Parameters
    ----------
    time : float, optional*]
      The simulation time before the time step.
      This is a mandatory input for transients.
    dt : float
      The time step size. This is a mandatory
      input for transients.
    method : str
      The time stepping method. This is a mandatory
      input for transients.
    u_half : numpy.ndarray (n_dofs,)
      An optional solution vector required for certain
      time stepping methods.
    """
    # steady state assembly
    if not self.problem.is_transient:
      # bcs applied in routines
      self.AssemblePhysics()
      self.AssembleSource()
      self.u[:] = spsolve(self.A, self.b)

    # transient assembly
    else:
      # recompute physics matrix, if coupled
      self.AssembleMass()
      self.AssemblePhysics()

      # shorthand
      A, M, rhs = self.A, self.M, self.b
      u_old = self.u_old
      f_old = A @ u_old
      
      # assemble system and rhs
      # forward euler
      if method == 'fwd_euler':
        self.AssembleSource(time)
        matrix = M/dt
        rhs += M/dt @ u_old - f_old

      # backward euler
      elif method == 'bwd_euler':
        self.AssembleSource(time+dt)
        matrix = A + M/dt
        rhs += M/dt @ u_old

      # crank nicholson
      elif method == 'cn':
        self.AssembleSource(time+dt/2)
        matrix = M/dt + A/2
        rhs += M/dt @ u_old - f_old/2

      # 2nd order BDF
      elif method == 'bdf2':
        assert u_half is not None, (
          "u_half must be provided for BDF2."
        )
        self.AssembleSource(time+dt)
        matrix = 1.5*M/dt + A
        rhs += 2*M/dt @ u_half - 0.5*M/dt @ u_old

      # apply bcs and solve
      self.ApplyBCs(matrix, rhs)
      self.u[:] = spsolve(matrix, rhs)
      
  def AssemblePhysics(self):
    """ Assemble the physics operator. """
    if self.is_nonlinear or self.A is None:
      Arows, Acols, Avals = [], [], []
      for cell in self.mesh.cells:
        rows, cols, vals = self.AssembleCellPhysics(cell)
        Arows += rows
        Acols += cols
        Avals += vals
      shape = (self.n_dofs, self.n_dofs)
      self.A = csr_matrix((Avals, (Arows, Acols)), shape)
      if not self.problem.is_transient:
        self.ApplyBCs(matrix=self.A)

  def AssembleMass(self):
    """ Assemble the time derivative operator. """
    if self.is_nonlinear or self.M is None:
      Mrows, Mvals, Mcols = [], [], []
      for cell in self.mesh.cells:
        rows, cols, vals = self.AssembleCellMass(cell)
        Mrows += rows
        Mcols += cols
        Mvals += vals
      shape = (self.n_dofs, self.n_dofs)
      self.M = csr_matrix((Mvals, (Mrows, Mcols)), shape)

  def AssembleSource(self, time=0):
    """ Assemble the forcing term vector at time. """
    self.b *= 0
    for cell in self.mesh.cells:
      rows, vals = self.AssembleCellSource(cell, time)
      if rows != []:
        self.b[rows] += vals
    if not self.problem.is_transient:
      self.ApplyBCs(vector=self.b)

  def OldPhysicsAction(self):
    """ Compute the old physics action. """
    if self.A is None:
      self.AssemblePhysics()
    return self.A @ self.u_old

  def AssembleCellPhysics(self, cell):
    raise NotImplementedError(
      "This method must be implemented in derived classes."
    )

  def AssembleCellMass(self, cell):
    raise NotImplementedError(
      "This method must be implemented in derived classes."
    )

  def AssembleCellSource(self, cell, time=0):
    raise NotImplementedError(
      "This method must be implemented in derived classes."
    )

  def ApplyBCs(self, matrix=None, vector=None):
    raise NotImplementedError(
      "This method must be implemented in derived classes."
    )
