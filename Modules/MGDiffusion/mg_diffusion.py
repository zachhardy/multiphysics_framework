#!/usr/bin/env python3

import sys
import numpy as np
from time import perf_counter
from scipy.sparse.linalg import spsolve

sys.path.append("../../src")
from physics_base import PhysicsBase
from material import NeutronicsMaterial as NeutMat

class MultiGroupDiffusion(PhysicsBase):
  """ Multigroup diffusion physics module 
  
  This is a base module for multigroup diffusion problems.
  Derived modules should be those of different discretization
  techniques, such as CFE, FV, etc.

  Parameters
  ----------
  problem : Problem object
    The problem this module is being added to.
  field : Field object
    The field attached to this module.
  bcs : List of BC
    The boundary conditions
  ics : The initial conditions, optional
    The initial conditions, if a transient.
  """
  def __init__(self, problem, field, bcs, ics=None):
    super().__init__(problem, field, bcs, ics)

    # get ordered neutronics materials
    self.materials = self.InitializeMaterials()

    # coupling information
    self.is_coupled = False

    # system storage
    self.A = None
    self.b = np.zeros(self.n_dofs)
    if ics is not None:
      # pre-compute mass matrix
      self.M = self.AssembleMass()

  def SolveSteadyState(self):
    """ Solve a steady state problem. """
    self.AssemblePhysics()
    self.AssembleSource()
    self.ApplyBCs(self.A, self.b)
    self.u[:] = spsolve(self.A, self.b)

  def SolveTimeStep(self, time, dt):
    """ Solve a time step of a transient problem. """
    # Recompute physics matrix, if coupled
    if self.is_coupled or self.A is None:
      self.AssemblePhysics()

    # Shorthand
    method = self.problem.method
    A, M, rhs = self.A, self.M, self.b
    u_old = self.u_old
    f_old = A @ u_old
    
    # Assemble time step
    if method == 'fwd_euler':
      self.AssembleSource(time)
      matrix = M/dt
      rhs += M/dt @ u_old - f_old

    elif method == 'bwd_euler':
      self.AssembleSource(time+dt)
      matrix = A + M/dt
      rhs += M/dt @ u_old

    elif method == 'cn':
      self.AssembleSource(time+dt/2)
      matrix = M/dt + A/2
      rhs += M/dt @ u_old - f_old/2

    elif method == 'tbdf2':
      # half step crank-nicholson
      self.AssembleSource(time+dt/4)
      matrix = A/2 + 2*M/dt
      rhs += 2*M/dt @ u_old - f_old/2
      # apply boundary values
      self.ApplyBCs(matrix, rhs)
      u_tmp = spsolve(matrix, rhs)

      # half step bdf2
      if self.is_coupled:
        self.AssemblePhysics()
      self.AssembleSource(time+dt)
      matrix = A + 3*M/dt 
      rhs += 4*M/dt @ u_tmp - M/dt @ u_old
  
    # apply boundary values 
    self.ApplyBCs(matrix, rhs)
    self.u[:] = spsolve(matrix, rhs)

  def AssemblePhysics(self):
    raise NotImplementedError(
      "This must be implemented in derived classes."
    )

  def AssembleMass(self):
    raise NotImplementedError(
      "This must be implemented in derived classes."
    )

  def AssembleSource(self, time=0.):
    raise NotImplementedError(
      "This must be implemented in derived classes."
    )

  def InitializeMaterials(self):
    """ Get neutronics properties and sort by zone. """
    materials = []
    for material in self.problem.materials:
      if isinstance(material, NeutMat):
          assert material.G == self.G, (
            "Incompatible group structures."
          )
          materials += [material]
    materials.sort(key=lambda x: x.material_id)
    return materials

  def ValidateBCs(self, bcs):
    try:
      G = self.field.components
      for bc in bcs:
        if bc.vals is not None:
          if len(bc.vals.shape) == 1:
            if len(bc.vals) != G:
              raise ValueError(
                "BC incompatible with G."
              )
          else:
            for i in range(len(bc.vals)):
              if len(bc.vals[i]) != G:
                raise ValueError(
                  "BC incompatible with G."
                )
        return bcs

    except ValueError as err:
      print(err.args[0])
      sys.exit(-1)

  def ValidateICs(self, ics):
    try:
      G = self.field.components
      if ics is not None:
        if not isinstance(ics, list):
          raise ValueError("ics must be a list.")
        if len(ics) != G:
          raise ValueError("ics do not agree with G.")
        for ic in ics:
          if not callable(ic):
            raise ValueError("Uncallable ic encountered.")
      return ics
      
    except ValueError as err:
      print(err.args[0])
      sys.exit(-1)