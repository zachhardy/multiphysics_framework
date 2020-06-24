#!/usr/bin/env python3

import sys
import numpy as np
from time import perf_counter
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve

from Physics.physics_system import PhysicsSystem
from material import NeutronicsMaterial as NeutMat

class MultiGroupDiffusion(PhysicsSystem):
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
    self.is_nonlinear = False
    
    
  def AssembleCellPhysics(self, cell):
    raise NotImplementedError(
      "This must be implemented in derived classes."
    )

  def AssembleCellMass(self, cell):
    raise NotImplementedError(
      "This must be implemented in derived classes."
    )

  def AssembleCellSource(self, cell, time=0):
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