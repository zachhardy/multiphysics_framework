#!/usr/bin/env python3

import sys
import numpy as np
from scipy.sparse.linalg import spsolve

from Physics.physics_system import PhysicsSystem
from material import HeatConductionMaterial as HeatMat

class HeatConduction(PhysicsSystem):
  """ Heat conduction physics module
  
  This is a base module for heat conduction problems.
  Derived modules should be those of different 
  discretization techniques such as CFE, FV, etc.
  
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

  def AssembleCellPhysics(self, cell):
    raise NotImplementedError(
      "This must be implemented in derived classes."
    )

  def AssembleCellMass(self, cell):
    raise NotImplementedError(
      "This must be implemented in derived classes."
    )

  def AssembleCellSource(self, cell, time=0.):
    raise NotImplementedError(
      "This must be implemented in derived classes."
    )

  def InitializeMaterials(self):
    """ Get neutronics properties and sort by zone. """
    materials = []
    for material in self.problem.materials:
      if isinstance(material, HeatMat):
          materials += [material]
    materials.sort(key=lambda x: x.material_id)
    return materials

  def ValidateBCs(self, bcs):
    return bcs

  def ValidateICs(self, ics):
    return ics