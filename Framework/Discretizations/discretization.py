#!/usr/bin/env python3

import numpy as np

class Discretization:
  """ Template class for discretizations.
  
  Parameters
  ----------
  mesh : MeshBase-like
  """
  def __init__(self, mesh):
    self.mesh = mesh
    self.dim = mesh.dim
    self.geom = mesh.geom
    self.n_nodes = 0
    self.grid = None

  def CreateGrid(self):
    raise NotImplementedError(
      "This method must be implemented in derived classes."
    )
    
