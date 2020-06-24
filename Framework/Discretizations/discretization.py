#!/usr/bin/env python3

import numpy as np

class Discretization:
  """ Base class to describe a spatial discretization. """
  def __init__(self, mesh):
    # store mesh reference
    self.mesh = mesh
    
    # general info
    self.dim = mesh.dim
    self.geom = mesh.geom

    # derived class attributes
    self.n_nodes = 0
    self.grid = None

  def CreateGrid(self):
    raise NotImplementedError(
      "This method must be implemented in derived classes."
    )
    
