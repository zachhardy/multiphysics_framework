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
    self.cell_views = []
    self.grid = None

  def CreateGrid(self):
    """ Create the grid coordinates of the unknowns. """
    grid = []
    for view in self.cell_views:
      grid.extend(view.nodes)
    grid = np.atleast_2d(grid)
    return np.unique(grid, axis=0)
