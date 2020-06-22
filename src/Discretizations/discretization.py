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
    self.cell_view = None
    self.neighbor_view = None
    self.grid = None

  def CreateGrid(self):
    """ Create the grid coordinates of the unknowns. """
    grid = []
    for cell in self.mesh.cells:
      self.cell_view.reinit(cell)
      grid.extend(self.cell_view.nodes)
    grid = np.atleast_2d(grid)
    return np.unique(grid, axis=0)
