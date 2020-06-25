#!/usr/bin/env python3

import numpy as np
from ..discretization import Discretization
from .fv_view import CellFVView1D

class FV(Discretization):
  """ Class doe Finite Volume discretizations. 
  
  Parameters
  ----------
  mesh : MeshBase object.
  """
  def __init__(self, mesh):
    super().__init__(mesh)
    # General information
    self.n_nodes = self.mesh.n_el
    self.nodes_per_cell = 1
    # Finite volume cell views
    fv_views = []
    for cell in mesh.cells:
      fv_views.append(CellFVView1D(self, cell))
    self.fv_views = fv_views
    # Grid
    self.grid = self.CreateGrid()

  def CreateGrid(self):
    """ Generate the grid of unknowns. """
    grid = []
    for fv_view in self.fv_views:
      grid.extend(fv_view.nodes)
    grid = np.atleast_2d(grid)
    return np.unique(grid, axis=0)

