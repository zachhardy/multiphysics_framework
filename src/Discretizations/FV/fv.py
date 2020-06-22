#!/usr/bin/env python3

import numpy as np
from ..discretization import Discretization
from .cell_fv_view_1d import CellFVView1D

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
    for cell in mesh.cells:
      self.cell_views.append(CellFVView1D(self, cell))
    # Grid
    self.grid = self.CreateGrid()

