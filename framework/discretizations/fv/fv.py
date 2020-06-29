import numpy as np
from ..discretization import Discretization
from .fv_view import CellFVView1D

class FV(Discretization):

  dtype = 'fv'
  
  def __init__(self, mesh):
    super().__init__(mesh)
    self.n_nodes = self.mesh.n_el
    self.nodes_per_cell = 1

    # Generate cell views
    for cell in mesh.cells:
      self.cell_views.append(CellFVView1D(self, cell))

    # Generate the grid of unique nodes
    self.grid = self.create_grid()

  def create_grid(self):
    grid = []
    for view in self.cell_views:
      grid.extend(view.nodes)
    grid = np.atleast_2d(grid)
    return np.unique(grid, axis=0)

