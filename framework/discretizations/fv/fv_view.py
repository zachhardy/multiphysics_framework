import numpy as np

class CellFVView1D:

  def __init__(self, discretization, cell):
    self.geom = discretization.geom
    self.n_nodes = discretization.n_nodes
    self.nodes_per_cell = 1

    # Node information
    self.node_ids = [cell.id]
    self.nodes = np.atleast_2d(
      np.average(cell.vertices, axis=0)
    )

  @property
  def dofs(self):
    return self.node_ids

  def cell_dof_map(self, component=0):
    return self.dofs[0] + component*self.n_nodes

  def average_solution(self, u):
    return float(u[self.node_ids])