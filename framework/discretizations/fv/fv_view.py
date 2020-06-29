#!/usr/bin/env python3

import numpy as np

class CellFVView1D:
  """ Finite volume cell view. 
  
  Parameters
  ----------
  cell : CellBase
  """
  def __init__(self, discretization, cell):
    # General information
    self.geom = discretization.geom
    self.n_nodes = discretization.n_nodes
    self.nodes_per_cell = 1

    # Node information
    self.node_ids = [cell.id]
    self.nodes = np.atleast_2d(
      np.average(cell.vertices, axis=0)
    )

  def cell_dof_map(self, component=0):
    """ Map the local cell dof to a global dof. """
    return self.node_ids[0] + component*self.n_nodes

  def average_solution(self, u):
    """ Return the solution on this cell. """
    return float(u[self.node_ids])