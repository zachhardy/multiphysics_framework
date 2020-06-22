#!/usr/bin/env python3

import numpy as np

class CellFVView1D:
  """ Finite volume cell view. 
  
  Parameters
  ----------
  cell : CellBase
  """
  def __init__(self, discretization, cell):
    # general information
    self.geom = discretization.geom
    self.n_nodes = discretization.n_nodes
    self.nodes_per_cell = 1

    # node information
    self.node_ids = [cell.id]
    self.nodes = np.atleast_2d(
      np.average(cell.vertices, axis=0)
    )

  def CellDoFMap(self, component=0):
    """ Map the local cell dof to a global dof. """
    return self.node_ids[0] + component*self.n_nodes