#!/usr/bin/env python3

import numpy as np
from ..discretization import Discretization
from .cell_cfe_view_1d import CellCFEView1D
from quadrature import Quadrature

class CFE(Discretization):
  """ Continuous finite element discretization.

  Parameters
  ----------
  mesh : MeshBase object.
  porder : int, optional
    The element order (default is 1).
  n_qpts : int, optional
    The number of quadrature points (default is 2).
  """
  def __init__(self, mesh, porder=1, n_qpts=2):
    super().__init__(mesh)
    # General information
    self.n_nodes = porder*mesh.n_el + 1
    self.nodes_per_cell = porder + 1
    self.porder = 1
    self.qrule = Quadrature(n_qpts)
    # Lagrange elements
    self._phi, self._grad_phi = LagrangeElements(porder)
    # Continuous finite element cell views
    for cell in mesh.cells:
      self.cell_views.append(CellCFEView1D(self, cell))
    # Grids
    self.grid = self.CreateGrid() 
    

# ====== End class definition


def LagrangeElements(porder):
  """ Generate Lagrange interpolants of order porder.

  Parameters
  ----------
  porder: int
    The polynomial order.

  Returns
  -------
  Polynomial object lists containing the interpolants and
  their derivatives.
  """
  from scipy.interpolate import lagrange

  bf, dbf = [], []
  xref = np.linspace(-1, 1, porder+1)
  for i in range(porder+1):
    yref = np.zeros(porder+1)
    yref[i] = 1
    bf.append(lagrange(xref, yref))
    dbf.append(bf[i].deriv())
  return bf, dbf