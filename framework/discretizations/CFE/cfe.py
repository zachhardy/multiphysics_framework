#!/usr/bin/env python3

import numpy as np
from ..discretization import Discretization
from .cfe_view import CellCFEView1D
from quadrature import GLQuadrature

class CFE(Discretization):
    """ Continuous finite element discretization. """
    def __init__(self, mesh, porder=1, n_qpts=2):
        super().__init__(mesh)
        self.n_nodes = porder*mesh.n_el + 1
        self.nodes_per_cell = porder + 1
        self.porder = 1
        self.qrule = GLQuadrature(n_qpts)
        
        # Lagrange elements
        tmp = lagrange_elements(porder)
        self._shape = tmp[0]
        self._grad_shape = tmp[1]

        fe_views = []
        for cell in mesh.cells:
            fe_views.append(CellCFEView1D(self, cell))
        self.fe_views = fe_views

        self.grid = self.create_grid() 

    def create_grid(self):
        """ Generate the grid of unknowns. """
        grid = []
        for fe_view in self.fe_views:
            grid.extend(fe_view.nodes)
        grid = np.atleast_2d(grid)
        return np.unique(grid, axis=0)
        

# ====== End class definition


def lagrange_elements(porder):
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