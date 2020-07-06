import numpy as np

from ..discretization import Discretization
from .cfe_view import CellCFEView1D
from quadrature import GLQuadrature

class CFE(Discretization):

    dtype = 'cfe'

    def __init__(self, mesh, porder=1, n_qpts=2):
        super().__init__(mesh)
        self.n_nodes = porder*mesh.n_el + 1
        self.nodes_per_cell = porder + 1
        self.porder = 1
        self.qrule = GLQuadrature(n_qpts)

        # Generate the finite elements
        tmp = lagrange_elements(porder)
        self._shape = tmp[0]
        self._grad_shape = tmp[1]

        # Generate the cell views
        for cell in mesh.cells:
            self.cell_views.append(CellCFEView1D(self, cell))

        # Generate the grid of unique unknowns
        self.grid = self.create_grid() 

    def create_grid(self):
        grid = []
        for view in self.cell_views:
            grid.extend(view.nodes)
        grid = np.atleast_2d(grid)
        return np.unique(grid, axis=0)
        

# ====== End class definition


def lagrange_elements(porder):
    from scipy.interpolate import lagrange

    bf, dbf = [], []
    xref = np.linspace(-1, 1, porder+1)
    for i in range(porder+1):
        yref = np.zeros(porder+1)
        yref[i] = 1
        bf.append(lagrange(xref, yref))
        dbf.append(bf[i].deriv())
    return bf, dbf