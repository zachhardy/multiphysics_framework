#!/usr/bin/env python3

import sys
import numpy as np
from time import perf_counter
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve

from .hc import HeatConduction
from Discretizations.CFE.cfe import CFE
from field import Field

class CFE_HeatConduction(HeatConduction):
  """ Continuous finite element heat conduction handler. """
  def __init__(self, problem, bcs, ics=None, porder=1):
    # initialize field
    sd = CFE(problem.mesh, porder, n_qpts=porder+1)
    field = Field('temperature', problem.mesh, sd, 1)
    # useful quantities
    self.porder = porder
    self.nodes_per_cell = sd.nodes_per_cell
    # cell matrix
    npc = self.nodes_per_cell
    self.cell_matrix = np.zeros((npc, npc))
    self.cell_vector = np.zeros(npc)
    # initialize the physics module
    super().__init__(problem, field, bcs, ics)

  def AssembleCellPhysics(self, cell):
    """ Assemble the spatial physics operator. """
    # iterate over cells
    rows, cols, vals = [], [], []
    # discretization view
    fe_view = self.sd.fe_views[cell.id]
    # cell info
    material = self.materials[cell.imat]

    # conductivity
    k = material.k
    if callable(k):
      if not self.is_coupled:
        T = fe_view.SolutionAtQuadrature(self.u)
        k = k(T)
      else:
        raise NotImplementedError(
          "Only T dependent conductivities allowed."
        )
    
    # construct cell matrix
    self.cell_matrix *= 0
    for i in range(self.nodes_per_cell):
      row = fe_view.CellDoFMap(i)
      for j in range(self.nodes_per_cell):
        col = fe_view.CellDoFMap(j)
        self.cell_matrix[i,j] += (
          fe_view.Integrate_GradPhiI_GradPhiJ(i, j, k)
        )
        rows += [row]
        cols += [col]
    vals += list(self.cell_matrix.ravel())
    return rows, cols, vals

  def AssembleCellSource(self, cell, time=0):
    """ Assemble the source vector.

    Parameters
    ----------
    time : float, optional
      The simulation time (default is 0).
    """
    rows, vals = [], []  
    # discretization view
    fe_view = self.sd.fe_views[cell.id]
    # cell info
    material = self.materials[cell.imat]

    # source
    if hasattr(material, 'q'):
      q = material.q
      if callable(q):
        q = q(time)
      
      # construct cell vector
      if q != 0:
        self.cell_vector *= 0
        for i in range(self.nodes_per_cell):
          row = fe_view.CellDoFMap(i)
          self.cell_vector[i] = fe_view.Integrate_PhiI(i, q)
          rows += [row]
        vals += list(self.cell_vector.ravel())
    return rows, vals

  def ApplyBCs(self, matrix=None, vector=None):
    """ Apply BCs to matrix and vector.

    Parameters
    ----------
    matrix : csr_matrix (n_dofs, n_dofs)
    vector : numpy.ndarray (n_dofs,)
    """
    # iterate over bndry cells and faces
    for cell in self.mesh.bndry_cells:
      fe_view = self.sd.fe_views[cell.id]

      for f, face in enumerate(cell.faces):        
        if face.flag == cell.flag:
          bc = self.bcs[face.flag-1]
          row = fe_view.FaceDoFMap(f)

          # neumann bc
          if bc.boundary_kind == 'neumann':
            if vector is not None:
              vector[row] += bc.vals

          # robin bc
          elif bc.boundary_kind == 'robin':
            if matrix is not None:
              matrix[row,row] += bc.vals[0]
            if vector is not None:
              vector[row] += bc.vals[1]
            
          # dirichlet bc
          elif bc.boundary_kind == 'dirichlet':
            if matrix is not None:
              matrix[row,row] = 1.0
              for col in matrix[row].nonzero()[1]:
                if row != col:
                  matrix[row,col] = 0.0
            if vector is not None:
              vector[row] = bc.vals



