#!/usr/bin/env python3

import sys
import numpy as np
from time import perf_counter
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve

sys.path.append('../../src')
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
    # initialize the physics module
    super().__init__(problem, field, bcs, ics)

  def AssemblePhysics(self):
    """ Assemble the spatial physics operator. """
    # iterate over cells
    Arows, Acols, Avals = [], [], []
    for cell in self.mesh.cells:
      # cell informations
      view = self.sd.cell_views[cell.id]
      material = self.materials[cell.imat]

      # material properties
      k = material.k
      if callable(k):
        if not self.is_coupled:
          T = view.SolutionAtQuadrature(self.u)
          k = k(T)
        else:
          raise NotImplementedError(
            "Only T dependent conductivities allowed."
          )
      
      # iterate over test/trial pairs
      self.cell_matrix *= 0
      for i in range(self.nodes_per_cell):
        row = view.CellDoFMap(i)
        for j in range(self.nodes_per_cell):
          col = view.CellDoFMap(j)

          # diffusion term
          self.cell_matrix[i,j] += (
            view.Integrate_GradPhiI_GradPhiJ(i, j, k)
          )

          Arows += [row]
          Acols += [col]
      Avals += list(self.cell_matrix.ravel())
    shape = (self.n_dofs, self.n_dofs) # shorthand
    self.A = csr_matrix((Avals, (Arows, Acols)), shape)

  def AssembleSource(self, time=0):
    """ Assemble the source vector.

    Parameters
    ----------
    time : float, optional
      The simulation time (default is 0).
    """
    self.b *= 0
    # iterate over cells
    for cell in self.mesh.cells:
      # cell information
      view = self.sd.cell_views[cell.id]
      material = self.materials[cell.imat]

      if hasattr(material, 'q'):
        # material source
        q = material.q
        # evaluate, if callable
        if callable(q):
          q = q(time)

        # iterate through test functions
        if q != 0:
          for i in range(self.nodes_per_cell):
            row = view.CellDoFMap(i)
            self.b[row] += view.Integrate_PhiI(i, q)

  def ApplyBCs(self, matrix, vector):
    """ Apply BCs to matrix and vector.

    Parameters
    ----------
    matrix : csr_matrix (n_dofs, n_dofs)
    vector : numpy.ndarray (n_dofs,)
    """
    # iterate over bndry cells and faces
    for cell in self.mesh.bndry_cells:
      view = self.sd.cell_views[cell.id]

      for f, face in enumerate(cell.faces):        
        if face.flag == cell.flag:
          bc = self.bcs[face.flag-1]
          row = view.FaceDoFMap(f)

          # neumann bc
          if bc.boundary_kind == 'neumann':
            vector[row] += bc.vals

          # robin bc
          elif bc.boundary_kind == 'robin':
            matrix[row,row] += bc.vals[0]
            vector[row] += bc.vals[1]
            
          # dirichlet bc
          elif bc.boundary_kind == 'dirichlet':
            matrix[row,row] = 1.0
            for col in matrix[row].nonzero()[1]:
              if row != col:
                matrix[row,col] = 0.0
            vector[row] = bc.vals



