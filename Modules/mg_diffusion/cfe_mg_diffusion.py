#!/usr/bin/env python3

import sys
import numpy as np
from time import perf_counter
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve

sys.path.append('../../src')
from .mg_diffusion import MultiGroupDiffusion
from discretizations.cfe.cfe import CFE
from field import Field

class CFE_MultiGroupDiffusion(MultiGroupDiffusion):
  """ Continuous finite element multigroup diffusion module. """
  def __init__(self, problem, G, bcs, ics=None, porder=1):
    # initialize field
    sd = CFE(problem.mesh, porder, n_qpts=porder+1)
    field = Field('flux', problem.mesh, sd, G)
    # group structure
    self.G = G
    # useful quantities
    self.porder = porder
    self.nodes_per_cell = sd.nodes_per_cell
    # cell matrix
    npc = self.nodes_per_cell
    self.cell_matrix = np.zeros((npc, npc))
    self.cell_vector = np.zeros(npc)
    # initialize physics
    super().__init__(problem, field, bcs, ics)

  def AssembleCellPhysics(self, cell):
    """ Assemble the spatial/energy physics operator. """
    rows, cols, vals = [], [], []
    # discretization view and material
    fe_view = self.sd.fe_views[cell.id]
    material = self.materials[cell.imat]

    # assemble group-wise
    for ig in range(self.G):
      # removal and diffusion
      sig_r = material.sig_r[ig]
      D = material.D[ig] 

      # assemble cell matrix
      self.cell_matrix *= 0
      for i in range(self.nodes_per_cell):
        row = fe_view.CellDoFMap(i, ig)
        for j in range(self.nodes_per_cell):
          col = fe_view.CellDoFMap(j, ig)
          self.cell_matrix[i,j] += (
            fe_view.Integrate_PhiI_PhiJ(i, j, sig_r)
            + fe_view.Integrate_GradPhiI_GradPhiJ(i, j, D)
          )
          rows += [row]
          cols += [col]
      vals += list(self.cell_matrix.ravel())

      # assemble energy coupled terms
      for jg in range(self.G):
        # scattering
        if hasattr(material, 'sig_s'):
          sig_s = material.sig_s[ig][jg]

          # assemble cell matrix
          if sig_s != 0:
            self.cell_matrix *= 0
            for i in range(self.nodes_per_cell):
              row = fe_view.CellDoFMap(i, ig)
              for j in range(self.nodes_per_cell):
                col = fe_view.CellDoFMap(j, jg)
                self.cell_matrix[i,j] -= \
                  fe_view.Integrate_PhiI_PhiJ(i, j, sig_s)
                rows += [row]
                cols += [col]
            vals += list(self.cell_matrix.ravel())

        # fission
        if hasattr(material, 'nu_sig_f'):
          chi = material.chi[ig]
          nu_sig_f = material.nu_sig_f[jg]

          # assemble cell matrix
          if chi*nu_sig_f != 0:
            self.cell_matrix *= 0
            for i in range(self.nodes_per_cell):
              row = fe_view.CellDoFMap(i, ig)
              for j in range(self.nodes_per_cell):
                col = fe_view.CellDoFMap(j, jg)
                self.cell_matrix[i,j] -= \
                  fe_view.Integrate_PhiI_PhiJ(i, j, chi*nu_sig_f)
                rows += [row]
                cols += [col]
            vals += list(self.cell_matrix.ravel())
    return rows, cols, vals
                  
  def AssembleCellMass(self, cell):
    """ Assemble the time derivative term. """
    rows, cols, vals = [], [], []
    # discretization view
    fe_view = self.sd.fe_views[cell.id]
    # cell info
    material = self.materials[cell.imat]

    # assemble group-wise
    for ig in range(self.G):
      # inverse velocity
      v = material.v[ig]

      # assemble cell matrix
      self.cell_matrix *= 0
      for i in range(self.nodes_per_cell):
        row = fe_view.CellDoFMap(i, ig)
        for j in range(self.nodes_per_cell):
          col = fe_view.CellDoFMap(j, ig)
          self.cell_matrix[i,j] += \
            fe_view.Integrate_PhiI_PhiJ(i, j, 1/v)
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

    # assemble group-wise
    for ig in range(self.G):
      # source
      if hasattr(material, 'q'):
        q = material.q[ig]
        if callable(q):
          q = q(time)
        
        # assemble cell vector
        if q != 0:
          self.cell_vector *= 0
          for i in range(self.nodes_per_cell):
            row = fe_view.CellDoFMap(i, ig)
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
        if face.flag > 0:
          bc = self.bcs[face.flag-1]
  
          # iterate over energy groups
          for ig in range(self.G):
            row = fe_view.FaceDoFMap(f, ig)

            # neumann bc
            if bc.boundary_kind == 'neumann':
              if vector is not None:
                vector[row] += bc.vals[ig]

            # robin bc
            elif bc.boundary_kind == 'robin':
              if matrix is not None:
                matrix[row,row] += 0.5
              if vector is not None:
                vector[row] += 2.0*bc.vals[ig]

            # dirichlet bc
            elif bc.boundary_kind == 'dirichlet':
              if matrix is not None:
                matrix[row,row] = 1.0
                for col in matrix[row].nonzero()[1]:
                  if row != col:
                    matrix[row,col] = 0.0
              if vector is not None:
                vector[row] = bc.vals[ig]
