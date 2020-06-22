#!/usr/bin/env python3

import sys
import numpy as np
from time import perf_counter
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve

sys.path.append('../../src')
from .mg_diffusion import MultiGroupDiffusion
from Discretizations.CFE.cfe import CFE
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
    # initialize physics
    super().__init__(problem, field, bcs, ics)

  def AssemblePhysics(self):
    """ Assemble the spatial/energy physics operator. """
    # iterate over cells 
    Arows, Acols, Avals = [], [], []
    for cell in self.mesh.cells:
      # cell information
      view = self.sd.cell_views[cell.id]
      material = self.materials[cell.imat]

      # iterate over energy groups
      for ig in range(self.G):

        # material properties
        sig_r = material.sig_r[ig]
        D = material.D[ig] 

        # iterate over test/trial pairs
        self.cell_matrix *= 0
        for i in range(self.nodes_per_cell):
          row = view.CellDoFMap(i, ig)
          for j in range(self.nodes_per_cell):
            col = view.CellDoFMap(j, ig)
            
            # diffusion + removal
            self.cell_matrix[i,j] += (
              view.Integrate_PhiI_PhiJ(i, j, sig_r)
              + view.Integrate_GradPhiI_GradPhiJ(i, j, D)
            )

            Arows += [row]
            Acols += [col]
        Avals += list(self.cell_matrix.ravel())

        # iterate over energy groups for coupling
        for jg in range(self.G):

          # scattering term
          if hasattr(material, 'sig_s'):

            # material property
            sig_s = material.sig_s[ig][jg]
            
            # construct cell matrix if non-zero property
            if sig_s != 0:
              # iterate over test/trial pairs
              self.cell_matrix *= 0
              for i in range(self.nodes_per_cell):
                row = view.CellDoFMap(i, ig)
                for j in range(self.nodes_per_cell):
                  col = view.CellDoFMap(j, jg)

                  self.cell_matrix[i,j] -= \
                    view.Integrate_PhiI_PhiJ(i, j, sig_s)
                  
                  Arows += [row]
                  Acols += [col]
              Avals += list(self.cell_matrix.ravel())

          # fission term
          if hasattr(material, 'nu_sig_f'):

            # material properties
            chi = material.chi[ig]
            nu_sig_f = material.nu_sig_f[jg]

            # construct cell matrix if non-zero property
            if chi*nu_sig_f != 0:
              # iterate over test/trial pairs
              self.cell_matrix *= 0
              for i in range(self.nodes_per_cell):
                row = view.CellDoFMap(i, ig)
                for j in range(self.nodes_per_cell):
                  col = view.CellDoFMap(j, jg)

                  self.cell_matrix[i,j] -= \
                    view.Integrate_PhiI_PhiJ(i, j, chi*nu_sig_f)
              
                  Arows += [row]
                  Acols += [col]
              Avals += list(self.cell_matrix.ravel())
    shape = (self.n_dofs, self.n_dofs) # shorthand
    self.A = csr_matrix((Avals, (Arows, Acols)), shape)
                  
  def AssembleMass(self):
    """ Assemble the time derivative term. """
    # iterate over cells
    Mrows, Mcols, Mvals = [], [], []
    for cell in self.mesh.cells:
      # cell information
      view = self.sd.cell_views[cell.id]
      material = self.materials[cell.imat]

      # iterate through groups
      for ig in range(self.G):
        # group velocity
        v = material.v[ig]

        # iterate through test/trial pairs
        self.cell_matrix *= 0
        for i in range(self.nodes_per_cell):
          row = view.CellDoFMap(i, ig)
          for j in range(self.nodes_per_cell):
            col = view.CellDoFMap(j, ig)
            # add to matrix
            self.cell_matrix[i,j] += \
              view.Integrate_PhiI_PhiJ(i, j, 1/v)
            
            Mrows += [row]
            Mcols += [col]
        Mvals += list(self.cell_matrix.ravel())
    shape = (self.n_dofs, self.n_dofs) # shorthand
    return csr_matrix((Mvals, (Mrows, Mcols)), shape)    

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

      # iterate over groups
      for ig in range(self.G):
        # if there is a source
        if hasattr(material, 'q'):
          # group source
          q = material.q[ig]
          # evaluate time dependency
          if callable(q):
            q = q(time)
          
          # iterate through test functions
          if q != 0:
            for i in range(self.nodes_per_cell):
              row = view.CellDoFMap(i, ig)
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
        if face.flag > 0:
          bc = self.bcs[face.flag-1]
  
          # iterate over energy groups
          for ig in range(self.G):
            row = view.FaceDoFMap(f, ig)

            # neumann bc
            if bc.boundary_kind == 'neumann':
              vector[row] += bc.vals[ig]

            # robin bc
            elif bc.boundary_kind == 'robin':
              matrix[row,row] += bc.vals[0][ig]
              vector[row] += bc.vals[1][ig]
              
            # dirichlet bc
            elif bc.boundary_kind == 'dirichlet':
              matrix[row,row] = 1.0
              for col in matrix[row].nonzero()[1]:
                if row != col:
                  matrix[row,col] = 0.0
              vector[row] = bc.vals[ig]
