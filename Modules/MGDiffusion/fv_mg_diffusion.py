#!/usr/bin/env python3

import sys
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve

sys.path.append('../../src')
from mg_diffusion import MultiGroupDiffusion
from Discretizations.FV.fv import FV
from field import Field

class FV_MultiGroupDiffusion(MultiGroupDiffusion):
  """ Finite volume multigroup diffusion handler. """
  def __init__(self, problem, G, bcs, ics=None):
    # initialize field
    sd = FV(problem.mesh)
    field = Field('flux', problem.mesh, sd, G)
    # group structure
    self.G = G
    # initialize physics
    super().__init__(problem, field, bcs, ics)
      
  def AssemblePhysics(self):
    """ Assemble the spatial/energy physics operator. """
    view = self.sd.cell_view
    nbr_view = self.sd.neighbor_view
    shape = (self.n_dofs, self.n_dofs)
    # iterate over cells
    Arows, Acols, Avals = [], [], []
    for cell in self.mesh.cells:
      # cell information
      view.reinit(cell)
      width = cell.width[0]
      volume = cell.volume
      material = self.materials[cell.imat]

      # iterate over energy groups
      for ig in range(self.G):
        row = view.CellDoFMap(ig)
        # removal term
        sig_r = material.sig_r[ig]
        # add to matrix
        Arows += [row]
        Acols += [row]
        Avals += [sig_r * volume]

        # iterate over energy groups for coupling
        for jg in range(self.G):
          col = view.CellDoFMap(jg)
          # scattering term
          if hasattr(material, 'sig_s'):
            sig_s = material.sig_s[ig][jg]
            # add to matrix
            if sig_s != 0:
              Arows += [row]
              Acols += [col]
              Avals += [-sig_s * volume]

          # fission term
          if hasattr(material, 'nu_sig_f'):
            chi = material.chi[ig]
            nu_sig_f = material.nu_sig_f[jg]
            # add to matrix
            if chi*nu_sig_f != 0:
              Arows += [row]
              Acols += [col]
              Avals += [-chi * nu_sig_f * volume]

      # iterate over faces for face terms
      for face in cell.faces:
        # interior diffusion
        if face.flag == 0:
          # neighbor cell information
          nbr_view.reinit(face.neighbor_cell)
          nbr_width = nbr_view.cell.width[0]
          nbr_material = self.materials[nbr_view.cell.imat]
          
          # iterate through groups
          for ig in range(self.G):
            row = view.CellDoFMap(ig)
            col = nbr_view.CellDoFMap(ig)
            # diffusion coefs
            D = material.D[ig]
            nbr_D = nbr_material.D[ig]
            # harmonic averaged quantities
            width_avg = 0.5*(width + nbr_width)
            D_avg = 2*width_avg/(width/D + nbr_width/nbr_D)
            val = face.area * D_avg/width_avg
            # add to matrix
            Arows += [row, row]
            Acols += [row, col]
            Avals += [val, -val]
    print(len(Avals), len(Arows), len(Acols))
    self.A = csr_matrix((Avals, (Arows, Acols)), shape)

  def AssembleMass(self):
    """ Assemble the time derivative term. """
    view = self.sd.cell_view
    shape = (self.n_dofs, self.n_dofs)
    # iterate over cells
    Mrows, Mcols, Mvals = [], [], []
    for cell in self.mesh.cells:
      # cell information
      view.reinit(cell)
      volume = cell.volume
      material = self.materials[cell.imat]

      # iterate through groups
      for ig in range(self.G):
        row = view.CellDoFMap(ig)
        # group velocity
        velocity = material.v[ig]
        # add to matrix
        Mrows += [row]
        Mcols += [row]
        Mvals += [volume / velocity]
    return csr_matrix((Mvals, (Mrows, Mcols)), shape)

  def AssembleSource(self, time=0):
    """ Assemble the source vector.

    Parameters
    ----------
    time : float, optional
      The simulation time (default is 0).
    """
    self.b *= 0 # clear vector
    view = self.sd.cell_view
    # iterate over cells
    for cell in self.mesh.cells:
      # cell information
      view.reinit(cell)
      volume = cell.volume
      material = self.materials[cell.imat]

      # iterate over groups
      for ig in range(self.G):
        row = view.CellDoFMap(ig)

        # source term
        if hasattr(material, 'q'):
          q = material.q[ig]
          if callable(q):
            q = q(time)
          if q != 0:
            self.b[row] += q * volume

  def ApplyBCs(self, matrix, vector):
    """ Apply BCs to matrix and vector.

    Parameters
    ----------
    matrix : csr_matrix (n_dofs, n_dofs)
    vector : numpy.ndarray (n_dofs,)
    """
    view = self.sd.cell_view
    # iterate through bndry cells
    for cell in self.mesh.bndry_cells:
      # get necessary cell info
      material = self.materials[cell.imat]
      width = cell.width

      # iterate over faces of bndry cells
      for face in cell.faces:
        # if this is the bndry face
        if face.flag == cell.flag:
          bc = self.bcs[face.flag-1]

          # iterate over energy groups
          for ig in range(self.G):
            row = view.CellDoFMap(ig)

            # reflective bc
            if bc.boundary_kind == 'reflective':
              pass

            # if a marshak-like bc
            elif bc.boundary_kind in ['source', 'zero_flux',
                                      'marshak, vacuum']:
              # get the diffusion coef
              D = material.D[ig]

              # compute the current term coef
              if bc.boundary_kind in ['source', 'zero_flux']:
                coef = 2*D/width
              elif bc.boundary_kind in ['marshak', 'vacuum']:
                coef = 2*D/(4*D+width)
              
              # all of these bcs change matrix
              matrix[row,row] += face.area * coef

              # only source and marshak change vector
              if bc.boundary_kind in ['source', 'marshak']:
                vector[row] += bc.vals[ig]
