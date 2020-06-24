#!/usr/bin/env python3

import sys
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve

from .mg_diffusion import MultiGroupDiffusion
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
      
  def AssembleCellPhysics(self, cell):
    """ Assemble the spatial/energy physics operator. """
    rows, cols, vals = [], [], []
    # discretization view
    fv_view = self.sd.fv_views[cell.id]
    # cell info
    width = cell.width[0]
    volume = cell.volume
    material = self.materials[cell.imat]

    # assemble group-wise
    for ig in range(self.G):
      row = fv_view.CellDoFMap(ig)

      # removal
      sig_r = material.sig_r[ig]
      rows += [row]
      cols += [row]
      vals += [sig_r * volume]

      # assemble coupling terms
      for jg in range(self.G):
        col = fv_view.CellDoFMap(jg)

        # scattering
        if hasattr(material, 'sig_s'):
          sig_s = material.sig_s[ig][jg]
          if sig_s != 0:
            rows += [row]
            cols += [col]
            vals += [-sig_s * volume]

        # fission
        if hasattr(material, 'nu_sig_f'):
          chi = material.chi[ig]
          nu_sig_f = material.nu_sig_f[jg]
          if chi*nu_sig_f != 0:
            rows += [row]
            cols += [col]
            vals += [-chi * nu_sig_f * volume]

    # assemble interior diffusion
    for face in cell.faces:
      if face.flag == 0:
        # neighbor cell and discretization
        nbr_cell = face.neighbor_cell
        nbr_fv_view = self.sd.fv_views[nbr_cell.id]
        # neighbor cell info
        nbr_width = nbr_cell.width[0]
        nbr_material = self.materials[nbr_cell.imat]
        
        # assemble group-wise
        for ig in range(self.G):
          row = fv_view.CellDoFMap(ig)
          col = nbr_fv_view.CellDoFMap(ig)
          # diffusion coefs
          D = material.D[ig]
          nbr_D = nbr_material.D[ig]
          # harmonic averaged quantities
          width_avg = 0.5*(width + nbr_width)
          D_avg = 2*width_avg/(width/D + nbr_width/nbr_D)
          # compute the contributions
          val = face.area * D_avg/width_avg
          rows += [row, row]
          cols += [row, col]
          vals += [val, -val]
    return rows, cols, vals

  def AssembleCellMass(self, cell):
    """ Assemble the time derivative term. """
    rows, cols, vals = [], [], []
    # discretization view
    fv_view = self.sd.fv_views[cell.id]
    # cell info
    volume = cell.volume
    material = self.materials[cell.imat]

    # assemble group-wise
    for ig in range(self.G):
      row = fv_view.CellDoFMap(ig)

      # inverse velocity scaling
      v = material.v[ig]
      rows += [row]
      cols += [row]
      vals += [volume / v]
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
    fv_view = self.sd.fv_views[cell.id]
    # cell info
    volume = cell.volume
    material = self.materials[cell.imat]

    # assemble group-wise
    for ig in range(self.G):
      row = fv_view.CellDoFMap(ig)

      # source
      if hasattr(material, 'q'):
        q = material.q[ig]
        if callable(q):
          q = q(time)
        if q != 0:
          rows += [row]
          vals += [q * volume]
    return rows, vals
    
  def ApplyBCs(self, matrix=None, vector=None):
    """ Apply BCs to matrix or vector.

    Parameters
    ----------
    matrix : csr_matrix (n_dofs, n_dofs)
    vector : numpy.ndarray (n_dofs,)
    """
    # iterate through bndry cells
    for cell in self.mesh.bndry_cells:
      # cell info
      fv_view = self.sd.fv_views[cell.id]
      material = self.materials[cell.imat]
      width = cell.width

      # iterate over faces of bndry cells
      for face in cell.faces:
        # if this is the bndry face
        if face.flag == cell.flag:
          bc = self.bcs[face.flag-1]

          # iterate over energy groups
          if bc.boundary_kind != 'reflective':
            for ig in range(self.G):
              row = fv_view.CellDoFMap(ig)
              
              # if a marshak-like bc
              if bc.boundary_kind in ['source', 'zero_flux',
                                        'marshak, vacuum']:
                # get the diffusion coef
                D = material.D[ig]
                # compute the current term coef
                if bc.boundary_kind in ['source', 'zero_flux']:
                  coef = 2*D/width
                elif bc.boundary_kind in ['marshak', 'vacuum']:
                  coef = 2*D/(4*D+width)
                
                # all of these bcs change matrix
                if matrix is not None:
                  matrix[row,row] += face.area * coef

                # add source contributions
                if vector is not None:
                  if bc.boundary_kind in ['source', 'marshak']:
                    vector[row] += face.area * coef * bc.vals[ig]
  
  def ValidateBCs(self, bcs):
    bc_kinds = [
      'reflective', 
      'marshak', 
      'vacuum',
      'source',
      'zero_flux'
    ]

    try:
      for bc in bcs:
        if bc.boundary_kind not in bc_kinds:
          msg = "Approved BCs are:\n"
          for kind in bc_kinds:
            msg += "{}\n".format(kind)
          raise ValueError(msg)
      bcs = super().ValidateBCs(bcs)
      return bcs
      

    except ValueError as err:
      print(err.args[0])
      sys.exit(-1)