#!/usr/bin/env python3

import sys
import numpy as np

from field import Field
from discretizations.fv.fv import FV
from physics.physics_system import PhysicsSystem
from .neutronics_material import NeutronicsMaterial
from .k_eigen import KEigenMixin


class FV_MultiGroupDiffusion(PhysicsSystem, KEigenMixin):
    """ Finite volume multigroup diffusion handler. """
    
    name = 'flux'
    material_type = NeutronicsMaterial.material_type
    bc_kinds = [
        'reflective', 
        'marshak', 
        'vacuum',
        'source',
        'zero_flux'
    ]

    def __init__(self, problem, G, bcs, ics=None):
        super().__init__(problem, bcs, ics)
        # Number of energy groups
        self.G = G 
        # Get relevant materials
        self.materials = self._parse_materials(self.material_type)
        # Initialize and register the field with problem.
        fv = FV(self.mesh)
        self.field = Field(self.name, self.mesh, fv, G)
        self._register_field()
        
        # Additional information
        self.is_nonlinear = False
            
    def assemble_cell_physics(self, cell, keigen=False):
        """ Assemble the spatial/energy physics operator. """
        rows, cols, vals = [], [], []
        fv_view = self.sd.cell_views[cell.id]
        width = cell.width[0]
        volume = cell.volume
        material = self.materials[cell.imat]

        for ig in range(self.G):
            row = fv_view.cell_dof_map(ig)

            # Assemble removal.
            sig_r = material.sig_r[ig]
            rows += [row]
            cols += [row]
            vals += [sig_r * volume]

            for jg in range(self.G):
                col = fv_view.cell_dof_map(jg)

                # Assemble scattering.
                if hasattr(material, 'sig_s'):
                    sig_s = material.sig_s[ig][jg]
                    if sig_s != 0:
                        rows += [row]
                        cols += [col]
                        vals += [-sig_s * volume]

                # Assemble fission
                if not keigen:
                    if hasattr(material, 'nu_sig_f'):
                        chi = material.chi[ig]
                        nu_sig_f = material.nu_sig_f[jg]
                        if chi*nu_sig_f != 0:
                            rows += [row]
                            cols += [col]
                            vals += [-chi * nu_sig_f * volume]

        # Assemble interior diffusion
        for face in cell.faces:
            if face.flag == 0:
                # Get information from the neighboring cell.
                nbr_cell = self.mesh.cells[face.neighbor]
                nbr_fv_view = self.sd.cell_views[nbr_cell.id]
                nbr_width = nbr_cell.width[0]
                nbr_material = self.materials[nbr_cell.imat]
                
                for ig in range(self.G):
                    row = fv_view.cell_dof_map(ig)
                    col = nbr_fv_view.cell_dof_map(ig)
                    # Get this and neighbor cell diffusion coefs.
                    D = material.D[ig]
                    nbr_D = nbr_material.D[ig]
                    # Compute the harmonic averaged cell width and
                    # diffusion coefficient to enforce continuity
                    # of current across the cell intergace.
                    width_avg = 0.5*(width + nbr_width)
                    D_avg = 2*width_avg/(width/D + nbr_width/nbr_D)
                    # Compute the edge current term.
                    val = face.area * D_avg/width_avg
                    rows += [row, row]
                    cols += [row, col]
                    vals += [val, -val]
        return rows, cols, vals

    def assemble_cell_fission(self, cell):
        """ Assemble the fission term on cell. """
        rows, cols, vals = [], [], []
        fv_view = self.sd.cell_views[cell.id]
        volume = cell.volume
        material = self.materials[cell.imat]

        for ig in range(self.G):
            row = fv_view.cell_dof_map(ig)
            for jg in range(self.G):
                col = fv_view.cell_dof_map(jg)
                if hasattr(material, 'nu_sig_f'):
                    chi = material.chi[ig]
                    nu_sig_f = material.nu_sig_f[jg]
                    if chi*nu_sig_f != 0:
                        rows += [row]
                        cols += [col]
                        vals += [chi * nu_sig_f * volume]
        return rows, cols, vals

    def assemble_cell_mass(self, cell):
        """ Assemble the time derivative term. """
        rows, cols, vals = [], [], []
        fv_view = self.sd.cell_views[cell.id]
        volume = cell.volume
        material = self.materials[cell.imat]

        for ig in range(self.G):
            row = fv_view.cell_dof_map(ig)

            # Assemble inverse velocity.
            v = material.v[ig]
            rows += [row]
            cols += [row]
            vals += [volume / v]
        return rows, cols, vals

    def assemble_cell_source(self, cell, time=0):
        """ Assemble the source vector.

        Parameters
        ----------
        time : float, optional
            The simulation time (default is 0).
        """
        rows, vals = [], []
        fv_view = self.sd.cell_views[cell.id]
        volume = cell.volume
        material = self.materials[cell.imat]

        for ig in range(self.G):
            row = fv_view.cell_dof_map(ig)

            # Assemble source.
            if hasattr(material, 'q'):
                q = material.q[ig]
                if callable(q):
                    q = q(time)
                if q != 0:
                    rows += [row]
                    vals += [q * volume]
        return rows, vals
        
    def apply_bcs(self, matrix=None, vector=None):
        """ Apply BCs to matrix or vector.

        Parameters
        ----------
        matrix : csr_matrix (n_dofs, n_dofs)
        vector : numpy.ndarray (n_dofs,)
        """
        for cell in self.mesh.bndry_cells:
            fv_view = self.sd.cell_views[cell.id]
            material = self.materials[cell.imat]
            width = cell.width[0]

            for face in cell.faces:
                if face.flag > 0:
                    bc = self.bcs[face.flag-1]

                    # Reflective bcs do not change the matrix 
                    # or vector, so pass if a reflective boundary.
                    if bc.boundary_kind != 'reflective':

                        for ig in range(self.G):
                            row = fv_view.cell_dof_map(ig)
 
                            # Compute the coefficient required to enforce
                            # a FV boundary condition. This comes from the
                            # boundary condition being part of the diffusion term.
                            D = material.D[ig]
                            if bc.boundary_kind in ['source', 'zero_flux']:
                                coef = 2*D/width
                            elif bc.boundary_kind in ['marshak', 'vacuum']:
                                coef = 2*D/(4*D+width)
                            
                            # All non-reflective bcs change the matrix.
                            if matrix is not None:
                                matrix[row,row] += face.area * coef

                            # Only source and marshak bcs change the vector.
                            if vector is not None:
                                if bc.boundary_kind in ['source', 'marshak']:
                                    vector[row] += face.area * coef * bc.vals[ig]
    
    def _validate_bcs(self, bcs):
        """ Validate the provided boundary conditions. """
        for bc in bcs:
            if bc.boundary_kind not in self.bc_kinds:
                msg = "Approved BCs are:\n"
                for kind in self.bc_kinds:
                    msg += "{}\n".format(kind)
                raise ValueError(msg)
        return bcs

    def _validate_ics(self, ics):
        return ics
