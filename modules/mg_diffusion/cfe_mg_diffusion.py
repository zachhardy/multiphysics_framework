#!/usr/bin/env python3

import sys
import numpy as np

from field import Field
from discretizations.cfe.cfe import CFE
from physics.physics_system import PhysicsSystem
from .neutronics_material import NeutronicsMaterial
from .k_eigen import KEigenMixin

class CFE_MultiGroupDiffusion(PhysicsSystem, KEigenMixin):
    """ Continuous finite element multigroup diffusion module. """

    name = 'flux'
    material_type = NeutronicsMaterial.material_type
    bc_kinds = [
        'neumann',
        'robin',
        'dirichlet'
    ]

    def __init__(self, problem, G, bcs, ics=None, porder=1):
        super().__init__(problem, bcs, ics)
        # Number of energy groups
        self.G = G 
        # Get relevant materials
        self.materials = self._parse_materials(self.material_type)
        # Initialize and register the field with problem.
        cfe = CFE(self.mesh, porder, porder+1)
        self.field = Field(self.name, self.mesh, cfe, G)
        self._register_field()
        # Store relevant CFE info.
        self.porder = porder
        npc = self.sd.nodes_per_cell
        self.cell_matrix = np.zeros((npc, npc))
        self.cell_vector = np.zeros(npc)
        
    def assemble_cell_physics(self, cell, keigen=False):
        """ Assemble the spatial/energy physics operator. 
        
        If keigen is True, the fission term is ommitted, else
        all neutron diffusion physics is considered.

        Parameters
        ----------
        cell : cell-like object
            The cell to assemble the physics operator on.
        """
        rows, cols, vals = [], [], []
        fe_view = self.sd.cell_views[cell.id]
        material = self.materials[cell.imat]

        for ig in range(self.G):
            # Assembal removal + diffusion
            sig_r = material.sig_r[ig]
            D = material.D[ig]           
            self.cell_matrix *= 0
            for i in range(self.porder+1):
                row = fe_view.cell_dof_map(i, ig)
                for j in range(self.porder+1):
                    col = fe_view.cell_dof_map(j, ig)
                    self.cell_matrix[i,j] += (
                        fe_view.intV_shapeI_shapeJ(i, j, sig_r)
                        + fe_view.intV_gradShapeI_gradShapeJ(i, j, D)
                    )
                    rows += [row]
                    cols += [col]
            vals += list(self.cell_matrix.ravel())

            for jg in range(self.G):
                # Assemble scattering
                if hasattr(material, 'sig_s'):
                    sig_s = material.sig_s[ig][jg]
                    if sig_s != 0:
                        self.cell_matrix *= 0
                        for i in range(self.porder+1):
                            row = fe_view.cell_dof_map(i, ig)
                            for j in range(self.porder+1):
                                col = fe_view.cell_dof_map(j, jg)
                                self.cell_matrix[i,j] -= (
                                    fe_view.intV_shapeI_shapeJ(
                                        i, j, sig_s
                                    )
                                )
                                rows += [row]
                                cols += [col]
                        vals += list(self.cell_matrix.ravel())

                # Assemble fission
                if not keigen:
                    if hasattr(material, 'nu_sig_f'):
                        chi = material.chi[ig]
                        nu_sig_f = material.nu_sig_f[jg]
                        if chi*nu_sig_f != 0:
                            self.cell_matrix *= 0
                            for i in range(self.porder+1):
                                row = fe_view.cell_dof_map(i, ig)
                                for j in range(self.porder+1):
                                    col = fe_view.cell_dof_map(j, jg)
                                    self.cell_matrix[i,j] -= (
                                        fe_view.intV_shapeI_shapeJ(
                                            i, j, chi*nu_sig_f
                                        )
                                    )
                                    rows += [row]
                                    cols += [col]
                            vals += list(self.cell_matrix.ravel())
        return rows, cols, vals

    def assemble_cell_fission(self, cell):
        """ Assemble the fission operator on this cell. 
        
        This routine is only used when calling a k-eigenvalue solver.

        Parameters
        ----------
        cell : cell-like object
            The cell to assemble the fission operator on.
        """
        rows, cols, vals = [], [], []
        fe_view = self.sd.cell_views[cell.id]
        material = self.materials[cell.imat]

        for ig in range(self.G):
            for jg in range(self.G):
                # Assemble fission
                if hasattr(material, 'nu_sig_f'):
                    chi = material.chi[ig]
                    nu_sig_f = material.nu_sig_f[jg]
                    if chi*nu_sig_f != 0:
                        self.cell_matrix *= 0
                        for i in range(self.porder+1):
                            row = fe_view.cell_dof_map(i, ig)
                            for j in range(self.porder+1):
                                col = fe_view.cell_dof_map(j, jg)
                                self.cell_matrix[i,j] += (
                                    fe_view.intV_shapeI_shapeJ(
                                        i, j, chi*nu_sig_f
                                    )
                                )
                                rows += [row]
                                cols += [col]
                        vals += list(self.cell_matrix.ravel())
        return rows, cols, vals
                                    
    def assemble_cell_mass(self, cell):
        """ Assemble the time derivative term. """
        rows, cols, vals = [], [], []
        fe_view = self.sd.cell_views[cell.id]
        material = self.materials[cell.imat]

        # assemble group-wise
        for ig in range(self.G):
            v = material.v[ig]

            # Assemble inverse velocity
            self.cell_matrix *= 0
            for i in range(self.porder+1):
                row = fe_view.cell_dof_map(i, ig)
                for j in range(self.porder+1):
                    col = fe_view.cell_dof_map(j, ig)
                    self.cell_matrix[i,j] += \
                        fe_view.intV_shapeI_shapeJ(i, j, 1/v)
                    rows += [row]
                    cols += [col]
            vals += list(self.cell_matrix.ravel())
        return rows, cols, vals  

    def assemble_cell_source(self, cell, time=0):
        """ Assemble the source vector.

        Parameters
        ----------
        time : float, optional
            The simulation time (default is 0).
        """
        rows, vals = [], []
        fe_view = self.sd.cell_views[cell.id]
        material = self.materials[cell.imat]

        for ig in range(self.G):
            # Assemble source
            if hasattr(material, 'q'):
                q = material.q[ig]
                q = q(time) if callable(q) else q
                if q != 0:
                    self.cell_vector *= 0
                    for i in range(self.porder+1):
                        row = fe_view.cell_dof_map(i, ig)
                        self.cell_vector[i] = fe_view.intV_shapeI(i, q)
                        rows += [row]
                    vals += list(self.cell_vector.ravel())
        return rows, vals

    def apply_bcs(self, matrix=None, vector=None):
        """ Apply BCs to matrix and vector.

        Parameters
        ----------
        matrix : csr_matrix (n_dofs, n_dofs)
        vector : numpy.ndarray (n_dofs,)
        """
        # iterate over bndry cells and faces
        for cell in self.mesh.bndry_cells:
            fe_view = self.sd.cell_views[cell.id]
            
            for f, face in enumerate(cell.faces):        
                if face.flag > 0:
                    bc = self.bcs[face.flag-1]
    
                    # iterate over energy groups
                    for ig in range(self.G):
                        row = fe_view.face_dof_map(f, ig)

                        # neumann bc
                        if bc.boundary_kind == 'neumann':
                            if vector is not None:
                                vector[row] += face.area * bc.vals[ig]

                        # robin bc
                        elif bc.boundary_kind == 'robin':
                            if matrix is not None:
                                matrix[row,row] += face.area * 0.5
                            if vector is not None:
                                vector[row] += face.area * 2.0*bc.vals[ig]

                        # dirichlet bc
                        elif bc.boundary_kind == 'dirichlet':
                            if matrix is not None:
                                matrix[row,row] = 1.0
                                for col in matrix[row].nonzero()[1]:
                                    if row != col:
                                        matrix[row,col] = 0.0
                            if vector is not None:
                                vector[row] = bc.vals[ig]

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
