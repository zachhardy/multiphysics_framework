#!/usr/bin/env python3

import sys
import numpy as np

from discretizations.cfe.cfe import CFE
from field import Field
from physics.physics_system import PhysicsSystem
from .hc_material import HeatConductionMaterial

bc_kinds = [
    'neumann',
    'robin',
    'dirichlet'
]

class CFE_HeatConduction(PhysicsSystem):
    """ Continuous finite element heat conduction handler. 
    
    Generate a heat conduction subproblem (physics) within 
    the global problem handler. 
    
    Parameters
    ----------
    problem : Problem object
        The global problem handler
    bcs : list of BC objects
        The bounday conditions for the heat conduction problem.
    ics : callable, default=None
        The initial condition for the heat conduction problem.
        If steady state, this should be None.
    porder : int, default=1
        The polynomial order of the finite elements.
    """

    name = 'temperature'
    material_type = HeatConductionMaterial.material_type

    def __init__(self, problem, bcs, ics=None, porder=1):
        super().__init__(problem, bcs, ics)

        # Get relevant materials
        self.materials = self._parse_materials(self.material_type)
        # Initialize and register the field with problem.
        cfe = CFE(self.mesh, porder, porder+1)
        self.field = Field(self.name, self.mesh, cfe, 1)
        self._register_field()
        # Store relevant CFE info.
        self.porder = porder
        npc = self.sd.nodes_per_cell
        self.cell_matrix = np.zeros((npc, npc))
        self.cell_vector = np.zeros(npc)
        # Determine nonlinearity
        for material in self.materials:
            if callable(material.k):
                self.is_nonlinear = True


    def assemble_cell_physics(self, cell):
        """ Assemble the spatial physics operator. 
        
        Parameters
        ----------
        cell : cell-like
        """
        rows, cols, vals = [], [], []
        fe_view = self.sd.fe_views[cell.id]
        material = self.materials[cell.imat]

        # Assemble diffusion
        k = material.k
        if callable(k):
            if not self.is_coupled:
                T = fe_view.quadrature_solution(self.u)
                k = k(T)
            else:
                raise NotImplementedError(
                    "Only T dependent conductivities allowed."
                )
        self.cell_matrix *= 0
        for i in range(self.porder+1):
            row = fe_view.cell_dof_map(i)
            for j in range(self.porder+1):
                col = fe_view.cell_dof_map(j)
                self.cell_matrix[i,j] += (
                    fe_view.intV_gradShapeI_gradShapeJ(i, j, k)
                )
                rows += [row]
                cols += [col]
        vals += list(self.cell_matrix.ravel())
        return rows, cols, vals

    def assemble_cell_source(self, cell, time=0):
        """ Assemble the source vector.

        Parameters
        ----------
        cell : cell-like object
        time : float, optional
            The simulation time (default is 0).
        """
        rows, vals = [], []  
        fe_view = self.sd.fe_views[cell.id]
        material = self.materials[cell.imat]

        # Assemble source
        if hasattr(material, 'q'):
            q = material.q
            q = q(time) if callable(q) else q
            if q != 0:
                self.cell_vector *= 0
                for i in range(self.porder+1):
                    row = fe_view.cell_dof_map(i)
                    self.cell_vector[i] = fe_view.intV_shapeI(i, q)
                    rows += [row]
                vals += list(self.cell_vector.ravel())
        return rows, vals

    def apply_bcs(self, matrix=None, vector=None):
        """ Apply boundary conditions to matrix and vector.

        This function requires either matrix, vector, or both
        to be provided. If one of the two are provided, bounndary
        conditions are only applied to the provided input.

        Parameters
        ----------
        matrix : csr_matrix (n_dofs, n_dofs), default=None
            The matrix to apply boundary conditions to.
        vector : numpy.ndarray (n_dofs,), default=None
            The vector to apply boundary conditions to.
        """
        for cell in self.mesh.bndry_cells:
            fe_view = self.sd.fe_views[cell.id]

            for f, face in enumerate(cell.faces):        
                if face.flag > 0:
                    bc = self.bcs[face.flag-1]
                    row = fe_view.face_dof_map(f)

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

    def _validate_bcs(self, bcs):
        """ Validate the boundary conditions.

        Parameters
        ----------
        bcs : list of BC objects
        """
        for bc in bcs:
            if bc.boundary_kind not in bc_kinds:
                msg = "Approved BCs are:\n"
                for kind in bc_kinds:
                    msg += "{}\n".format(kind)
                raise ValueError(msg)
        return bcs

    def _validate_ics(self, ics):
        """ Validate the initial condition.

        Parameters
        ----------
        ics : callable, None
            The initial condition, or lack there of.
        """
        if ics is not None:
            if not callable(ics):
                msg = "Initial condition must be callable."
                raise ValueError(msg)
        return ics


