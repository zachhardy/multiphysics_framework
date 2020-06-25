#!/usr/bin/env python3

import sys
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
from .physics_base import PhysicsBase

class PhysicsSystem(PhysicsBase):
    """ Class for physics that use a system of equation. """
    def __init__(self, problem, field, bcs, ics=None):
        super().__init__(problem, field, bcs, ics)

        # Stup system storage
        self.A = None
        self.b = np.zeros(self.n_dofs)
        if ics is not None:
            self.M = None

    def SolveSystem(
            self, time=None, dt=None, 
            method=None, u_half=None):
        """ Solve a time step of a transient problem. 
        
        Parameters
        ----------
        time : float, optional*]
            The simulation time before the time step.
            This is a mandatory input for transients.
        dt : float
            The time step size. This is a mandatory
            input for transients.
        method : str
            The time stepping method. This is a mandatory
            input for transients.
        u_half : numpy.ndarray (n_dofs,)
            An optional solution vector required for certain
            time stepping methods.
        """
        # Assemble and solve a steady state system.
        if not self.problem.is_transient:
            # Note that for steady state, boundary conditions
            # are applied in the assembly routines.
            self.assemble_physics()
            self.assemble_source()
            self.u[:] = spsolve(self.A, self.b)

        # Assemble and solve a transient system.
        else:
            # recompute physics matrix, if coupled
            self.assemble_mass()
            self.assemble_physics()

            # Shorthand for common use variables
            A, M, rhs = self.A, self.M, self.b
            u_old = self.u_old
            f_old = A @ u_old
            
            # Assemble a Forward Euler system.
            if method == 'fwd_euler':
                self.AssembleSource(time)
                matrix = M/dt
                rhs += M/dt @ u_old - f_old

            # Assemble a Backward Euler system.
            elif method == 'bwd_euler':
                self.AssembleSource(time+dt)
                matrix = A + M/dt
                rhs += M/dt @ u_old

            # Assemble a Crank Nicholson system.
            elif method == 'cn':
                self.AssembleSource(time+dt/2)
                matrix = M/dt + A/2
                rhs += M/dt @ u_old - f_old/2

            # Assemble a BDF2 system.
            elif method == 'bdf2':
                assert u_half is not None, (
                    "u_half must be provided for BDF2."
                )
                self.AssembleSource(time+dt)
                matrix = 1.5*M/dt + A
                rhs += 2*M/dt @ u_half - 0.5*M/dt @ u_old

            # Apply boundary conditions and solve.
            self.ApplyBCs(matrix, rhs)
            self.u[:] = spsolve(matrix, rhs)
            
    def assemble_physics(self):
        """ Assemble the physics operator. """
        if self.is_nonlinear or self.A is None:
            Arows, Acols, Avals = [], [], []
            for cell in self.mesh.cells:
                rows, cols, vals = self.assemble_cell_physics(cell)
                Arows += rows
                Acols += cols
                Avals += vals
            shape = (self.n_dofs, self.n_dofs)
            self.A = csr_matrix((Avals, (Arows, Acols)), shape)
            if not self.problem.is_transient:
                self.ApplyBCs(matrix=self.A)

    def assemble_mass(self):
        """ Assemble the time derivative operator. """
        if self.is_nonlinear or self.M is None:
            Mrows, Mvals, Mcols = [], [], []
            for cell in self.mesh.cells:
                rows, cols, vals = self.assemble_cell_mass(cell)
                Mrows += rows
                Mcols += cols
                Mvals += vals
            shape = (self.n_dofs, self.n_dofs)
            self.M = csr_matrix((Mvals, (Mrows, Mcols)), shape)

    def assemble_source(self, time=0):
        """ Assemble the forcing term vector at time. """
        self.b *= 0
        for cell in self.mesh.cells:
            rows, vals = self.assemble_cell_source(cell, time)
            if rows != []:
                self.b[rows] += vals
        if not self.problem.is_transient:
            self.ApplyBCs(vector=self.b)

    def old_physics_action(self):
        """ Compute the old physics action. """
        if self.A is None:
            self.assemble_physics()
        return self.A @ self.u_old

    def assemble_cell_physics(self, cell):
        raise NotImplementedError

    def assemble_cell_mass(self, cell):
        raise NotImplementedError

    def assemble_cell_source(self, cell, time=0):
        raise NotImplementedError

    def apply_bcs(self, matrix=None, vector=None):
        raise NotImplementedError