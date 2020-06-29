import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
from .physics_base import PhysicsBase

class PhysicsSystem(PhysicsBase):

    def __init__(self, problem, bcs, ics=None):
        super().__init__(problem)
        # Validate boundary and initial conditions.
        self.bcs = self._validate_bcs(bcs)
        self.ics = self._validate_ics(ics)
        # Setup system storage
        self.A = None
        self.b = None
        if ics is not None:
            self.f_old = None
            self.M = None

    def solve_system(self, time=None, dt=None, 
                     method=None, u_half=None):
        # Assemble and solve a steady state system.
        if not self.problem.is_transient:
            # Note that for steady state, boundary conditions
            # are applied in the assembly routines.
            self.assemble_physics()
            self.assemble_source()
            self.field.u[:] = spsolve(self.A, self.b)

        # Assemble and solve a transient system.
        else:
            # recompute physics matrix, if coupled
            self.assemble_mass()
            self.assemble_physics()

            # Shorthand for common use variables
            A, M, rhs = self.A, self.M, self.b
            u_old, f_old = self.field.u_old, self.f_old
            
            # Assemble a Forward Euler system.
            if method == 'fwd_euler':
                self.assemble_source(time)
                matrix = M/dt
                rhs += M/dt @ u_old - f_old

            # Assemble a Backward Euler system.
            elif method == 'bwd_euler':
                self.assemble_source(time+dt)
                matrix = A + M/dt
                rhs += M/dt @ u_old

            # Assemble a Crank Nicholson system.
            elif method == 'cn':
                self.assemble_source(time+dt/2)
                matrix = M/dt + A/2
                rhs += M/dt @ u_old - f_old/2

            # Assemble a BDF2 system.
            elif method == 'bdf2':
                assert u_half is not None, (
                    "u_half must be provided for BDF2."
                )
                self.assemble_source(time+dt)
                matrix = 1.5*M/dt + A
                rhs += 2*M/dt @ u_half - 0.5*M/dt @ u_old

            # Apply boundary conditions and solve.
            self.apply_bcs(matrix, rhs)
            self.field.u[:] = spsolve(matrix, rhs)
            
    def assemble_physics(self):
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
                self.apply_bcs(matrix=self.A)

    def assemble_mass(self):
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
        self.b = np.zeros(self.n_dofs) if self.b is None else 0*self.b
        for cell in self.mesh.cells:
            rows, vals = self.assemble_cell_source(cell, time)
            if rows != []:
                self.b[rows] += vals
        if not self.problem.is_transient:
            self.apply_bcs(vector=self.b)

    def recompute_old_physics_action(self):
        if self.A is None:
            self.assemble_physics()
        self.f_old[:] = self.A @ self.field.u_old

    def assemble_cell_physics(self, cell):
        raise NotImplementedError

    def assemble_cell_mass(self, cell):
        raise NotImplementedError

    def assemble_cell_source(self, cell, time=0):
        raise NotImplementedError

    def apply_bcs(self, matrix=None, vector=None):
        raise NotImplementedError

    def _register_field(self):
        self.b = np.zeros(self.field.n_dofs)
        self.f_old = np.zeros(self.b.shape)
        super()._register_field()

    def _validate_bcs(self, bcs):
        raise NotImplementedError

    def _validate_ics(self, ics):
        raise NotImplementedError