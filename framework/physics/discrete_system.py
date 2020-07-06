import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve

class DiscreteSystem:

    def __init__(self, field, bcs):
        # Objects
        self.field = field
        self.mesh = field.mesh
        self.discretization = field.discretization
        self.bcs = bcs

        # System information
        self.A = None
        self.M = None
        self.f_ell = np.zeros(self.field.n_dofs)
        self.f_old = np.zeros(self.field.n_dofs)
        self.rhs = np.zeros(self.field.n_dofs)

    def assemble_physics(self):
        raise NotImplementedError

    def assemble_mass(self):
        raise NotImplementedError

    def assemble_forcing(self, time=0):
        raise NotImplementedError

    def assemble_lagged_sources(self, old=False):
        raise NotImplementedError

    def apply_bcs(self, matrix=None, vector=None):
        raise NotImplementedError

    def solve_steady_state(self):
        self.assemble_physics()
        self.assemble_forcing()
        self.assemble_lagged_sources()
        self.rhs -= self.f_ell
        self.apply_bcs(vector=self.rhs)
        self.field.u[:] = spsolve(self.A, self.rhs)

    def solve_time_step(self, time, dt, method, u_tmp):
        # Assemble matrices
        self.assemble_physics()
        self.assemble_mass()

        # Shorthand
        A, M, rhs = self.A, self.M, self.rhs
        f_ell, f_old = self.f_ell, self.f_old
        u_old = self.field.u_old

        # Forward Euler
        if method == 'fwd_euler':
            self.assemble_forcing(time)
            matrix = M/dt
            rhs += M/dt @ u_old - f_old
            
        # Backward Euler
        elif method == 'bwd_euler':
            self.assemble_forcing(time+dt)
            self.assemble_lagged_sources()
            matrix = M/dt + A
            rhs += M/dt @ u_old - f_ell
        # Crank Nicholson
        elif method == 'cn':
            self.assemble_forcing(time+dt/2)
            self.assemble_lagged_sources()
            matrix = M/dt + A/2
            rhs += M/dt @ u_old - (f_ell + f_old)/2
        # BDF2
        elif method == 'bdf2':
            assert u_tmp is not None, (
                    "u_tmp must be provided for BDF2."
                )
            u_tmp = u_tmp[self.field.dofs]
            self.assemble_forcing(time+dt)
            self.assemble_lagged_sources()
            matrix = 1.5*M/dt + A
            rhs += 2*M/dt @ u_tmp - 0.5*M/dt @ u_old - f_ell

        # Apply boundary conditions and solve
        self.apply_bcs(matrix, rhs)
        self.field.u[:] = spsolve(matrix, rhs)

    def compute_old_physics_action(self):
        if self.A is None:
            self.assemble_physics()
        self.f_old += self.A @ self.field.u_old
