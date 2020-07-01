import numpy as np
from scipy.sparse.linalg import spsolve

class DiscreteSystem:

    # System information
    A = None
    M = None
    f_ell = None
    f_old = None
    rhs = None
    # Objects
    mesh = None
    discretization = None
    field = None
    # Boundary and initial conditions
    bcs = []
    ics = []

    def solve_steady_state(self):
        self.assemble_physics()
        self.assemble_forcing()
        self.lagged_operator_action(True, self.f_ell)
        self.rhs -= self.f_ell
        self.apply_bcs(vector=self.rhs)
        self.field.u[:] = spsolve(self.A, self.rhs)

    def solve_time_step(self, time, dt, method, u_tmp):
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
            self.lagged_operator_action(True, f_ell)
            matrix = M/dt + A
            rhs += M/dt @ u_old - f_ell
        # Crank Nicholson
        elif method == 'cn':
            self.assemble_forcing(time+dt/2)
            self.lagged_operator_action(True, f_ell)
            matrix = M/dt + A/2
            rhs += M/dt @ u_old - (f_ell + f_old)/2
        # BDF2
        elif method == 'bdf2':
            assert u_tmp is not None, (
                    "u_tmp must be provided for BDF2."
                )
            u_tmp = u_tmp[self.field.dofs]
            self.assemble_forcing(time+dt)
            self.lagged_operator_action(True, f_ell)
            matrix = 1.5*M/dt + A
            rhs += 2*M/dt @ u_tmp - 0.5*M/dt @ u_old - f_ell

        # Apply boundary conditions and solve
        self.apply_bcs(matrix, rhs)
        self.field.u[:] = spsolve(matrix, rhs)

    def assemble_physics(self):
        if self.discretization.dtype == 'fv':
            self.assemble_fv_physics()
        elif self.discretization.dtype == 'cfe':
            self.assemble_cfe_physics()

    def assemble_mass(self):
        if self.discretization.dtype == 'fv':
            self.assemble_fv_mass()
        elif self.discretization.dtype == 'cfe':
            self.assemble_cfe_mass()

    def assemble_forcing(self, time=0):
        self.rhs[:] = 0
        if self.discretization.dtype == 'fv':
            self.assemble_fv_forcing(time)
        elif self.discretization.dtype == 'cfe':
            self.assemble_cfe_forcing(time)

    def apply_bcs(self, matrix=None, vector=None):
        if self.discretization.dtype == 'fv':
            self.apply_fv_bcs(matrix, vector)
        elif self.discretization.dtype == 'cfe':
            self.apply_cfe_bcs(matrix, vector)

    def compute_old_physics_action(self):
        if self.A is None:
            self.assemble_physics()
        self.f_old += self.A @ self.field.u_old

    def lagged_operator_action(self, ell, f):
        raise NotImplementedError

    def assemble_fv_physics(self):
        raise NotImplementedError

    def assemble_cfe_physics(self):
        raise NotImplementedError

    def assemble_fv_mass(self):
        raise NotImplementedError

    def assemble_cfe_mass(self):
        raise NotImplementedError

    def assemble_fv_forcing(self, time):
        raise NotImplementedError

    def assemble_cfe_forcing(self, time):
        raise NotImplementedError

    def apply_fv_bcs(self, matrix=None, vector=None):
        raise NotImplementedError

    def apply_cfe_bcs(self, matrix=None, vector=None):
        raise NotImplementedError

    