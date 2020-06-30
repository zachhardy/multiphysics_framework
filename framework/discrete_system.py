from scipy.sparse.linalg import spsolve

class DiscreteSystem:

    # System information
    A = None
    b = None
    M = None
    f_old = None
    # Objects
    mesh = None
    discretization = None
    field = None
    # Boundary and initial conditions
    bcs = []
    ics = []

    def solve_time_step(self, time, dt, method, u_half):
        A = self.A
        M = self.M
        u_old = self.field.u_old
        f_old = self.f_old

        # Forward Euler
        if method == 'fwd_euler':
            self.assemble_source(time)
            matrix = M/dt
            self.b += M/dt @ u_old - f_old
        # Backward Euler
        elif method == 'bwd_euler':
            self.assemble_source(time+dt)
            matrix = M/dt + A
            self.b += M/dt @ u_old
        # Crank Nicholson
        elif method == 'cn':
            self.assemble_source(time+dt/2)
            matrix = M/dt + A/2
            self.b += M/dt @ u_old - f_old/2
        # BDF2
        elif method == 'bdf2':
            assert u_half is not None, (
                    "u_half must be provided for BDF2."
                )
            u_half = u_half[self.field.dofs]
            self.assemble_source(time+dt)
            matrix = 1.5*M/dt + A
            self.b += 2*M/dt @ u_half - 0.5*M/dt @ u_old

        # Apply boundary conditions and solve
        self.apply_bcs(matrix, self.b)
        self.field.u[:] = spsolve(matrix, self.b)

    def assemble_physics(self):
        if self.discretization.dtype == 'fv':
            self.assemble_fv_physics()
        elif self.discretization.dtype == 'cfe':
            pass

    def assemble_mass(self):
        if self.discretization.dtype == 'fv':
            self.assemble_fv_mass()
        elif self.discretization.dtype == 'cfe':
            pass

    def assemble_source(self, time=0):
        if self.discretization.dtype == 'fv':
            self.assemble_fv_source(time)
        elif self.discretization.dtype == 'cfe':
            pass

    def apply_bcs(self, matrix=None, vector=None):
        if self.discretization.dtype == 'fv':
            self.apply_fv_bcs(matrix, vector)
        elif self.discretization.dtype == 'cfe':
            pass

    def compute_old_physics_action(self):
        if self.A is None:
            self.assemble_physics()
        self.f_old = self.A @ self.field.u_old

    def assemble_fv_physics(self):
        raise NotImplementedError

    def assemble_fv_mass(self):
        raise NotImplementedError

    def assemble_fv_source(self, time):
        raise NotImplementedError

    def apply_fv_bcs(self, matrix=None, vector=None):
        raise NotImplementedError

    