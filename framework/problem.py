import numpy as np
import matplotlib.pyplot as plt
from solvers.operator_splitting import OperatorSplitting

class Problem:

    # Objects
    mesh = None
    materials = []
    physics = []
    fields = []
    solver = None
    # Field information
    n_dofs = 0
    n_fields = 0
    # Solution vectors
    u = np.array([])
    u_ell = np.array([])
    u_old = np.array([])
    # Other parameters
    tol = 1e-6
    maxit = 100
    verbosity = 0
    
    def __init__(self, mesh, materials):
        self.mesh = mesh
        self.materials = materials
        
    def run_steady_state(self, solver_type='os', tol=1e-6, 
                         maxit=100, verbosity=2):
        self.verbosity = verbosity
        self.is_transient = False
        self.tol = tol
        self.maxit = maxit
        self.u_ell = np.copy(self.u)
        # Initialize a solver and solve.
        self.solver = OperatorSplitting(self)
        self.solver.solve_system()

    def run_transient(self, solver_type='os', t0=0, tend=0.1, 
                      dt=2e-3, method='tbdf2', tol=1e-6, 
                      maxit=100, verbosity=0):
        self.verbosity = verbosity
        self.is_transient = True
        self.tol = tol
        self.maxit = maxit
        # Initialize time stepping parameters.
        time = t0
        dt = dt
        step = 0
        # Create additional storage for mulit-step methods.
        if method == 'tbdf2':
            u_tmp = np.copy(self.u)

        # Initialize a solver and start time stepping.
        self.evauluate_ics()
        self.solver = OperatorSplitting(self)
        while time < tend:
            step += 1
            if self.verbosity > 0:
                msg = "* Time Step {} *".format(step)
                msg = "\n".join(["\n", "*"*len(msg), msg, "*"*len(msg)])
                msg += "\ntime:\t{:.5f} micro-sec".format(time+dt)
                msg += "\ndt:\t{:.5f} micro-sec".format(dt)
                msg += "\n"+"="*25
                print(msg)

            # Compute old physics action
            self.compute_old_physics_action()

            # Handle single step methods.
            if method != 'tbdf2':
                self.solver.solve_system(time, dt, method)

            # Handle multi-step methods. Currently, the only multi-step
            # method available is TBDF2.
            else:
                # Take a half step with Crank Nicholson.
                self.solver.solve_system(time, dt/2, 'cn')
                # Store the result for use in BDF2.
                u_tmp[:] = self.u 
                # Take another half step with BDF2.
                self.solver.solve_system(time+dt/2, dt/2, 'bdf2', u_tmp)

            time += dt
            # If the next time step pushes time beyond tend, change the
            # time step to ensure the simulaltion ends at tend.
            if time + dt > tend:
                dt = tend - time
            self.u_old[:] = self.u

    def evauluate_ics(self):
        for physic in self.physics:
            fields = physic.fields
            for f, field in enumerate(fields):
                for c in range(field.components):
                    dofs = field.component_dofs(c)
                    grid = field.grid.ravel()
                    # NOTE: This implies that fields have only 
                    # one component. If several fields live in 
                    # one physics and any of those fields have 
                    # multiple components, this will fail. A
                    # potential fix is to make ICs a list of 
                    # lists instead of just a list.
                    self.u[dofs] = physic.ics[f](grid)
        self.u_ell = np.copy(self.u)
        self.u_old = np.copy(self.u)

    def compute_old_physics_action(self):
        for physic in self.physics:
            physic.compute_old_physics_action()

        