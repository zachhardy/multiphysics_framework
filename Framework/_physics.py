#!/usr/bin/env python3

import sys
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve, gmres

class PhysicsBase:
    """ Template class for physics modules.

    Parameters
    ----------
    mesh : MeshBase object
    fields : list of Field
    materials : list of Material
    """
    def __init__(self, mesh, fields, materials):
        self.mesh = mesh
        self.fields = fields
        self.materials = materials

        # General information
        self.n_el = mesh.n_el

        # Discretization information
        self.n_nodes = None
        self.n_dofs = None

    def run_steady_state(self, bcs, nonlinear=False, u0=None,
                         tol=1e-6, maxit=100, solver='picard',
                         verbosity=2):
        """ Run a steady state problem.
        
        Parameters
        ----------
        bcs: list
            BCs list where each BC is a dict.
        nonlinear: boolean
            Flag for linear/nonlinear problems (default is False).
        u0: numpy.ndarray (n_dofs,) 
            An initial guess (default is None).
        tol: float 
            The convergence tolerance (default is 1e-6).
        maxit: int 
            Maximum number of iterations (default is 100).
        solver: str 
            Solver type. Options are 'picard' or 'newton'
            (default is 'picard').
        verbosity: int
            Level of screen output (default is 2).
        """
        self.transient = False
        self.nonlinear = nonlinear
        self.verbosity = verbosity
        
        self.bcs = self._validate_bcs(bcs)
        
        if nonlinear:
            solver = self._validate_solver(solver)
            self.solve_nonlinear_system(u0, tol, maxit, solver)
        
        else:
            self.solve_linear_system()
        
        return self.u
        
    def run_transient(self, bcs, ics, t0=0., tend=0.1, 
                      dt=2e-3, method='tbdf2', nonlinear=False, 
                      tol=1e-6, maxit=100, solver='picard', 
                      verbosity=2):
        """
        Run a transient problem.
        
        Parameters
        ----------
        bcs: list
            BCs list where each BC is a dict.
        ics: list
            ICs list where each IC is callable.
        t0: float
            Initial simulation time (default is 0).
        tend: float
            Final simulation time (default is 0.1).
        dt: float
            Time step size (default is 2e-3).
        method: str
            The time stepping method to use. Options are
            backward_euler, forward_euler, crank_nicholson,
            and tbdf2 (default is tbdf2).
        nonlinear: boolean
            Flag for linear/nonlinear problems (default is False)
        tol: float 
            The convergence tolerance (default is 1e-6).
        maxit: int 
            Maximum number of iterations (default is 100).
        solver: str 
            Solver type. Options are 'picard' or 'newton'
            (default is 'picard').
        verbosity: int
            Level of screen output (default is 2).
        """
        self.transient = True
        self.nonlinear = nonlinear
        self.verbosity = verbosity
        
        # Parse inputs
        self.bcs = self._validate_bcs(bcs)
        self.ics = self._validate_ics(ics)
        method = self._validate_method(method)
        if nonlinear is True:
            solver = self._validate_solver(solver) 
            
        self.evaluate_ics()

        # Initializations
        u_old = np.copy(self.u) # old solution vector
        rhs = np.zeros(u_old.shape) # right hand side vector
        A = self.assemble_physics() # physics matrix w/ bcs
        M = self.assemble_mass() # mass matrix for time deriv
        
        # Time stepping loop
        time, istep = t0, 0
        while time < tend:
            # Increment step counter
            istep += 1

            # Print time step info
            if self.verbosity > 0:
                msg = "* Time Step {} *".format(istep)
                msg = "\n".join(["", "*"*len(msg), msg, "*"*len(msg)])
                msg += "\ntime:\t{:.3e} micro-sec".format(time+dt)
                msg += "\ndt:\t{:.4e} micro-sec".format(dt)
                print(msg)

            # Solve a time step
            if not nonlinear:
                self.solve_linear_system(
                    time=time, dt=dt, method=method, 
                    u_old=u_old, rhs=rhs, A=A, M=M
                )
            else:
                self.solve_nonlinear_system(u_old, tol, maxit, solver, 
                    time=time, dt=dt, method=method, u_old=u_old, 
                    rhs=rhs, A=None, M=M
                )
            
            # Reinit old solution vector
            u_old[:] = self.u
            time += dt
            if time + dt > tend:
                dt = tend - time
        
        return self.u

    def solve_linear_system(self, **kwargs):
        """
        Solve a linear system. This is the basic method used
        to solve linear/nonlinear, transient/steady state problem.
        
        Parameters
        ----------
        kwargs: Parameters for time stepping.
        """
        matrix, rhs = self.assemble_system(**kwargs)
        self.u = spsolve(matrix, rhs)


    def solve_nonlinear_system(self, u0=None, tol=1e-6, maxit=100, 
                               solver='picard', **kwargs):
        """
        Solve a nonlinear system. This is the basic method used for both
        transient and steady state nonlinear problems. The kwargs consist
        of the parameters used for time stepping.

        Parameters
        ----------
        u0: numpy.ndarray (n_dofs,) 
            An initial guess. Default is None.

        tol: float 
            The convergence tolerance. Default is 1e-6.

        maxit: int 
            Maximum number of iterations. Default is 100.

        solver: str 
            Solver type. Options are `picard` or `newton`.
            Default is `picard`.

        kwargs: Parameters for time stepping.
        """
        if u0 is not None:
            assert len(u0) == self.n_dofs
        
        # Init solution vector to initial guess
        self.u[:] = np.copy(u0)

        diff, nit = 1e6, 0
        while diff > tol and nit < maxit:
            # Picard iteration
            if solver == 'picard':
                # Assemble and solve
                self.solve_linear_system(**kwargs)
                # Check convergence
                diff = np.linalg.norm(self.u-u0, ord=2)
                # Reinit guess
                u0[:] = self.u

            # Newton iteration
            elif 'newton' in solver:
                # Compute Jacobian system
                J, R = self.numerical_jacobian(self.u, **kwargs)
                # Solve system
                du = spsolve(J, -R)
                # Increment solution
                self.u += du
                # Check convergence
                diff = np.linalg.norm(du, ord=2)

            # Increment iteration counter
            nit += 1

            # Print outs
            if not self.transient and self.verbosity > 0:
                msg = "\nIteration {}".format(nit)
                msg = "\n".join([msg, "-"*len(msg)])
                msg += "\nAbsolute Difference:\t{:.4e}\n".format(diff)
                print(msg)
            elif self.transient and self.verbosity > 1:
                msg = "\n\tIteration {}".format(nit)
                msg = "\n".join([msg, "\t"+"-"*len(msg)])
                msg += "\n\tAbsolute Difference:\t{:.4e}\n".format(diff)
                print(msg)

    def residual(self, u, **kwargs):
        """
        Compute the residual associated with vector u.

        Parameters
        ----------
        u: numpy.ndarray (n_dofs,) 
            A solution vector.

        Returns
        -------
        numpy.ndarray (n_dofs,); The residual from vector u.
        """
        # Checks
        assert len(u) == self.n_dofs
        # Assemble system
        matrix, rhs = self.assemble_system(**kwargs)
        return matrix @ u - rhs

    def numerical_jacobian(self, u, **kwargs):
        """
        Compute the Jacobian matrix.

        Parameters
        ----------
        u: numpy.ndarray (n_dofs,) 
            A solution vector.

        Returns
        -------
        J: csr_matrix (n_dofs, n_dofs)
            Approximate Jacobian matrix.

        R: numpy.ndarray (n_dofs,)
            A reference residual vector.
        """
        # Checks
        assert len(u) == self.n_dofs
        # Compute machine precision
        eps_m = np.finfo(float).eps
        # Compute residual
        R = self.residual(u, **kwargs)

        # Build Jacobian
        J = np.zeros((self.n_dofs, self.n_dofs))
        for dof in range(self.n_dofs):
            # Eps is a unit vector
            eps = np.zeros(self.n_dofs)
            eps[dof] = (1 + np.abs(u[dof])) * np.sqrt(eps_m)

            # Residual at u + eps
            Rp = self.residual(u+eps, **kwargs)

            # idof column of J is the forward difference formula
            J[:,dof] = (Rp - R) / eps[dof]
        return csr_matrix(J), R

    def assemble_system(self, u=None, time=None, dt=None, method=None, 
                        u_old=None, rhs=None, A=None, M=None):
        """
        Solve a time step.
        
        Parameters
        ----------
        u: numpy.ndarray
            A vector used for evaluating nonlinearities. This will 
            be resized to be shape (n_nodes, -1). Default is None.

        time: float
            The simulation time at the start of the time step.
            Default is None.

        dt: float
            The time step size. Default is None.

        method: str
            The time stepping method. See `run_transient` for documentation.
            Default is None.

        u_old: numpy.ndarray (n_dofs,) 
            The old solution vector.

        rhs: numpy.ndarray (n_dofs,) 
            The right hand side of the system. This vector is filled by
            this routine. Default is None.

        A: csr_matrix (n_dofs, n_dofs) 
            The physics matrix. Default is None. If None, the matrix
            is constructed here.

        M: csr_matrix (n_dofs, n_dofs) 
            The mass maxtix. Default is None. If None, the matrix
            is constructed here.
        """
        # Assemble steady state system
        if self.transient is False:
            matrix = self.assemble_physics(u)
            rhs = self.assemble_source(u)
        
        else:
            # Form matrices for the time step
            A = self.assemble_physics(u) if A is None else A
            M = self.assemble_mass() if M is None else M
            
            # Forward Euler systems
            if method == 'forward_euler':
                matrix = M/dt
                self.assemble_source(time, rhs)
                rhs = (M/dt - A) @ u_old

            # Backward Euler systems
            elif method == 'backward_euler':
                matrix = M/dt + A
                self.assemble_source(time+dt, rhs)
                rhs += M/dt @ u_old

            # Crank Nicholson systems
            elif method == 'crank_nicholson':
                matrix = M/dt + 0.5*A
                self.assemble_source(time+dt/2, rhs)
                rhs += (M/dt - 0.5*A) @ u_old

            # TBDF-2 systems
            elif method == 'tbdf2':
                # Half step crank nicholson
                matrix = 2*M/dt + 0.5*A
                self.assemble_source(time+dt/4, rhs)
                rhs += (2*M/dt - 0.5*A) @ u_old
                self.apply_dirichlet_bcs(matrix, rhs)
                u = spsolve(matrix, rhs)

                # Half step BDF2
                matrix = 3*M/dt + A
                self.assemble_source(time+dt, rhs)
                rhs += 4*M/dt @ u - M/dt @ u_old
        return self.apply_dirichlet_bcs(matrix, rhs)
    
    def assemble_physics(self, u=None):
        raise NotImplementedError(
            "This must be implemented in derived classes."
        )

    def assemble_mass(self):
        raise NotImplementedError(
            "This must be implemented in derived classes."
        )

    def assemble_source(self, time=0, rhs=None):
        raise NotImplementedError(
            "This must be implemented in derived classes."
        )

    def apply_matrix_bcs(self, matrix):
        raise NotImplementedError(
            "This must be implemented in derived classes."
        )
    
    def apply_vector_bcs(self, matrix):
        raise NotImplementedError(
            "This must be implemented in derived classes."
        )
    
    def apply_dirichlet_bcs(self, matrix, rhs):
        raise NotImplementedError(
            "This must be implemented in derived classes."
        )

    def evaluate_ics(self):
        raise NotImplementedError(
            "This must be implemented in derived classes."
        )
 
    def _validate_bcs(self, bcs):
        raise NotImplementedError(
            "This method must be implemented in derived classes."
        )

    def _validate_ics(self, ics):
        raise NotImplementedError(
            "This method must be implemented in derived classes."
        )

    @staticmethod
    def _validate_method(method):
        try:
            if method not in ['forward_euler',
                              'backward_euler',
                              'crank_nicholson',
                              'tbdf2']:
                raise ValueError("Unrecognized method.")
        except ValueError as err:
            print(err.args[0])
            sys.exit(-1)
        return method

    @staticmethod
    def _validate_solver(solver):
        try:
            if solver not in ['picard', 'single_newton', 'newton']:
                raise ValueError("Unrecognized solver.")
        except ValueError as err:
            print(err.args[0])
            sys.exit(-1)