#!/usr/bin/env python3

import numpy as np
from .solver_base import SolverBase

class OperatorSplitting(SolverBase):
    """ Operator splitting solver.

    Parameters
    ----------
    problem : Problem
    """
    def __init__(self, problem):
        super().__init__(problem)
        # Nonlinear parameters
        self.tol = problem.tol
        self.maxit = problem.maxit  
        # Solution vectors  
        self.u = problem.u
        self.u_ell = np.copy(self.u)
        
    def solve_system(self, *args):
        """ Run a steady state problem. 
        
        Parameters
        ----------
        args : tuple
            Inputs for transient systems. The ordering
            should be time, dt, method, and u_half.
            The first three are mandatory and the last
            is defaulted to None.
        """
        # Initialize u_ell with current solution
        self.u_ell[:] = self.u
        # Nonlinear iterations
        diff, nit = 1., 0
        while diff > self.tol and nit < self.maxit:
            for physic in self.problem.physics:
                physic.solve_system(*args)
            diff = np.linalg.norm(self.u-self.u_ell, ord=2)
            self.u_ell[:] = self.u
            nit += 1
            
            if self.problem.verbosity > 0:
                msg = "Nonlinear Iteration {}".format(nit)
                msg = "\n".join(['', msg, "-"*len(msg), ''])
                msg += "Absolute Diff:\t{:.3e}".format(diff)
                print(msg)
