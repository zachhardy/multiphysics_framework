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
    self.u = problem.u    

  def SolveSystem(self, *args):
    """ Run a steady state problem. 
    
    Parameters
    ----------
    args : tuple
      Inputs for transient systems. The ordering
      should be time, dt, method, and u_half.
      The first three are mandatory and the last
      is defaulted to None.
    """
    # shorthand
    tol = self.problem.tol
    maxit = self.problem.maxit
    # nonlinear iterations
    u_ell = np.copy(self.u)
    diff, nit = 1., 0 # book-keeping
    while diff > tol and nit < maxit:
      # iteration through physics
      for physic in self.problem.physics:
        physic.SolveSystem(*args)

      # compute error, increment step
      diff = np.linalg.norm(self.u-u_ell, ord=2)
      nit += 1

      # print-out
      if self.problem.verbosity > 0:
        msg = "Nonlinear Iteration {}".format(nit)
        msg = "\n".join(['', msg, "-"*len(msg), ''])
        msg += "Absolute Diff:\t{:.3e}".format(diff)
        print(msg)

      # Reset nonlinear iteration vector
      u_ell[:] = self.u
