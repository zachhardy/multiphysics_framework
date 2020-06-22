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

  def SolveSteadyState(self):
    """ Run a steady state problem. """
    # shorthand
    tol = self.problem.tol
    maxit = self.problem.maxit
    # nonlinear iterations
    u_ell = np.copy(self.u)
    diff, nit = 1., 0 # book-keeping
    while diff > tol and nit < maxit:
      # iteration through physics
      for physic in self.problem.physics:
        physic.SolveSteadyState()

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

  def SolveTimeStep(self, time, dt):
    """ Run a time step of a transient problem. """
    # shorthand
    tol = self.problem.tol
    maxit = self.problem.maxit

    # nonlinear iteration loop
    u_ell = np.copy(self.u)
    diff, nit = 1., 0 # book-keeping
    while diff > tol and nit < maxit:
      # loop through physics
      for physic in self.problem.physics:
        physic.SolveTimeStep(time, dt)

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
    