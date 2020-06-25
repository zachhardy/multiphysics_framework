#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

class Problem:
  """ Class for managing a problem.

  This class is meant to solve general problems
  in an agnostic way. What this means is that it 
  depends on each physics module having similarly
  named routines to perform higher level tasks such
  as solving a time step, solving a steady state system,
  etc.

  Parameters
  ----------
  mesh : MeshBase
  fields : list of Field
  materials : list of MaterialBase
  """
  def __init__(self, mesh, materials):
    self.mesh = mesh
    self.materials = materials

    # Physics
    self.physics = []

    # Fields
    self.fields = []

    # Solver
    self.solver = None

    # General info
    self.n_dofs = 0
    self.n_fields = 0
    
    # Solution vector
    self.u = np.array([])

    # Nonlinear parameters
    self.tol = 1e-6
    self.maxit = 100

    # Time stepping parameters
    self.method = None

    # Verbosity
    self.verbosity = 0

  def RunSteadyState(self, tol=1e-6, maxit=100, verbosity=2):
    """ Run a steady state problem.

    Parameters
    ----------
    tol : float
      Nonlinear convergence tolerance. Default is 1e-6.
    maxit : int
      Maximum nonlinear iteration. Default is 100.
    verbosity : int
      Level of screen printout. Default is 2.
    """
    self.verbosity = verbosity
    self.is_transient = False
    # nonlinear parameters
    self.tol = tol
    self.maxit = maxit
    # run problem
    self.solver.SolveSystem()

  def RunTransient(self, t0=0, tend=0.1, dt=2e-3, 
           method='tbdf2', tol=1e-6, maxit=100,
           verbosity=0):
    """ Run a steady state problem.

    Parameters
    ----------
    t0 : float
      Simulation start time. Default is 0.
    tend : float
      Simulation end time. Default is 0.1.
    dt : float
      Simulation time step. Default is 2e-3.
    method : str
      The time stepping method. The options are
      'fwd_euler', 'bwd_euler', 'cn', and 'tbdf2'.
      Default is 'tbdf2'.
    tol : float
      Nonlinear convergence tolerance. Default is 1e-6.
    maxit : int
      Maximum nonlinear iteration. Default is 100.
    verbosity : int
      Level of screen printout. Default is 2.
    """
    self.verbosity = verbosity
    self.is_transient = True
    # store nonlinear parameters
    self.tol = tol
    self.maxit = maxit
    # store time parameters
    time = t0
    dt = dt
    step = 0

    # additional storage for half step
    if method == 'tbdf2':
      u_half = np.copy(self.u)
    
    # initial conditions
    self.EvaluateICs()
    self.u_old = np.copy(self.u)

    # time stepping          
    while time < tend:
      step += 1
      # print time step info
      if self.verbosity > 0:
        msg = "* Time Step {} *".format(step)
        msg = "\n".join(["\n", "*"*len(msg), msg, "*"*len(msg)])
        msg += "\ntime:\t{:.5f} micro-sec".format(time+dt)
        msg += "\ndt:\t{:.5f} micro-sec".format(dt)
        msg += "\n"+"="*25
        print(msg)

      # solve the time step
      if method != 'tbdf2':
        self.solver.SolveSystem(time, dt, method)
      else:
        # half step crank nicholson
        self.solver.SolveSystem(time, dt/2, 'cn')
        u_half[:] = self.u # half step result
        # half step BDF2
        self.solver.SolveSystem(time+dt/2, dt/2, 'bdf2', u_half)

      # book-keeping 
      time += dt
      if time + dt > tend:
        dt = tend - time
      self.u_old[:] = self.u

  def EvaluateICs(self):
    """ Evaluate initial conditions. """
    for physic in self.physics:
      field = physic.field
      for c in range(field.components):
        dofs = field.ComponentDoFs(c)
        grid = field.grid.ravel()
        self.u[dofs] = physic.ics[c](grid)
    self.u_old = np.copy(self.u)

      
    