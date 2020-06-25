#!/usr/bin/env python3

import numpy as np
import numpy.linalg as npla

class PhysicsBase:
  """ Template class for physics modules. """
  def __init__(self, problem, field, bcs, ics=None):
    problem.physics.append(self)

    # reference to problem
    self.problem = problem

    # refrence to mesh
    self.mesh = problem.mesh

    # reference to field
    self.field = field

    # reference to materials
    self.materials = self.InitializeMaterials()

    # reference to discretization
    self.sd = field.sd
    # discretization info
    self.grid = field.grid
    self.n_nodes = field.n_nodes
    self.n_dofs = field.n_dofs

    # field dof map start
    field.dof_start = problem.n_dofs
    # update problem info
    problem.fields.append(field)
    problem.n_fields += 1
    problem.n_dofs += field.n_dofs
    problem.u.resize(problem.n_dofs)
    # field dof map end
    field.dof_end = problem.n_dofs

    # boundary condition list
    self.bcs = self.ValidateBCs(bcs)
    self.ics = self.ValidateICs(ics)

    # booleans
    self.is_coupled = False
    self.is_nonlinear = True

  @property
  def u(self):
    """ Get the solution vector for this physics. """
    dofs = self.field.dofs
    return self.problem.u[dofs[0]:dofs[-1]+1]

  @property
  def u_old(self):
    """ Get the old solution vector for this physics. """
    dofs = self.field.dofs
    return self.problem.u_old[dofs[0]:dofs[-1]+1]

  @property
  def f_old(self):
    """ Get the old physics action. """
    return self.OldPhysicsAction()

  def OldPhysicsAction(self):
    raise NotImplementedError(
      "This method must be implemented in derived classes."
    )

  def SolveSteadyState(self):
    raise NotImplementedError(
      "This method must be implemented in derived classes."
    )
    
  def SolveTimeStep(self):
    raise NotImplementedError(
      "This method must be implemented in derived classes."
    )
  def InitializeMaterials(self):
    raise NotImplementedError(
      "This method must be implemented in derived classes."
    )

  def ValidateBCs(self, bcs):
    raise NotImplementedError(
      "This method must be implemented in derived classes."
    )

  def ValidateICs(self, ics):
    raise NotImplementedError(
      "This method must be implemented in derived classes."
    )

  

    
