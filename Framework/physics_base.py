#!/usr/bin/env python3

from field import Field
from Discretizations.FV.fv import FV
from Discretizations.CFE.cfe import CFE

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

    # reference to discretization
    self.sd = field.sd
    # discretization info
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

  @property
  def u(self):
    """ The current solution of field.

    This routine grabs the solution of the field
    associated with this physics from the global 
    solution vector.
    """
    start = self.field.dof_start
    end = self.field.dof_end
    return self.problem.u[start:end]

  @property
  def u_old(self):
    """ The old solution of field.

    This routine grabs the solution of the field
    associated with this physics from the global 
    solution vector.
    """
    start = self.field.dof_start
    end = self.field.dof_end
    return self.problem.u_old[start:end]

  def SolveSteadyState(self):
    raise NotImplementedError(
      "This method must be implemented in "
      "derived classes."
    )
    
    
  def SolveTimeStep(self):
    raise NotImplementedError(
      "This method must be implemented in "
      "derived classes."
    )

  
  def SolvePhysics(self):
    raise NotImplementedError(
      "This method must be implemented in "
      "derived classes."
    )

  def ValidateBCs(self, bcs):
    raise NotImplementedError(
      "This method must be implemented in "
      "derived classes."
    )

  def ValidateICs(self, ics):
    raise NotImplementedError(
      "This method must be implemented in "
      "derived classes."
    )

  

    
