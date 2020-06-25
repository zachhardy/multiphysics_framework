#/usr/bin/env python3

class SolverBase:
  """ Base class for multiphysics solvers.

  Parameters
  ----------
  problem : Problem
  """
  def __init__(self, problem):
    self.problem = problem

  def old_physics_action(self):
    """ Compute the old physics action for each physics. """
    for physics in self.problem.physics:
      physics.OldPhysicsAction()

  def solve_system(self):
    raise NotImplementedError

  