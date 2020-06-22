#/usr/bin/env python3

class SolverBase:
    """ Base class for multiphysics solvers.

    Parameters
    ----------
    problem : Problem
    """
    def __init__(self, problem):
        problem.solver = self
        self.problem = problem

    def SolveTimeStep(self):
        raise NotImplementedError(
            "This method must be implemented in "
            "derived classes."
        )

    def SolveSteadyState(self):
        raise NotImplementedError(
            "This method must be implemented in "
            "derived classes."
        )

    def SolvePhysics(self, physic, u_old, f_old):
        raise NotImplementedError(
            "This method must be implemented in "
            "derived classes."
        )

    