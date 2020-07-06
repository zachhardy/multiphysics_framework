class SolverBase:
 
    def __init__(self, problem):
        self.problem = problem

    def old_physics_action(self):
        for physics in self.problem.physics:
            physics.OldPhysicsAction()

    def solve_system(self):
        raise NotImplementedError

    