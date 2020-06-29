import numpy as np
import numpy.linalg as npla

class PhysicsBase:

    def __init__(self, problem):
        problem.physics.append(self)
        self.problem = problem
        self.mesh = problem.mesh
        self.materials = None
        self.discretization = None
        self.field = None

        # Discritization info
        self.grid = []
        self.n_nodes = 0
        self.n_dofs = 0

        # Validate boundary and initial conditions.
        self.bcs = None
        self.ics = None

        # Booleans for coupling and nonlinearity.
        self.is_coupled = False
        self.is_nonlinear = False
        self.is_transient = False

    def _register_field(self):
        # Add the field to the field stack.
        self.problem.fields.append(self.field)
        # Add the discretization to the physics
        self.discretization = self.field.discretization
        self.grid = self.discretization.grid
        self.n_nodes = self.discretization.n_nodes
        self.n_dofs = self.field.n_dofs
        # Set the beginning of the field dof range.
        self.field.dof_start = self.problem.n_dofs
        # Add the field to the problem, and update the
        # appropriate attributes.
        self.problem.fields.append(self.field)
        self.problem.n_fields += 1
        self.problem.n_dofs += self.field.n_dofs
        self.problem.u.resize(self.problem.n_dofs)
        # Set the end of the field dof range.
        self.field.dof_end = self.problem.n_dofs

    def recompute_old_physics_action(self):
        raise NotImplementedError

    def solve_system(self):
        raise NotImplementedError
        
    def _parse_materials(self, material_type):
        materials = []
        for material in self.problem.materials:
            if material.material_type == material_type:
                materials += [material]
        materials.sort(key=lambda x: x.material_id)
        return materials
