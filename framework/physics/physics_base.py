#!/usr/bin/env python3

import numpy as np
import numpy.linalg as npla

class PhysicsBase:
    """ Template class for physics modules. """
    def __init__(self, problem, field, bcs, ics=None):
        problem.physics.append(self)
        self.problem = problem
        self.mesh = problem.mesh
        self.field = field
        self.materials = self.initialize_materials()
        self.sd = field.sd

        # Discritization info
        self.grid = field.grid
        self.n_nodes = field.n_nodes
        self.n_dofs = field.n_dofs

        # Set the beginning of the field dof range.
        field.dof_start = problem.n_dofs
        # Add the field to the problem, and update the
        # appropriate attributes.
        problem.fields.append(field)
        problem.n_fields += 1
        problem.n_dofs += field.n_dofs
        problem.u.resize(problem.n_dofs)
        # Set the end of the field dof range.
        field.dof_end = problem.n_dofs

        # Validate boundary and initial conditions.
        self.bcs = self._validate_bcs(bcs)
        self.ics = self._validate_ics(ics)

        # Booleans for coupling and nonlinearity.
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

    def old_physics_action(self):
        raise NotImplementedError

    def solve_system(self):
        raise NotImplementedError
        
    def initialize_materials(self):
        raise NotImplementedError

    def _validate_bcs(self, bcs):
        raise NotImplementedError

    def _validate_ics(self, ics):
        raise NotImplementedError

    

        
