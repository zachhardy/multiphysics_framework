#!/usr/bin/env python3

import sys
import numpy as np

class Field:
    """ Field function object.
    
    Parameters
    ----------
    name : str
        A name for this field function.
    mesh : MeshBase
        The mesh this field function is defined on.
    discretization : Discretization
        The discretization applied to this field
    components : int, optional
        The number of components this field has
        (default is 1).
    """
    def __init__(self, name, problem, discretization, components=1):
        self.name = name
        # Discretization information
        self.problem = problem
        self.mesh = problem.mesh
        self.discretization = discretization
        self.grid = discretization.grid
        self.n_nodes = discretization.n_nodes
        # Compnents information
        self.components = components
        self.n_dofs = components * self.n_nodes
        # Global DoF information
        self.dof_start = 0
        self.dof_end = 0

    @property
    def u(self):
        """ Get the solution vector for this physics. """
        return self.problem.u[self.dofs[0]:self.dofs[-1]+1]

    @property
    def u_old(self):
        """ Get the old solution vector for this physics. """
        return self.problem.u_old[self.dofs[0]:self.dofs[-1]+1]

    @property
    def dofs(self):
        """ Return the dofs for this field. """
        return list(range(self.dof_start, self.dof_end))
    
    def component_dofs(self, component=0):
        """ Return the dofs for a given component. """
        start = self.dof_start + component*self.n_nodes
        end = start + self.n_nodes
        return self.dofs[start:end]
