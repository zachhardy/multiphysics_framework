#!/usr/bin/env python3

import sys
import numpy as np
from Discretizations.FV.fv import FV
from Discretizations.CFE.cfe import CFE

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
    def __init__(self, name, mesh, discretization, components=1):
        # Field name
        self.name = name

        # Discretization information
        self.mesh = mesh
        self.sd = discretization
        self.grid = discretization.grid
        self.n_nodes = discretization.n_nodes
        
        # Compnents information
        self.components = components
        self.n_dofs = components * self.n_nodes

        # Global DoF information
        self.dof_start = 0
        self.dof_end = 0
