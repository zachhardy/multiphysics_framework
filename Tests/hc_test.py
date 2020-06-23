#!/usr/bin/env python3

import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.extend(['../Framework', '../Modules'])
from problem import Problem
from Mesh.mesh import Mesh1D
from field import Field
from material import HeatConductionMaterial
from material import NeutronicsMaterial
from bc import BC
from Solvers.operator_splitting import OperatorSplitting
from HeatConduction.cfe_hc import CFE_HeatConduction

def k(T):
    return 1.5 + (2510 / (215 + T))

### Mesh
# parameters
r_b = 0.45 # domain width
n_cells = 100 # number of cells
# create mesh
mesh = Mesh1D([0., r_b], [n_cells], [0], geom='cylinder')

### Materials
# properties
props = {'k': k, 'q': 3e4}
materials = [HeatConductionMaterial(**props)]

### Problem
problem = Problem(mesh, materials)

### Physics
bc_L = BC('neumann', 0, 0.)
bc_R = BC('dirichlet', 1, 300.)
bcs = [bc_L, bc_R]
hc = CFE_HeatConduction(problem, bcs)

### Solver
solver = OperatorSplitting(problem)

problem.RunSteadyState(verbosity=2)

plt.plot(hc.field.grid, hc.u)
plt.show()
