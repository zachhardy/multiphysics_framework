import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.extend(['../framework', '../modules'])
from problem import Problem
from mesh.mesh import Mesh1D
from bc import BC
from heat_conduction.cfe_hc import CFE_HeatConduction
from heat_conduction.hc_material import HeatConductionMaterial

def k(T):
    return 1.5 + (2510 / (215 + T))

### Mesh
# parameters
r_b = 0.45 # domain width
n_cells = 100 # number of cells
# create mesh
mesh = Mesh1D([0., r_b], [n_cells], [0], geom='slab')

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

problem.run_steady_state(verbosity=2)

plt.plot(hc.field.grid, hc.u)
plt.show()
