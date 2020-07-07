import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.extend(['../framework', '../modules'])
from problem import Problem
from mesh.mesh import Mesh1D
from bc import BC
from discretizations.cfe.cfe import CFE
from heat_conduction.heat_conduction import HeatConduction
from heat_conduction.hc_material import HeatConductionMaterial
from heat_conduction.hc_source import HeatConductionSource

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
props = {'k': k}
src = {'q': 3e4}
materials = [HeatConductionMaterial(**props)]
sources = [HeatConductionSource(**src)]

### Problem
problem = Problem(mesh, materials, sources)

### Physics
bcs = [BC('neumann', 0, 0.), BC('dirichlet', 1, 300.)]
cfe = CFE(mesh)
hc = HeatConduction(problem, cfe, bcs)

### Execute
problem.run_steady_state(verbosity=2)

for field in problem.fields:
    plt.plot(field.grid, field.u)
plt.show()