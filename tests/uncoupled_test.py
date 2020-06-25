#!/usr/bin/env python3

import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.extend(['../framework', '../modules'])
from problem import Problem
from mesh.mesh import Mesh1D
from field import Field
from material import HeatConductionMaterial
from material import NeutronicsMaterial
from bc import BC
from heat_conduction.cfe_hc import CFE_HeatConduction
from mg_diffusion.cfe_mg_diffusion import CFE_MultiGroupDiffusion

def k(T):
    return 1.5 + (2510 / (215 + T))

### Mesh
mesh = Mesh1D([0, 0.45], [50], [0], geom='slab')

### Materials
# neutronics
xs = {'D': [2], 'sig_r': [0.5], 'q': [10.]}
materials = [NeutronicsMaterial(**xs)]
# heat conduction
props = {'k': k, 'q': 3e4}
materials += [HeatConductionMaterial(**props)]

### Problem
problem = Problem(mesh, materials)

### Physics
# neutronics
phi_bcs = [BC('neumann', 0, [0]), BC('robin', 1, [[0.5],[2.]])]
mgd = CFE_MultiGroupDiffusion(problem, 1, phi_bcs)
# conduction
T_bcs = [BC('neumann', 0, 0), BC('dirichlet', 1, 300.)]
hc = CFE_HeatConduction(problem, T_bcs)

### Run
problem.RunSteadyState()

plt.figure(0)
plt.plot(hc.field.grid, hc.u)

plt.figure(1)
plt.plot(mgd.field.grid, mgd.u)

plt.show()



