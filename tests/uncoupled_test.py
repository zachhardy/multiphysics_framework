#!/usr/bin/env python3

import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.extend(['../framework', '../modules'])
from problem import Problem
from mesh.mesh import Mesh1D
from bc import BC
from discretizations.cfe.cfe import CFE
from discretizations.fv.fv import FV
from heat_conduction.hc_material import HeatConductionMaterial
from heat_conduction.heat_conduction import HeatConduction
from mg_diffusion.mg_diffusion import MultiGroupDiffusion
from mg_diffusion.neutronics_material import NeutronicsMaterial

def k(T):
    return 1.5 + (2510 / (215 + T))

### Mesh
mesh = Mesh1D([0, 0.45], [50], [0], geom='slab')

### Materials
# neutronics
xs = {'D': [2], 'sig_r': [0.5], 'q': [10.]}
materials = [NeutronicsMaterial(**xs)]
# heat conduction
hc_properties = {'k': k, 'q': 3e4}
materials += [HeatConductionMaterial(**hc_properties)]

### Problem
problem = Problem(mesh, materials)

### Physics
# neutronics
phi_bcs = [BC('robin', 0, [0]), BC('robin', 1, [0])]
cfe = CFE(mesh)
mgd = MultiGroupDiffusion(problem, cfe, phi_bcs, maxit=1000, tol=1e-7)
# conduction
T_bcs = [BC('dirichlet', 0, 300), BC('dirichlet', 1, 300.)]
hc = HeatConduction(problem, cfe, T_bcs)

### Run
problem.run_steady_state()

for field in problem.fields:
    plt.figure()
    plt.plot(field.grid, field.u)
    plt.title(field.name)
plt.show()
