#!/usr/bin/env python3

import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.extend(['../framework', '../modules'])
from problem import Problem
from mesh.mesh import Mesh1D
from bc import BC
from mg_diffusion.fv_mg_diffusion \
    import FV_MultiGroupDiffusion
from mg_diffusion.neutronics_material \
    import NeutronicsMaterial


### Mesh
# parameters
r_b = 6.0 # domain width (cm)
n_cells = 40 # number of cells
geom = 'sphere'
# create mesh
mesh = Mesh1D([0, r_b], [n_cells], [0], geom=geom)

### Materials
G = 3
# properties
xs = {
  'Na': 0.05, 'v': [2000, 100, 2.2],
  'chi': [1, 0, 0], 'sig_t': [7.71, 50, 25.6],
  'nu_sig_f': [5.4, 60.8, 28], 
  'sig_r': [3.31, 36.2, 13.6],
  'sig_s': [[0, 0, 0],
        [1.46, 0, 0],
        [0, 0, 0]],
  'q': [0, 0, 0]
}
materials = [NeutronicsMaterial(**xs)]

### Problem
problem = Problem(mesh, materials)

### Physics
bc_L = BC('reflective', 0, [0., 0., 0.])
bc_R = BC('marshak', 1, [0., 0., 0.])
bcs = [bc_L, bc_R]
ics = [lambda r: (r_b**2 - r**2) / r_b**2,
       lambda r: (r_b**2 - r**2) / r_b**2,
       lambda r: 0]
mgd = FV_MultiGroupDiffusion(problem, G, bcs, ics)

k, u = mgd.solve_k_eigen_problem(verbosity=2)

# problem.run_transient(verbosity=1, method='cn')

for g in range(G):
  beg = g * mgd.n_nodes
  end = beg + mgd.n_nodes
  plt.plot(mgd.grid, u[beg:end])
plt.show()
