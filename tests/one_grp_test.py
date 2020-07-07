#!/usr/bin/env python3

import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.extend(['../framework', '../modules'])
from problem import Problem
from mesh.mesh import Mesh1D
from discretizations.cfe.cfe import CFE
from bc import BC
from mg_diffusion.neutronics_source import NeutronicsSource
from mg_diffusion.neutronics_material import NeutronicsMaterial
from mg_diffusion.mg_diffusion_base import MultiGroupDiffusion

mesh = Mesh1D([0, 5, 13], [20, 60], [0, 1], [0, 1], geom='slab')

xs0 = {'material_id': 0, 'D': [1.3], 'sig_r': [1.4]}
xs1 = {'material_id': 1, 'D': [43], 'sig_r': [0.4]}
materials = [NeutronicsMaterial(**xs0), NeutronicsMaterial(**xs1)]
sources = [NeutronicsSource(0, [5.2]), NeutronicsSource(1, [3.2])]

problem = Problem(mesh, materials, sources)

bcs = [BC('dirichlet', 0, [1.]), BC('robin', 1, [0.])]
cfe = CFE(mesh)
mgd = MultiGroupDiffusion(problem, cfe, bcs, opt='full')

problem.run_steady_state()

for field in problem.fields:
    plt.plot(field.grid, field.u)
plt.show()
