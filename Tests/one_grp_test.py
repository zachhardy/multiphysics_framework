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
from MGDiffusion.cfe_mg_diffusion import CFE_MultiGroupDiffusion

mesh = Mesh1D([0, 100], [100], [0], geom='slab')

xs = {'D': [2.], 'sig_r': [0.5], 'q': [10.]}
materials = [NeutronicsMaterial(**xs)]

problem = Problem(mesh, materials)

bcs = [BC('dirichlet', 0, [5.]), BC('dirichlet', 1, [5.])]
mgd = CFE_MultiGroupDiffusion(problem, 1, bcs)

OperatorSplitting(problem)

problem.RunSteadyState()

plt.plot(mgd.grid, mgd.u)
plt.show()
