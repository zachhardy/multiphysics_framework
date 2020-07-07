import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.extend(['../framework', '../modules'])
from problem import Problem
from mesh.mesh import Mesh1D
from bc import BC
from discretizations.fv.fv import FV
from discretizations.cfe.cfe import CFE
from mg_diffusion.mg_diffusion_base import MultiGroupDiffusion
from mg_diffusion.neutronics_material import NeutronicsMaterial

# Parameters
r_b = 6.0
n_cells = [50]
geom = 'sphere'
n_grps = 3
xs = {
  'Na': 0.05, 'v': [2000, 100, 2.2],
  'chi_p': [1, 0, 0], 'sig_t': [7.71, 50, 25.6],
  'nu_sig_f': [5.4, 60.8, 28], 
  'sig_r': [3.31, 36.2, 13.6],
  'sig_s': [[0, 1.46, 0],
            [0, 0, 0],
            [0, 0, 0]],
  'q': [0, 0, 0]
}

# Initialize problem
mesh = Mesh1D([0, r_b], n_cells, [0], geom=geom)
materials = [NeutronicsMaterial(material_id=0, **xs)]
problem = Problem(mesh, materials)

# Define physics
bcs = [BC('reflective', 0, np.zeros(n_grps)), 
       BC('zero_flux', 1, np.zeros(n_grps))]
ics = [
    lambda r: (r_b**2 - r**2)/r_b**2,
    lambda r: (r_b**2 - r**2)/r_b**2,
    lambda r: 0
]
fv = FV(mesh)
mgd = MultiGroupDiffusion(problem, fv, bcs, ics=ics, opt='full')

# # Problem execution
# problem.run_steady_state(verbosity=1, maxit=100)
# problem.run_transient(verbosity=1, method='tbdf2', tend=0.1)
mgd.compute_k_eigenvalue(verbosity=2)



plt.figure()
for group in mgd.groups:
    plt.plot(group.field.grid, group.field.u)
# plt.figure()
# plt.plot(mgd.grid, mgd.fission_rate)
plt.show()

