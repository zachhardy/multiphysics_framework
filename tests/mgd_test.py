import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.extend(['../framework', '../modules'])
from problem import Problem
from mesh.mesh import Mesh1D
from bc import BC
from discretizations.fv.fv import FV
from mgd.neutronics_material import NeutronicsMaterial
from mgd.neutronics_source import NeutronicsSource
from mgd.mg_diffusion import MultiGroupDiffusion


# Parameters
r_b = 6.0
n_cells = [50]
geom = 'sphere'
n_grps = 3
xs = {
  'material_id': 0,
  'Na': 0.05, 'v': [2000, 100, 2.2],
  'chi_p': [1, 0, 0], 'sig_t': [7.71, 50, 25.6],
  'nu_sig_f': [5.4, 60.8, 28], 
  'sig_r': [3.31, 36.2, 13.6],
  'sig_s': [[0, 1.46, 0],
            [0, 0, 0],
            [0, 0, 0]],
}
src = {'source_id': 0, 'q': np.zeros(n_grps)}

# Initialize problem
mesh = Mesh1D([0, r_b], n_cells, [0], geom=geom)
materials = [NeutronicsMaterial(**xs)]
sources = [NeutronicsSource(**src)]
problem = Problem(mesh, materials, sources)

# Define physics
bcs = [BC('reflective', 0, np.zeros(n_grps)), 
       BC('zero_flux', 1, np.zeros(n_grps))]
ics = [
    lambda r: (r_b**2 - r**2)/r_b**2,
    lambda r: (r_b**2 - r**2)/r_b**2,
    lambda r: 0
]
fv = FV(mesh)
mgd = MultiGroupDiffusion(
    problem, fv, bcs, ics=ics, solve_opt='group', tol=1e-12)

# # Problem execution
# problem.run_steady_state(verbosity=1, maxit=100)
problem.run_transient(verbosity=1, method='tbdf2', dt=1e-3, tend=1e-1)
# mgd.compute_k_eigenvalue(verbosity=1)

plt.figure()
for field in problem.fields:
    plt.plot(field.grid, field.u, label=field.name)
plt.legend()
plt.show()