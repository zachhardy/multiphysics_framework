import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.extend(['../framework', '../modules'])
from problem import Problem
from mesh.mesh import Mesh1D
from discretizations.cfe.cfe import CFE
from discretizations.fv.fv import FV
from bc import BC
from mg_diffusion.neutronics_source import NeutronicsSource
from mg_diffusion.neutronics_material import NeutronicsMaterial
from mg_diffusion.mg_diffusion_base import MultiGroupDiffusion

problem_type = sys.argv[1]

if problem_type == '0':
  mesh = Mesh1D([0, 100], [100], geom='slab')
  sd = CFE(mesh)
  materials = [NeutronicsMaterial(D=[2.], sig_r=[0.5], v=[100.])]
  sources = [NeutronicsSource(q=[10.])]
  bcs = [BC('dirichlet', 0, [5.]), BC('dirichlet', 1, [5.])]
  ics = [lambda r: 1.0]

elif problem_type == '1':    
  mesh = Mesh1D([0, 5, 13], [20, 60], [0, 1], [0, 1], geom='slab')
  sd = CFE(mesh)
  materials = [
    NeutronicsMaterial(material_id=0, D=[1.3], sig_r=[1.4], v=[100.]),
    NeutronicsMaterial(material_id=1, D=[43], sig_r=[0.4], v=[100.])
  ]
  sources = [
    NeutronicsSource(source_id=0, q=[5.2]), 
    NeutronicsSource(source_id=1, q=[3.2])
  ]
  bcs = [BC('dirichlet', 0, [1.]), BC('robin', 1, [0.])]
  ics = [lambda r: 1.0]

elif problem_type == '2':
  mesh = Mesh1D([0, 1.0], [10], geom='slab')
  sd = FV(mesh)
  xs = {
    'sig_t': [0.83], 'sig_r': [0.125], 
    'nu_sig_f': [0.125], 'v': [1.]
  }
  precursors = {'decay_const': [0.1], 'beta': [600e-5]}
  materials = [
    NeutronicsMaterial(material_id=0, **xs, **precursors)
  ]
  sources = [NeutronicsSource(source_id=0, q=[0.0])]
  bcs = [BC('reflective', 0), BC('reflective', 1)]
  ics = [lambda r: 1.0, lambda r: 600e-5*0.125/0.1]


problem = Problem(mesh, materials, sources)
mgd = MultiGroupDiffusion(
  problem, sd, bcs, ics, solve_opt='gw', sub_precursors=True,
  tol=1e-8, maxit=500
)

if sys.argv[2] == '0':
  mgd.compute_k_eigenvalue(verbosity=1)
elif sys.argv[2] == '1':
  problem.run_steady_state()
elif sys.argv[2] == '2':
  problem.run_transient(
    tend=1, dt=0.1, method='tbdf2', verbosity=1
  )

if sys.argv[2] in ['1', '2']:
  # plt.figure()
  # plt.plot(mgd.fission_rate.grid, mgd.fission_rate.u)
  for field in problem.fields:
    plt.figure()
    plt.plot(field.grid, field.u, label=field.name)
    plt.legend()
  plt.show()
