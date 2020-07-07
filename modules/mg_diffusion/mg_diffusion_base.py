import numpy as np
import numpy.linalg as npla
from scipy.sparse import lil_matrix
from scipy.sparse import bmat
from scipy.sparse.linalg import spsolve

from physics.field import Field
from physics.physics_base import PhysicsBase
from .neutronics_material import NeutronicsMaterial
from .neutronics_source import NeutronicsSource
from .group import Group

valid_fv_bcs = ['reflective', 'marshak', 'vacuum',
                'source', 'zero_flux']
valid_cfe_bcs = ['dirichlet', 'neumann', 'robin']
valid_solve_options = ['full', 'group']

def MultiGroupDiffusion(problem, discretization, bcs, ics=None, 
                        opt='gw', tol=1e-8, maxit=500):
  if opt == 'gw':
    from .mg_diffusion_gw import MultiGroupDiffusionGroupWise
    return MultiGroupDiffusionGroupWise(
      problem, discretization, bcs, ics, tol, maxit
    )
  elif opt == 'full':
    from .mg_diffusion_full import MultiGroupDiffusionFull
    return MultiGroupDiffusionFull(problem, discretization, bcs, ics)


class MultiGroupDiffusionBase(PhysicsBase):

  name = 'scalar_flux'
  material_type = NeutronicsMaterial.material_type
  source_type = NeutronicsSource.source_type
  n_grps = 0

  def __init__(self, problem, discretization, bcs, ics=None):
    super().__init__(problem)
    self.materials = self._parse_materials(self.material_type)
    self.sources = self._parse_sources(self.source_type)
    self._validate_materials_and_sources()
    self.discretization = discretization
    self.bcs = self._validate_bcs(bcs)
    self.ics = self._validate_ics(ics)
    self.n_grps = self.materials[0].n_grps

    # Initialize group opjects
    self.groups = []
    for g in range(self.n_grps):
      gname = self.name + '_g{}'.format(g+1)
      field = Field(gname, self.problem, self.discretization, 1)
      self._register_field(field)
      self.groups.append(Group(self, field, g))

    # # Initialize precursor objects
    # self.precursors = []
    # for material in self.materials:
    #   if material.n_precursors > 0:
        
  def compute_fission_rate(self):
    self.fission_rate[:] = 0
    for group in self.groups:
      group.compute_fission_rate(self.fission_rate)

  def compute_fission_power(self):
    fission_source = 0
    for group in self.groups:
      fission_source += group.compute_fission_power()
    return fission_source

  def _validate_materials(self, materials):
    n_grps = materials[0].n_grps
    if len(materials) > 1:
      for material in materials[1:]:
        assert material.n_grps==n_grps, (
          "All materials must have the same group structure."
        )
    return materials

  def _validate_sources(self, sources):
    n_grps = sources[0].n_grps
    if len(sources) > 1:
      for source in sources[1:]:
        assert source.n_grps==n_grps, (
          "All sources must have the same group structure."
        )
    return sources

  def _validate_materials_and_sources(self):
    if self.materials[0].n_grps != self.sources[0].n_grps:
      raise ValueError("Incompatible group structures.")

  def _validate_bcs(self, bcs):
    for bc in bcs:
      if self.discretization.dtype == 'fv':
        valid_bcs = valid_fv_bcs
      elif self.discretization.dtype == 'cfe':
        valid_bcs = valid_cfe_bcs
      dscrt = self.discretization.dtype
      if bc.boundary_kind not in valid_bcs:
        msg = "\nApproved BCs for {} ".format(dscrt)
        msg += "multigroup diffusion are:\n"
        for kind in valid_bcs:
          msg += "--- {}\n".format(kind)
        raise ValueError(msg)
    return bcs
  
  def _validate_ics(self, ics):
    if ics is None:
      return ics
    for ic in ics:
      assert callable(ic), (
        "All initial conditions must be callable."
      )
    return ics

  @staticmethod
  def print_k_iter_summary(nit, k_eff, k_error):
    msg = "\nIteration {}".format(nit)
    delim = '-'*len(msg)
    msg = '\n'.join(['', msg, delim])
    print(msg)
    print('k-eff:\t\t{:.3e}'.format(k_eff))
    print('k Error:\t{:.3e}'.format(k_error))

  @staticmethod
  def print_k_calc_summary(converged, nit, k_eff, k_error):
    if converged:
      print("\n*** Simulation converged in {} iterations ***".format(nit))
      print("Converged k:\t\t{:.5e}".format(k_eff))
    else:
      print("\n*** WARNING: Simulation did not converge ***")
      print("Unconverged k:\t\t{:.5e}".format(k_eff))
    print("Final k Error:\t\t{:.3e}".format(k_error))
