import sys
import numpy as np
import numpy.linalg as npla
from scipy.sparse import lil_matrix
from scipy.sparse import bmat
from scipy.sparse.linalg import spsolve

from physics.field import Field, AuxField
from discretizations.fv.fv import FV
from physics.physics_base import PhysicsBase
from .neutronics_material import NeutronicsMaterial
from .neutronics_source import NeutronicsSource
from .group import Group
from .precursor import Precursor

valid_fv_bcs = ['reflective', 'marshak', 'vacuum',
                'source', 'zero_flux']
valid_cfe_bcs = ['dirichlet', 'neumann', 'robin']
valid_solve_options = ['full', 'group']

def MultiGroupDiffusion(problem, discretization, bcs, ics=None, 
                        solve_opt='gw', **kwargs):
  if solve_opt == 'gw':
    from .mg_diffusion_gw import MultiGroupDiffusionGroupWise
    return MultiGroupDiffusionGroupWise(
      problem, discretization, bcs, ics, **kwargs
    )
  elif solve_opt == 'full':
    from .mg_diffusion_full import MultiGroupDiffusionFull
    return MultiGroupDiffusionFull(
      problem, discretization, bcs, ics, **kwargs
    )
  else:
    raise NotImplementedError


class MultiGroupDiffusionBase(PhysicsBase):

  name = 'scalar_flux'
  material_type = NeutronicsMaterial.material_type
  source_type = NeutronicsSource.source_type
  n_grps = 0

  def __init__(self, problem, discretization, bcs, ics=None, **kwargs):
    super().__init__(problem)
    self.materials = self._parse_materials(self.material_type)
    self.sources = self._parse_sources(self.source_type)
    self.discretization = discretization
    self.n_nodes = discretization.n_nodes
    self.grid = discretization.grid
    self.bcs = self._validate_bcs(bcs)
    self.ics = self._validate_ics(ics)

    # Groups and precursors initialization
    self.groups = []
    self.precursors = []
    self.n_grps = self.materials[0].n_grps
    self.n_precursors = self._count_precursors()
    
    # Precursor options
    self.use_precursors = True if self.n_precursors>0 else False
    if 'use_precursors' in kwargs:
      self.use_precursors = kwargs['use_precursors']
      assert isinstance(self.use_precursors, bool), (
        "sub_precursors option must be a boolean."
      )
    self.sub_precursors = True
    if 'sub_precursors' in kwargs:
      self.sub_precursors = kwargs['sub_precursors']
      assert isinstance(self.sub_precursors, bool), (
        "sub_precursors option must be a boolean."
      )

    # Initialize fields
    self.init_fields()
    if self.discretization.dtype == 'fv':
      self.fission_rate = AuxField(
        'fission_rate', problem, discretization
      )

  def compute_fission_rate(self, old=False):
    u = self.fission_rate.u_old if old else self.fission_rate.u
    u[:] = 0.0
    for group in self.groups:
      group.compute_fission_rate(u, old)

  def compute_fission_power(self):
    fission_source = 0
    for group in self.groups:
      fission_source += group.compute_fission_power()
    return fission_source

  def init_fields(self):
    # Init groups
    for g in range(self.n_grps):
      gname = self.name + '_g{}'.format(g+1)
      field = Field(gname, self.problem, self.discretization, 1)
      self._register_field(field)
      self.groups.append(Group(self, field, g))
    
    # Init precursors
    if self.use_precursors:
      for material in self.materials:
        if material.n_precursors > 0:
          imat = material.material_id
          for j in range(material.n_precursors):
            pname = "precursor_mat{}_{}".format(imat, j)
            field = Field(pname, self.problem, self.discretization, 1)
            self._register_field(field)
            self.precursors.append(Precursor(self, field, imat, j))    

  def _count_precursors(self):
    n_precursors = 0
    for material in self.materials:
        n_precursors += material.n_precursors
    return n_precursors

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
    assert self.materials[0].n_grps==n_grps, (
      "Material properties and sources must have the same "
      "group structure."
    )
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
