import sys
import numpy as np
import numpy.linalg as npla
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve

from physics.time_stepper import TimeStepperMixin
from physics.physics_base import PhysicsBase
from .neutronics_material import NeutronicsMaterial
from .neutronics_source import NeutronicsSource
from physics.field import Field
from .group import Group
from .precursor import Precursor

valid_fv_bcs = ['reflective', 'marshak', 'vacuum',
                'source', 'zero_flux']
valid_solve_options = ['full', 'group']

class MultiGroupDiffusion(TimeStepperMixin, PhysicsBase):

  mtype = NeutronicsMaterial.material_type
  stype = NeutronicsSource.source_type
  
  def __init__(self, problem, discretization, bcs, 
               ics=None, **kwargs):

    super().__init__(problem)

    # Materials and sources
    self.materials = self._parse_materials(self.mtype)
    self.sources = self._parse_sources(self.stype)
    self._validate_materials_and_sources()

    # Discretization information
    if discretization.dtype != 'fv':
      msg = "Only finite volume discretizations permitted."
      raise TypeError(msg)
    self.discretization = discretization
    self.bcs = self._validate_bcs(bcs)
    self.ics = self._validate_ics(ics)

    # Keyword arguments
    self.use_precursors = True if self.n_precursors>0 else False
    self.is_nonlinear = False
    self.tol = 1e-8
    self.maxit = 250
    self.solve_opt = 'full'
    self._parse_kwargs(kwargs)

    # Groups and precursors
    self.groups, self.precursors = [], []
    for g in range(self.n_grps):
      gname = 'scalar_flux_g{}'.format(g+1)
      field = Field(gname, problem, discretization)
      self._register_field(field)
      self.groups += [Group(self, field, g)]
    if self.use_precursors:
      for material in self.materials:
        if material.n_precursors > 0:
          imat = material.material_id
          for j in range(material.n_precursors):
            pname = 'precursor_j{}_imat{}'.format(j, imat)
            field = Field(pname, problem, discretization)
            self._register_field(field)
            self.precursors += [Precursor(self, field, imat, j)]
    
    # Fission rate
    self.fission_rate = np.zeros(self.n_nodes)
    self.fission_rate_old = np.zeros(self.n_nodes)

    # Full solvers
    if self.solve_opt == 'full':
      self.A = None
      self.M = None

      # Initialize
      self.assemble_physics_matrix()
      self.assemble_mass_matrix()

  def solve_system(self, opt=0):
    if self.solve_opt == 'full':
      if self.problem.is_transient:
        self.assemble_rhs(opt)
        self.solve_time_step(opt)
        if self.use_precursors:
          self.update_precursors(opt)
    else:
      converged = False
      for nit in range(self.maxit):        
        diff = 0
        if self.problem.is_transient:
          self.assemble_rhs(opt)
          for group in self.groups:
            group.solve_time_step(opt)
          
        for group in self.groups:
          diff += npla.norm(group.u-group.u_ell, ord=2)
          group.u_ell[:] = group.u
        if self.use_precursors:
          self.update_precursors(opt)
          for precursor in self.precursors:
            diff += npla.norm(precursor.u-precursor.u_ell, ord=2)
            precursor.u_ell[:] = precursor.u

        # Check convergence
        if diff < self.tol:
          converged = True
          break

      # Iteration summary
      if converged:
        if self.problem.verbosity > 0:
          print("\n*** Diffusion Solver Converged ***")
          print("Iterations: {}".format(nit))
          print("Final Error: {:.3e}".format(diff))
      else:
        if self.problem.verbosity > 0:
          print("\n*** WARNING: DID NOT CONVERGE. ***")

    for group in self.groups:
      group.u_ell[:] = group.u
    if self.use_precursors:
      for precursor in self.precursors:
        precursor.u_ell[:] = precursor.u
        
  def assemble_physics_matrix(self, opt=0):
    if (self.is_nonlinear or self.A is None
        or self.method=='tbdf2' and self.use_precursors):
      self.A = lil_matrix((self.n_dofs, self.n_dofs))
      for igroup in self.groups:
        i = igroup.g
        istart, iend = i*self.n_nodes, (i+1)*self.n_nodes
        igroup.assemble_physics_matrix()
        self.A[istart:iend, istart:iend] += igroup.A
 
        for jgroup in self.groups:
          j = jgroup.g
          jstart, jend = j*self.n_nodes, (j+1)*self.n_nodes
          self.A[istart:iend, jstart:jend] += (
            igroup.assemble_cross_group_matrix(jgroup, opt)
          )
      self.A = self.A.tocsr()
  
  def assemble_mass_matrix(self):
    if self.M is None:
      self.M = lil_matrix((self.n_dofs, self.n_dofs))
      for group in self.groups:
        i = group.g
        istart, iend = i*self.n_nodes, (i+1)*self.n_nodes
        group.assemble_mass_matrix()
        self.M[istart:iend, istart:iend] += group.M
      self.M = self.M.tocsr()

  def assemble_rhs(self, old=False, opt=0):
    for group in self.groups:
      group.assemble_rhs(old, opt)
      
  def assemble_forcing(self, time=0.0):
    for group in self.groups:
      group.assemble_forcing(time)

  def assemble_old_physics_action(self):
    self.compute_fission_rate(old=True)
    self.assemble_rhs(old=True)
  
  def apply_bcs(self, matrix=None, vector=None):
    for group in self.groups:
      offset = group.g*group.n_dofs
      group.apply_bcs(matrix, vector, offset)

  def update_precursors(self, opt=0):
    self.compute_fission_rate(old=False)
    for precursor in self.precursors:
      precursor.update_precursor(opt)

  def compute_fission_rate(self, old=False):
    FR = self.fission_rate_old if old else self.fission_rate
    FR[:] = 0.0
    for cell in self.mesh.cells:
      view = self.discretization.cell_views[cell.id]
      dof = view.dofs[0]
      material = self.materials[cell.imat]
      
      # If fissile material, compute fission rate
      if hasattr(material, 'nu_sig_f'):
        for group in self.groups:
          u_g = group.u_old if old else group.u
          nu_sig_f = material.nu_sig_f[group.g]
          FR[dof] += nu_sig_f * u_g[dof]

  @property
  def f(self):
    return np.block([g.f for g in self.groups])

  @property
  def f_old(self):
    return np.block([g.f_old for g in self.groups])

  @property
  def b(self):
    return np.block([g.b for g in self.groups])

  @property
  def u(self):
    start = min([g.field.dof_start for g in self.groups])
    end = max([g.field.dof_end for g in self.groups])
    return self.problem.u[start:end]

  @property
  def u_ell(self):
    start = min([g.field.dof_start for g in self.groups])
    end = max([g.field.dof_end for g in self.groups])
    return self.problem.u_ell[start:end]
  
  @property
  def u_half(self):
    start = min([g.field.dof_start for g in self.groups])
    end = max([g.field.dof_end for g in self.groups])
    return self.problem.u_half[start:end]

  @property
  def u_old(self):
    start = min([g.field.dof_start for g in self.groups])
    end = max([g.field.dof_end for g in self.groups])
    return self.problem.u_old[start:end]

  @property
  def power(self):
    power = 0.0
    self.compute_fission_rate()
    for cell in self.mesh.cells:
      power += self.fission_rate[cell.id] * cell.volume
    return power

  @property
  def grid(self):
    return self.discretization.grid

  @property
  def n_dofs(self):
    return self.n_grps * self.n_nodes

  @property
  def n_nodes(self):
    return self.discretization.n_nodes

  @property
  def n_grps(self):
    return self.materials[0].n_grps
  
  @property
  def n_precursors(self):
    return sum([mat.n_precursors for mat in self.materials])

  @property
  def method(self):
    return self.problem.method
  
  @property
  def dt(self):
    dt = self.problem.dt
    return 0.5*dt if self.method=='tbdf2' else dt

  @property
  def time(self):
    return self.problem.time

  @staticmethod
  def _validate_bcs(bcs):
    for bc in bcs:
      if bc.boundary_kind not in valid_fv_bcs:
        msg = "\nApproved BCs for "
        msg += "multigroup diffusion are:\n"
        for kind in valid_fv_bcs:
          msg += "--- {}\n".format(kind)
        raise ValueError(msg)
    return bcs

  @staticmethod
  def _validate_ics(ics):
    if ics is None:
      return ics
    for ic in ics:
      assert callable(ic), (
        "All initial conditions must be callable."
      )
    return ics
    
  @staticmethod
  def _validate_materials(materials):
    n_grps = materials[0].n_grps
    if len(materials) > 1:
      for material in materials[1:]:
        assert material.n_grps==n_grps, (
          "All materials must have the same group structure."
        )
    return materials

  @staticmethod
  def _validate_sources(sources):
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

  def _parse_kwargs(self, kwargs):
    for key, val in kwargs.items():
      if key == 'use_precursors':
        self.use_precursors = val
      elif key == 'has_feedback':
        self.is_nonlinear = val
      elif key == 'solve_opt':
        self.solve_opt = val
      elif key == 'tol':
        self.tol = val
      elif key == 'maxit':
        self.maxit = val
