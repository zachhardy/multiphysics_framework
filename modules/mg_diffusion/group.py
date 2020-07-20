import sys
import numpy as np
from scipy.sparse import lil_matrix

from physics.time_stepper import TimeStepperMixin

class Group(TimeStepperMixin):

  def __init__(self, mgd, field, g):
    
    self.mgd = mgd
    self.field = field
    self.g = g

    # Initialize system
    self.A = None
    self.M = None
    self.b = np.zeros(self.n_dofs)
    self.f = np.zeros(self.n_dofs)
    self.f_old = np.zeros(self.n_dofs)

    # Initialize matrices
    self.assemble_physics_matrix()
    self.assemble_mass_matrix()

  def assemble_physics_matrix(self, opt=0):
    if self.is_nonlinear or self.A is None:
      self.A = lil_matrix((self.n_dofs, self.n_dofs))
      for cell in self.mesh.cells:
        view = self.discretization.cell_views[cell.id]
        dof = view.dofs[0]
        # Geometric properties
        V = cell.volume
        # Material properties
        material = self.materials[cell.imat]
        sig_r = material.sig_r[self.g]
        D = material.D[self.g]
        
        # Contribute removal term
        self.A[dof,dof] += sig_r * V

        # Contribute interior diffusion term
        width = cell.width[0]
        for face in cell.faces:
          if face.flag == 0:
            # Geometric properties
            A = face.area
            # Neighbor information
            nbr_cell = self.mesh.cells[face.neighbor]
            nbr_width = nbr_cell.width[0]
            nbr_view = self.discretization.cell_views[nbr_cell.id]
            nbr_dof = nbr_view.dofs[0]
            # Neighbor material properties
            nbr_material = self.materials[nbr_cell.imat]
            nbr_D = nbr_material.D[self.g]
            # Effective edge quantities
            eff_width = 0.5*(width + nbr_width)
            eff_D = 2*eff_width / (width/D + nbr_width/nbr_D)
            self.A[dof,dof] += A * eff_D/eff_width
            self.A[dof,nbr_dof] -= A * eff_D/eff_width
      self.A = self.A.tocsr()

  def assemble_cross_group_matrix(self, group, opt=0):
    A = lil_matrix((self.n_dofs, self.n_dofs))
    for cell in self.mesh.cells:
      view = self.discretization.cell_views[cell.id]
      dof = view.dofs[0]
      # Geometric properties
      V = cell.volume
      # Material properties
      material = self.materials[cell.imat]
      beta_total = material.beta_total
      beta_total = beta_total if self.use_precursors else 0.0
      nu_sig_f, chi_p = 0.0, 0.0
      if hasattr(material, 'nu_sig_f'):
        chi_p = material.chi_p[self.g]
        nu_sig_f = material.nu_sig_f[group.g]
      sig_s = 0.0
      if hasattr(material, 'sig_s'):
        sig_s = material.sig_s[group.g][self.g]

      if chi_p*nu_sig_f != 0.0 or sig_s != 0.0:
        A[dof,dof] -= (
          ((1-beta_total) * chi_p * nu_sig_f + sig_s) * V
        )

      if self.use_precursors:
        for precursor in self.precursors:
          if cell.imat == precursor.imat:
            # Material properties
            chi_d = material.chi_d[precursor.j][self.g]
            decay_const = material.decay_const[precursor.j]
            beta = material.beta[precursor.j]
            
            if chi_d*nu_sig_f != 0.0:
              if self.method=='bwd_euler':
                coef = 1 / (1 + decay_const*self.dt)
              elif self.method=='cn' or self.method=='tbdf2' and opt==0:
                coef = 0.5 / (1 + 0.5*decay_const*self.dt) 
              elif self.method=='tbdf2' and opt==1:
                coef = 2/3 / (1 + 2/3*decay_const*self.dt)

              A[dof,dof] -= chi_d * decay_const * coef * (
                beta * self.dt * nu_sig_f * V
              )
    return A.tocsr()

  def assemble_mass_matrix(self):
    if self.M is None:
      self.M = lil_matrix((self.n_dofs, self.n_dofs))
      for cell in self.mesh.cells:
        view = self.discretization.cell_views[cell.id]
        dof = view.dofs[0]
        # Geometric properties
        V = cell.volume
        # Material properties
        material = self.materials[cell.imat]
        v = material.v[self.g]

        # Contribute inverse velocity term
        self.M[dof,dof] += V / v
      self.M = self.M.tocsr()

  def assemble_forcing(self, time=0.0):
    self.b[:] = 0
    sd = self.discretization
    for cell in self.mesh.cells:
      view = sd.cell_views[cell.id]
      source = self.sources[cell.isrc]
      q = source.q[self.g]
      q = q(time) if callable(q) else q
      if q != 0:
        dof = view.dofs[0]
        self.b[dof] += q * cell.volume

  def assemble_rhs(self, old=False, opt=0):
    f = self.f_old if old else self.f
    f[:] = 0
    if old:
      f -= self.A @ self.u_old

    for cell in self.mesh.cells:
      view = self.discretization.cell_views[cell.id]
      dof = view.dofs[0]
      # Geometric properties
      V = cell.volume
      # Material properties
      material = self.materials[cell.imat]
      beta_total = material.beta_total
      beta_total = beta_total if self.use_precursors else 0.0

      # Contribute group-coupling terms
      if old or self.solve_opt == 'group':
        for group in self.groups:
          u = group.u_old[dof] if old else group.u_ell[dof]
          # Material properties
          chi_p, nu_sig_f = 0.0, 0.0
          if hasattr(material, 'nu_sig_f'):
            chi_p = material.chi_p[self.g]
            nu_sig_f = material.nu_sig_f[group.g]
          sig_s = 0.0
          if hasattr(material, 'sig_s'):
            sig_s = material.sig_s[group.g][self.g]

          if chi_p * nu_sig_f != 0.0 or sig_s != 0.0:
            f[dof] += (
              ((1 - beta_total) * chi_p * nu_sig_f + sig_s) * u * V
            )

      # Contribute delayed precursor terms
      if self.use_precursors:
        for precursor in self.precursors:
          if cell.imat == precursor.imat:
            # Material properties
            chi_d = material.chi_d[precursor.j][self.g]
            beta = material.beta[precursor.j]
            decay_const = material.decay_const[precursor.j]

            if chi_d != 0.0:
              coef = chi_d * decay_const * V
              Cj_old = precursor.u_old[dof]

              # Contribute old precursor contribution
              if old:
                f[dof] += coef * Cj_old

              # Contribute lagged precursor contribution
              # NOTE: This is a substitution from the precursor 
              # equation where the unknown, end time step precursor
              # concentration is represented as a function of the
              # old precursor concentration, the end time step
              # fission rate, and the old fission rate.
              else:    
                if self.solve_opt == 'group':
                  FR = self.fission_rate[dof]

                if self.method=='bwd_euler':
                  coef *= 1 / (1 + decay_const*self.dt)
                  f[dof] += coef * Cj_old
                  if self.solve_opt == 'group':
                    f[dof] += coef * beta * self.dt * FR

                elif self.method=='cn' or self.method=='tbdf2' and opt==0:
                  coef *= 1 / (1 + 0.5*decay_const*self.dt)
                  f[dof] += coef * (1 - 0.5*decay_const*self.dt) * Cj_old
                  if self.solve_opt == 'group':
                    FR_old = self.fission_rate_old[dof]
                    f[dof] += 0.5 * coef * beta * self.dt * (FR + FR_old)

                elif self.method=='tbdf2' and opt==1:
                  Cj_half = precursor.u_half[dof]
                  coef *= 1 / (1 + 2/3*decay_const*self.dt)
                  f[dof] += coef * (4*Cj_half - Cj_old) / 3
                  if self.solve_opt == 'group':
                    f[dof] += 2/3 * coef * beta * self.dt * FR

  def apply_bcs(self, matrix=None, vector=None, offset=0):
    assert matrix is not None or vector is not None, (
      "Either a matrix, vector, or both must be provided."
    )
    for cell in self.mesh.bndry_cells:
      view = self.discretization.cell_views[cell.id]
      for face in cell.faces:
        if face.flag > 0:
          bc = self.bcs[face.flag-1]

          if bc.boundary_kind != 'reflective':
            dof = view.dofs[0] + offset
            width = cell.width[0]
            material = self.materials[cell.imat]
            D = material.D[self.g]

            # Compute coefficient for the bc
            if bc.boundary_kind in ['source', 'zero_flux']:
              coef = 2*D / width
            elif bc.boundary_kind in ['marshak', 'vacuum']:
              coef = 2*D / (4*D + width)
            
            # Apply matrix bcs
            if matrix is not None:
              matrix[dof,dof] += face.area * coef

            # Apply vector bcs
            if vector is not None:
              if bc.boundary_kind in ['source', 'marshak']:
                val = bc.vals[self.g]
                vector[dof] += face.area * coef * val

  @property
  def u(self):
    return self.field.u

  @property
  def u_ell(self):
    return self.field.u_ell

  @property
  def u_half(self):
    return self.mgd.problem.u_half[self.field.dofs]

  @property
  def u_old(self):
    return self.field.u_old
  
  @property
  def fission_rate(self):
    return self.mgd.fission_rate

  @property
  def fission_rate_old(self):
    return self.mgd.fission_rate_old

  @property
  def materials(self):
    return self.mgd.materials

  @property
  def sources(self):
    return self.mgd.sources

  @property
  def groups(self):
    return self.mgd.groups

  @property
  def precursors(self):
    return self.mgd.precursors

  @property
  def bcs(self):
    return self.mgd.bcs

  @property
  def mesh(self):
    return self.field.mesh

  @property
  def discretization(self):
    return self.field.discretization

  @property
  def n_dofs(self):
    return self.field.n_dofs

  @property
  def n_nodes(self):
    return self.field.n_nodes

  @property
  def use_precursors(self):
    return self.mgd.use_precursors

  @property
  def is_nonlinear(self):
    return self.mgd.is_nonlinear

  @property
  def solve_opt(self):
    return self.mgd.solve_opt

  @property
  def method(self):
    return self.mgd.method

  @property
  def dt(self):
    return self.mgd.dt
    
  @property
  def time(self):
    return self.mgd.time