import numpy as np
from scipy.sparse import lil_matrix
from physics.time_stepper import TimeStepperMixin

class Group(TimeStepperMixin):

  def __init__(self, mgd, field, g):
    # Objects
    self.problem = mgd.problem
    self.mgd = mgd
    self.field = field
    self.mesh = mgd.mesh
    self.materials = mgd.materials
    self.sources = mgd.sources
    self.discretization = field.discretization
    self.bcs = mgd.bcs

    self.g = g # group number
    self.n_dofs = field.n_dofs
    
    # Precursor information
    self.use_precursors = mgd.use_precursors
    self.sub_precursors = mgd.sub_precursors

    # Boolean flags
    self.is_nonlinear = mgd.is_nonlinear
    self.is_coupled = mgd.is_coupled

    # Initialize discrete system
    self.A_ = None
    self.M_ = None
    self.b = np.zeros(self.n_dofs)
    self.f = np.zeros(self.n_dofs)
    self.f_old = np.zeros(self.n_dofs)

  @property
  def A(self):
    """ Assemble the diffusion + removal operator. """
    if self.is_nonlinear or self.A_ is None:
      return self.assemble_within_group_matrix()
    return self.A_

  @property
  def M(self):
    """ Assemble the velocity mass matrix. """
    if self.M_ is None:
      return self.assemble_mass_matrix()
    return self.M_
  
  def assemble_within_group_matrix(self):
    sd = self.discretization
    self.A_ = lil_matrix(tuple([self.n_dofs]*2))
    for cell in self.mesh.cells:
      view = sd.cell_views[cell.id]
      material = self.materials[cell.imat]
      # Material properties
      sig_r = material.sig_r[self.g]
      D = material.D[self.g]
    
      if sd.dtype == 'fv':
        row = view.cell_dof_map()

        # Removal
        self.A_[row,row] += sig_r * cell.volume

        # Interior diffusion
        width = cell.width[0]
        for face in cell.faces:
          if face.flag == 0:
            # Neighbor information
            nbr_cell = self.mesh.cells[face.neighbor]
            nbr_width = nbr_cell.width[0]
            nbr_view = sd.cell_views[nbr_cell.id]
            nbr_material = self.materials[nbr_cell.imat]
            nbr_D = nbr_material.D[self.g]
            col = nbr_view.cell_dof_map()
            # Compute the effective edge quantities
            eff_width = 0.5*(width + nbr_width)
            eff_D = 2*eff_width / (width/D + nbr_width/nbr_D)
            # Add to the matrix
            self.A[row,row] += face.area * eff_D/eff_width
            self.A[row,col] -= face.area * eff_D/eff_width

    self.A_ = self.A_.tocsr()
    return self.A_

  def assemble_mass_matrix(self):
    sd = self.discretization
    self.M_ = lil_matrix(tuple([self.n_dofs]*2))
    for cell in self.mesh.cells:
      view = sd.cell_views[cell.id]
      material = self.materials[cell.imat]
      v = material.v[self.g]

      if sd.dtype == 'fv':
        row = view.cell_dof_map()

        # Inverse velocity
        self.M_[row,row] += cell.volume / v
        
    return self.M_.tocsr()

  def assemble_cross_group_matrix(self, group):
    """ Assemble the matrix for coupling from other groups. """
    A = lil_matrix(tuple([self.n_dofs]*2))    
    sd = self.discretization
    for cell in self.mesh.cells:
      view = sd.cell_views[cell.id]
      material = self.materials[cell.imat]
      # Material properties
      beta_total = material.beta_total
      beta_total = beta_total if self.use_precursors else 0.0
      chi_p, nu_sig_f = 0.0, 0.0
      if hasattr(material, 'nu_sig_f'):
         chi_p = material.chi_p[self.g]
         nu_sig_f = material.nu_sig_f[group.g]
      sig_s = 0.0
      if hasattr(material, 'sig_s'):
        sig_s = material.sig_s[group.g][self.g]
      
      # Assemble if non-zero
      if chi_p*nu_sig_f != 0.0 or sig_s != 0.0:
        if sd.dtype == 'fv':
          row = view.cell_dof_map()
          A[row,row] += (1-beta_total) * chi_p * nu_sig_f * cell.volume
          A[row,row] += sig_s * cell.volume
    return A.tocsr()

  def assemble_precursor_matrix(self, precursor):
    """ Assemble to precursor source matrix. """
    sd = self.discretization
    A = lil_matrix(tuple([self.n_dofs]*2))
    for cell in self.mesh.cells:
      if cell.imat == precursor.imat:
        view = sd.cell_views[cell.id]
        material = self.materials[cell.imat]

        # Precursor properties
        chi_d = material.chi_d[precursor.j][self.g]
        decay_const = material.decay_const[precursor.j]

        # Assemble if non-zero
        if chi_d != 0.0:
          if sd.dtype == 'fv':
            row = view.cell_dof_map()
            A[row,row] += chi_d * decay_const * cell.volume
    return A.tocsr()

  def assemble_substitution_matrix(self, group, opt=0):
    method = self.problem.method
    dt = self.problem.dt
    dt = 0.5*dt if method=='tbdf2' else dt
    sd = self.discretization
    A = lil_matrix(tuple([self.n_dofs]*2))
    for cell in self.mesh.cells:
      for precursor in self.mgd.precursors:
        if cell.imat == precursor.imat:
          view = sd.cell_views[cell.id]
          material = self.materials[cell.imat]

          # Material properties
          beta = material.beta[precursor.j]
          decay_const = material.decay_const[precursor.j]
          chi_d = material.chi_d[precursor.j][self.g]
          nu_sig_f = 0.0
          if hasattr(material, 'nu_sig_f'):
            nu_sig_f = material.nu_sig_f[group.g]
          
          if chi_d * nu_sig_f != 0.0:
            coef  = chi_d * decay_const * precursor.coef[opt]
            if method == 'bwd_euler':
              coef *= beta * dt
            elif method=='cn' or method=='tbdf2' and opt==0:
              coef *= 0.5 * beta * dt
            elif method=='tbdf2' and opt==1:
              coef *= 2.0/3.0 * beta * dt

            if sd.dtype == 'fv':
              row = view.cell_dof_map()
              A[row,row] += coef * nu_sig_f * cell.volume
    return A.tocsr()

  def assemble_cross_group_rhs(self, old=False):
    """ Assemble the fission source. """
    f = self.f_old if old else self.f
    sd = self.discretization
    for cell in self.mesh.cells:
      view = sd.cell_views[cell.id]
      material = self.materials[cell.imat]
      beta_total = material.beta_total
      beta_total = beta_total if self.use_precursors else 0.0

      # Contributions from fission/scattering
      for group in self.mgd.groups:
          u = group.u_old if old else group.u_ell

          # Material properties
          chi_p, nu_sig_f = 0.0, 0.0
          if hasattr(material, 'nu_sig_f'):
            chi_p = material.chi_p[self.g]
            nu_sig_f = material.nu_sig_f[group.g]
          sig_s = 0.0
          if hasattr(material, 'sig_s'):
            sig_s = material.sig_s[group.g][self.g]
        
          # Assemble if non-zero
          if chi_p * nu_sig_f != 0.0 or sig_s != 0.0:
              if sd.dtype == 'fv':
                row = view.cell_dof_map()
                f[row] +=  u[row] * cell.volume * (
                  (1-beta_total) * chi_p * nu_sig_f + sig_s
                )

  def assemble_precursor_rhs(self, old=False):
    """ Assemble the precursor source. """
    f = self.f_old if old else self.f
    sd = self.discretization
    for cell in self.mesh.cells:
      view = sd.cell_views[cell.id]
      material = self.materials[cell.imat]

      # Contributions from each precursor
      for precursor in self.mgd.precursors:
        if cell.imat == precursor.imat:
          # Precursor vector
          u = precursor.u_old if old else precursor.u_ell
          chi_d = material.chi_d[precursor.j][self.g]
          decay_const = material.decay_const[precursor.j]
          
          # Assemble if non-zero
          if chi_d != 0.0:
            if sd.dtype == 'fv':
              row = view.cell_dof_map()
              f[row] += chi_d * decay_const * u[row] * cell.volume
  
  def assemble_gw_substitution_rhs(self, opt=0):
    method = self.problem.method
    dt = self.problem.dt
    dt = 0.5*dt if method=='tbdf2' else dt
    sd = self.discretization
    for cell in self.mesh.cells:
      for precursor in self.mgd.precursors:
        if cell.imat == precursor.imat:
          view = sd.cell_views[cell.id]
          material = self.materials[cell.imat]
          decay_const = material.decay_const[precursor.j]
          beta = material.beta[precursor.j]
          chi_d = material.chi_d[precursor.j][self.g]

          if chi_d != 0.0:
            coef = chi_d * decay_const * cell.volume
            coef *= precursor.coef[opt]
            
            if sd.dtype == 'fv':
              row = view.cell_dof_map()
              fr = self.mgd.fission_rate.u[row]
              Cj_old = precursor.u_old[row]
              
              if method == 'bwd_euler':
                self.f[row] += coef * (Cj_old + beta*dt * fr)
              elif method=='cn' or method=='tbdf2' and opt==0:
                fr_old = self.mgd.fission_rate.u_old[row]
                self.f[row] += coef * (
                  (1 - 0.5*decay_const*dt) * Cj_old
                  + 0.5*beta*dt * (fr + fr_old)
                )
              elif method=='tbdf2' and opt==1:
                Cj_half = precursor.u_half[row]
                self.f[row] += coef * (
                  (4.0*Cj_half - Cj_old)/3.0
                  + 2.0/3.0*beta*dt * fr
                )

  def assemble_full_substitution_rhs(self, opt=0):
    self.f[:] = 0
    method = self.problem.method
    dt = self.problem.dt
    dt = 0.5*dt if method=='tbdf2' else dt
    sd = self.discretization
    for cell in self.mesh.cells:
      for precursor in self.mgd.precursors:
        if cell.imat == precursor.imat:
          view = sd.cell_views[cell.id]
          material = self.materials[cell.imat]
          decay_const = material.decay_const[precursor.j]
          chi_d = material.chi_d[precursor.j][self.g]

          if chi_d != 0.0:
            coef = chi_d * decay_const * cell.volume
            coef *= precursor.coef[opt]

            if sd.dtype == 'fv':
              row = view.cell_dof_map()
              Cj_old = precursor.u_old[row]

              if method == 'bwd_euler':
                self.f[row] += coef * Cj_old
              elif method=='cn' or method=='tbdf2' and opt==0:
                fr_old = self.mgd.fission_rate.u_old[row]
                beta = material.beta[precursor.j]
                self.f[row] += coef * (
                  (1 - 0.5*decay_const*dt) * Cj_old
                  + 0.5*beta*dt * fr_old
                )
              elif method=='tbdf2' and opt==1:
                Cj_half = precursor.u_half[row]
                self.f[row] += coef * (
                  (4.0*Cj_half - Cj_old) / 3.0
                )

  def assemble_forcing(self, time=0):
    """ Assemble the inhomogeneous source. """
    self.b[:] = 0
    sd = self.discretization
    for cell in self.mesh.cells:
      view = sd.cell_views[cell.id]
      source = self.sources[cell.isrc]
      q = source.q[self.g]
      q = q(time) if callable(q) else q
      if q != 0:

          #  Finite volume
          if sd.dtype == 'fv':
            row = view.cell_dof_map()
            self.b[row] += q * cell.volume
    if not self.problem.is_transient:
      self.apply_bcs(vector=self.b)
    return self.b

  def apply_bcs(self, matrix=None, vector=None, offset=0):
    """ Apply boundary conditions to the matrix and/or vector. """
    # --- Input checks
    assert matrix is not None or vector is not None, (
      "Either a matrix, vector, or both must be provided."
    ) 
    # ---
    sd = self.discretization
    for cell in self.mesh.bndry_cells:
      view = sd.cell_views[cell.id]
      for face in cell.faces:
        if face.flag > 0:
          bc = self.bcs[face.flag-1]

          # Finite volume
          if sd.dtype == 'fv':
            # No changes for reflective bcs
            if bc.boundary_kind == 'reflective':
              continue

            # Handle non-reflective bcs
            row = view.cell_dof_map() + offset
            material = self.materials[cell.imat]
            width = cell.width[0]
            material = self.materials[cell.imat]
            D = material.D[self.g]
            # Compute the coefficient used for bcs
            if bc.boundary_kind in ['source', 'zero_flux']:
              coef = 2 * D / width
            elif bc.boundary_kind in ['marshak', 'vacuum']:
              coef = 2*D / (4*D + width)
            # Apply to matrix
            if matrix is not None:
              matrix[row,row] += face.area * coef
            # Apply to vector
            if vector is not None:
              if bc.boundary_kind in ['source', 'marshak']:
                val = bc.vals[self.g]
                vector[row] += face.area * coef * val

  def compute_fission_rate(self, fission_rate, old=False):
    sd = self.discretization
    u = self.u_old if old else self.u
    for cell in self.mesh.cells:
      view = sd.cell_views[cell.id]
      material = self.materials[cell.imat]
      nu_sig_f = 0.0
      if hasattr(material, 'nu_sig_f'):
        nu_sig_f = material.nu_sig_f[self.g]
      
      if sd.dtype == 'fv':
        row = view.cell_dof_map()
        fission_rate[row] += nu_sig_f * u[row]


  def compute_fission_power(self):
    """ Compute the total fission power in the problem.

    This routine is generally used in k-calculations.
    """
    fission_power = 0
    sd = self.discretization
    for cell in self.mesh.cells:
      view = sd.cell_views[cell.id]
      material = self.materials[cell.imat]
      if hasattr(material, 'nu_sig_f'):
        nu_sig_f = material.nu_sig_f[self.g]
        if nu_sig_f != 0:

          # Finite volume
          if sd.dtype == 'fv':
            row = view.cell_dof_map()
            u_i = self.field.u[row]
            fission_power += nu_sig_f * u_i * cell.volume
    return fission_power

  @property
  def u(self):
    return self.field.u

  @property
  def u_ell(self):
    return self.field.u_ell

  @property
  def u_half(self):
    return self.problem.u_half[self.field.dofs]

  @property
  def u_old(self):
    return self.field.u_old
