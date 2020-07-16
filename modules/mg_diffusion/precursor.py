import numpy as np
from scipy.sparse import lil_matrix
from physics.time_stepper import TimeStepperMixin

class Precursor(TimeStepperMixin):

  def __init__(self, mgd, field, imat, j):
    self.problem = mgd.problem
    self.mgd = mgd
    self.field = field
    self.mesh = mgd.mesh
    self.materials = mgd.materials
    self.discretization = field.discretization
    self.bcs = mgd.bcs
    
    self.imat = imat # material number
    self.j = j # precursor number
    self.n_dofs = field.n_dofs

    # Initialize system
    self.A_ = None
    self.M_ = None
    self.b = np.zeros(self.n_dofs)
    self.f = np.zeros(self.n_dofs)
    self.f_old = np.zeros(self.n_dofs)

  @property
  def A(self):
    if self.A_ is None:
      return self.assemble_decay_matrix()
    return self.A_

  @property
  def M(self):
    if self.M_ is None:
      return self.assemble_mass_matrix()
    return self.M_

  def assemble_decay_matrix(self):
    sd = self.discretization
    self.A_ = lil_matrix(tuple([self.n_dofs]*2))
    for cell in self.mesh.cells:
      if cell.imat == self.imat:
        view = sd.cell_views[cell.id]
        material = self.materials[self.imat]
        decay_const = material.decay_const[self.j]

        # Finite volume
        if sd.dtype == 'fv':
          row = view.cell_dof_map()
          self.A_[row,row] += decay_const * cell.volume
    self.A_ = self.A_.tocsr()
    return self.A_

  def assemble_mass_matrix(self):
    sd = self.discretization
    self.M_ = lil_matrix(tuple([self.n_dofs]*2))
    for cell in self.mesh.cells:
      if cell.imat == self.imat:
        view = sd.cell_views[cell.id]

        # Finite volume
        if sd.dtype == 'fv':
          row = view.cell_dof_map()
          self.M_[row,row] += cell.volume
    self.M_ = self.M_.tocsr()
    return self.M_

  def assemble_production_matrix(self, group):
    sd = self.discretization
    A = lil_matrix(tuple([self.n_dofs]*2))
    for cell in self.mesh.cells:
      if cell.imat == self.imat:
        view = sd.cell_views[cell.id]
        material = self.materials[cell.imat]
    
        # Material properties
        beta = material.beta[self.j]
        nu_sig_f = 0.0
        if hasattr(material, 'nu_sig_f'):
          nu_sig_f = material.nu_sig_f[group.g]

        if nu_sig_f != 0.0:
          if sd.dtype == 'fv':
            row = view.cell_dof_map()
            A[row,row] += beta * nu_sig_f * cell.volume
    return A.tocsr()

  def assemble_production_rhs(self, old=False):
    f = self.f_old if old else self.f
    sd = self.discretization
    for cell in self.mesh.cells:
      if cell.imat == self.imat:
        view = sd.cell_views[cell.id]
        material = self.materials[cell.imat]
        beta = material.beta[self.j] # delayed fraction

        fr = self.mgd.fission_rate.u_old if old \
             else self.mgd.fission_rate.u
        if sd.dtype == 'fv':
          row = view.cell_dof_map()
          f[row] += beta * fr[row] * cell.volume

  def update_precursor(self, opt=0):
    self.u[:] = 0.0
    method = self.problem.method
    dt = self.problem.dt
    dt = 0.5*dt if method=='tbdf2' else dt
    sd = self.discretization
    for cell in self.mesh.cells:
      if cell.imat == self.imat:
        view = sd.cell_views[cell.id]
        material = self.materials[cell.imat]
        decay_const = material.decay_const[self.j]
        beta = material.beta[self.j]
        coef = self.coef[opt]

        if sd.dtype == 'fv':
          row = view.cell_dof_map()
          fr = self.mgd.fission_rate.u[row]

          # Contributions from the old concentration
          if method=='bwd_euler':
            self.u[row] += coef * (
              self.u_old[row] + beta*dt * fr
            )
          elif method=='cn' or method=='tbdf2' and opt==0:
            fr_old = self.mgd.fission_rate.u_old[row]
            self.u[row] += coef * (
              (1 - 0.5*decay_const*dt) * self.u_old[row]
              + 0.5*beta*dt * (fr + fr_old)
            )
          elif method=='tbdf2' and opt==1:
            self.u[row] += coef * (
              (4.0*self.u_half[row] - self.u_old[row])/3.0
              + 2.0/3.0*beta*dt * fr
            )


  def assemble_forcing(self, time=0.0):
    pass

  def apply_bcs(self, matrix=None, vector=None):
    pass

  @property
  def coef(self):
    method = self.problem.method
    dt = self.problem.dt
    dt = 0.5*dt if method=='tbdf2' else dt
    # Material properties
    material = self.materials[self.imat]
    decay_const = material.decay_const[self.j]
    # Form coefficients
    if method == 'bwd_euler':
      return [1 / (1 + decay_const*dt)]
    elif method == 'cn':
      return [1 / (1 + 0.5*decay_const*dt)]
    elif method == 'tbdf2':
      coef1 = 1 / (1 + 0.5*decay_const*dt)
      coef2 = 1 / (1 + 2.0/3.0*decay_const*dt)
      return [coef1, coef2]

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