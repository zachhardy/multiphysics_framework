import sys
import numpy as np

class Precursor:

  def __init__(self, mgd, field, imat, j):

    self.mgd = mgd
    self.field = field
    self.imat = imat
    self.j = j

  def update_precursor(self, opt=0):
    self.u[:] = 0.0
    for cell in self.mesh.cells:
      if cell.imat == self.imat:
        view = self.discretization.cell_views[cell.id]
        dof = view.dofs[0]
        # Material properties
        material = self.materials[cell.imat]
        decay_const = material.decay_const[self.j]
        beta = material.beta[self.j]
        FR = self.fission_rate[dof]
        u_old = self.u_old[dof]

        if self.method=='bwd_euler':
          coef = 1 / (1 + decay_const*self.dt)
          self.u[dof] += coef * (u_old + beta*self.dt*FR)
        elif self.method=='cn' or self.method=='tbdf2' and opt==0:
          FR_old = self.fission_rate_old[dof]
          coef = 1 / (1 + 0.5*decay_const*self.dt)
          self.u[dof] += coef * (
            (1 - 0.5*decay_const*self.dt) * u_old
            + 0.5 * beta * self.dt * (FR + FR_old)
          )
        elif self.method=='tbdf2' and opt==1:
          coef = 1 / (1 + 2.0/3.0*decay_const*self.dt)
          self.u[dof] += coef * (
            (4.0*self.u_half[dof] - self.u_old[dof]) / 3.0
            + 2.0/3.0 * beta * self.dt * self.fission_rate[dof]
          )
      
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
  def groups(self):
    return self.mgd.groups

  @property
  def precursors(self):
    return self.mgd.precursors

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
  def method(self):
    return self.mgd.problem.method

  @property
  def dt(self):
    dt = self.mgd.problem.dt
    return 0.5*dt if self.method=='tbdf2' else dt