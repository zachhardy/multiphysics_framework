import sys
import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve

class TimeStepperMixin:

  def solve_time_step(self, opt=0):
    time = self.time
    dt = self.dt

    self.assemble_physics_matrix(opt)
    self.assemble_mass_matrix()

    if self.method=='fwd_euler':
      self.assemble_forcing(time)
      matrix = self.M/dt
      rhs = self.b + self.M/dt @ self.u_old
      rhs += self.f_old

    elif self.method=='bwd_euler':
      self.assemble_forcing(time+dt)
      matrix = self.M/dt + self.A
      rhs = self.b + self.M/dt @ self.u_old
      rhs += self.f
      
    elif self.method=='cn' or self.method=='tbdf2' and opt==0:
      self.assemble_forcing(time+dt/2)
      matrix = self.M/dt + 0.5*self.A
      rhs = self.b + self.M/dt @ self.u_old
      rhs += 0.5 * (self.f + self.f_old)
      
    elif self.method=='tbdf2' and opt==1:
      self.assemble_forcing(time+2*dt)
      matrix = 1.5*self.M/dt + self.A
      rhs = self.b + 2*self.M/dt @ self.u_half
      rhs -= 0.5*self.M/dt @ self.u_old
      rhs += self.f

    self.apply_bcs(matrix=matrix, vector=rhs)
    self.u[:] = spsolve(matrix, rhs)
