import sys
import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve

class TimeStepperMixin:

  def solve_time_step(self, opt=0):
    method = self.problem.method
    time = self.problem.time
    dt = self.problem.dt

    if method == 'fwd_euler':
      self.assemble_forcing(time)
      matrix = self.M/dt
      rhs = self.b + self.M/dt @ self.u_old
      rhs += self.f_old

    elif method == 'bwd_euler':
      self.assemble_forcing(time+dt)
      matrix = self.M/dt + self.A
      rhs = self.b + self.M/dt @ self.u_old
      rhs += self.f

    elif method == 'cn' or method == 'tbdf2' and opt == 0:
      dt = dt if method=='cn' else dt/2
      self.assemble_forcing(time+dt/2)
      matrix = self.M/dt + 0.5*self.A
      rhs = self.b + self.M/dt @ self.u_old
      rhs += 0.5 * (self.f + self.f_old)
      
    elif method == 'tbdf2' and opt == 1:
      dt /= 2.0
      self.assemble_forcing(time+2*dt)
      matrix = 1.5*self.M/dt + self.A
      rhs = self.b + 2*self.M/dt @ self.u_half
      rhs -= 0.5*self.M/dt @ self.u_old
      rhs += self.f

    self.apply_bcs(matrix=matrix, vector=rhs)
    return spsolve(matrix, rhs)
