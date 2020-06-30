import numpy as np
from scipy.sparse import lil_matrix
from .group_fv_mixin import GroupFVMixin
from .group_cfe_mixin import GroupCFEMixin
from discrete_system import DiscreteSystem

class Group(GroupFVMixin, GroupCFEMixin, DiscreteSystem):

    def __init__(self, mgd, field, group_num):
        self.problem = mgd.problem
        self.mgd = mgd
        self.field = field
        self.mesh = mgd.mesh
        self.materials = mgd.materials
        self.discretization = field.discretization
        self.bcs = mgd.bcs
        self.ics = mgd.ics
        self.group_num = group_num
        # Boolean flags
        self.is_nonlinear = mgd.is_nonlinear
        self.is_coupled = mgd.is_coupled
        # Initialize vectors
        self.rhs = np.zeros(self.field.n_dofs)
        self.f_ell = np.zeros(self.field.n_dofs)
        self.f_old = np.zeros(self.field.n_dofs)

    def lagged_operator_action(self, ell, f):
        f[:] = 0
        u = self.problem.u_ell if ell else self.problem.u_old
        if self.discretization.dtype == 'fv':
            self.fv_fission_and_scattering_source(u, f)
        elif self.discretization.dtype == 'cfe':
            pass

    def compute_old_physics_action(self):
        self.lagged_operator_action(False, self.f_old)
        super().compute_old_physics_action()
        
    def compute_fission_power(self):
        if self.discretization.dtype == 'fv':
            return self.compute_fv_fission_power()
        elif self.discretization.dtype == 'cfe':
            pass

    
