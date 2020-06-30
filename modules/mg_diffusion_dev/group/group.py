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

    def assemble_source(self, time=0):
        if self.b is None:
            self.b = np.zeros(self.field.n_dofs)
        self.b *= 0
        for group in self.mgd.groups:
            if self.discretization.dtype == 'fv':
                self.assemble_fv_source(group, time)
            elif self.discretization.dtype == 'cfe':
                pass

    def compute_fission_source(self):
        if self.discretization.dtype == 'fv':
            return self.compute_fv_fission_source()
        elif self.discretization.dtype == 'cfe':
            pass

