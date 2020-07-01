import numpy as np
from scipy.sparse.linalg import spsolve

from field import Field
from discrete_system import DiscreteSystem
from physics.physics_base import PhysicsBase
from .hc_cfe_mixin import HeatConductionCFEMixin
from .hc_material import HeatConductionMaterial

valid_bcs = ['neumann', 'robin', 'dirichlet']

class HeatConduction(PhysicsBase, HeatConductionCFEMixin, 
                     DiscreteSystem):

    name = 'temperature'
    material_type = HeatConductionMaterial.material_type

    def __init__(self, problem, discretization, bcs, ics=None):
        super().__init__(problem)
        self.materials = self._parse_materials(self.material_type)
        self.bcs = self._validate_bcs(bcs)
        self.ics = self._validate_ics(ics)
        # Initialize and register the field with problem.
        field = Field(self.name, problem, discretization, 1)
        self._register_field(field)
        self.field = self.fields[0]
        # Initialize vectors
        self.rhs = np.zeros(self.field.n_dofs)
        self.f_old = np.zeros(self.field.n_dofs)
        # Determine nonlinearity
        for material in self.materials:
            if callable(material.k):
                self.is_nonlinear = True

    def solve_system(self, time=None, dt=None, method=None, u_tmp=None):
        if not self.problem.is_transient:
            self.assemble_physics()
            self.assemble_forcing()
            self.apply_bcs(vector=self.rhs)
            self.field.u[:] = spsolve(self.A, self.rhs)

    def _validate_materials(self, materials):
        return materials

    def _validate_bcs(self, bcs):
        for bc in bcs:
            if bc.boundary_kind not in valid_bcs:
                msg = "Approved BCs are:\n"
                for kind in valid_bcs:
                    msg += "{}\n".format(kind)
                raise ValueError(msg)
        return bcs

    def _validate_ics(self, ics):
        if ics is not None:
            if not callable(ics):
                msg = "Initial condition must be callable."
                raise ValueError(msg)
        return ics


