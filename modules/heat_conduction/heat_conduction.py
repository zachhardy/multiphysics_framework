import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve

from physics.field import Field
from physics.discrete_system import DiscreteSystem
from physics.physics_base import PhysicsBase
from .hc_material import HeatConductionMaterial
from .hc_source import HeatConductionSource

valid_bcs = ['neumann', 'robin', 'dirichlet']

class HeatConduction(DiscreteSystem, PhysicsBase):

  name = 'temperature'
  material_type = HeatConductionMaterial.material_type
  source_type = HeatConductionSource.source_type

  def __init__(self, problem, discretization, bcs, ics=None):
    PhysicsBase.__init__(self, problem)
    self.materials = self._parse_materials(self.material_type)
    self.sources = self._parse_sources(self.source_type)
    self.bcs = self._validate_bcs(bcs)
    self.ics = self._validate_ics(ics)
    # Initialize and register the field with problem.
    field = Field(self.name, problem, discretization, 1)
    self._register_field(field)
    # Initialize discrete system
    DiscreteSystem.__init__(self, self.n_dofs, bcs)
    # Determine nonlinearity
    for material in self.materials:
      if callable(material.k):
        self.is_nonlinear = True

  def solve_system(self, time=None, dt=None, method=None, u_tmp=None):
    if not self.problem.is_transient:
      self.u[:] = self.solve_steady_state()

  def assemble_physics(self):
    if self.is_nonlinear or self.A is None:
      sd = self.discretization
      self.A = lil_matrix(tuple([self.n_dofs]*2))
      for cell in self.mesh.cells:
        view = sd.cell_views[cell.id]
        material = self.materials[cell.imat]
        k = material.k
        if callable(k):
          T = view.quadrature_solution(self.u)
          k = k(T)

        ### Finite element
        if sd.dtype == 'cfe':
          for i in range(sd.porder+1):
            row = view.cell_dof_map(i)
            for j in range(sd.porder+1):
              col = view.cell_dof_map(j)
              self.A[row,col] += (
                view.intV_gradShapeI_gradShapeJ(i, j, k)
              )
      self.A = self.A.tocsr()
      if not self.problem.is_transient:
        self.apply_bcs(matrix=self.A)

  def assemble_forcing(self, time=0):
    self.rhs[:] = 0
    sd = self.discretization
    for cell in self.mesh.cells:
      view = self.discretization.cell_views[cell.id]
      source = self.sources[cell.isrc]
      q = source.q
      q = q(time) if callable(q) else q
      if q != 0:
        
        ### Finite element
        if sd.dtype == 'cfe':
          for i in range(sd.porder+1):
            row = view.cell_dof_map(i)
            self.rhs[row] += view.intV_shapeI(i, q)
  
  def apply_bcs(self, matrix=None, vector=None):
    # --- Input checks
    assert matrix is not None or vector is not None, (
      "Either a matrix, vector, or both must be provided."
    ) 
    # ---
    sd = self.discretization
    for cell in self.mesh.bndry_cells:
      view = self.discretization.cell_views[cell.id]
      for iface, face in enumerate(cell.faces):
        if face.flag > 0:
          bc = self.bcs[face.flag-1]

          if sd.dtype == 'cfe':
            row = view.face_dof_map(iface)

            if bc.boundary_kind == 'neumann':
              if vector is not None:
                vector[row] += face.area * bc.vals

            elif bc.boundary_kind == 'robin':
              msg = "Robin bcs for heat conduction have not "
              msg += "been implemented."
              raise NotImplementedError(msg)

            elif bc.boundary_kind == 'dirichlet':
              if matrix is not None:
                matrix[row,row] = 1.0
                for col in matrix[row].nonzero()[1]:
                  if row != col:
                    matrix[row,col] = 0.0
              if vector is not None:
                vector[row] = bc.vals

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
