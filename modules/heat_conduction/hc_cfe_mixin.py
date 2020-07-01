import numpy as np
from scipy.sparse import lil_matrix

class HeatConductionCFEMixin:
    
    def assemble_cfe_physics(self):
        if self.is_nonlinear or self.A is None:
            p = self.discretization.porder
            self.A = lil_matrix(tuple([self.field.n_dofs]*2))
            for cell in self.mesh.cells:
                view = self.discretization.cell_views[cell.id]
                material = self.materials[cell.imat]
                k = material.k
                if callable(k):
                    T = view.quadrature_solution(self.field.u)
                    k = k(T)
                
                for i in range(p+1):
                    row = view.cell_dof_map(i)
                    for j in range(p+1):
                        col = view.cell_dof_map(j)
                        self.A[row,col] += (
                            view.intV_gradShapeI_gradShapeJ(i, j, k)
                        )
            self.A = self.A.tocsr()
            if not self.problem.is_transient:
                self.apply_cfe_bcs(matrix=self.A)

    def assemble_cfe_forcing(self, time=0):
        porder = self.discretization.porder
        for cell in self.mesh.cells:
            view = self.discretization.cell_views[cell.id]
            material = self.materials[cell.imat]
            if hasattr(material, 'q'):
                q = material.q
                q = q(time) if callable(q) else q
                if q != 0:
                    for i in range(porder+1):
                        row = view.cell_dof_map(i)
                        self.rhs[row] += view.intV_shapeI(i, q)
    
    def apply_cfe_bcs(self, matrix=None, vector=None):
        # --- Input checks
        assert matrix is not None or vector is not None, (
            "Either a matrix, vector, or both must be provided."
        ) 
        # ---
        for cell in self.mesh.bndry_cells:
            view = self.discretization.cell_views[cell.id]
            for iface, face in enumerate(cell.faces):
                if face.flag > 0:
                    bc = self.bcs[face.flag-1]
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




