import numpy as np
from scipy.sparse import lil_matrix

class GroupFVMixin:

    def assemble_fv_physics(self):
        if self.is_nonlinear or self.A is None:
            sd = self.discretization
            self.A = lil_matrix(tuple([self.field.n_dofs]*2))
            for cell in self.mesh.cells:
                view = sd.cell_views[cell.id]
                material = self.materials[cell.imat]
                sig_r = material.sig_r[self.group_num]
                D = material.D[self.group_num]
                row = view.cell_dof_map()
                # Removal
                self.A[row,row] += sig_r * cell.volume
                # Interior diffusion
                width = cell.width[0]
                for face in cell.faces:
                    if face.flag == 0:
                        nbr_cell = self.mesh.cells[face.neighbor]
                        nbr_width = nbr_cell.width[0]
                        nbr_view = sd.cell_views[nbr_cell.id]
                        nbr_material = self.materials[nbr_cell.imat]
                        nbr_D = nbr_material.D[self.group_num]
                        col = nbr_view.cell_dof_map()
                        # Compute the effective edge quantities
                        width_eff = (width + nbr_width)/2.0
                        D_eff = 2*width_eff / (width/D + nbr_width/nbr_D)
                        # Add the diffusion term to the matrix
                        self.A[row,row] += face.area * D_eff/width_eff
                        self.A[row,col] -= face.area * D_eff/width_eff
            self.A = self.A.tocsr()
            if not self.problem.is_transient:
                self.apply_fv_bcs(matrix=self.A)   
    
    def assemble_fv_mass(self):
        if self.is_nonlinear or self.M is None:
            self.M = lil_matrix(tuple([self.field.n_dofs]*2))
            for cell in self.mesh.cells:
                view = self.discretization.cell_views[cell.id]
                material = self.materials[cell.imat]
                v = material.v[self.group_num]
                row = view.cell_dof_map()
                self.M[row,row] += cell.volume / v
            self.M = self.M.tocsr()

    def assemble_fv_forcing(self, time):
        for cell in self.mesh.cells:
            view = self.discretization.cell_views[cell.id]
            material = self.materials[cell.imat]
            row = view.cell_dof_map()
            if hasattr(material, 'q'):
                q = material.q[self.group_num]
                if q != 0:
                    q = q(time) if callable(q) else q
                    self.rhs[row] += q * cell.volume

    def fv_fission_and_scattering_source(self, u, f):
        for cell in self.mesh.cells:
            view = self.discretization.cell_views[cell.id]
            material = self.materials[cell.imat]
            row = view.cell_dof_map()
            # Fission
            if hasattr(material, 'nu_sig_f'):
                for group in self.mgd.groups:
                    gprime = group.group_num
                    chi = material.chi[self.group_num]
                    nu_sig_f = material.nu_sig_f[gprime]
                    if chi*nu_sig_f != 0:
                        u_i = u[group.field.dofs[row]]
                        f[row] -= chi*nu_sig_f * u_i * cell.volume
            # Scattering
            if hasattr(material, 'sig_s'):
                for group in self.mgd.groups:
                    gprime = group.group_num
                    sig_s = material.sig_s[gprime][self.group_num]
                    if sig_s != 0:
                        u_i = u[group.field.dofs[row]]
                        f[row] -= sig_s * u_i * cell.volume

    def compute_fv_fission_power(self):
        fission_power = 0
        for cell in self.mesh.cells:
            view = self.discretization.cell_views[cell.id]
            material = self.materials[cell.imat]
            if hasattr(material, 'nu_sig_f'):
                nu_sig_f = material.nu_sig_f[self.group_num]
                if nu_sig_f != 0:
                    row = view.cell_dof_map()
                    u = self.field.u[row]
                    fission_power += nu_sig_f * u * cell.volume
        return fission_power

    def apply_fv_bcs(self, matrix=None, vector=None):
        # --- Input checks
        assert matrix is not None or vector is not None, (
            "Either a matrix, vector, or both must be provided."
        ) 
        # ---
        for cell in self.mesh.bndry_cells:
            view = self.discretization.cell_views[cell.id]
            for face in cell.faces:
                if face.flag > 0:
                    bc = self.bcs[face.flag-1]
                    # If the boundary condition is reflecetive 
                    if bc.boundary_kind != 'reflective':
                        row = view.cell_dof_map()
                        width = cell.width[0]
                        material = self.materials[cell.imat]
                        D = material.D[self.group_num]
                        # Compute the coefficient used in the bc
                        if bc.boundary_kind in ['source', 'zero_flux']:
                            coef = 2*D/width
                        elif bc.boundary_kind in ['marshak', 'vacuum']:
                            coef = 2*D/(4*D+width)
                        # Make changes to matrix
                        if matrix is not None:
                            matrix[row,row] += face.area * coef
                        # Make changes to vector
                        if vector is not None:
                            if bc.boundary_kind in ['marshak', 'source']:
                                bcval = bc.vals[self.group_num]
                                vector[row] += face.area * coef * bcval
                        

