import numpy as np
from scipy.sparse import lil_matrix
from physics.discrete_system import DiscreteSystem

class Group(DiscreteSystem):

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
        # Initialize discrete system
        DiscreteSystem.__init__(self, field, self.bcs)

    def compute_old_physics_action(self):
        """ Compute the old action of the physics. 
        
        First, assemble the old fission, scattering, and 
        delayed neutron sources, then add to it the old 
        leakage plus removal sinks.
        """
        self.assemble_lagged_sources(True)
        super().compute_old_physics_action()

    def assemble_physics(self):
        """ Assemble the diffusion + removal operator. """
        if self.is_nonlinear or self.A is None:
            sd = self.discretization
            self.A = lil_matrix(tuple([self.field.n_dofs]*2))
            for cell in self.mesh.cells:
                view = sd.cell_views[cell.id]
                material = self.materials[cell.imat]
                sig_r = material.sig_r[self.group_num]
                D = material.D[self.group_num]

                ### Finite volume assembly
                if sd.dtype == 'fv':
                    row = view.cell_dof_map()
                    # Removal term
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
                            eff_width = 0.5*(width + nbr_width)
                            eff_D = 2*eff_width / (width/D + nbr_width/nbr_D)
                            # Add to the matrix
                            self.A[row,row] += face.area * eff_D/eff_width
                            self.A[row,col] -= face.area * eff_D/eff_width
                
                ### Finite element assembly
                if sd.dtype == 'cfe': 
                    for i in range(sd.porder+1):
                        row = view.cell_dof_map(i)
                        for j in range(sd.porder+1):
                            col = view.cell_dof_map(j)
                            self.A[row,col] += (
                                view.intV_shapeI_shapeJ(i, j, sig_r)
                                + view.intV_gradShapeI_gradShapeJ(i, j, D)
                            )
            self.A = self.A.tocsr()
            if not self.problem.is_transient:
                self.apply_bcs(matrix=self.A)

    def assemble_mass(self):
        """ Assemble the velocity mass matrix. """
        if self.is_nonlinear or self.M is None:
            sd = self.discretization
            self.M = lil_matrix(tuple([self.field.n_dofs]*2))
            for cell in self.mesh.cells:
                view = sd.cell_views[cell.id]
                material = self.materials[cell.imat]
                v = material.v[self.group_num]

                ### Finite volume assembly
                if sd.dtype == 'fv':
                    row = view.cell_dof_map()
                    self.M[row,row] += cell.volume / v

                ### Finite element assembly
                if sd.dtype == 'cfe':
                    for i in range(sd.porder+1):
                        row = view.cell_dof_map(i)
                        for j in range(sd.porder+1):
                            col = view.cell_dof_map(j)
                            self.M[row,col] += (
                                view.intV_shapeI_shapeJ(i, j, 1/v)
                            )
            self.M = self.M.tocsr()

    def assemble_forcing(self, time=0):
        """ Assemble the inhomogeneous source. """
        self.rhs[:] = 0
        sd = self.discretization
        for cell in self.mesh.cells:
            view = sd.cell_views[cell.id]
            material = self.materials[cell.imat]
            if hasattr(material, 'q'):
                q = material.q[self.group_num]
                q = q(time) if callable(q) else q
                if q != 0:
                    
                    ### Finite volume assembly
                    if sd.dtype == 'fv':
                        row = view.cell_dof_map()
                        self.rhs[row] += q * cell.volume

                    ### Finite element assembly
                    elif sd.dtype == 'cfe':
                        for i in range(sd.porder+1):
                            row = view.cell_dof_map(i)
                            self.rhs[row] += view.intV_shapeI(i, q)
    
    def assemble_lagged_sources(self, old=False):
        """ Assemble lagged sources. 

        Lagged sources for multi-group diffusion includes
        all cross-group couplings and delayed neutron emmission.
        """
        # Get the correct vectors
        f = self.f_old if old else self.f_ell
        u = self.problem.u_old if old else self.problem.u_ell
        # Clear the destination vector and rebuild
        f[:] = 0
        for cell in self.mesh.cells:
            self.assemble_fission_source(cell, u, f)
            self.assemble_scattering_source(cell, u, f)
            self.assemble_delayed_neutron_source(cell, u, f)
            
    def assemble_fission_source(self, cell, u, f):
        """ Assemble the fission source on a cell. """
        sd = self.discretization
        view = sd.cell_views[cell.id]
        material = self.materials[cell.imat]
        if hasattr(material, 'nu_sig_f'):
            for group in self.mgd.groups:
                gprime = group.group_num
                chi = material.chi[self.group_num]
                nu_sig_f = material.nu_sig_f[gprime]
                beta_total = material.beta_total
                if chi * nu_sig_f != 0:
                    u_g = u[group.field.dofs]

                    ### Finite volume
                    if sd.dtype == 'fv':
                        row = view.cell_dof_map()
                        f[row] -= (
                            (1-beta_total) * chi * nu_sig_f 
                            * u_g[row] * cell.volume
                        )

                    ### Finite element
                    elif sd.dtype == 'cfe':
                        u_qp = view.quadrature_solution(u_g)
                        fis = (1-beta_total) * chi * nu_sig_f * u_qp
                        for i in range(sd.porder+1):
                            row = view.cell_dof_map(i)
                            f[row] -= view.intV_shapeI(i, fis)

    def assemble_scattering_source(self, cell, u, f):
        """ Assemble the scattering source on a cell. """
        sd = self.discretization
        view = sd.cell_views[cell.id]
        material = self.materials[cell.imat]
        if hasattr(material, 'sig_s'):
            for group in self.mgd.groups:
                gprime = group.group_num
                sig_s = material.sig_s[gprime][self.group_num]
                if sig_s != 0:
                    u_g = u[group.field.dofs]

                    ### Finite volume assembly
                    if sd.dtype == 'fv':
                        row = view.cell_dof_map()
                        f[row] -= sig_s * u_g[row] * cell.volume

                    ### Finite element assembly
                    elif sd.dtype == 'cfe':
                        u_qp = view.quadrature_solution(u_g)
                        sctr = sig_s * u_qp
                        for i in range(sd.porder+1):
                            row = view.cell_dof_map(i)
                            f[row] -= view.intV_shapeI(i, sctr)

    def assemble_delayed_neutron_source(self, cell, u, f):
        """ Assemble the delayed neutron source on a cell. """
        sd = self.discretization
        view = sd.cell_views[cell.id]
        material = self.materials[cell.imat]
        if material.n_delayed > 0:
            for precursor in self.mgd.precursors:
                p = precursor.precursor_num
                decay_const = material.decay_const[p]
                chi_d = material.chi_d[self.group_num]
                C = u[precursor.field.dofs]

                ### Finite volume
                if sd.dtype == 'fv':
                    row = view.cell_dof_map()
                    f[row] -= (
                        chi_d * decay_const * C[row] * cell.volume
                    )
                
                ### Finite element
                elif sd.dtype == 'cfe':
                    C_qp = view.quadrature_solution(C)
                    n_d = chi_d * decay_const * C_qp
                    for i in range(sd.porder+1):
                        row = view.cell_dof_map(i)
                        f[row] -= view.intV_shapeI(i, n_d)

    def apply_bcs(self, matrix=None, vector=None):
        """ Apply boundary conditions to the matrix and/or vector. """
        # --- Input checks
        assert matrix is not None or vector is not None, (
            "Either a matrix, vector, or both must be provided."
        ) 
        # ---
        sd = self.discretization
        for cell in self.mesh.bndry_cells:
            view = sd.cell_views[cell.id]
            for iface, face in enumerate(cell.faces):
                if face.flag > 0:
                    bc = self.bcs[face.flag-1]

                    ### Finite volume
                    if sd.dtype == 'fv':
                        # No changes for reflective bcs
                        if bc.boundary_kind == 'reflective':
                            continue

                        # Handle non-reflective bcs
                        row = view.cell_dof_map()
                        material = self.materials[cell.imat]
                        width = cell.width[0]
                        material = self.materials[cell.imat]
                        D = material.D[self.group_num]
                        # Compute the coefficient used for bcs
                        if bc.boundary_kind in ['source', 'zero_flux']:
                            coef = 2 * D / width
                        elif bc.boundary_kind in ['marshak', 'vacuum']:
                            coef = 2*D / (4*D + width)
                        # Apply to matrix
                        if matrix is not None:
                            matrix[row,row] += face.area * coef
                        # Apply to vector
                        if vector is not None:
                            if bc.boundary_kind in ['source', 'marshak']:
                                val = bc.vals[self.group_num]
                                vector[row] += face.area * coef * val

                    ### Finite element
                    if sd.dtype == 'cfe':
                        row = view.face_dof_map(iface)
                        # Handle Neumann bcs
                        if bc.boundary_kind == 'neumann':
                            if vector is not None:
                                vector[row] += face.area * bc.vals[self.group_num]

                        # Handle Robin bcs
                        if bc.boundary_kind == 'robin':
                            if matrix is not None:
                                matrix[row,row] += face.area * 0.5
                            if vector is not None:
                                val = bc.vals[self.group_num]
                                vector[row] += 2.0 * face.area * val
                        
                        # Handle Dirichlet bcs
                        if bc.boundary_kind == 'dirichlet':
                            if matrix is not None:
                                matrix[row,row] = 1.0
                                for col in matrix[row].nonzero()[1]:
                                    if row != col:
                                        matrix[row,col] = 0.0
                            if vector is not None:
                                vector[row] = bc.vals[self.group_num]
    
    def compute_fission_power(self):
        """ Compute the total fission power in the problem.

        This routine is generally used in k-calculations.
        """
        fission_power = 0
        sd = self.discretization
        for cell in self.mesh.cells:
            view = sd.cell_views[cell.id]
            material = self.materials[cell.imat]
            if hasattr(material, 'nu_sig_f'):
                nu_sig_f = material.nu_sig_f[self.group_num]
                if nu_sig_f != 0:

                    ### Finite volume
                    if sd.dtype == 'fv':
                        row = view.cell_dof_map()
                        u_i = self.field.u[row]
                        fission_power += nu_sig_f * u_i * cell.volume

                    ### Finite element
                    elif sd.dtype == 'cfe':
                        u_qp = view.quadrature_solution(self.field.u)
                        for i in range(sd.porder+1):
                            for qp in range(view.n_qpts):
                                fission_power += (
                                    view.Jcoord[qp] * view.JxW[qp] 
                                    * nu_sig_f * u_qp[qp]
                                )
        return fission_power

