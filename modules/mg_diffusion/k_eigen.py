import sys
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve

from bc import BC


class KEigenMixin:

    def solve_k_eigen_problem(self, tol=1e-6, maxit=100, verbosity=0):
        """ Solve a k-eigenvalue problem.

        This routine solves a k-eigenvalue problem for a 
        neutronics problem using power iteration.

        Parameters
        ----------
        tol : float, default=1e-6
            The convergence tolerance.
        maxit : int, default=100
            The maximum number of iterations.
        verbosity : int, default=0
            The level of screen output.
        """

        # Total volume
        V = sum([cell.volume for cell in self.mesh.cells])

        def integrate_volume(phig):
            phig = phig.reshape(self.G, self.n_nodes)
            phi = sum(phig)
            phi_tot = 0
            for cell in self.mesh.cells:
                view = self.sd.cell_views[cell.id]
                phi_tot += view.average_solution(phi)*cell.volume
            return phi_tot 
        
       
        # Initial guesses
        u0 = np.ones(self.n_dofs)
        u0 /= integrate_volume(u0) / V
        k0 = 1

        # Compute loss and fission operators
        L, F = self.assemble_k_eigen_system()

        # Inverse power iteration
        k_err, u_err, nit = 1e2, 1e2, 0
        while k_err > tol and u_err > tol and nit < maxit:
            # Compute the next flux and k iterate
            u = spsolve(L, F @ u0)
            k = integrate_volume(u) / V
            # Renormalize the flux to a unit particle
            u /= k

            
            k_err = np.fabs(k-k0) / k
            u_err = np.sqrt(
                integrate_volume((u-u0)**2)
                / integrate_volume(u**2)
            )

            u0[:] = u
            k0 = k
            nit += 1

            if verbosity > 0:
                msg = "Iteration {}".format(nit)
                delim = '-'*len(msg)
                msg = '\n'.join(['', msg, delim])
                print(msg)
                print('k:\t\t{:.3e}'.format(k))
                print('k Error:\t{:.3e}'.format(k_err))
                print('Flux Error:\t{:.3e}'.format(u_err))
                print(delim)

        if nit < maxit:
            print("\nConverged k:\t\t{:.5e}".format(k))
        else:
            print("\n*** Warning, simulation did not converge ***")
            print("\nUnconvered k:\t\t{:.5e}".format(k))
        print("Final k Error:\t\t{:.3e}".format(k_err))
        print("Final Flux Error:\t{:.3e}".format(u_err))

        return k, u

    def assemble_k_eigen_system(self):
        """ Assemble a k-eigenvalue system. """
        Lrows, Lcols, Lvals = [], [], []
        Frows, Fcols, Fvals = [], [], []
        for cell in self.mesh.cells:
            # Loss operator
            rows, cols, vals = self.assemble_cell_physics(cell, True)
            Lrows.extend(rows)
            Lcols.extend(cols)
            Lvals.extend(vals)
            # Fission operator
            rows, cols, vals = self.assemble_cell_fission(cell)
            Frows.extend(rows)
            Fcols.extend(cols)
            Fvals.extend(vals)

        shape = (self.n_dofs, self.n_dofs)
        # Form loss operator
        L = csr_matrix((Lvals, (Lrows, Lcols)), shape)
        self.apply_bcs(matrix=L)
        # Form fission operator
        F = csr_matrix((Fvals, (Frows, Fcols)), shape)
        self.apply_dirichlet_bcs(matrix=F)
        return L, F

    def apply_dirichlet_bcs(self, matrix=None, vector=None):
        """ Apply a dirichlet value to the fission matrix. """
        for cell in self.mesh.bndry_cells:
            try:
                view = self.sd.fe_views[cell.id]
            except: 
                return
            
            for f, face in enumerate(cell.faces):
                if face.flag > 0:
                    bc = self.bcs[face.flag-1]

                    if bc.boundary_kind == 'dirichlet':
                        for ig in range(self.G):
                            row = view.face_dof_map(f, ig)
                            
                            if matrix is not None:
                                matrix[row,row] = 1.0
                                for col in matrix[row].nonzero()[1]:
                                    if row != col:
                                        matrix[row,col] = 0.0
                            if vector is not None:
                                vector[row] = 0.0

            
        

