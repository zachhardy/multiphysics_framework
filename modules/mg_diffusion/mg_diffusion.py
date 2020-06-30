import numpy as np
from numpy.linalg import norm
from scipy.sparse.linalg import spsolve
import time


from field import Field
from .group.group import Group
from physics.physics_base import PhysicsBase
from .neutronics_material import NeutronicsMaterial

valid_fv_bcs = ['reflective', 'marshak', 'vacuum',
                'source', 'zero_flux']
valid_cfe_bcs = ['dirichlet', 'neumann', 'robin']

class MultiGroupDiffusion(PhysicsBase):

    # Multigroup specific items
    name = 'scalar_flux'
    material_type = NeutronicsMaterial.material_type
    n_grps = 0
    groups = []

    def __init__(self, problem, discretization, bcs, ics=None,
                 tol=1e-8, maxit=100):
        super().__init__(problem)
        self.materials = self._parse_materials(self.material_type)
        self.discretization = discretization
        self.bcs = self._validate_bcs(bcs)
        self.ics = self._validate_ics(ics)
        # Convergence parameters
        self.tol = tol
        self.maxit = maxit
        # Initialize group objects
        self.groups = []
        self.n_grps = self.materials[0].n_grps
        for g in range(self.n_grps):
            gname = self.name + '_g{}'.format(g+1)
            field = Field(gname, problem, discretization, 1)
            self._register_field(field)
            self.groups.append(Group(self, field, g))
        
    def solve_system(self, time=None, dt=None, method=None, u_half=None):
        converged = False
        for nit in range(self.maxit):
            diff = 0
            for group in self.groups:
                group.assemble_physics()
                # Handle steady state problem
                if not self.problem.is_transient:
                    group.solve_steady_state()
                # Handle time step of a transient
                else:
                    group.assemble_mass()
                    group.solve_time_step(time, dt, method, u_half)
                # Compute the difference and reinit
                diff += norm(group.field.u-group.field.u_ell, ord=2)
                group.field.u_ell[:] = group.field.u
            if diff < self.tol:
                converged = True
                break
        if converged:
            print("*** Converged in {} iterations. ***".format(nit))
        else:
            print("*** WARNING: DID NOT CONVERGE. ***")

    def compute_old_physics_action(self):
        for group in self.groups:
            group.compute_old_physics_action()

    def compute_fission_power(self):
        fission_source = 0
        for group in self.groups:
            fission_source += group.compute_fission_power()
        return fission_source

    def compute_k_eigenvalue(self, tol=1e-8, maxit=100, verbosity=0):
        # Prepare problem by setting to steady state, setting
        # the inhomogeneous source to zero, and initializing a 
        # an iteration vector.
        self.problem.is_transient = False
        for material in self.materials:
            if hasattr(material, 'q'):
                material.q = np.zeros(self.n_grps)

        # Initialize initial guesses and operators
        for group in self.groups:
            group.assemble_physics()
            group.field.u_ell[:] = 1
        k_eff_old = 1
        
        # Inverse power iterations
        converged = False
        for nit in range(maxit):
            # Solve group-wise and compute new k-eff
            for group in self.groups:
                group.solve_steady_state()
                group.field.u_ell[:] = group.field.u
            k_eff = self.compute_fission_power()
            # Reinit and normalize group fluxes
            for group in self.groups:
                group.field.u_ell[:] = group.field.u / k_eff
            # Compute the change in k-eff and reinit
            k_error = np.abs(k_eff-k_eff_old) / np.abs(k_eff)
            k_eff_old = k_eff
            # Check convergence
            if k_error < tol:
                converged = True
                break
            
            # Iteration printouts
            if verbosity > 1:
                self.print_k_iter_summary(nit, k_eff, k_error)
        self.print_k_calc_summary(converged, nit, k_eff, k_error)
        

    def _validate_materials(self, materials):
        n_grps = materials[0].n_grps
        if len(materials) > 1:
            for material in materials[1:]:
                assert material.n_grps==n_grps, (
                    "All materials must have the same group structure."
                )
        return materials

    def _validate_bcs(self, bcs):
        for bc in bcs:
            if self.discretization.dtype == 'fv':
                valid_bcs = valid_fv_bcs
            elif self.discretization.dtype == 'cfe':
                valid_bcs = valid_cfe_bcs
            dscrt = self.discretization.dtype
            if bc.boundary_kind not in valid_bcs:
                msg = "\nApproved BCs for {} ".format(dscrt)
                msg += "multigroup diffusion are:\n"
                for kind in valid_bcs:
                    msg += "--- {}\n".format(kind)
                raise ValueError(msg)
        return bcs
    
    def _validate_ics(self, ics):
        if ics is None:
            return ics
        for ic in ics:
            assert callable(ic), (
                "All initial conditions must be callable."
            )
        return ics

    @staticmethod
    def print_k_iter_summary(nit, k_eff, k_error):
        msg = "\nIteration {}".format(nit)
        delim = '-'*len(msg)
        msg = '\n'.join(['', msg, delim])
        print(msg)
        print('k-eff:\t\t{:.3e}'.format(k_eff))
        print('k Error:\t{:.3e}'.format(k_error))

    @staticmethod
    def print_k_calc_summary(converged, nit, k_eff, k_error):
        if converged:
            print("\n*** Simulation converged in {} iterations ***".format(nit))
            print("Converged k:\t\t{:.5e}".format(k_eff))
        else:
            print("\n*** WARNING: Simulation did not converge ***")
            print("Unconverged k:\t\t{:.5e}".format(k_eff))
        print("Final k Error:\t\t{:.3e}".format(k_error))