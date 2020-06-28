import sys
import numpy as np

from material import MaterialBase

class NeutronicsMaterial(MaterialBase):
    """ Class for a multigroup neutronics material. 
        
    There is much variability in neutronics properties. At a 
    minimum, sig_r and either D or sig_t must be defined. Note
    that if Na is provided, microscopic cross-sections should be 
    specified, else macroscopic.

    Parameters
    ----------
    Na : float, default=1
        The atom density (default is 1).
    sig_r : list of float/callable
        Removal cross section.
    sig_t, D : list of float/callable
        Total cross section, diffusion coefficient. Note that 
        one must be provided. If D is not provided, it is
        defaulted to 1/(3.sig_t).
    chi, nu_sig_f : list of float/callable, optional
        Fission spectrum, fission source cross section. Note 
        that if a multigroup material, chi must be specified,
        else is is 1.
    sig_s : list of list of float/callable, optional
        Scattering cross section.
    v : list of float, optional
        Group velocities.
    q : list of float/callable, optional
        Neutron source.
    """

    material_type = 'neutronics'
    
    def __init__(self, material_id=0, Na=1., sig_r=[], sig_t=[], 
                 D=[], sig_s=[], nu_sig_f=[],
                 chi=[], v=[], q=[]):
        super().__init__(material_id)
        
        # Atom density
        self.Na = Na

        # Removal cross section
        assert sig_r != [], "sig_r must be specified."
        self.sig_r = Na * np.atleast_1d(sig_r)
        self.G = len(sig_r)
        
        # Total cross section
        if sig_t != []:
            assert len(sig_t)==self.G, "Invalid group structure."
            self.sig_t = Na * np.atleast_1d(sig_t)

        # Diffusion coefficient
        if D != []:
            assert len(D)==self.G, "Invalid group structure."
            self.D = np.atleast_1d(D)/Na 
        else:
            assert sig_t!=[], "If D is not specified, sig_t must be."
            self.D = 1/(3*self.sig_t)

        # Scattering cross section
        if sig_s != []:
            sig_s = np.atleast_2d(sig_s)
            assert sig_s.shape[0]==self.G, "Invalid group structure."
            assert sig_s.shape[0]==sig_s.shape[1], "Invalid group structure."
            self.sig_s = Na * sig_s

        # Fission cross sections
        if nu_sig_f != []:
            assert len(nu_sig_f)==self.G, "Invalid group structure."
            self.nu_sig_f = Na * np.atleast_1d(nu_sig_f)
            # Fission spectrum
            if self.G > 1:
                assert len(chi)==self.G, "Invalid group structure."
                self.chi = np.atleast_1d(chi)
            else:
                self.chi = np.array([1.])

        # Velocity
        if v != []:
            assert len(v)==self.G, "Invalid group structure."
            self.v = np.atleast_1d(v)

        # Source
        if q != []:
            assert len(q)==self.G, "Invalid group structure."
            self.q = np.atleast_1d(q)