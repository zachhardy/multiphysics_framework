import sys
import numpy as np

from material import MaterialBase

class NeutronicsMaterial(MaterialBase):
    """ Class for a multigroup neutronics material. 
        
    There is much variability in neutronics properties. At a 
    minimum, sig_r and either D or sig_t must be defined. Note
    that if Na is provided, microscopic cross-sections should be 
    specified, else macroscopic.
    """

    material_type = 'neutronics'
    
    def __init__(self, material_id=0, Na=1., sig_r=[], sig_t=[], 
                 D=[], sig_s=[], nu_sig_f=[],
                 chi=[], v=[], q=[]):
        super().__init__(material_id)
        # Group structure
        self.n_grps = len(sig_r)
        # Atom density
        self.Na = Na
        # Removal cross section
        self.sig_r = Na * np.atleast_1d(sig_r)        
        # Total cross section
        if sig_t != []:
            self.sig_t = Na * np.atleast_1d(sig_t)
        # Diffusion coefficient
        if D != []:
            self.D = np.atleast_1d(D)/Na 
        else:
            assert sig_t!=[], "If D is not specified, sig_t must be."
            self.D = 1/(3*self.sig_t)
        # Scattering cross section
        if sig_s != []:
            sig_s = np.atleast_2d(sig_s)
            assert sig_s.shape[0]==self.n_grps, "Invalid group structure."
            assert sig_s.shape[0]==sig_s.shape[1], "Invalid group structure."
            self.sig_s = Na * sig_s
        # Fission cross sections
        if nu_sig_f != []:
            assert len(nu_sig_f)==self.n_grps, "Invalid group structure."
            self.nu_sig_f = Na * np.atleast_1d(nu_sig_f)
            # Fission spectrum
            if self.n_grps> 1:
                assert len(chi)==self.n_grps, "Invalid group structure."
                self.chi = np.atleast_1d(chi)
            else:
                self.chi = np.array([1.])
        # Velocity
        if v != []:
            assert len(v)==self.n_grps, "Invalid group structure."
            self.v = np.atleast_1d(v)
        # Source
        if q != []:
            assert len(q)==self.n_grps, "Invalid group structure."
            self.q = np.atleast_1d(q)