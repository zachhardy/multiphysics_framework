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
                 D=[], sig_s=[], nu_sig_f=[], chi_p=[], v=[], 
                 decay_const=[], beta=[], chi_d=[], q_prec=[]):
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
      # Neutrons per fission
      # Fission spectrum
      if self.n_grps > 1:
        assert len(chi_p)==self.n_grps, "Invalid group structure."
        self.chi_p = np.atleast_1d(chi_p)
      else:
        self.chi_p = np.array([1.])
    # Velocity
    if v != []:
      assert len(v)==self.n_grps, "Invalid group structure."
      self.v = np.atleast_1d(v)
    # Delayed neutron decay constants
    if decay_const != []:
      self.n_precursors= len(decay_const)
      self.decay_const = np.atleast_1d(decay_const)
      # Delayed neutron fractions
      assert beta!=[], (
        "beta must be provided is delayed neutrons are enabled."
      )
      assert len(beta)==self.n_grps, (
        "Invalid number of delayed fractions."
      )
      self.beta = np.atleast_1d(beta)
      self.beta_total = sum(self.beta)
      # Delayed neutron spectrum
      if self.n_grps == 1:
        self.chi_d = np.ones((self.n_precursors, 1))
      else:
        assert len(chi_d)==self.n_precursors, (
          "Delayed neutron spectra must be provided for "
          "each dnp group for multigroup calculations."
        )
        assert len(chi_d[0])==self.n_grps, (
          "Delayed neutron spectra have invalid group structure."
        )
        self.chi_d = np.atleast_2d(chi_d)
      if q_prec != []:
        assert len(q_prec)==self.n_precursors, (
          "Artificial precursor source does not agree with the "
          "number of precursors provided."
        )
        self.q_prec = np.atleast_1d(q_prec)
    else:
      self.n_precursors = 0
      self.beta_total = 0
