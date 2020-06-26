import sys
import numpy as np

from material import MaterialBase

class HeatConductionMaterial(MaterialBase):
    """ Class for a heat conduction material.

    A heat conduction material is defined at a minimum
    with a conductivity and can include specific heats 
    and sources.

    Parameters
    ----------
    k : float or callable
        The thermal conductivity.
    C_v : float or callable, optional
        The specific heat. 
    q : float or callable, optional
        The heat source.
    """

    material_type = 'hc'

    def __init__(self, material_id=0, k=[], C_v=[], q=[]):
        super().__init__(material_id)
        
        # Conductivity
        assert k != [], "k must be specified."
        self.k = k

        # Specific heat
        if C_v != []:
            self.C_v = C_v
        
        # Source
        if q != []:
            self.q = q
