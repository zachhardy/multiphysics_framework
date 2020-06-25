#!/usr/bin/env python3

import numpy as np

kinds = [
    'reflective',
    'marshak',
    'source',
    'vacuum',
    'zero_flux',
    'dirichlet',
    'neumann',
    'robin'
]

class BC:
    """ Class for boundary conditions

    Parameters
    ----------
    boundary_kind : str
        The boundary kind. Options specified above.
    boundary_id : int
        The boundary id this BC belongs to.
    vals : array-like, optional
        The values associated with this BC.
    """
    def __init__(self, boundary_kind, boundary_id, vals=None):
        # Checks
        assert boundary_kind in kinds, 'Unrecognized BC kind.'
        if boundary_kind not in ['reflective', 
                             'vacuum', 
                             'zero_flux']:
            assert vals is not None, (
                "vals must be given for this boundary kind."
            )

        self.boundary_kind = boundary_kind
        self.boundary_id = boundary_id
        self.vals = None if vals is None else np.array(vals)