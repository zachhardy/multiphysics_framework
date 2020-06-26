#!/usr/bin/env python3

import sys
import numpy  as np

class GLQuadrature:
    """
    Gauss-Legendre quadrature.
    
    Parameters
    ----------
    n_qpts: int, default=2
        The number of quadrature points.
    """

    def __init__(self, n_qpts=2):
        assert isinstance(n_qpts, int), "n_qpts must be int."
        assert n_qpts > 0, "n_qpts must be greater than 0."
        
        self.n_qpts = n_qpts 
        self.Lq = 2.0 # interval size

        # Generate gauss legendre quadrature
        [self.qpoints, self.weights] = np.polynomial.legendre.leggauss(self.n_qpts)
