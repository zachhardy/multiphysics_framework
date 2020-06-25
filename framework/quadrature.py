#!/usr/bin/env python3

import sys
import numpy  as np

class Quadrature:
    """
    Gauss-Legendre quadrature.
    
    Parameters
    ----------
    n_qpts: int; The number of points. Default is 2.
    """

    def __init__(self, n_qpts=2):
        assert isinstance(n_qpts, int), "n_qpts must be int."
        assert n_qpts > 0, "n_qpts must be greater than 0."
        
        self.n_qpts = n_qpts 
        self.Lq = 2.0 # interval size

        # Generate gauss legendre quadrature
        [self.qpoints, self.weights] = np.polynomial.legendre.leggauss(self.n_qpts)
