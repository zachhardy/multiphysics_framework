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
        # ===== Input checks
        try:
            if not isinstance(n_qpts, int):
                raise ValueError("n_qpts must be an int.")
            if n_qpts < 0:
                raise ValueError("n_qpts must be positive.")
        except ValueError as err:
            msg = "Aborting program due to ValueError:\n\t{}"
            print(msg.format(err.args[0]))
            sys.exit(-1)
            
        self.n_qpts = n_qpts # number of quadrature points
        self.Lq = 2.0 # quadrature interval length

        # Generate gauss legendre quadrature
        [self.qpoints, self.weights] = np.polynomial.legendre.leggauss(self.n_qpts)
