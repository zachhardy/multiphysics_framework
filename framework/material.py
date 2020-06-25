#!/usr/bin/env python3

import sys
import numpy as np

class MaterialBase:
    """ Material base class. """
    def __init__(self, material_id):
        self.material_id = material_id
