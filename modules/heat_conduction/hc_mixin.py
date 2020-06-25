#!/usr/bin/env python3

import sys
import numpy as np
from scipy.sparse.linalg import spsolve

from physics.physics_system import PhysicsSystem
from material import HeatConductionMaterial

class HCMixin:

    def _parse_materials(self):
        """ Get neutronics properties and sort by zone. """
        materials = []
        for material in self.problem.materials:
            if isinstance(material, HeatConductionMaterial):
                materials += [material]
        materials.sort(key=lambda x: x.material_id)
        return materials
