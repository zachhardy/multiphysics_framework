#!/usr/bin/env python3

import numpy as np
from .Cell.cell_1d import Cell1D

class MeshBase:
  """ Template class for simple, rectangular meshes.

  Parameters
  ----------
  zone_edges : list of float
    List of zone edges from left to right.
  zone_subdivs : list of int
    List of elements in each zone.
  material_zones : list of int, optional
    List of material ids for each zone. Default is [0].
  geom : 'slab', 'cylinder', or 'sphere', optional
    Default is 'slab'.
  """
  def __init__(self, zone_edges, zone_subdivs, 
        material_zones=[0],
        geom='slab'):

    # Checks
    assert len(zone_edges) == len(zone_subdivs)+1, (
      "zone_edges and zone_subdivs are incompatible."
    )
    assert len(material_zones) >= 1, "Invalid material_zones input."
    assert geom in ['slab', 'cylinder', 'sphere'], "Invalid geom input."

    self._zone_edges = zone_edges
    self._zone_subdivs = zone_subdivs
    self._material_zones = material_zones
    self.n_zones = len(zone_subdivs)
    
    self.dim = 0
    self.geom = geom
    self.n_el = 0
    self.n_faces = 0
    self.vcoords = []
    self.iel2mat = []
    self.iel2vids = []
    self.iel2vcoords = []
    self.iel2flags = []
    self.iel2neighbors = []

  def refine(self):
    raise NotImplementedError(
      "This method must be implemented in derived classes."
    )
