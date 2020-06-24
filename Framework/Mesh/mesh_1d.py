#!/usr/bin/env python3

import numpy as np
from .mesh_base import MeshBase
from .Cell.cell_1d import Cell1D

class Mesh1D(MeshBase):
  """ One-dimensional mesh.

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
         material_zones=[0], geom='slab'):
    super().__init__(
      zone_edges, zone_subdivs, 
      material_zones, geom
    )
    self.dim = 1
    self.n_el = sum(self._zone_subdivs)
    self.n_faces = self.n_el + 1

    # create a list of the element edge coordinates
    zedges, zsubdivs = self._zone_edges, self._zone_subdivs
    vcoords = np.linspace(zedges[0], zedges[1], zsubdivs[0]+1)
    for izn in range(1, self.n_zones):
      tmp = np.linspace(zedges[izn], zedges[izn+1], zsubdivs[izn]+1)
      vcoords = np.insert(vcoords, len(vcoords), tmp[1:])
    vcoords = vcoords.reshape(self.n_el+1, -1)
    self.vcoords = vcoords

    # define element-wise boundary indicator flags
    iel2flags = np.zeros((self.n_el, 2), dtype=int)
    iel2flags[0,0] = 1
    iel2flags[self.n_el-1,1] = 2
    self.iel2flags = iel2flags

    # define element-wise vertex information
    iel2vids = np.zeros((self.n_el, 2), dtype=int)
    iel2vcoords = np.zeros((self.n_el, 2, self.dim))
    iel2neighbors = np.zeros((self.n_el, 2), dtype=int)
    for iel in range(self.n_el):
      iel2vids[iel] = [iel, iel+1]
      iel2vcoords[iel] = [vcoords[iel], vcoords[iel+1]]
      iel2neighbors[iel,:] = [iel-1, iel+1]

    # correct neighbors for boundaries
    iel2neighbors[0,0] = -1
    iel2neighbors[self.n_el-1,1] = -2
    self.iel2vids = iel2vids
    self.iel2vcoords = iel2vcoords
    self.iel2neighbors = iel2neighbors

    # define element-wise material id
    iel2mat = []
    for izn in range(self.n_zones):
      iel2mat += [material_zones[izn]] * zone_subdivs[izn]
    self.iel2mat = iel2mat

    # generate cells
    cells, faces = [], []
    for iel in range(self.n_el):
      cells += [Cell1D(self, iel)]
      faces += [cells[-1].faces[0]]
    faces += [cells[-1].faces[1]]
    self.cells, self.faces = cells, faces
    self.bndry_cells = [self.cells[0], self.cells[-1]]
