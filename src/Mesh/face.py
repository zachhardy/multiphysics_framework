#!/usr/bin/env python3

import numpy as np

class FaceBase:
  """ Base class for a face.

  Parameters
  ----------
  mesh : Derived class of MeshBase
  """
  def __init__(self, cell, iface):
    # General info
    self._mesh = cell._mesh
    self._cell = cell
    self.dim = cell._mesh.dim
    self.geom = cell._mesh.geom
    self.vertices_per_face = None
    self.flag = []
    self.neighbor = -1
    # Vertex info
    self.vertex_ids = []
    self.vertices = []
    # Geometric info
    self.area = 0.
    self.normal = []

  @property
  def neighbor_cell(self):
    if self.flag == 0:
      return self._mesh.cells[self.neighbor]
    else: 
      return self._cell


class Face1D(FaceBase):
  """ One-dimensional face. """
  def __init__(self, cell, iface):
    super().__init__(cell, iface)
    # General info
    self.id = cell.id + iface
    self.vertices_per_face = 1
    self.flag = self._mesh.iel2flags[cell.id][iface]
    self.neighbor = cell.neighbors[iface]
    # Vertex info
    self.vertex_ids = cell.vertex_ids[iface]
    self.vertices = cell.vertices[iface]
    # Geometry info
    self.area = cell.face_areas[iface]
    self.normal = -1 if iface==0 else 1

  @property
  def neighbor_cell(self):
    if self.flag == 0:
      return self._mesh.cells[self.neighbor]
    else: 
      return self._cell

  def GetArea(self):
    """ Compute the area of this face. """
    if self.geom == 'slab':
      return 1.
    elif self.geom == 'cylinder':
      return 2*np.pi*self.vertices[0]
    elif self.geom == 'sphere':
      return 4*np.pi*self.vertices[0]**2
