#!/usr/bin/env python3

from .face_base import FaceBase

class Face1D(FaceBase):
  """ One-dimensional face. 
  
  Parameters
  ----------
  cell : CellBase-like
  iface : int
    The face index.
  """
  def __init__(self, cell, iface):
    super().__init__(cell, iface)
    self.id = cell.id + iface
    self.vertices_per_face = 1
    self.flag = self.mesh.iel2flags[cell.id][iface]
    self.neighbor = cell.neighbors[iface]
    # vertex info
    self.vertex_ids = cell.vertex_ids[iface]
    self.vertices = cell.vertices[iface]
    # geometry info
    self.area = cell.face_areas[iface]
    self.normal = -1 if iface==0 else 1