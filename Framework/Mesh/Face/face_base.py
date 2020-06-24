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
    self.mesh = cell.mesh
    self.cell = cell
    self.dim = cell.mesh.dim
    self.geom = cell.mesh.geom
    self.vertices_per_face = None
    self.flag = []
    self.neighbor = -1
    # vertex info
    self.vertex_ids = []
    self.vertices = []
    # geometric info
    self.area = 0.
    self.normal = []

  @property
  def neighbor_cell(self):
    if self.flag == 0:
      return self.mesh.cells[self.neighbor]
    else: 
      return self.cell
