#!/usr/bin/env python3

import numpy as np
from ..Face.face_1d import Face1D

class CellBase:
  """ Base class for a cell.

  Parameters
  ----------
  mesh : MeshBase-like
  iel : int
    The cell index
  """
  def __init__(self, mesh, iel):
    self.mesh = mesh
    self.dim = mesh.dim
    self.geom = mesh.geom
    self.id = iel
    self.vertices_per_cell = 0
    self.faces_per_cell = 0
    self.imat = []
    self.flag = 0
    self.neighbors = []
    # vertex info
    self.vertex_ids = []
    self.vertices = []
    # geometric info
    self.width = []
    self.volume = 0.
    self.face_areas = []
    # faces
    self.faces = []

  def GetCellVolume(self):
    raise NotImplementedError(
      "This method must be implemented in derived classes."
    )

  def GetFaceAreas(self):
    raise NotImplementedError(
      "This method must be implemented in derived classes."
    )