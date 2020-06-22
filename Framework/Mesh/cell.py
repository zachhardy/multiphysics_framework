#!/usr/bin/env python3

import numpy as np
from Mesh.face import Face1D

class CellBase:
  """ Base class for a cell.

  Parameters
  ----------
  mesh : MeshBase object.
  """
  def __init__(self, mesh):
    # General info
    self._mesh = mesh
    self.dim = mesh.dim
    self.geom = mesh.geom
    self.vertices_per_cell = None
    self.faces_per_cell = None
    self.imat = []
    self.flag = 0
    self.neighbors = []
    # Vertex info
    self.vertex_ids = []
    self.vertices = []
    # Geometric info
    self.width = []
    self.volume = 0.
    self.face_areas = []

    # Faces
    self.faces = []


class Cell1D(CellBase):
  """ One-dimensional cell. """
  def __init__(self, mesh, iel):
    super().__init__(mesh)
    # General info
    self.id = iel
    self.vertices_per_cell = 2
    self.faces_per_cell = 2
    self.imat = mesh.iel2mat[iel]
    self.flag = max(mesh.iel2flags[iel])
    self.neighbors = mesh.iel2neighbors[iel]
    # Vertex info
    self.vertex_ids = mesh.iel2vids[iel]
    self.vertices = mesh.iel2vcoords[iel]
    # Geometric info
    self.width = self.vertices[1] - self.vertices[0]
    self.volume = self.GetVolume()
    self.face_areas = self.GetFaceAreas()
    # Face objects
    self.faces = [Face1D(self, 0), 
                  Face1D(self, 1)]
    
  def GetVolume(self):
    """ Compute the volume of the cell. """
    if self.geom == 'slab':
      return self.width
    elif self.geom == 'cylinder':
      return np.pi*(self.vertices[1][0]**2-self.vertices[0][0]**2)
    elif self.geom == 'sphere':
      return 4/3*np.pi*(self.vertices[1][0]**3-self.vertices[0][0]**3)

  def GetFaceAreas(self):
    """ Compute the area of a face on the cell. """
    A = np.zeros(self.faces_per_cell)
    for iface in range(self.faces_per_cell):
      if self.geom == 'slab':
        A[iface] = 1.
      elif self.geom == 'cylinder':
        A[iface] = 2*np.pi*self.vertices[iface][0]
      elif self.geom == 'sphere':
        A[iface] = 4*np.pi*self.vertices[iface][0]**2
    return A
