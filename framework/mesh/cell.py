import numpy as np
from .face import Face1D

class Cell1D:
    
    dim = 1 

    def __init__(self, mesh, iel):
        self.geom = mesh.geom
        self.id = iel
        self.imat = mesh.iel2mat[iel]
        self.flags = mesh.iel2flags[iel]
        self.neighbors = mesh.iel2neighbors[iel]
        # Vertex information
        self.vertices_per_cell = 2
        self.vertex_ids = mesh.iel2vids[iel]
        self.vertices = mesh.iel2vcoords[iel]
        # Geometric information
        self.width = self.vertices[1] - self.vertices[0]
        self.volume = self.GetCellVolume()
        # Face information
        self.faces_per_cell = 2
        self.face_areas = self.GetFaceAreas()
        self.faces = [
            Face1D(self, 0), 
            Face1D(self, 1)
        ]
        
    def GetCellVolume(self):
        if self.geom == 'slab':
            return self.width[0]
        elif self.geom == 'cylinder':
            return np.pi * (
                self.vertices[1][0]**2
                - self.vertices[0][0]**2
            )
        elif self.geom == 'sphere':
            return 4/3*np.pi * (
                self.vertices[1][0]**3
                - self.vertices[0][0]**3
            )

    def GetFaceAreas(self):
        A = np.zeros(self.faces_per_cell)
        for iface in range(self.faces_per_cell):
            if self.geom == 'slab':
                A[iface] = 1.
            elif self.geom == 'cylinder':
                A[iface] = 2*np.pi*self.vertices[iface][0]
            elif self.geom == 'sphere':
                A[iface] = 4*np.pi*self.vertices[iface][0]**2
        return A
