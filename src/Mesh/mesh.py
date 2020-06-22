#!/usr/bin/env python3

import numpy as np
from Mesh.cell import Cell1D

class MeshBase:
    """ Base class for a mesh.

    This class is meant to generate zoned rectangular meshes.

    Parameters
    ----------
    zone_edges : list of float
        List of zone edges from left to right.
    zone_subdivs : list of int
        List of elements in each zone.
    material_zones : list of int, optional
        Material identifiers within each zone (default is [0],
        which implies a single material zone).
    geom : str 'slab', 'cylinder', or 'sphere'), optional
        The geometry type (default is 'slab').
    """
    def __init__(self, zone_edges, zone_subdivs, 
                material_zones=[0],
                geom='slab'):

        # --- Checks
        assert len(zone_edges) == len(zone_subdivs)+1, (
            "zone_edges and zone_subdivs are incompatible."
        )
        assert len(material_zones) >= 1, "Invalid material_zones input."
        assert geom in ['slab', 'cylinder', 'sphere'], "Invalid geom input."
        # ---

        # Private attributes
        self._zone_edges = zone_edges
        self._zone_subdivs = zone_subdivs
        self._material_zones = material_zones

        # Public attributes
        self.n_zones = len(zone_subdivs)
        self.dim = 1
        self.geom = geom
        self.n_el = None
        self.n_faces = None
        self.vcoords = None
        self.iel2mat = None
        self.iel2vids = None
        self.iel2vcoords = None
        self.iel2flags = None
        self.iel2neighbors = None

    def refine(self):
        raise NotImplementedError(
            "This method must be implemented in derived classes."
        )


class Mesh1D(MeshBase):
    """ One-dimensional mesh.

    For additional documentation, see MeshBase.
    """
    def __init__(self, zone_edges, zone_subdivs,
                 material_zones=[0],
                 geom='slab'):
        super().__init__(zone_edges, zone_subdivs,
                         material_zones, geom)
        self.n_el = sum(self._zone_subdivs)
        self.n_faces = self.n_el + 1

        # Create a list of the element edge coordinates
        zedges, zsubdivs = self._zone_edges, self._zone_subdivs
        vcoords = np.linspace(zedges[0], zedges[1], zsubdivs[0]+1)
        for izn in range(1, self.n_zones):
            tmp = np.linspace(zedges[izn], zedges[izn+1], zsubdivs[izn]+1)
            vcoords = np.insert(vcoords, len(vcoords), tmp[1:])
        vcoords = vcoords.reshape(self.n_el+1, -1)
        self.vcoords = vcoords

        # Define element-wise boundary indicator flags
        iel2flags = np.zeros((self.n_el, 2), dtype=int)
        iel2flags[0,0] = 1
        iel2flags[self.n_el-1,1] = 2
        self.iel2flags = iel2flags

        # Define element-wise vertex information
        iel2vids = np.zeros((self.n_el, 2), dtype=int)
        iel2vcoords = np.zeros((self.n_el, 2, self.dim))
        iel2neighbors = np.zeros((self.n_el, 2), dtype=int)
        for iel in range(self.n_el):
            iel2vids[iel] = [iel, iel+1]
            iel2vcoords[iel] = [vcoords[iel], vcoords[iel+1]]
            iel2neighbors[iel,:] = [iel-1, iel+1]
        # Correct neighbors for boundaries
        iel2neighbors[0,0] = -1
        iel2neighbors[self.n_el-1,1] = -2
        self.iel2vids = iel2vids
        self.iel2vcoords = iel2vcoords
        self.iel2neighbors = iel2neighbors

        # Define element-wise material id
        iel2mat = []
        for izn in range(self.n_zones):
            iel2mat += [material_zones[izn]] * zone_subdivs[izn]
        self.iel2mat = iel2mat

        # Generate cells
        cells, faces = [], []
        for iel in range(self.n_el):
            cells += [Cell1D(self, iel)]
            faces += [cells[-1].faces[0]]
        faces += [cells[-1].faces[1]]
        self.cells, self.faces = cells, faces
        self.bndry_cells = [self.cells[0], self.cells[-1]]
