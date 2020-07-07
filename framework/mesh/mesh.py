import numpy as np

from .cell import Cell1D

geom_types = [
  'slab', 'cylinder', 'sphere'
]

class Mesh1D:

  dim = 1

  def __init__(self, zone_edges, zone_subdivs,
                 material_zones=[0], source_zones=[0],
                 geom='slab'):
    self.geom = geom
    self.zone_edges = zone_edges
    self.zone_subdivs = zone_subdivs
    self.material_zones = material_zones
    self.source_zones = source_zones
    
    # Counts of things
    self.n_zones = len(zone_subdivs)
    self.n_el = sum(zone_subdivs)
    self.n_faces = self.n_el + 1
    self.n_vertices = self.n_faces

    # Checks
    if len(material_zones) != self.n_zones:
      msg = "There must be n_zones material zones"
      raise ValueError(msg)
    if len(source_zones) != self.n_zones:
      msg = "There must be n_zones source zones"
      raise ValueError(msg)

    # Create a list of the element edge coordinates.
    vcoords = np.linspace(
      zone_edges[0], zone_edges[1], zone_subdivs[0]+1
    )
    for izn in range(1, self.n_zones):
      tmp = np.linspace(
        zone_edges[izn], zone_edges[izn+1], 
        zone_subdivs[izn]+1
      )
      vcoords = np.insert(vcoords, len(vcoords), tmp[1:])
    vcoords = vcoords.reshape(self.n_el+1, -1)
    self.vcoords = vcoords

    # Define element-wise vertex information.
    iel2vids = np.zeros((self.n_el, 2), dtype=int)
    iel2vcoords = np.zeros((self.n_el, 2, self.dim))
    iel2neighbors = np.zeros((self.n_el, 2), dtype=int)
    for iel in range(self.n_el):
        iel2vids[iel] = [iel, iel+1]
        iel2vcoords[iel] = [vcoords[iel], vcoords[iel+1]]
        iel2neighbors[iel,:] = [iel-1, iel+1]

    # Correct neighbors for boundaries.
    iel2neighbors[0,0] = -1
    iel2neighbors[self.n_el-1,1] = -2
    self.iel2vids = iel2vids
    self.iel2vcoords = iel2vcoords
    self.iel2neighbors = iel2neighbors

    # Define element-wise boundary indicator flags.
    iel2flags = np.zeros((self.n_el, 2), dtype=int)
    iel2flags[0,0] = 1
    iel2flags[self.n_el-1,1] = 2
    self.iel2flags = iel2flags

    # Define element-wise material ids.
    iel2mat, iel2src = [], []
    for izn in range(self.n_zones):
        iel2mat += [material_zones[izn]] * zone_subdivs[izn]
        iel2src += [source_zones[izn]] * zone_subdivs[izn]
    self.iel2mat = iel2mat
    self.iel2src = iel2src

    # Generate the cells, faces, and bndry faces.
    cells, faces = [], []
    for iel in range(self.n_el):
        cells += [Cell1D(self, iel)]
        faces += [cells[-1].faces[0]]
    faces += [cells[-1].faces[1]]
    self.cells, self.faces = cells, faces
    self.bndry_cells = [self.cells[0], self.cells[-1]]
