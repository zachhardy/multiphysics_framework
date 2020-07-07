class Face1D:

  dim = 1
  
  def __init__(self, cell, iface):
    self.geom = cell.geom
    self.id = cell.id + iface
    self.vertices_per_face = 1
    self.flag = cell.flags[iface]
    self.neighbor = cell.neighbors[iface]
    self.vertex_ids = cell.vertex_ids[iface]
    self.vertices = cell.vertices[iface]
    self.area = cell.face_areas[iface]
    self.normal = [-1] if iface==0 else [1]