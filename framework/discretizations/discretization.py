class Discretization:

  def __init__(self, mesh):
    self.mesh = mesh
    self.dim = mesh.dim
    self.geom = mesh.geom
    self.n_nodes = 0
    self.cell_views = []
    self.grid = None

  def create_grid(self):
    raise NotImplementedError
    
