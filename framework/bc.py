import numpy as np

class BC:

  def __init__(self, boundary_kind, boundary_id, vals=None):
    self.boundary_kind = boundary_kind
    self.boundary_id = boundary_id
    self.vals = None if vals is None else np.array(vals)