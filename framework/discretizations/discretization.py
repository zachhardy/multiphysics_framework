#!/usr/bin/env python3

import numpy as np

class Discretization:
  """ Base class to describe a spatial discretization. """
  def __init__(self, mesh):
    self.mesh = mesh
    self.dim = mesh.dim
    self.geom = mesh.geom
    self.n_nodes = 0
    self.grid = None

  def create_grid(self):
    raise NotImplementedError
    
