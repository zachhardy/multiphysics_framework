#!/usr/bin/env python3

import numpy as np

class CFEView1D:
  """ Continuous finite element cell view.

  Parameters
  ----------
  disctetization : CFE object.
  """
  def __init__(self, discretization, cell):
    self.porder = discretization.porder
    self.geom = discretization.geom
    self.qrule = discretization.qrule
    self.n_qpts = self.qrule.n_qpts
    self.n_nodes = discretization.n_nodes
    self.nodes_per_cell = self.porder + 1
    self.volume = cell.volume
    self.width = cell.width

    # nodes
    self.node_ids = np.arange(
      cell.id*self.porder,
      (cell.id+1)*self.porder+1,
      dtype=int)
    self.nodes = np.linspace(
      cell.vertices[0],
      cell.vertices[1],
      self.porder+1)

    # jacobian
    self.J = cell.width/self.qrule.Lq
    self.Jinv = 1/self.J
    self.Jdet = self.J
    self.JxW = self.Jdet * self.qrule.wq

    # global coordinates/coordinate system
    self.qpoint_glob = self.GetGlobalQPoints()
    self.Jcoord = self.GetCoordSysJacobian()

    # shape functions
    self._shape = discretization._shape
    self._grad_shape = discretization._grad_shape
    self.shape_vals = self.GetShapeValues()
    self.grad_shape_vals = self.GetGradShapeValues()

  def CellDoFMap(self, local_id, component=0):
    """ Map a local cell dof to a global.
    
    Parameters
    ----------
    local_id : int
      The local dof index on the cell.
    component : int, optional
      The global component for the dof. Default is 0.

    Returns
    -------
    int, The mapped dof.
    """
    assert local_id < self.nodes_per_cell, "Invalid local_id."
    return self.node_ids[local_id] + component*self.n_nodes

  def FaceDoFMap(self, face_id, component=0):
    """ Map a local face dof to a global.
    
    Parameters
    ----------
    face_id : int
      The face on the cell.
    component : int, optional
      The component of the solution.
    """
    assert face_id < 2, "Invalid face_id."
    if face_id == 0:
      return self.node_ids[0] + component*self.n_nodes
    elif face_id == 1:
      return self.node_ids[-1] + component*self.n_nodes

  def Integrate_PhiI_PhiJ(self, i, j, coef=None):
    """ Integrate a reaction-like term over the cell.
    
    Parameters
    ----------
    i : int
      Test function index.
    j : int
      Trial function index.
    coef : float, array-like, None
      Quadrature point-wise coefficients for 
      integration. If float, the constant is 
      mapped to a n_qpts vector. If None, the
      coefficients are unity.
    """
    val = 0
    coef = self.FormatCoef(coef)
    for qp in range(self.n_qpts):
      val += (
        self.Jcoord[qp] * self.JxW[qp]
        * coef[qp] * self.shape_vals[qp][i] 
        * self.shape_vals[qp][j]
      )
    return val

  def Integrate_GradPhiI_GradPhiJ(self, i, j, coef=None):
    """ Integrate a diffusion-like term over a cell volume.
    
    Parameters
    ----------
    i : int
      Test function index.
    j : int
      Trial function index.
    coef : float, array-like, None
      Quadrature point-wise coefficients for 
      integration. If float, the constant is 
      mapped to a n_qpts vector. If None, the
      coefficients are unity.
    """
    val = 0
    coef = self.FormatCoef(coef)
    for qp in range(self.n_qpts):
      val += (
        self.Jcoord[qp] * self.JxW[qp] 
        * coef[qp] * self.grad_shape_vals[qp][i] 
        * self.grad_shape_vals[qp][j]
      )
    return val

  def Integrate_PhiI(self, i, coef=None):
    """ Integrate a source- or lumped-like term over a cell.
    
    Parameters
    ----------
    i : int
      Test function index.
    coef : float, array-like, None
      Quadrature point-wise coefficients for 
      integration. If float, the constant is 
      mapped to a n_qpts vector. If None, the
      coefficients are unity.
    """
    val = 0
    coef = self.FormatCoef(coef)
    for qp in range(self.n_qpts):
      val += (
        self.Jcoord[qp] * self.JxW[qp] 
        * coef[qp] * self.shape_vals[qp][i]
      )
    return val

  def SolutionAtQuadrature(self, u):
    """ Solution at quadrature points on the cell.

    Parameters
    ----------
    u : numpy.ndarray
    """
    u_qp = np.zeros(self.n_qpts)
    for qp in range(self.n_qpts):
      u_qp[qp] = np.dot(u[self.node_ids], self.shape_vals[qp])
    return u_qp

  def SolutionAverage(self, u):
    """ Average of the solution on this cell.

    Parameters
    ----------
    u : numpy.ndarray
    """
    u_avg = 0
    for qp in range(self.n_qpts):
      u_avg += (
        self.Jcoord[qp] * self.JxW[qp]
        * np.dot(u[self.node_ids], self.shape_vals[qp])
      )
    return u_avg / self.volume

  def FormatCoef(self, coef):
    """ Format a coefficient for quadrature integration.

    Parameters
    ----------
    coef : float, array-like, or None
    
    Returns
    -------
    numpy.ndarry (n_qpts,)
      The formatted coefficients.
    """
    if isinstance(coef, float):
      return coef * np.ones(self.n_qpts)
    elif coef is None:
      return np.ones(self.n_qpts)
    else:
      if len(coef) != self.n_qpts:
        raise ValueError("Improperly sized coef.")
      return coef

  def GetGlobalQPoints(self):
    """ Get the quadrature points in global coordinates. """
    return self.J * (self.qrule.xq + 1) + self.nodes[0]

  def GetCoordSysJacobian(self):
    """ Get the coordinate system jacobians. """
    Jcoord = np.zeros(self.n_qpts)
    for qp in range(self.n_qpts):
      if self.geom == 'slab':
        Jcoord[qp] = 1.
      elif self.geom == 'cylinder':
        Jcoord[qp] = 2 * np.pi * self.qpoint_glob[qp]
      elif self.geom == 'sphere':
        Jcoord[qp] = 4 * np.pi * self.qpoint_glob[qp]**2
    return Jcoord

  def GetShapeValues(self):
    """ Shape values at the quadrature points. """
    qpoint = self.qrule.xq
    shape_vals = np.zeros((self.n_qpts, self.porder+1))
    for qp in range(self.n_qpts):
      for i, shape in enumerate(self._shape):
        shape_vals[qp][i] = shape(qpoint[qp])
    return shape_vals
  
  def GetGradShapeValues(self):
    """ Gradient of shape functions at the quadrature points. """
    qpoint = self.qrule.xq
    grad_shape_vals = np.zeros((self.n_qpts, self.porder+1))
    for qp in range(self.n_qpts):
      for i, grad_shape in enumerate(self._grad_shape):
        grad_shape_vals[qp][i] = grad_shape(qpoint[qp])*self.Jinv
    return grad_shape_vals
