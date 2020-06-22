#!/usr/bin/env python3

import numpy as np

class CellCFEView1D:
  """ Continuous finite element cell view.

  Parameters
  ----------
  disctetization : CFE object.
  """
  def __init__(self, discretization):
    # General information
    self.porder = discretization.porder
    self.geom = discretization.geom
    self.qrule = discretization.qrule
    self.n_qpts = self.qrule.n_qpts
    self.n_nodes = discretization.n_nodes
    self.nodes_per_cell = self.porder + 1

    # Cell storage
    self.cell = None

    # Node information
    self.node_ids = None
    self.nodes = None

    # Jacobian information
    self.J = None
    self.Jinv = None
    self.Jdet = None
    self.JxW = None

    # Global coordinates
    self.qpoint_glob = np.zeros(self.n_qpts)
    self.Jcoord = np.zeros(self.n_qpts)

    # Shape function information
    self._phi = discretization._phi
    self._grad_phi = discretization._grad_phi
    self.phi = np.zeros((self.n_qpts, self.porder+1))
    self.grad_phi = np.zeros((self.n_qpts, self.porder+1))

    # Precompute phi values
    self.GetPhiValues()

  def reinit(self, cell):
    """ Reinit the cell view for cell. """
    # store the cell
    self.cell = cell
    
    # node information
    self.node_ids = np.arange(
      cell.id*self.porder,
      (cell.id+1)*self.porder+1,
      dtype=int)
    self.nodes = np.linspace(
      cell.vertices[0],
      cell.vertices[1],
      self.porder+1)
    
    # jacobian information
    self.J = cell.width/self.qrule.Lq
    self.Jinv = 1/self.J
    self.Jdet = self.J
    self.JxW = self.Jdet * self.qrule.wq

    # global coordinates
    self.GetGlobalQPoints()
    self.GetCoordSysJacobian()

    # compute grad shape function values
    self.GetGradPhiValues()

  def CellDoFMap(self, local_id, component=0):
    """ Map a local cell dof to a global dof.
    
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
    """ Get a face dof for a given component. 
    
    Parameters
    ----------
    face_id : int
      The face number to get the dof from.
    component : int, optional
      The global component for the dof. Default is 0.
    
    Returns
    -------
    int, The mapped dof.
    """
    assert face_id < len(self.cell.faces), "Invalid face_id."
    if face_id == 0:
      return self.node_ids[0] + component*self.n_nodes
    elif face_id == 1:
      return self.node_ids[-1] + component*self.n_nodes

  def Integrate_PhiI_PhiJ(self, i, j, coef=None):
    """ Integrate coef * phi_i * phi_j over a cell volume.
    
    Parameters
    ----------
    i : int
      Test function index.
    j : int
      Trial function index.
    coef : numpy.ndarray (n_qpts,)
      Coefficients at each quadrature point.
    
    Returns
    -------
    float, The integration result.
    """
    n_qpts = self.n_qpts # shorthand
    phi = self.phi # shorthand

    if isinstance(coef, float):
      coef *= np.ones(n_qpts)
    elif coef is None:
      coef = np.ones(n_qpts)
    elif len(coef) != n_qpts:
      raise ValueError("Invalid coef input.")
    
    val = 0 # init integral result
    for qp in range(n_qpts):
      val += (
        self.Jcoord[qp] * self.JxW[qp]
        * coef[qp] * phi[qp][i] * phi[qp][j]
      )
    return val

  def Integrate_GradPhiI_GradPhiJ(self, i, j, coef=None):
    """ Integrate coef * grad_phi_i * grad_phi_j over a cell volume.
    
    Parameters
    ----------
    i : int
      Test function index.
    j : int
      Trial function index.
    coef : numpy.ndarray (n_qpts,)
      Coefficients at each quadrature point.
    
    Returns
    -------
    float, The integration result.
    """
    n_qpts = self.n_qpts # shorthand
    dphi = self.grad_phi # shorthand
    coef = self.FormatCoef(coef)
    # Quadrature integration
    val = 0 # init integral result
    for qp in range(n_qpts):
      val += (
        self.Jcoord[qp] * self.JxW[qp] 
        * coef[qp] * dphi[qp][i] * dphi[qp][j]
      )
    return val

  def Integrate_PhiI(self, i, coef=None):
    """ Integrate coef * phi_i over a cell volume.
    
    Parameters
    ----------
    i : int
      Test function index.
    coef : numpy.ndarray (n_qpts,)
      Coefficients at each quadrature point.
    
    Returns
    -------
    float, The integration result.
    """
    n_qpts = self.n_qpts # shorthand
    phi = self.phi # shorthand
    coef = self.FormatCoef(coef)
    # Quadrature integration
    val = 0 # init integral result
    for qp in range(n_qpts):
      val += (
        self.Jcoord[qp] * self.JxW[qp] 
        * coef[qp] * phi[qp][i]
      )
    return val

  def SolutionAtQuadrature(self, u):
    """ Get the solution at quadrature points.

    Parameters
    ----------
    u : numpy.ndarray
      A solution vector

    Returns
    -------
    numpy.ndarray (n_qpts, -1)
    """
    n_qpts = self.n_qpts # shorthand
    u_qp = np.zeros(n_qpts)
    for qp in range(n_qpts):
      u_qp[qp] = np.dot(u[self.node_ids], self.phi[qp])
    return u_qp

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
    self.qpoint_glob = self.J * (self.qrule.xq + 1) + self.nodes[0]

  def GetCoordSysJacobian(self):
    """ Get the coordinate system jacobians. """
    x = self.qpoint_glob # shorthand
    for qp in range(self.n_qpts):
      if self.geom == 'slab':
        self.Jcoord[qp] = 1.
      elif self.geom == 'cylinder':
        self.Jcoord[qp] = 2 * np.pi * x[qp]
      elif self.geom == 'sphere':
        self.Jcoord[qp] = 4 * np.pi * x[qp]**2

  def GetPhiValues(self):
    """ Compute phi at the quadrature points. """
    x = self.qrule.xq # shorthand
    for qp in range(self.n_qpts):
      for i in range(len(self._phi)):
        self.phi[qp][i] = self._phi[i](x[qp])
  
  def GetGradPhiValues(self):
    """ Compute grad_phi at the quadrature points. """
    x = self.qrule.xq # shorthand
    self.grad_phi *= 0 # clear grad phi values
    for qp in range(self.n_qpts):
      for i in range(len(self._grad_phi)):
        self.grad_phi[qp][i] = self._grad_phi[i](x[qp])*self.Jinv
