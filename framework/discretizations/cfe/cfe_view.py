import numpy as np

class CellCFEView1D:
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

        # Node information
        self.node_ids = np.arange(
            cell.vertex_ids[0]*self.porder,
            (cell.vertex_ids[0]+1)*self.porder+1,
            dtype=int
        )
        self.nodes = np.linspace(
            cell.vertices[0],
            cell.vertices[1],
            self.porder+1
        )

        # Jacobian to reference element
        self.J = cell.width/self.qrule.Lq
        self.Jinv = 1/self.J
        self.Jdet = self.J
        self.JxW = self.Jdet * self.qrule.weights

        # Global coordinate system information
        self.qpoint_glob = self.get_global_qpoints()
        self.Jcoord = self.get_coord_sys_jacobian()

        # Shape function information
        self._shape = discretization._shape
        self._grad_shape = discretization._grad_shape
        self.shape_values = self.get_shape_values()
        self.grad_shape_values = self.get_grad_shape_values()

    def cell_dof_map(self, local_id, component=0):
        """ Map a local cell dof to a global dof.
        
        Parameters
        ----------
        local_id : int
            The local dof index on the cell.
        component : int, optional
            The global component for the dof. Default is 0.
        """
        assert local_id < self.nodes_per_cell, "Invalid local_id."
        return self.node_ids[local_id] + component*self.n_nodes

    def face_dof_map(self, face_id, component=0):
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
        assert face_id < 2, "Invalid face_id."
        if face_id == 0:
            return self.node_ids[0] + component*self.n_nodes
        elif face_id == 1:
            return self.node_ids[-1] + component*self.n_nodes

    def intV_shapeI_shapeJ(self, i, j, coef=None):
        """ Integrate a reaction-like term over a cell volume.
        
        Parameters
        ----------
        i : int
            Test function index.
        j : int
            Trial function index.
        coef : numpy.ndarray (n_qpts,)
            Coefficients at each quadrature point.
        """
        val = 0 # init integral result
        coef = self.format_coef(coef)
        for qp in range(self.n_qpts):
            val += (
                self.Jcoord[qp] * self.JxW[qp]
                * coef[qp] * self.shape_values[qp][i] 
                * self.shape_values[qp][j]
            )
        return val

    def intV_gradShapeI_gradShapeJ(self, i, j, coef=None):
        """ Integrate a diffusion-like term over a cell volume.
        
        Parameters
        ----------
        i : int
            Test function index.
        j : int
            Trial function index.
        coef : numpy.ndarray (n_qpts,)
            Coefficients at each quadrature point.
        """
        val = 0
        coef = self.format_coef(coef)
        for qp in range(self.n_qpts):
            val += (
                self.Jcoord[qp] * self.JxW[qp] 
                * coef[qp] * self.grad_shape_values[qp][i] 
                * self.grad_shape_values[qp][j]
            )
        return val

    def intV_shapeI(self, i, coef=None):
        """ Integrate a source-like term over a cell volume.
        
        Parameters
        ----------
        i : int
            Test function index.
        coef : numpy.ndarray (n_qpts,)
            Coefficients at each quadrature point.
        """
        val = 0
        coef = self.format_coef(coef)
        for qp in range(self.n_qpts):
            val += (
                self.Jcoord[qp] * self.JxW[qp] 
                * coef[qp] * self.shape_values[qp][i]
            )
        return val

    def quadrature_solution(self, u):
        """ Get the solution at quadrature points.

        Parameters
        ----------
        u : numpy.ndarray
        """
        u_qp = np.zeros(self.n_qpts)
        for qp in range(self.n_qpts):
            u_qp[qp] = np.dot(
                u[self.node_ids], 
                self.shape_values[qp]
            )
        return u_qp

    def average_solution(self, u):
        """ Get the average solution on this cell.

        Parameters
        ----------
        u : numpy.ndarray
        """
        u_avg = 0
        u_qp = self.quadrature_solution(u)
        for qp in range(self.n_qpts):
            u_avg += self.Jcoord[qp] * self.JxW[qp] * u_qp[qp]
        return u_avg / self.volume

    def format_coef(self, coef):
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

    def get_global_qpoints(self):
        """ Get the quadrature points in global coordinates. """
        return self.J * (self.qrule.qpoints + 1) + self.nodes[0]

    def get_coord_sys_jacobian(self):
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

    def get_shape_values(self):
        """ Compute phi at the quadrature points. """
        qpoints = self.qrule.qpoints
        shape_vals = np.zeros((self.n_qpts, self.porder+1))
        for qp in range(self.n_qpts):
            for i, shape in enumerate(self._shape):
                shape_vals[qp][i] = shape(qpoints[qp])
        return shape_vals
    
    def get_grad_shape_values(self):
        """ Compute grad_phi at the quadrature points. """
        qpoints = self.qrule.qpoints
        grad_shape_vals = np.zeros((self.n_qpts, self.porder+1))
        for qp in range(self.n_qpts):
            for i, grad_shape in enumerate(self._grad_shape):
                grad_shape_vals[qp][i] = (
                    grad_shape(qpoints[qp]) * self.Jinv
                ) 
        return grad_shape_vals
