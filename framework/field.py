class Field:

    def __init__(self, name, problem, discretization, components=1):
        self.name = name
        # Discretization information
        self.problem = problem
        self.mesh = problem.mesh
        self.discretization = discretization
        self.grid = discretization.grid
        self.n_nodes = discretization.n_nodes
        # Compnents information
        self.components = components
        self.n_dofs = components * self.n_nodes
        # Global DoF information
        self.dof_start = 0
        self.dof_end = 0

    @property
    def u(self):
        return self.problem.u[self.dofs[0]:self.dofs[-1]+1]

    @property
    def u_ell(self):
        return self.problem.u_ell[self.dofs[0]:self.dofs[-1]+1]

    @property
    def u_old(self):
        return self.problem.u_old[self.dofs[0]:self.dofs[-1]+1]

    @property
    def dofs(self):
        return list(range(self.dof_start, self.dof_end))
    
    def component_dofs(self, component=0):
        start = component*self.n_nodes
        end = start + self.n_nodes
        return self.dofs[start:end]
