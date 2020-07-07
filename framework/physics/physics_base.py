class PhysicsBase:

  def __init__(self, problem):
    # Add the physics to the physics stack
    problem.physics.append(self)
    self.problem = problem
    self.mesh = problem.mesh
    self.materials = []
    self.fields = []

    # Discritization info
    self.grid = []
    self.n_nodes = 0
    self.n_dofs = 0

    # Boundary and initial conditions
    self.bcs = []
    self.ics = []

    # Boolean flags
    self.is_coupled = False
    self.is_nonlinear = False

  def _register_field(self, field):
    # Add the field to the field stack.
    self.problem.fields.append(field)
    self.fields.append(field)
    # Add the discretization to the physics
    self.discretization = field.discretization
    self.grid = self.discretization.grid
    self.n_nodes = self.discretization.n_nodes
    self.n_dofs += field.n_dofs
    # Set the beginning of the field dof range.
    field.dof_start = self.problem.n_dofs
    # Add the field to the problem, and update the
    # appropriate attributes.
    self.problem.n_fields += 1
    self.problem.n_dofs += field.n_dofs
    self.problem.u.resize(self.problem.n_dofs)
    self.problem.u_ell.resize(self.problem.n_dofs)
    # Set the end of the field dof range.
    field.dof_end = self.problem.n_dofs

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
    start = min([field.dof_start for field in self.fields])
    end = max([field.dof_end for field in self.fields])
    return list(range(start, end))

  def compute_old_physics_action(self):
    raise NotImplementedError

  def solve_system(self):
    raise NotImplementedError
    
  def _parse_materials(self, material_type):
    materials = []
    for material in self.problem.materials:
      if material.material_type == material_type:
        materials += [material]
    materials.sort(key=lambda x: x.material_id)
    return self._validate_materials(materials)
  
  def _validate_materials(self, materials):
    raise NotImplementedError
