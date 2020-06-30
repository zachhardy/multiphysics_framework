class PhysicsBase:

    problem = None
    mesh = None
    materials = []
    discretization = None
    fields = []

    # Discritization info
    grid = []
    n_nodes = 0
    n_dofs = 0

    # Boundary and initial conditions
    bcs = []
    ics = []

    # Boolean flags
    is_coupled = False
    is_nonlinear = False

    def __init__(self, problem):
        # Add the physics to the physics stack
        problem.physics.append(self)
        self.problem = problem
        self.mesh = problem.mesh

    def _register_field(self, field):
        # Add the field to the field stack.
        self.problem.fields.append(field)
        self.fields.append(field)
        # Add the discretization to the physics
        self.discretization = field.discretization
        self.grid = self.discretization.grid
        self.n_nodes = self.discretization.n_nodes
        self.n_dofs = field.n_dofs
        # Set the beginning of the field dof range.
        field.dof_start = self.problem.n_dofs
        # Add the field to the problem, and update the
        # appropriate attributes.
        self.problem.fields.append(field)
        self.problem.n_fields += 1
        self.problem.n_dofs += field.n_dofs
        self.problem.u.resize(self.problem.n_dofs)
        # Set the end of the field dof range.
        field.dof_end = self.problem.n_dofs

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
