class Params:
    """
    A structured parameter container for simulation job configuration and control.

    This class encapsulates all relevant runtime and model-specific parameters for finite element
    simulations (e.g., valve leaflets, beams). It provides a clean and validated interface for passing
    metadata to downstream modules, ensuring consistency across the simulation workflow.

    Attributes:
        percent (int): Parameter used to define material reduction or percentage-based operation (e.g., damage, curvature).
        problem_name (str): Type of the simulation problem. Must be one of:
             'leaflet_single', 'leaflet_contact', 'default'.
        ID (int): Simulation identifier for tracking or versioning.
        cpus (int): Number of CPU threads to allocate for the simulation job.
        mesh_step (float): Spatial resolution (step size) for mesh generation.
        baseName (str): Base filename for simulation input/output files.
        Slim (float): Simulation threshold or constraint parameter.
        direction (str): Boundary condition flip orientation; must be either 'direct' or 'reverse'.
        valve_position (str): Anatomical reference for the valve ('ao' for aortic, etc.).
        DIA (float): Leaflet apparatus diameter.
        Lift (float): lift of leaflet apparatus to simulate actual frame (basicly, set 0 if don`t care).
        SEC (float): sector in degrees one leaflet was occupied.
        EM (float): Elastic (Yungus) modulus, MPa.
        density (float): Density of material (1.02e-9 tonn/mm3 for water-like),
        material_name (str): name of material for leaflet.

    Raises:
        Exception: If `problem_name` or `direction` is invalid.
    """
    percent = 0
    problem_name = ''
    ID = 0
    cpus = 1
    mesh_step = 0.35
    baseName = ''
    Slim = 0
    direction = 'direct'
    valve_position = 'ao'
    DIA = 0
    Lift = 0
    SEC = 0
    EM = 0
    density = 1.02e-9
    material_name = 'test material'
    material_csv_path = 'path'
    abq_tang_beh = 0
    abq_norm_beh = 0
    global_path = ''
    def __init__(self,
                 percent, problem_name, id, cpus, mesh_step,
                 baseName, Slim, direction, valve_position,
                 DIA, Lift, SEC, EM, density, material_name, material_csv_path,
                 abq_tangent_behavior, abq_normal_behavior, global_path):
        self.percent = percent
        if (
                problem_name.lower() == 'leaflet_single'
                or problem_name.lower() == 'leaflet_contact'
                or problem_name.lower() == 'default'
        ):
            self.problem_name = problem_name
        else:
            raise (Exception(f'Wrong problem name: {problem_name}! '
                             f'Allowed \'Leaflet_Single\',\'Leaflet_Contact\'\n'))
        if (
                direction.lower() != 'direct' and direction.lower() != 'reverse'
        ):
            raise (Exception(f'Wrong direction for leaflet BC flip. Entered {direction}'))

        self.ID = id
        self.cpus = cpus
        self.mesh_step = mesh_step
        self.baseName = baseName
        self.Slim = Slim
        self.direction = direction
        self.valve_position = valve_position
        self.DIA = DIA
        self.Lift = Lift
        self.SEC = SEC
        self.EM = EM
        self.density = density
        self.material_name = material_name
        self.material_csv_path = material_csv_path
        self.abq_tang_beh = abq_tangent_behavior
        self.abq_norm_beh = abq_normal_behavior
        self.global_path = global_path


params = Params(
    percent=0,
    problem_name='default',
    id=0,
    cpus=1,
    mesh_step=0.35,
    baseName='changeIt',
    Slim=0,
    direction='direct',
    valve_position='ao',
    DIA=29,
    Lift=0,
    SEC=120,
    EM=1.48,
    density=1e-9,
    material_name='Change it! Material name',
    material_csv_path='no path',
    abq_tangent_behavior=1,
    abq_normal_behavior=0.2,
    global_path=''
)


# Setters
def set_DIA(val: float):
    params.DIA = val


def set_Lift(val: float):
    params.Lift = val


def set_SEC(val: float):
    params.SEC = val


def set_EM(val: float):
    params.EM = val


def set_density(val: float):
    params.density = val


def set_material_name(val: str):
    params.material_name = val


def set_tangent_behavior(val: float):
    params.abq_tang_beh = val


def set_normal_behavior(val: float):
    params.abq_norm_beh = val


def set_percent(val: float):
    params.percent = val


def set_problem_name(val: str):
    params.problem_name = val


def set_id(val: int):
    params.ID = val


def set_cpus(val: int):
    params.cpus = val


def set_mesh_step(val: float):
    params.mesh_step = val


def set_base_name(val: str):
    params.baseName = val


def set_s_lim(val: float):
    params.Slim = val


def set_material_csv_path(val: str):
    params.material_csv_path = val


def set_valve_position(val: str):
    params.valve_position = val


def set_global_path(val: str):
    params.global_path = val


# Getters

def get_id() -> int:
    return params.ID


def get_cpus() -> int:
    return params.cpus


def get_percent() -> int:
    return params.percent


def get_problem_name() -> str:
    return params.problem_name


def get_base_name() -> str:
    return params.baseName


def get_mesh_step() -> float:
    return params.mesh_step


def get_s_lim() -> float:
    return params.Slim


def get_direction() -> str:
    return params.direction


def get_valve_position() -> str:
    return params.valve_position

def get_DIA() -> float:
    return params.DIA


def get_Lift() -> float:
    return params.Lift


def get_SEC() -> float:
    return params.SEC


def get_EM() -> float:
    return params.EM


def get_density() -> float:
    return params.density


def get_material_name() -> str:
    return params.material_name


def get_material_csv_path() -> str:
    return params.material_csv_path

def get_tangent_behavior() -> float:
    return params.abq_tang_beh


def get_normal_behavior() -> float:
    return params.abq_norm_beh


def get_global_path() -> str:
    return params.global_path


def reset_direction() -> None:
    params.valve_position = 'direct'


def change_direction():
    if params.direction.lower() == 'direct':
        params.direction = 'reverse'
    else:
        params.direction = 'direct'
