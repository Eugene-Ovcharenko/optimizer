class Params:
    percent = 0
    problem_name = ''
    ID = 0
    cpus = 1
    mesh_step = 0.35
    baseName = ''
    Slim = 0
    dead_objects = 0
    direction = 'direct'
    valve_position = 'ao'

    def __init__(self, percent, problem_name, id, cpus, mesh_step, baseName, Slim, dead_obj, direction, valve_position):
        self.percent = percent
        if (
                problem_name.lower() == 'beam'
                or problem_name.lower() == 'leaflet_single'
                or problem_name.lower() == 'leaflet_contact'
                or problem_name.lower() == 'default'
        ):
            self.problem_name = problem_name
        else:
            raise (Exception(f'Wrong problem name: {problem_name}! '
                             f'Allowed \'Beam\', \'Leaflet_Single\',\'Leaflet_Contact\'\n'))
        if (
                direction.lower() != 'direct' and direction.lower() != 'reverse'
        ):
            raise (Exception(f'Wrong direction for leaflet BC flip. Entered {direction}'))

        self.ID = id
        self.cpus = cpus
        self.mesh_step = mesh_step
        self.baseName = baseName
        self.Slim = Slim
        self.dead_objects = dead_obj
        self.direction = direction
        self.valve_position = valve_position


params = Params(
    percent=0,
    problem_name='default',
    id=0,
    cpus=1,
    mesh_step=0.35,
    baseName='changeIt',
    Slim=0,
    dead_obj=0,
    direction='direct',
    valve_position='ao'
)


def set_percent(val: float):
    params.percent = val


def set_dead_objects(val: int):
    params.dead_objects = val


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


def set_valve_position(val: str):
    params.valve_position = val

def change_direction():
    if params.direction.lower() == 'direct':
        params.direction = 'reverse'
    else:
        params.direction = 'direct'


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


def get_dead_objects() -> int:
    return params.dead_objects


def get_direction() -> str:
    return params.direction

def get_valve_position() -> str:
    return params.valve_position