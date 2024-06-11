class Params:
    percent = 0
    problem_name = ''
    ID = 0
    cpus = 1
    mesh_step = 0.35
    baseName = ''
    Slim = 0
    dead_objects = 0

    def __init__(self, percent, problem_name, id, cpus, mesh_step, baseName, Slim, dead_obj):
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

        self.ID = id
        self.cpus = cpus
        self.mesh_step = mesh_step
        self.baseName = baseName
        self.Slim = Slim
        self.dead_objects = dead_obj


params = Params(percent=0, problem_name='default', id=0, cpus=1, mesh_step=0.35, baseName='changeIt', Slim=0, dead_obj=0)


def set_percent(val):
    params.percent = val

def set_dead_objects(val):
    params.dead_objects = val



def set_problem_name(val):
    params.problem_name = val


def set_id(val):
    params.ID = val


def set_cpus(val):
    params.cpus = val


def set_mesh_step(val):
    params.mesh_step = val


def set_base_name(val):
    params.baseName = val


def set_s_lim(val):
    params.Slim = val


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