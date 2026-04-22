import numpy as np
from glob2 import glob
import datetime
import open3d as o3d
import trimesh
import pathlib
from pymoo.problems import get_problem
from utils.logger_leaflet import log_message
from utils.global_variable import *
from utils.compute_utils import get_history_output as get_history_output
from utils.compute_utils import run_abaqus
from utils.fea_results_utils import read_data
from utils.project_utils import purgeFiles
from utils.create_geometry_utils_v2 import generateShell
from utils.mesh_utils import triangulate_points_pca, evaluate_leaflet_developability
from utils.create_input_files import write_inp_shell, write_inp_contact
import os

now = str(datetime.datetime.now()).replace(' ', '_').replace(':', '-').split('.')[0]
now = now[:-3]

pathToAbaqus = str(pathlib.Path(__file__).parent.resolve()) + '/abaqusWF/'
path_utils = str(pathlib.Path(__file__).parent.resolve())
path_project = path_utils[:-6]
if os.sys.platform == 'win32':
    pathToAbaqus = pathToAbaqus.replace('\\','/')
    path_utils = path_utils.replace('\\','/')
    path_project = path_project.replace('\\','/')

# procedure parameters class
class Procedure:
    baseName = None
    cpus = 4
    mesh_step = None

    def __init__(self, cpus, baseName, mesh_step):
        self.baseName = baseName
        self.cpus = cpus
        self.mesh_step = mesh_step

    # ==================================================================================================================
    # __________________________________________RUN PROCEDURE___________________________________________________________
    # ==================================================================================================================

    def run_procedure(self, params) -> dict:

        def run_leaflet_single(self, params) -> dict:
            ID = get_id()
            try:
                baseName = get_base_name() + '_' + now
                reset_direction()
                tt1 = datetime.datetime.now()

                if 'HGT' in get_parameters_list():
                    HGT = params[get_parameters_list().index('HGT')]
                else:
                    HGT = get_HGT()

                if 'Lstr' in get_parameters_list():
                    Lstr = params[get_parameters_list().index('Lstr')]
                else:
                    Lstr = get_Lstr()

                if 'THK' in get_parameters_list():
                    THK = params[get_parameters_list().index('THK')]
                else:
                    THK = get_THK()

                if 'ANG' in get_parameters_list():
                    ANG = params[get_parameters_list().index('ANG')]
                else:
                    ANG = get_ANG()

                if 'CVT' in get_parameters_list():
                    CVT = params[get_parameters_list().index('CVT')]
                else:
                    CVT = get_CVT()

                if 'LAS' in get_parameters_list():
                    LAS = params[get_parameters_list().index('LAS')]
                else:
                    LAS = get_LAS()

                if 'DIA' in get_parameters_list():
                    DIA = params[get_parameters_list().index('DIA')]
                else:
                    DIA = get_DIA()

                if 'Lift' in get_parameters_list():
                    Lift = params[get_parameters_list().index('Lift')]
                else:
                    Lift = get_Lift()
                try:
                    if 'FCVT' in get_parameters_list():
                        from utils.create_geometry_utils_v2 import generate_leaflet_pointcloud
                        FCVT = params[get_parameters_list().index('FCVT')]
                    else:
                        from utils.create_geometry_utils import generate_leaflet_pointcloud
                        FCVT = get_Lift()
                except:
                    pass

                if 'SEC' in get_parameters_list():
                    SEC = params[get_parameters_list().index('SEC')]
                else:
                    SEC = get_SEC()

                EM = get_EM()  # Formlabs elastic 50A
                mesh_step = self.mesh_step
                tangent_behavior = get_tangent_behavior()
                normal_behavior = get_normal_behavior()

                Dens = get_density()
                MaterialName = get_material_name()
                PressType = get_valve_position()  # can be 'vent'
                fileName = baseName + '.inp'
                # 1. Geometry Generation (Existing)
                try:
                    if 'FCVT' in get_parameters_list():
                        pointsInner, _, _, _, pointsHullLower, _, points, _, finalRad, currRad, message = \
                            generate_leaflet_pointcloud(HGT=HGT, Lstr=Lstr, SEC=SEC, DIA=DIA, THK=0.35,
                                                        ANG=ANG, Lift=Lift, CVT=CVT, LAS=LAS,
                                                        mesh_step=mesh_step, FCVT=FCVT)
                    else:
                        pointsInner, _, _, _, pointsHullLower, _, points, _, finalRad, currRad, message = \
                            generate_leaflet_pointcloud(HGT=HGT, Lstr=Lstr, SEC=SEC, DIA=DIA, THK=0.35,
                                                        ANG=ANG, Lift=Lift, CVT=CVT, LAS=LAS, mesh_step=mesh_step)
                except Exception as e:
                    raise e

                # 2. New Meshing (Replacing Open3D)
                try:
                    pts = pointsInner.T if pointsInner.shape[0] == 3 else pointsInner
                    shellEle = triangulate_points_pca(pts, mesh_step, filter_options=['area'])
                    shellNode = pts
                    
                    # Identify fixed BC nodes
                    fixed_bc = []
                    phl = pointsHullLower.T if pointsHullLower.shape[0] == 3 else pointsHullLower
                    for i in range(phl.shape[0]):
                        dists = np.linalg.norm(shellNode - phl[i], axis=1)
                        idx = np.argmin(dists)
                        if dists[idx] < mesh_step * 0.1:
                            fixed_bc.append(idx)
                    fixed_bc = np.unique(fixed_bc)
                    
                    message = 'done'
                    log_message(
                        message + '. Mesh nodes is ' + str(len(shellNode)) + '. Elements is ' + str(len(shellEle)))
                except Exception as e:
                    raise e

                tt2 = datetime.datetime.now()

                del mesh
                inpFileName = str(inpDir) + str(baseName)
                jobName = str(baseName) + '_Job'
                modelName = str(baseName) + '_Model'
                partName = str(baseName) + '_Part'
                outFEATime = 0
                if get_direction().lower() == 'direct':
                    try:
                        write_inp_shell(
                            fileName=inpFileName + '.inp', Nodes=shellNode, Elements=shellEle,
                            BCfix=fixed_bc, THC=THK, Emod=EM, Dens=9e-10, JobName=jobName,
                            ModelName=modelName, partName=partName, MaterialName='PVA', PressType='vent',
                            press_overclosure='linear', tangent_behavior=tangent_behavior,
                            normal_behavior=normal_behavior
                        )
                        message = run_abaqus(pathToAbaqus, jobName, inpFileName, self.cpus)
                        outFEATime = datetime.datetime.now() - tt1
                    except Exception as e:
                        os.chdir(path_project)
                        raise e

                    # парсим odb, считываем поля, считаем максимумы и площадь открытия, пишем в outFileName
                    try:
                        get_history_output(pathName=pathToAbaqus, odbFileName=jobName + '.odb', cpus=self.cpus)
                    except:
                        os.chdir(path_project)
                        raise 'Odb parse problem'

                    try:
                        endPath = pathToAbaqus + 'results/'
                        LMN_op, LMN_cl, Smax, VMS, perf_index, _ = read_data(
                            pathToAbaqus=pathToAbaqus, endPath=endPath, partName=partName
                        )
                        purgeFiles(endPath, partName, pathToAbaqus, jobName)
                    except Exception as e:
                        os.chdir(path_project)
                        raise e

                    if LMN_op < 0:
                        change_direction()
                        log_message(f'Changed direction to {get_direction()}! LMN_op is {LMN_op}\m')
                        try:
                            write_inp_shell(
                                fileName=inpFileName + '.inp', Nodes=shellNode, Elements=shellEle,
                                BCfix=fixed_bc, THC=THK, Emod=EM, Dens=9e-10, JobName=jobName,
                                ModelName=modelName, partName=partName, MaterialName='PVA', PressType='vent',
                                press_overclosure='linear', tangent_behavior=tangent_behavior,
                                normal_behavior=normal_behavior
                            )
                            message = run_abaqus(pathToAbaqus, jobName, inpFileName, self.cpus)
                            outFEATime = datetime.datetime.now() - tt1
                        except Exception as e:
                            os.chdir(path_project)
                            raise e

                        # парсим odb, считываем поля, считаем максимумы и площадь открытия, пишем в outFileName
                        try:
                            get_history_output(pathName=pathToAbaqus, odbFileName=jobName + '.odb', cpus=self.cpus)
                        except:
                            os.chdir(path_project)
                            raise 'Odb parse problem'

                        try:
                            endPath = pathToAbaqus + 'results/'

                            LMN_op, LMN_cl, Smax, VMS, perf_index, _ = read_data(
                                pathToAbaqus=pathToAbaqus, endPath=endPath, partName=partName
                            )
                            purgeFiles(endPath, partName, pathToAbaqus, jobName)
                        except Exception as e:
                            os.chdir(path_project)
                            raise e

                        change_direction()
                        log_message(f'Swap back direction to {get_direction()}! LMN_op is {LMN_op}')

                del fixed_bc, partName, jobName, endPath, modelName, inpFileName
                del tt1, tt2

                if LMN_cl < 0:
                    out_lmn_cl = 0
                else:
                    out_lmn_cl = LMN_cl

                objectives_dict = {
                    'LMN_open': out_lmn_cl,
                    "LMN_closed": LMN_cl,
                    "Smax": Smax,
                    'VMS': VMS
                }

                return {"results": objectives_dict}
            except Exception as exept:
                log_message(f'Exception: {exept}')

                objectives_dict = {
                    'LMN_open': 0.0,
                    "LMN_closed": 2,
                    "Smax": 50,
                    'VMS': 50
                }

                return {"results": objectives_dict}

        def run_leaflet_contact(self, params) -> dict:
            log_message(f"current ID is {get_id()}. current time is {str(datetime.datetime.now()).split('.')[0]}\n")
            try:
                log_message(f"current ID is {get_id()}\n")
                baseName = get_base_name() + '_' + now
                reset_direction()
                tt1 = datetime.datetime.now()

                if 'HGT' in get_parameters_list():
                    HGT = params[get_parameters_list().index('HGT')]
                else:
                    HGT = get_HGT()

                if 'Lstr' in get_parameters_list():
                    Lstr = params[get_parameters_list().index('Lstr')]
                else:
                    Lstr = get_Lstr()

                if 'THK' in get_parameters_list():
                    THK = params[get_parameters_list().index('THK')]
                else:
                    THK = get_THK()

                if 'ANG' in get_parameters_list():
                    ANG = params[get_parameters_list().index('ANG')]
                else:
                    ANG = get_ANG()

                if 'CVT' in get_parameters_list():
                    CVT = params[get_parameters_list().index('CVT')]
                else:
                    CVT = get_CVT()

                if 'LAS' in get_parameters_list():
                    LAS = params[get_parameters_list().index('LAS')]
                else:
                    LAS = get_LAS()

                if 'DIA' in get_parameters_list():
                    DIA = params[get_parameters_list().index('DIA')]
                else:
                    DIA = get_DIA()

                if 'Lift' in get_parameters_list():
                    Lift = params[get_parameters_list().index('Lift')]
                else:
                    Lift = get_Lift()
                try:
                    if 'FCVT' in get_parameters_list():
                        from utils.create_geometry_utils_v2 import generate_leaflet_pointcloud
                        FCVT = params[get_parameters_list().index('FCVT')]
                    else:
                        from utils.create_geometry_utils import generate_leaflet_pointcloud
                        FCVT = get_Lift()
                except:
                    pass

                if 'SEC' in get_parameters_list():
                    SEC = params[get_parameters_list().index('SEC')]
                else:
                    SEC = get_SEC()


                # try:
                #     HGT, Lstr, THK, ANG, CVT, LAS = params
                # except:
                #     Lstr, ANG, CVT, LAS = params
                #     HGT = 11
                #     THK = 0.3
                # DIA = get_DIA()
                # Lift = get_Lift()
                EM = get_EM()  # Formlabs elastic 50A
                mesh_step = self.mesh_step
                tangent_behavior = get_tangent_behavior()
                normal_behavior = get_normal_behavior()

                Dens = get_density()
                MaterialName = get_material_name()
                PressType = get_valve_position()  # can be 'vent'
                fileName = baseName + '.inp'
                # 1. Geometry Generation (Existing)
                try:
                    if 'FCVT' in get_parameters_list():
                        pointsInner, _, _, _, pointsHullLower, _, points, _, finalRad, currRad, message = \
                            generate_leaflet_pointcloud(HGT=HGT, Lstr=Lstr, SEC=SEC, DIA=DIA, THK=0.35,
                                                        ANG=ANG, Lift=Lift, CVT=CVT, LAS=LAS,
                                                        mesh_step=mesh_step, FCVT=FCVT)
                    else:
                        pointsInner, _, _, _, pointsHullLower, _, points, _, finalRad, currRad, message = \
                            generate_leaflet_pointcloud(HGT=HGT, Lstr=Lstr, SEC=SEC, DIA=DIA, THK=0.35,
                                                        ANG=ANG, Lift=Lift, CVT=CVT, LAS=LAS, mesh_step=mesh_step)
                except Exception as e:
                    raise e

                # 2. New Meshing (Replacing Open3D)
                try:
                    pts = pointsInner.T if pointsInner.shape[0] == 3 else pointsInner
                    shellEle = triangulate_points_pca(pts, mesh_step, filter_options=['edge', 'area'])
                    shellNode = pts
                    
                    # Identify fixed BC nodes
                    fixed_bc = []
                    phl = pointsHullLower.T if pointsHullLower.shape[0] == 3 else pointsHullLower
                    for i in range(phl.shape[0]):
                        dists = np.linalg.norm(shellNode - phl[i], axis=1)
                        idx = np.argmin(dists)
                        if dists[idx] < mesh_step * 0.1:
                            fixed_bc.append(idx)
                    fixed_bc = np.unique(fixed_bc)
                    
                    message = 'done'
                    log_message(
                        message + '. Mesh nodes is ' + str(len(shellNode)) + '. Elements is ' + str(len(shellEle)))
                except Exception as e:
                    raise e

                tt2 = datetime.datetime.now()

                results = {}
                K_max = None
                if get_check_unfolding():
                    # Use new robust developability assessment
                    dev_index, is_dev, dev_results = evaluate_leaflet_developability(
                        points=shellNode,
                        elements=shellEle,
                        DIA=DIA,
                        points_to_exclude=pointsHullLower,
                        mesh_step=mesh_step
                    )

                    if is_dev:
                        K_max = 0
                    else:
                        K_max = dev_results['k_rms'] # Using RMS as the scale-independent metric
                        
                        if (
                                'K_max' in get_objectives_list()
                                or 'K_max' in get_constraints_list()
                                or 'K_max' in get_parameters_list()
                        ):
                            if K_max > dev_results['t_adj']:
                                res_dict = {
                                    'LMN_open': 0.0,
                                    "LMN_closed": 2,
                                    "Smax": get_s_lim()*2,
                                    'HELI': 3,
                                    'VMS': 5,
                                    'K_max': K_max
                                }

                                return {"results": res_dict}
                del results

                inpFileName = str(inpDir) + str(baseName)
                jobName = str(baseName) + '_Job'
                modelName = str(baseName) + '_Model'
                partName = str(baseName) + '_Part'
                outFEATime = 0

                try:
                    write_inp_contact(
                        fileName=inpFileName + '.inp', Nodes=shellNode, Elements=shellEle,
                        BCfix=fixed_bc, THC=THK, Emod=EM, Dens=Dens, JobName=jobName,
                        ModelName=modelName, partName=partName, MaterialName=MaterialName, PressType=PressType,
                        press_overclosure='linear', tangent_behavior=tangent_behavior,
                        normal_behavior=normal_behavior
                    )
                    message = run_abaqus(pathToAbaqus, jobName, inpFileName, self.cpus)
                    outFEATime = datetime.datetime.now() - tt1
                except Exception as e:
                    os.chdir(path_project)
                    raise e

                # парсим odb, считываем поля, считаем максимумы и площадь открытия, пишем в outFileName
                try:
                    get_history_output(pathName=pathToAbaqus, odbFileName=jobName + '.odb')
                except Exception as e:
                    os.chdir(path_project)
                    raise f'Odb parse problem. Error: {e}'

                try:
                    endPath = pathToAbaqus + 'results/'
                    LMN_op, LMN_cl, Smax, VMS, perf_index, heli = read_data(
                        pathToAbaqus=pathToAbaqus, endPath=endPath, partName=partName
                    )
                    purgeFiles(endPath, partName, pathToAbaqus, jobName)
                except Exception as e:
                    change_direction()
                    try:
                        write_inp_contact(
                            fileName=inpFileName + '.inp', Nodes=shellNode, Elements=shellEle,
                            BCfix=fixed_bc, THC=THK, Emod=EM, Dens=Dens, JobName=jobName,
                            ModelName=modelName, partName=partName, MaterialName=MaterialName, PressType=PressType,
                            press_overclosure='linear', tangent_behavior=tangent_behavior,
                            normal_behavior=normal_behavior
                        )
                        message = run_abaqus(pathToAbaqus, jobName, inpFileName, self.cpus)
                        outFEATime = datetime.datetime.now() - tt1
                    except Exception as e:
                        os.chdir(path_project)
                        raise e

                    try:  # already in try: and switched direction
                        get_history_output(pathName=pathToAbaqus, odbFileName=jobName + '.odb')
                    except Exception as e:
                        os.chdir(path_project)
                        raise f'Odb parse problem. Reverse direction! Error: {e}'

                    try:
                        endPath = pathToAbaqus + 'results/'
                        LMN_op, LMN_cl, Smax, VMS, perf_index, heli = read_data(
                            pathToAbaqus=pathToAbaqus, endPath=endPath, partName=partName
                        )
                        purgeFiles(endPath, partName, pathToAbaqus, jobName)
                    except:
                        os.chdir(path_project)
                        raise e
                    delta = datetime.datetime.now() - tt1
                    raise e
                del fixed_bc, partName, jobName, endPath, modelName, inpFileName
                del tt1, tt2

                res_dict = {
                    'LMN_open': LMN_op,
                    "LMN_closed": LMN_cl,
                    "Smax": Smax,
                    'HELI': heli,
                    'VMS': VMS,
                    'K_max': K_max
                }

                return {"results": res_dict}
            except Exception as exept:
                log_message(f'Exception: {exept}')

                res_dict = {
                    'LMN_open': 0.0,
                    "LMN_closed": 2,
                    "Smax": get_s_lim()*2,
                    'HELI': 3,
                    'VMS': 5,
                    'K_max': 5
                }

                return {"results": res_dict}

        def run_pymoo(self, params) -> dict:
            problem = get_problem("welded_beam")
            param_array = np.array(list(params))
            result = problem.evaluate(param_array)
            objective_values = result[0]
            objectives_dict = {
                "objective1": objective_values[0],
                "objective2": objective_values[1]
            }
            constraint_values = result[1]
            constraints_dict = {
                "constraint1": constraint_values[0],
                "constraint2": constraint_values[1],
                "constraint3": constraint_values[2],
                "constraint4": constraint_values[3]
            }

            # cleanup_log_leaflet()
            # set_id(ID + 1)

            return {"objectives": objectives_dict, "constraints": constraints_dict}

        problem_name = get_problem_name().lower()
        if problem_name == 'leaflet_single':
            res = run_leaflet_single(self, params)
        elif problem_name == 'leaflet_contact':
            res = run_leaflet_contact(self, params)
        elif problem_name == 'test':
            res = run_pymoo(self, params)

        set_id(get_id() + 1)
        return res


# ====================================================================================================================
# ___________________________________________INIT PROCEDURE___________________________________________________________
# ====================================================================================================================

# абсолютный путь к папке ипутов /inps/, расположенной в папке проекта
inpDir = str(pathlib.Path(__file__).parent.resolve()) + '/inps/'


def init_procedure(param_array):
    problem_name = get_problem_name().lower()
    if problem_name == 'leaflet_single' or problem_name == 'leaflet_contact':
        # prepare folders for xlsx, inps, logs, geoms
        folders = ['inps', 'logs', 'geoms']
        for folder in folders:
            os.makedirs('utils/' + folder, exist_ok=True)
        del folders
        os.makedirs(str(pathlib.Path(__file__).parent.resolve()) + '/logs/', exist_ok=True)

        problem = Procedure(cpus=get_cpus(), baseName=get_base_name(), mesh_step=get_mesh_step())
    elif problem_name == 'test':
        problem = get_problem("welded_beam")
    else:
        raise Exception(f'Wrong problem name: {problem_name}')
    return problem
