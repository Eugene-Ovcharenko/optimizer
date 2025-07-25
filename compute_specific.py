import matplotlib.pyplot as plt
from openpyxl import load_workbook
import pandas as pd
import numpy as np
from glob2 import glob
import datetime
import random
import open3d as o3d
import trimesh
import pathlib
from random import random
import hydra
from omegaconf import DictConfig
from utils.global_variable import *
from utils.create_input_files import write_inp_contact
from utils.gaussian_curvature_v2 import evaluate_developability
from utils.parce_cfg import parce_cfg
from utils.fea_results_utils import read_data
from utils.compute_utils import run_abaqus, get_history_output
import os


def run_leaflet_contact(params):
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
            from utils.create_geometry_utils_v2 import generateShell
            from utils.create_geometry_utils_v2 import generate_leaflet_pointcloud as createGeometry
            FCVT = params[get_parameters_list().index('FCVT')]
        else:
            from utils.create_geometry_utils import generateShell
            from utils.create_geometry_utils import generate_leaflet_pointcloud as createGeometry
            FCVT = get_Lift()
    except:
        pass

    if 'SEC' in get_parameters_list():
        SEC = params[get_parameters_list().index('SEC')]
    else:
        SEC = get_SEC()

    reset_direction()
    ID = get_id()
    baseName = get_base_name()
    EM = get_EM()  # Formlabs elastic 50A
    mesh_step = get_mesh_step()
    tangent_behavior = get_tangent_behavior()
    normal_behavior = get_normal_behavior()
    Dens = get_density()
    MaterialName = get_material_name()
    PressType = get_valve_position()  # can be 'vent'
    fileName = baseName + '.inp'
    if 'FCVT' in get_parameters_list():
        pointsInner, _, _, _, pointsHullLower, _, points, _, finalRad, currRad, message = \
                createGeometry(HGT=HGT, Lstr=Lstr, SEC=SEC, DIA=DIA, THK=THK,
                               ANG=ANG, Lift=Lift, CVT=CVT, LAS=LAS, mesh_step=mesh_step, FCVT=FCVT)
    else:
        pointsInner, _, _, _, pointsHullLower, _, points, _, finalRad, currRad, message = \
                createGeometry(HGT=HGT, Lstr=Lstr, SEC=SEC, DIA=DIA, THK=THK,
                               ANG=ANG, Lift=Lift, CVT=CVT, LAS=LAS, mesh_step=mesh_step)
    os.makedirs('inps', exist_ok=True)
    with open(f'./inps/fixed_bottom_{get_base_name()}_{ID}.txt','w') as writer:
        for point in pointsHullLower.T:
            writer.write("%6.6f %6.6f %6.6f\n" % (point[0], point[1], point[2]))

    k = 1.1
    flag_calk_k = True
    os.makedirs('utils/geoms', exist_ok=True)
    while flag_calk_k:
        try:
            o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel(0))
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points.T)
            mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha=0.5 * k)
            _ = o3d.io.write_triangle_mesh('./utils/geoms/temp_' + '.ply', mesh, write_vertex_normals=True)

            mesh = trimesh.load_mesh('./utils/geoms/temp_' + '.ply')
            mesh.fix_normals()  # fix wrong normals
            mesh.export(f'./inps/{get_base_name()}_{ID}' + '.stl')

            flag_calk_k = False
        except:
            k += 0.1

    tt2 = datetime.datetime.now()

    fig = plt.figure(figsize=(10, 10))
    ax1 = fig.add_subplot(111, projection='3d')
    ax1.scatter(pointsInner[0], pointsInner[1], pointsInner[2], c='k', s=10)
    ax1.scatter(pointsHullLower[0], pointsHullLower[1], pointsHullLower[2], c='r', s=10)
    ax1.set_xlim((-DIA/2+2,DIA/2+2))
    ax1.set_ylim((-DIA/2+2,DIA/2+2))
    ax1.set_zlim((0,HGT+2))
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    plt.show()

    shellNode, shellEle, fixed_bc = generateShell(points=mesh.vertices, elements=mesh.faces,
                                                  pointsInner=pointsInner,
                                                  pointsHullLower=pointsHullLower, meshStep=mesh_step)
    with open(f'./inps/shellNode_{get_base_name()}_{ID}.txt','w') as writer:
        for point in shellNode:
            writer.write("%6.6f %6.6f %6.6f\n" % (point[0], point[1], point[2]))
    with open(f'./inps/shellEle_{get_base_name()}_{ID}.txt','w') as writer:
        for point in shellEle:
            writer.write("%6.6f %6.6f %6.6f\n" % (point[0], point[1], point[2]))

    tt2 = datetime.datetime.now()
    message = 'done'

    res = evaluate_developability(points_inner=shellNode, shell_elements=shellEle, visualize=True, method="pca")
    print(res['is_developable'])
    del mesh
    pathToAbaqus = str(pathlib.Path(__file__).parent.resolve()) + '/utils/abaqusWF/'
    inpFileName = str(pathlib.Path(__file__).parent.resolve()) + str('/utils/inps/') + f'{get_base_name()}_{ID}'
    jobName = str(baseName) + '_Job'
    modelName = str(baseName) + '_Model'
    partName = str(baseName) + '_Part'
    outFEATime = 0


    write_inp_contact(
        fileName=inpFileName + '.inp', Nodes=shellNode, Elements=shellEle,
        BCfix=fixed_bc, THC=THK, Emod=EM, Dens=Dens, JobName=jobName,
        ModelName=modelName, partName=partName, MaterialName=MaterialName, PressType=PressType,
        press_overclosure='linear', tangent_behavior=tangent_behavior,
        normal_behavior=normal_behavior
    )
    # run_abaqus(pathToAbaqus.replace('\\','/'), jobName, inpFileName, 3)
    # get_history_output(pathName=pathToAbaqus, odbFileName=jobName + '.odb')
    #
    # endPath = pathToAbaqus + 'results/'
    # LMN_op, LMN_cl, Smax, VMS, perf_index, heli = read_data(
    #     pathToAbaqus=pathToAbaqus, endPath=endPath, partName=partName
    # )
    #     purgeFiles(endPath, partName, pathToAbaqus, jobName)
    # except Exception as e:
    #     change_direction()
    #     try:
    #         write_inp_contact(
    #             fileName=inpFileName + '.inp', Nodes=shellNode, Elements=shellEle,
    #             BCfix=fixed_bc, THC=THK, Emod=EM, Dens=Dens, JobName=jobName,
    #             ModelName=modelName, partName=partName, MaterialName=MaterialName, PressType=PressType,
    #             press_overclosure='linear', tangent_behavior=tangent_behavior,
    #             normal_behavior=normal_behavior
    #         )
    #         message = run_abaqus(pathToAbaqus, jobName, inpFileName, self.cpus)
    #         outFEATime = datetime.datetime.now() - tt1
    #     except Exception as e:
    #         pass
    #     # парсим odb, считываем поля, считаем максимумы и площадь открытия, пишем в outFileName
    #     try:  # already in try: and switched direction
    #         _ = get_history_output(pathName=pathToAbaqus, odbFileName=jobName + '.odb')
    #     except:
    #         pass
    #
    #     try:
    #         endPath = pathToAbaqus + 'results/'
    #         LMN_op, LMN_cl, Smax, VMS, perf_index, heli = read_data(
    #             pathToAbaqus=pathToAbaqus, endPath=endPath, partName=partName, Slim=3.23,
    #             HGT=HGT, Lstr=Lstr, DIA=DIA, THK=THK, ANG=ANG, Lift=Lift, CVT=CVT, LAS=LAS, EM=EM, SEC=SEC,
    #             tangent_behavior=tangent_behavior, normal_behavior=normal_behavior,
    #             mesh_step=mesh_step, outFEATime=outFEATime, fileName=fileName,
    #             sheet_short=sheet_short, sheet_desc=sheet_desc, wbResults=wbResults,
    #             outFileNameResult=outFileNameResults, outFileNameGeom=outFileNameLog, tt1=tt1
    #         )
    #         purgeFiles(endPath, partName, pathToAbaqus, jobName)
    #     except:
    #         pass
    #     delta = datetime.datetime.now() - tt1
    #     try:
    #         with open(pathToAbaqus + '/results/' + partName[0:-5].upper() + '/FramesCount.txt',
    #                   'r') as DataFile:
    #             frames = int(DataFile.read())
    #     except:
    #         frames = 0
    #     append(
    #         [fileName, HGT, Lstr, SEC, DIA, THK, ANG, Lift, CVT, LAS, EM, tangent_behavior,
    #          normal_behavior,
    #          frames,
    #          str(e), delta])
    #     wbLog.save(outFileNameLog)
    #     wbLog.close()
    #     purgeFiles(pathToAbaqus + 'results/', partName, pathToAbaqus, jobName)
    #     del delta
    #     raise e
    # del fixed_bc, partName, jobName, endPath, modelName, inpFileName
    # del tt2

config_name='config_leaf_NSGA2_koka.yaml'

@hydra.main(config_path="configuration", config_name=config_name, version_base=None)
def main(cfg:DictConfig) -> None:

    parameters, objectives, constraints = parce_cfg(cfg=cfg, globalPath=str(pathlib.Path(__file__).parent.resolve()))

    trade_off_df = pd.read_excel(os.path.join('results/005_26_06_2025','history.xlsx'), sheet_name='Sheet1')

    for index, row in trade_off_df.iterrows():
        if row['Unnamed: 0'] in [1260]:
            set_id(f'{row["generation"]}_{row["Unnamed: 0"]}')

            params = {f'{param}': row[param] for param in parameters}

            run_leaflet_contact(list(params.values()))


if __name__ == '__main__':
    main()