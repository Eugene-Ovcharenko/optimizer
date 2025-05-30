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
from utils.global_variable import *
from pymoo.problems import get_problem
from utils.get_history_output import get_history_output as get_history_output
from utils.run_abaqus import run_abaqus as run_abaqus
from utils.read_data import read_data
from utils.purgeFiles import purgeFiles
from utils.logger_leaflet import configure_log_leaflet, cleanup_log_leaflet
from utils.generateShell import generateShell
from utils.createGeometry import createGeometry
from utils.logger_leaflet import log_message
from utils.write_inp import write_inp_shell, write_inp_contact
from optimizer import *
from utils.problem import *
import os


def run_leaflet_contact(params):

    try:
        HGT, Lstr, THK, ANG, CVT, LAS = params
    except:
        Lstr, ANG, CVT, LAS = params
        HGT = 11
        THK = 0.3
    ID = get_id()
    DIA = 29 - 2 * 1.5
    Lift = 0
    EM = 1.88 # Formlabs elastic 50A
    mesh_step = get_mesh_step()
    baseName = get_base_name()
    tangent_behavior = 1
    normal_behavior = 0.2
    SEC = 119
    Dens = 1.02e-9
    MaterialName = 'FormLabs Elasctic 50A'
    PressType = get_valve_position()  # can be 'vent'
    fileName = baseName + '.inp'
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


    shellNode, shellEle, fixed_bc = generateShell(points=mesh.vertices, elements=mesh.faces,
                                                  pointsInner=pointsInner,
                                                  pointsHullLower=pointsHullLower, meshStep=mesh_step)
    tt2 = datetime.datetime.now()
    message = 'done'


    del mesh
    pathToAbaqus = str(pathlib.Path(__file__).parent.resolve()) + 'utils/abaqusWF/'
    inpFileName = str('./inps/') + f'{get_base_name()}_{ID}'
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
    # _ = run_abaqus(pathToAbaqus, jobName, inpFileName, self.cpus)
    # _ = get_history_output(pathName=pathToAbaqus, odbFileName=jobName + '.odb')
    # try:
    #     endPath = pathToAbaqus + 'results/'
    #     LMN_op, LMN_cl, Smax, VMS, perf_index, heli = read_data(
    #         pathToAbaqus=pathToAbaqus, endPath=endPath, partName=partName, Slim=3.23,
    #         HGT=HGT, Lstr=Lstr, DIA=DIA, THK=THK, ANG=ANG, Lift=Lift, CVT=CVT, LAS=LAS, EM=EM, SEC=SEC,
    #         tangent_behavior=tangent_behavior, normal_behavior=normal_behavior,
    #         mesh_step=mesh_step, outFEATime=outFEATime, fileName=fileName,
    #         sheet_short=self.sheet_short, sheet_desc=self.sheet_desc, wbResults=self.wbResults,
    #         outFileNameResult=self.outFileNameResults, outFileNameGeom=self.outFileNameLog, tt1=tt1
    #     )
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

if __name__ == '__main__':
    from utils.global_variable import set_mesh_step, set_base_name
    set_mesh_step(0.5)
    set_base_name('opt_geom')
    set_s_lim(3.23)  # Formlabs elastic 50A
    set_cpus(3)
    typeof = 'Contact'
    set_problem_name('leaflet_contact')
    set_percent(0)
    import pandas as pd

    trade_off_df = pd.read_excel(os.path.join('results','test','test.xlsx'))

    for index, row in trade_off_df.iterrows():
        if row['generation'] in [1, 100, 200, 300, 400]:
            set_id(row['generation'])

            parameters = {
                'HGT': row['HGT'],
                'Lstr': row['Lstr'],
                'THK':  row['THK'],
                'ANG': row['ANG'],
                'CVT': row['CVT'],
                'LAS': row['LAS']
            }
            run_leaflet_contact(parameters.values())
            # problem = init_procedure(param_array=parameters.values())
            # problem.run_procedure(parameters.values())

