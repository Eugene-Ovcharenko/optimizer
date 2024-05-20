from openpyxl import load_workbook
import pandas as pd
import numpy as np
from glob2 import glob
import datetime
import random
from .write_inp import write_inp
from .get_history_output import get_history_output as get_history_output
from .runabaqus import runabaqus_minute_walltime as runabaqus
from .read_data import read_data
from .purgeFiles import purgeFiles
from .logger_leaflet import configure_log_leaflet, cleanup_log_leaflet
import pathlib

from .logger_leaflet import log_message

import os

now = str(datetime.datetime.now()).replace(' ', '_').replace(':', '-').split('.')[0]
now = now[:-3]

pathToAbaqus = str(pathlib.Path(__file__).parent.resolve()) + '/abaqusWF/'

ID = None

# procedure parameters class
class Procedure:
    baseName = None
    cpus = 4
    logging = True
    wbResults = None
    wbLog = None
    outFileNameResults = None
    outFileNameLog = None
    sheet_short = 'short'

    def __init__(self, cpus, logging, baseName, wbResults, wbLog, outFileNameResults, outFileNameLog,
                 sheet_short, sheet_desc):
        self.baseName = baseName
        self.cpus = cpus
        self.logging = logging
        self.wbResults = wbResults
        self.wbLog = wbLog
        self.outFileNameResults = outFileNameResults
        self.outFileNameLog = outFileNameLog
        self.sheet_short = sheet_short
        self.sheet_desc = sheet_desc


    def run_procedure(self, params) -> dict:
        global ID
        # try:
        baseName = self.baseName + '_' + now
        tt1 = datetime.datetime.now()
        Width, THK = params
        fileName = baseName + f'_{ID}' + '.inp'

        inpFileName = str(inpDir) + str(baseName) + f'_{ID}'
        jobName = str(baseName) + f'_{ID}' + '_Job'
        modelName = str(baseName) + f'_{ID}' + '_Model'
        partName = str(baseName) + f'_{ID}' + '_Part'

        try:
            write_inp(fileName=inpFileName + '.inp',
                      JobName=jobName, ModelName=modelName, partName=partName,
                      Width=Width, THK=THK)
            message = runabaqus(pathToAbaqus, jobName, inpFileName, self.cpus)
            outFEATime = datetime.datetime.now() - tt1
        except Exception as e:
            delta = datetime.datetime.now() - tt1
            sheet = self.wbLog['log']
            sheet.append(
                [fileName, ID, Width, THK, 0, str(e), delta])
            self.wbLog.save(self.outFileNameLog)
            self.wbLog.close()
            del delta, sheet
            raise e

        # парсим odb, считываем поля, считаем максимумы и площадь открытия, пишем в outFileName
        try:
            get_history_output(pathName=pathToAbaqus, odbFileName=jobName + '.odb', cpu=self.cpus)
        except:
            delta = datetime.datetime.now() - tt1
            # wbGeom = load_workbook(filename=self.outFileNameLog)
            sheet = self.wbLog['log']
            with open(pathToAbaqus + '/results/' + partName[0:-5] + '/FramesCount.txt', 'r') as DataFile:
                frames = int(DataFile.read())
            sheet.append(
                [fileName, ID, Width, THK, frames, 'Odb parse problem', delta])
            self.wbLog.save(self.outFileNameLog)
            self.wbLog.close()
            # purgeFiles(pathToAbaqus + 'results/', partName, pathToAbaqus, jobName)
            del delta, sheet
            raise 'Odb parse problem'
        try:
            with open(pathToAbaqus + '/results/' + partName[0:-5].upper() + '/FramesCount.txt',
                      'r') as DataFile:
                frames = int(DataFile.read())
        except:
            frames = 0
        try:

            endPath = pathToAbaqus + 'results/'
            delta = datetime.datetime.now() - tt1
            Smax, Displacement = read_data(
                pathToAbaqus=pathToAbaqus, endPath=endPath, partName=partName,
                Width=Width, ID=ID, THK=THK, fileName=fileName, frames=frames, delta=delta,
                sheet_short=self.sheet_short, sheet_desc=self.sheet_desc, wbResults=self.wbResults,
                outFileNameResult=self.outFileNameResults, outFileNameGeom=self.outFileNameLog
            )
            purgeFiles(endPath, partName, pathToAbaqus, jobName)
        except Exception as e:
            delta = datetime.datetime.now() - tt1
            sheet = self.wbLog['log']

            sheet.append([fileName, ID, Width, THK, frames, str(e), delta])
            self.wbLog.save(self.outFileNameLog)
            self.wbLog.close()
            purgeFiles(pathToAbaqus + 'results/', partName, pathToAbaqus, jobName)
            del delta, sheet
            raise e
        del partName, jobName, endPath, modelName, inpFileName
        del tt1

        objectives_dict = {
            'Displacement': np.abs(Displacement),
            "Mass": Width * THK
        }

        constraints_dict = {
            "THK_constr": THK - 5,
            "Width_constr": Width - 10,
            "Smax_constr": Smax - 500
        }

        return {"objectives": objectives_dict, "constraints": constraints_dict}
        # except Exception as exept:
        #     log_message(f'Exception: {exept}')
        #
        #     objectives_dict = {
        #         'Displacement': -1,
        #         "Smax": 500000
        #     }
        #
        #     constraints_dict = {
        #         "THK_constr": 50,
        #         "Width_constr": 50,
        #         "Smax_constr": 500000
        #     }

        cleanup_log_leaflet()
        ID += 1

        return {"objectives": objectives_dict, "constraints": constraints_dict}


# ====================================================================================================================
# ___________________________________________PROGRAMM_________________________________________________________________
# ====================================================================================================================

# абсолютный путь к папке ипутов /inps/, расположенной в папке проекта
inpDir = str(pathlib.Path(__file__).parent.resolve()) + '/inps/'

def init_procedure(cpus=10, logging=True, baseName='test'):
    global ID
    ID = 0
    # prepare folders for xlsx, inps, logs, geoms
    folders = ['inps', 'logs']
    for folder in folders:
        if not os.path.exists('utils/'+folder):
            os.makedirs('utils/'+folder)
    del folders
    base_folder = str(pathlib.Path(__file__).parent.resolve()) + '/logs/'
    existing_subfolders = [f for f in os.listdir(base_folder) if os.path.isdir(os.path.join(base_folder, f))]
    subfolders_number = len(existing_subfolders) + 1
    current_time = datetime.datetime.now()
    formatted_date = current_time.strftime('%d_%m_%Y')
    subfolder_name = f"{subfolders_number:03}_{formatted_date}"
    folder_path = os.path.join(base_folder, subfolder_name)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    print(f"Created folder: {folder_path}")

    # имена и путь xls файлов; базовое имя для инпутов и .odb
    outFileNameGeom = folder_path + '/geom_' + str(baseName) + '.xlsx'
    outFileNameResult = folder_path + '/odb_' + str(baseName) + '.xlsx'
    if logging:
        configure_log_leaflet(folder_path + '/log')
    # подготовка таблиц
    colNamesRes = pd.DataFrame(
        {'fileName': [], 'ID': [], 'Width, mm': [], 'THK, mm': [], 'Displacement, mm': [], 'Smax, MPa': [],
         'VMS, MPa': [], 'S11, MPa': [], 'S22, MPa': [], 'Smid, MPa': [], 'Smin, MPa': [],
         'VMS_1, MPa': [],
         'S11_1, MPa': [],
         'S22_1, MPa': [],
         'Smax_1, MPa': [],
         'Smid_1, MPa': [],
         'Smin_1, MPa': [],
         'Frames': [],
         'Total time, hh:mm:ss.ms': []})

    colNamesRes_short = pd.DataFrame(
        {'fileName': [], 'ID': [], 'Width, mm': [], 'THK, mm': [], 'Displacement, mm': []})

    writerRes = pd.ExcelWriter(str(outFileNameResult), engine='xlsxwriter')
    colNamesRes_short.to_excel(writerRes, sheet_name='short', index=False)
    colNamesRes.to_excel(writerRes, sheet_name='descriptive', index=False)
    writerRes._save()

    colNamesGeoms = pd.DataFrame(
        {'fileName': [], 'ID': [], 'Width': [], 'THK': [], 'Frames': [], 'Message': [], 'Exec time': []})

    writerGeom = pd.ExcelWriter(str(outFileNameGeom), engine='xlsxwriter')
    colNamesGeoms.to_excel(writerGeom, sheet_name='log', index=False)
    writerGeom._save()

    wbResults = load_workbook(filename=outFileNameResult)
    wbGeom = load_workbook(filename=outFileNameGeom)
    sheet_short = wbResults['short']
    sheet_desc = wbResults['descriptive']

    problem = Procedure(cpus=cpus, logging=logging, baseName=baseName, wbResults=wbResults,
                        wbLog=wbGeom,
                        outFileNameResults=outFileNameResult, outFileNameLog=outFileNameGeom, sheet_short=sheet_short,
                        sheet_desc=sheet_desc)

    return problem
