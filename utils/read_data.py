import numpy as np
from glob2 import glob
import os
import datetime
from openpyxl import load_workbook
from .logger_leaflet import log_message


# ====================================================================================================================
# _____________________________________________CLASS__________________________________________________________________
# ====================================================================================================================

# обертка для хранения и обработки результатов парсинга odb
# для сетки
class readedMesh:
    def __init__(self, lnodes, lelements):
        self.nodes = lnodes
        self.elements = lelements

    def repNodes(self, lnodes):
        self.nodes = lnodes

    def repElements(self, lelements):
        self.elements = lelements

    def addNodes(self, lnodes):
        self.nodes = np.append(self.nodes, lnodes)

    def addElements(self, lelements):
        self.elements = np.append(self.elements, lelements)

    def getNodes(self):
        return self.nodes

    def getElements(self):
        return self.elements


def read_data_single(pathToAbaqus=None, endPath=None, partName=None):
    foldPath = endPath + partName[0:-5].upper() + '/' + partName.upper()

    mesh1 = readedMesh(lnodes=np.loadtxt(foldPath + '-1/Nodes.txt'),
                       lelements=np.loadtxt(foldPath + '-1/Elements.txt'))
    stressFiles = glob(foldPath + '-1/' + 'Stress_*.txt')

    newstr = ''.join((ch if ch in '0123456789' else ' ') for ch in stressFiles[0])
    one_step = ([int(i) for i in newstr.split()])[-1]

    s_max1_1step = np.loadtxt(foldPath + '-1/' + 'Stress_' + str(one_step) + '.txt')

    eleCount = len(mesh1.getElements())

    disp1 = np.loadtxt(foldPath + '-1/' + 'U_' + str(one_step) + '.txt')

    return (s_max1_1step, disp1)


def read_data(pathToAbaqus=None, endPath=None, partName=None, ID=-1,
              Width=-1, THK=-1, fileName='', frames=-1, delta=-1,
              sheet_short=None, sheet_desc=None, wbResults=None, outFileNameResult=None,
              outFileNameGeom=None):
    subfolders = [f.path for f in os.scandir(endPath + partName[:-5].upper()) if f.is_dir()]

    (s_max1_1step, disp1) = read_data_single(
        pathToAbaqus=pathToAbaqus, endPath=endPath, partName=partName
    )
    VMS = s_max1_1step[-1]
    S11 = s_max1_1step[0]
    S22 = s_max1_1step[1]
    Smax = s_max1_1step[3]
    Smid = s_max1_1step[4]
    Smin = s_max1_1step[5]

    disp1 = disp1[-1][2]

    new_row_short = [fileName, ID, Width, THK, disp1]
    new_row_desc = [
        fileName, ID, Width, THK, disp1, Smax,
        VMS, S11, S22, Smid, Smin,
        s_max1_1step[-1],  # VMS
        s_max1_1step[0],  # S11
        s_max1_1step[1],  # S22
        s_max1_1step[3],  # Smax
        s_max1_1step[4],  # Smid
        s_max1_1step[5],  # Smin
        frames, delta
    ]

    sheet_short.append(new_row_short)
    sheet_desc.append(new_row_desc)
    wbResults.save(outFileNameResult)
    wbResults.close()

    wbGeom = load_workbook(filename=outFileNameGeom)
    sheet = wbGeom['log']
    sheet.append([fileName, ID, Width, THK, frames, str('Done'), delta])
    wbGeom.save(outFileNameGeom)
    wbGeom.close()

    return Smax, disp1
