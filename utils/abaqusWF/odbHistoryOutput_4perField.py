from odbAccess import *
from sys import argv,exit
from math import fabs

import sys
from sys import path
import os
import platform
import time
import numpy as np

f=open('req.txt','r')
req=f.readline()
f.close()
req=req.split(',')

odb = openOdb(path=req[0] + '/' + req[1])
myAssembly = odb.rootAssembly
instance_iter = 0
for instance in odb.rootAssembly.instances.keys():
    # Mesh output
    Nodes = odb.rootAssembly.instances[instance].nodes
    Elements = odb.rootAssembly.instances[instance].elements
    if len(odb.rootAssembly.instances.keys()) > 0:
        try:
            path2 = req[0] + '/results/' + instance[0:-7] + '/' + instance + '/'
            os.makedirs(path2)
        except:
            path2 = req[0] + '/results/' + instance[0:-7] + '/' + instance + '/'
            print('folder already exist')
    else:
        path2 = req[0] + '/results/'

    with open(path2 + 'Nodes.txt', 'w') as DataFile:
        for ldata in Nodes:
            DataFile.write("%10d %10f %10f %10f\n" % (
                ldata.label, ldata.coordinates[0], ldata.coordinates[1], ldata.coordinates[2]))

    with open(path2 + 'Elements.txt', 'w') as DataFile:
        for ldata in Elements:
            connectivity = ldata.connectivity
            DataFile.write("%10d " % (ldata.label))
            for i in connectivity:
                DataFile.write("%10d " % (i))
            DataFile.write('\n')

    # Field output
    with open(req[0] + '/results/' + instance[0:-7] + '/FramesCount.txt', 'w') as DataFile:
        DataFile.write("%d " % (len(odb.steps['Step-1'].frames)))

    if len(odb.steps['Step-1'].frames) >= 27:
        ids = [9, 27]
    else:
        if len(odb.steps['Step-1'].frames) >= 14:
            ids = [5, 14]
        else:
            ids = [5]

    for l_id in ids:
        currFrame = odb.steps['Step-1'].frames[l_id]
        displacement=currFrame.fieldOutputs['U']
        with open(path2 + 'U_' + str(l_id) + '.txt', 'w') as DataFile:
            for i in range(0,len(displacement.values)):
                v = displacement.values[i]
                DataFile.write('%5d %10.6f %10.6f %10.6f\n' % (v.nodeLabel, v.data[0], v.data[1], v.data[2]))

        axes = odb.rootAssembly.datumCsyses.keys()
        csys = odb.rootAssembly.datumCsyses[axes[instance_iter]]
        fieldName = 'OUTPUTELEMENTS'
        misesUntransformed = currFrame.fieldOutputs['S'].getSubset(region=odb.rootAssembly.elementSets[fieldName])
        misesField = misesUntransformed.getTransformedField(csys)
        s11 = np.array(range(len(misesField.values)), dtype='float64')
        s22 = np.array(range(len(misesField.values)), dtype='float64')
        s33 = np.array(range(len(misesField.values)), dtype='float64')
        s_max = np.array(range(len(misesField.values)), dtype='float64')
        s_mid = np.array(range(len(misesField.values)), dtype='float64')
        s_min = np.array(range(len(misesField.values)), dtype='float64')
        s_mis = np.array(range(len(misesField.values)), dtype='float64')
        for i in range(0, len(misesField.values)):
            s11[i] = misesField.values[i].data[0]
            s22[i] = misesField.values[i].data[1]
            s33[i] = misesField.values[i].data[2]
            s_max[i] = misesField.values[i].maxPrincipal
            s_mid[i] = misesField.values[i].midPrincipal
            s_min[i] = misesField.values[i].minPrincipal
            s_mis[i] = misesField.values[i].mises

        with open(path2 + 'Stress_' + str(l_id) + '.txt', 'w') as DataFile:
            DataFile.write('%10.6f %10.6f %10.6f %10.6f %10.6f %10.6f %10.6f\n' %
                           (max(s11), max(s22), max(s33), max(s_max), max(s_mid), max(s_min), max(s_mis)))
        del s11, s22, s33, s_max, s_mid, s_min, s_mis, misesField, misesUntransformed

        leUntransformed = currFrame.fieldOutputs['LE'].getSubset(region=odb.rootAssembly.elementSets[fieldName])
        leField = leUntransformed.getTransformedField(csys)
        le11 = np.array(range(len(leField.values)), dtype='float64')
        le22 = np.array(range(len(leField.values)), dtype='float64')
        le33 = np.array(range(len(leField.values)), dtype='float64')
        le_max = np.array(range(len(leField.values)), dtype='float64')
        le_mid = np.array(range(len(leField.values)), dtype='float64')
        le_min = np.array(range(len(leField.values)), dtype='float64')
        for i in range(0, len(leField.values)):
            le11[i] = leField.values[i].data[0]
            le22[i] = leField.values[i].data[1]
            le33[i] = leField.values[i].data[2]
            le_max[i] = leField.values[i].maxPrincipal
            le_mid[i] = leField.values[i].midPrincipal
            le_min[i] = leField.values[i].minPrincipal
        with open(path2 + 'LE_' + str(l_id) + '.txt', 'w') as DataFile:
            # LE11 LE22 LE33 maxPr midPr minPr
            DataFile.write('%10.6f %10.6f %10.6f %10.6f %10.6f %10.6f\n' % (
                max(le11), max(le22), max(le33), max(le_max), max(le_mid), max(le_min)))
        del le11, le22, le33, le_max, le_mid, le_min, leField, leUntransformed
    instance_iter += 1
odb.close()
