from datetime import datetime
from .logger_leaflet import log_message
from .global_variable import get_direction
import numpy as np

def write_inp_beam(fileName=None, JobName='test', ModelName='test', partName='test', Width=10, THK=3):
    fileID = open(fileName, 'w')

    fileID.write('*Heading\n')
    fileID.write('** Job name: %s Model name: %s\n' % (JobName, ModelName))
    fileID.write('** Generated by: LNB\n')
    fileID.write('*Preprint, echo=NO, model=NO, history=NO, contact=NO\n')
    fileID.write('**\n')
    fileID.write('** PARTS\n')
    fileID.write('**\n')
    fileID.write('*Part, name=%s\n' % (partName))
    fileID.write(f'*Node\n')
    fileID.write(f'      1,           0.,           0.\n')
    fileID.write(f'      2,          10.,           0.\n')
    fileID.write(f'      3,          20.,           0.\n')
    fileID.write(f'      4,          30.,           0.\n')
    fileID.write(f'      5,          40.,           0.\n')
    fileID.write(f'      6,          50.,           0.\n')
    fileID.write(f'      7,          60.,           0.\n')
    fileID.write(f'      8,          70.,           0.\n')
    fileID.write(f'      9,          80.,           0.\n')
    fileID.write(f'     10,          90.,           0.\n')
    fileID.write(f'     11,         100.,           0.\n')
    fileID.write(f'*Element, type=B21\n')
    fileID.write(f' 1,  1,  2\n')
    fileID.write(f' 2,  2,  3\n')
    fileID.write(f' 3,  3,  4\n')
    fileID.write(f' 4,  4,  5\n')
    fileID.write(f' 5,  5,  6\n')
    fileID.write(f' 6,  6,  7\n')
    fileID.write(f' 7,  7,  8\n')
    fileID.write(f' 8,  8,  9\n')
    fileID.write(f' 9,  9, 10\n')
    fileID.write(f'10, 10, 11\n')
    fileID.write(f'*Nset, nset=_PickedSet3, internal, generate\n')
    fileID.write(f'  1,  11,   1\n')
    fileID.write(f'*Elset, elset=_PickedSet3, internal, generate\n')
    fileID.write(f'  1,  10,   1\n')
    fileID.write(f'** Section: Shelf_section  Profile: Rect_profile\n')
    fileID.write(f'*Beam Section, elset=_PickedSet3, material=Steel, poisson = 0.3, temperature=GRADIENTS, section=RECT\n')
    fileID.write(f'{Width}, {THK}\n')
    fileID.write(f'0.,0.,-1.\n')
    fileID.write(f'*End Part\n')
    fileID.write(f'**\n')

    fileID.write(f'**\n')
    fileID.write(f'** ASSEMBLY\n')
    fileID.write(f'**\n')
    fileID.write(f'*Assembly, name=Assembly\n')
    fileID.write(f'**\n')
    fileID.write(f'*Instance, name=%s-1, part=%s\n' % (partName, partName))
    fileID.write(f'*End Instance\n')
    fileID.write(f'**\n')
    fileID.write(f'*Nset, nset=_PickedSet4, internal, instance=%s-1\n' % (partName))
    fileID.write(f' 1,\n')
    fileID.write(f'*Nset, nset=_PickedSet5, internal, instance=%s-1\n' % (partName))
    fileID.write(f' 11,\n')
    fileID.write(f'*End Assembly\n')
    fileID.write(f'*Amplitude, name=Amp-1\n')
    fileID.write(f'             0.,              0.,              1.,              1.\n')

    fileID.write(f'**\n')
    fileID.write(f'** MATERIALS\n')
    fileID.write(f'**\n')
    fileID.write(f'*Material, name=Steel\n')
    fileID.write(f'*Elastic\n')
    fileID.write(f'200000., 0.3\n')
    fileID.write(f'** ----------------------------------------------------------------\n')

    fileID.write(f'**\n')
    fileID.write(f'** STEP: Load\n')
    fileID.write(f'**\n')
    fileID.write(f'*Step, name=Load, nlgeom=NO, inc=1000\n')
    fileID.write(f'*Static\n')
    fileID.write(f'0.1, 1., 1e-05, 0.1\n')

    fileID.write(f'**\n')
    fileID.write(f'** BOUNDARY CONDITIONS\n')
    fileID.write(f'**\n')
    fileID.write(f'** Name: Fix Type: Symmetry/Antisymmetry/Encastre\n')
    fileID.write(f'*Boundary\n')
    fileID.write(f'_PickedSet4, ENCASTRE\n')

    fileID.write(f'**\n')
    fileID.write(f'** LOADS\n')
    fileID.write(f'**\n')
    fileID.write(f'** Name: Load_1kN   Type: Concentrated force\n')
    fileID.write(f'*Cload, amplitude=Amp-1\n')
    fileID.write(f'_PickedSet5, 2, -100.\n')

    fileID.write(f'**\n')
    fileID.write(f'** OUTPUT REQUESTS\n')
    fileID.write(f'**\n')
    fileID.write(f'*Restart, write, frequency=0\n')

    fileID.write(f'**\n')
    fileID.write(f'** FIELD OUTPUT: F-Output-1\n')
    fileID.write(f'**\n')
    fileID.write('*Output, field\n')
    fileID.write('*Node Output\n')
    fileID.write('U,\n')
    fileID.write('*Output, field\n')
    fileID.write('*Element Output, directions=YES\n')
    fileID.write('S,\n')


    fileID.write(f'**\n')
    fileID.write(f'** HISTORY OUTPUT: H-Output-1\n')
    fileID.write(f'**\n')
    fileID.write(f'*Output, history, variable=PRESELECT\n')
    fileID.write(f'*End Step\n')
    fileID.close()


def write_inp_shell(fileName=None, Nodes=None, Elements=None, BCfix=None, THC=None, Emod=-1, Dens=1e-9,
                        JobName=None, ModelName=None, partName=None, tangent_behavior=0.2, normal_behavior=0.2,
                        Time=1.0, frequencyOutput=40, scaligFactor=49, MaterialName='Biomat', PressType='ao',
                        press_overclosure='hard', poison_ratio=0.45):
    tt2 = datetime.now()
    fileID = open(fileName, 'w')
    ElementType = 'S3'
    fileID.write('*Heading\n')
    fileID.write('** Job name: %s Model name: %s\n' % (JobName, ModelName))
    fileID.write('** Generated by: LNB\n')
    fileID.write('*Preprint, echo=NO, model=NO, history=NO, contact=NO\n')
    fileID.write('**\n')
    fileID.write('** PARTS\n')
    fileID.write('**\n')
    fileID.write('*Part, name=%s\n' % (partName))
    fileID.write('*Node\n')
    for i in np.arange(0, len(Nodes)):
        fileID.write('\t%d' % (i + 1))
        for j in np.arange(0, 3):
            fileID.write(',\t%14.10f' % (Nodes[i, j]))
        fileID.write('\n')

    fileID.write('*Element, type=%s\n' % (ElementType))
    for i in np.arange(0, len(Elements)):
        fileID.write('\t%6d' % (i + 1))
        for j in np.arange(0, 3):
            fileID.write(',\t%6d' % (Elements[i, j] + 1))
        fileID.write('\n')

    fileID.write('*Nset, nset=Set-1, generate\n')
    fileID.write('\t%d,\t%d,\t%d\n' % (1, len(Nodes), 1))
    fileID.write('*Elset, elset=Set-2, generate\n')
    fileID.write('\t%d,\t%d,\t%d\n' % (1, len(Elements), 1))
    fileID.write('** Section: Section-1-Set-2\n')
    fileID.write('*Shell Section, elset=Set-2, material=Biomat\n%f, 9\n*End Part\n' % (THC))
    fileID.write('**\n')
    fileID.write('**\n')
    fileID.write('** ASSEMBLY\n')
    fileID.write('**\n')
    fileID.write('*Assembly, name=Assembly\n')
    fileID.write('**\n')
    fileID.write('*Instance, name=%s-1, part=%s\n' % (partName, partName))
    fileID.write('*End Instance\n')
    fileID.write('**\n')
    fileID.write('*Nset, nset=FixedSet, instance=%s-1\n' % (partName))
    for i in np.arange(0, len(BCfix)):
        if np.mod(i, 16) != 0:
            fileID.write('\t%d,' % (BCfix[i] + 1))
        else:
            fileID.write('\t%d\n' % (BCfix[i] + 1))
    if get_direction().lower() == 'direct':
        fileID.write('\n*Elset, elset=ElSet_1_SNEG, instance=%s-1, generate\n' % (partName))
        fileID.write('\t1,\t%d,\t1\n' % (len(Elements)))
        fileID.write('*Surface, type=ELEMENT, name=ElSet_Aort\n')
        fileID.write('ElSet_1_SNEG, SNEG\n')
        fileID.write('*Elset, elset=ElSet_2_SPOS, instance=%s-1, generate\n' % (partName))
        fileID.write('\t1,\t%d,\t1\n' % (len(Elements)))
        fileID.write('*Surface, type=ELEMENT, name=ElSet_Vent\n')
        fileID.write('ElSet_2_SPOS, SPOS\n')
    else:
        fileID.write('\n*Elset, elset=ElSet_1_SNEG, instance=%s-1, generate\n' % (partName))
        fileID.write('\t1,\t%d,\t1\n' % (len(Elements)))
        fileID.write('*Surface, type=ELEMENT, name=ElSet_Aort\n')
        fileID.write('ElSet_1_SNEG, SPOS\n')
        fileID.write('*Elset, elset=ElSet_2_SPOS, instance=%s-1, generate\n' % (partName))
        fileID.write('\t1,\t%d,\t1\n' % (len(Elements)))
        fileID.write('*Surface, type=ELEMENT, name=ElSet_Vent\n')
        fileID.write('ElSet_2_SPOS, SNEG\n')
    fileID.write('*Nset, nset=OutputNodes, instance=%s-1, generate\n' % (partName))
    fileID.write('	1,	%d,	1\n' % (len(Nodes)))
    fileID.write('*Elset, elset=OutputElements, instance=%s-1, generate\n' % (partName))
    fileID.write('	1,	%d,	1\n' % (len(Elements)))
    fileID.write('*Transform, nset=FixedSet, type=C\n')
    fileID.write('          0.,           0.,           0.,           0.,           0.,           1.\n')
    fileID.write('*End Assembly\n')
    fileID.write('*Amplitude, name=Amp-Aortic\n')
    fileID.write(
        '             0.,          0.0103,           0.025,          0.0107,            0.05,          0.0113,           0.075,          0.0124\n')
    fileID.write(
        '            0.1,           0.014,           0.125,          0.0153,            0.15,          0.0165,           0.175,          0.0172\n')
    fileID.write('            0.2,          0.0178\n')
    fileID.write('*Amplitude, name=Amp-Ventricular\n')
    fileID.write(
        '             0.,          0.0093,           0.025,          0.0129,            0.05,          0.0131,           0.075,           0.014\n')
    fileID.write(
        '            0.1,          0.0149,           0.125,          0.0156,            0.15,          0.0165,           0.175,          0.0173\n')
    fileID.write('            0.2,          0.0179\n')
    fileID.write('**\n')
    fileID.write('** MATERIALS\n')
    fileID.write('**\n')
    fileID.write('*Material, name=Biomat\n')
    fileID.write('*Density\n')
    fileID.write(' 1e-09,\n')
    fileID.write('*Elastic\n')
    fileID.write('%f, %f\n' % (Emod, poison_ratio))
    fileID.write('**\n')
    fileID.write('** BOUNDARY CONDITIONS\n')
    fileID.write('**\n')
    fileID.write('** Name: BC-1 Type: Symmetry/Antisymmetry/Encastre\n')
    fileID.write('*Boundary\n')
    fileID.write('FixedSet, ENCASTRE\n')
    fileID.write('** ----------------------------------------------------------------\n')
    fileID.write('**\n')
    fileID.write('** STEP: Step-1\n')
    fileID.write('**\n')
    fileID.write('*Step, name=Step-1, nlgeom=YES\n')
    fileID.write('*Dynamic, Explicit\n')
    fileID.write(', 0.2\n')
    fileID.write('*Bulk Viscosity\n')
    fileID.write('0.06, 1.2\n')
    fileID.write('**\n')
    fileID.write('** LOADS\n')
    fileID.write('**\n')

    fileID.write('** Name: LoadVentricular   Type: Pressure\n')
    fileID.write('*Dsload, amplitude=Amp-Ventricular\n')
    fileID.write('ElSet_Vent, P, 1\n')

    fileID.write('** Name: LoadAortic   Type: Pressure\n')
    fileID.write('*Dsload, amplitude=Amp-Aortic\n')
    fileID.write('ElSet_Aort, P, 1\n')

    fileID.write('** Name: ViscoPress  Type: Pressure\n')
    fileID.write('*Dsload\n')
    fileID.write('ElSet_Vent, VP, 1e-06\n')

    fileID.write('**\n')
    fileID.write('** OUTPUT REQUESTS\n')
    fileID.write('**\n')
    fileID.write('*Restart, write, number interval=1, time marks=NO\n')

    fileID.write('** \n')
    fileID.write('** FIELD OUTPUT: F-Output-1\n')
    fileID.write('** \n')
    fileID.write('*Output, field, frequency=%d\n' % (20))
    fileID.write('*Element Output, position=NODES, elset=OUTPUTELEMENTS, directions=YES\n')
    fileID.write('LE, S\n')

    fileID.write('** \n')
    fileID.write('** FIELD OUTPUT: F-Output-2\n')
    fileID.write('** \n')
    fileID.write('*Output, field, frequency=%d\n' % (20))
    fileID.write('*Node Output, nset=OUTPUTNODES\n')
    fileID.write('U, \n')

    fileID.write('** \n')
    fileID.write('** HISTORY OUTPUT: H-Output-1\n')
    fileID.write('** \n')
    fileID.write('*Output, history\n')
    fileID.write('*Energy Output\n')
    fileID.write('ALLAE, ALLIE, ALLKE, ALLSE, ALLWK, ETOTAL\n')
    fileID.write('*End Step\n')
    fileID.close()
    delta = datetime.now() - tt2
    log_message('Total time is > ' + str(delta.seconds) + ' seconds')


def write_inp_contact(fileName=None, Nodes=None, Elements=None, BCfix=None, THC=None, Emod=-1, Dens=1e-9,
                      JobName=None, ModelName=None, partName=None, tangent_behavior=0.2, normal_behavior=0.2,
                      Time=1.0, frequencyOutput=40, scaligFactor=49, MaterialName='Biomat', PressType='ao',
                      press_overclosure='hard'):
    tt2 = datetime.now()
    fileID = open(fileName, 'w')
    ElementType = 'S3'
    fileID.write('*Heading\n')
    fileID.write('** Job name: %s Model name: %s\n' % (JobName, ModelName))
    fileID.write('** Generated by: LNB\n')
    fileID.write('*Preprint, echo=NO, model=NO, history=NO, contact=NO\n')
    fileID.write('**\n')
    fileID.write('** PARTS\n')
    fileID.write('**\n')
    fileID.write('*Part, name=%s\n' % (partName))
    fileID.write('*Node\n')
    for i in np.arange(0, len(Nodes)):
        fileID.write('\t%d' % (i + 1))
        for j in np.arange(0, 3):
            fileID.write(',\t%14.10f' % (Nodes[i, j]))
        fileID.write('\n')

    fileID.write('*Element, type=%s\n' % (ElementType))
    for i in np.arange(0, len(Elements)):
        fileID.write('\t%6d' % (i + 1))
        for j in np.arange(0, 3):
            fileID.write(',\t%6d' % (Elements[i, j] + 1))
        fileID.write('\n')

    fileID.write('*Nset, nset=Set-1, generate\n')
    fileID.write('\t%d,\t%d,\t%d\n' % (1, len(Nodes), 1))
    fileID.write('*Elset, elset=Set-2, generate\n')
    fileID.write('\t%d,\t%d,\t%d\n' % (1, len(Elements), 1))
    fileID.write('** Section: Section-1-Set-2\n')
    fileID.write('*Shell Section, elset=Set-2, material=%s, offset=SPOS\n%f, 9\n*End Part\n' % (MaterialName, THC))
    fileID.write('**\n')

    fileID.write('**\n')
    fileID.write('** ASSEMBLY\n')
    fileID.write('**\n')
    fileID.write('*Assembly, name=Assembly\n')
    fileID.write('**\n')
    fileID.write('*Instance, name=%s-1, part=%s\n' % (partName, partName))
    fileID.write('*End Instance\n')
    fileID.write('**\n')
    fileID.write('*Instance, name=%s-2, part=%s\n' % (partName, partName))
    fileID.write('0., 0., 0.\n')
    fileID.write('0., 0., 0., 0., 0., -1, 120.\n')
    fileID.write('*End Instance\n')
    fileID.write('**\n')
    fileID.write('*Instance, name=%s-3, part=%s\n' % (partName, partName))
    fileID.write('0., 0., 0.\n')
    fileID.write('0., 0., 0., 0., 0., 1, 120.\n')
    fileID.write('*End Instance\n')
    fileID.write('**\n')
    for j in range(1, 4):
        fileID.write('\n*Nset, nset=FixedSet_%d, instance=%s-%d\n' % (j, partName, j))
        for i in np.arange(1, len(BCfix) + 1):
            if np.mod(i, 16) != 0:
                fileID.write('\t%6d,' % (BCfix[i - 1] + 1))
            else:
                fileID.write('\t%6d\n' % (BCfix[i - 1] + 1))
    fileID.write('\n')
    for i in range(1, 4):
        fileID.write('*Elset, elset=_ElSet_%d_SNEG, internal, instance=%s-%d, generate\n' % (i, partName, i))
        fileID.write('\t1,\t%d,\t1\n' % (len(Elements)))
        fileID.write('*Elset, elset=_ElSet_%d_SPOS, internal, instance=%s-%d, generate\n' % (i, partName, i))
        fileID.write('\t1,\t%d,\t1\n' % (len(Elements)))
        if get_direction().lower() == 'direct':
            if PressType == 'ao':
                fileID.write('*Surface, type=ELEMENT, name=ElSet_Inflow_%d\n' % (i))
                fileID.write('\t_ElSet_%d_SNEG, SNEG\n' % (i))
                fileID.write('*Surface, type=ELEMENT, name=ElSet_Outflow_%d\n' % (i))
                fileID.write('\t_ElSet_%d_SPOS, SPOS\n' % (i))
            else:
                fileID.write('*Surface, type=ELEMENT, name=ElSet_Outflow_%d\n' % (i))
                fileID.write('\t_ElSet_%d_SNEG, SNEG\n' % (i))
                fileID.write('*Surface, type=ELEMENT, name=ElSet_Inflow_%d\n' % (i))
                fileID.write('\t_ElSet_%d_SPOS, SPOS\n' % (i))
        else:
            if PressType == 'ao':
                fileID.write('*Surface, type=ELEMENT, name=ElSet_Outflow_%d\n' % (i))
                fileID.write('\t_ElSet_%d_SNEG, SNEG\n' % (i))
                fileID.write('*Surface, type=ELEMENT, name=ElSet_Inflow_%d\n' % (i))
                fileID.write('\t_ElSet_%d_SPOS, SPOS\n' % (i))
            else:
                fileID.write('*Surface, type=ELEMENT, name=ElSet_Inflow_%d\n' % (i))
                fileID.write('\t_ElSet_%d_SNEG, SNEG\n' % (i))
                fileID.write('*Surface, type=ELEMENT, name=ElSet_Outflow_%d\n' % (i))
                fileID.write('\t_ElSet_%d_SPOS, SPOS\n' % (i))
    for i in range(1, 4):
        fileID.write('*Nset, nset=OutputNodes_%d, instance=%s-%d, generate\n' % (i, partName, i))
        fileID.write('	1,	%d,	1\n' % (len(Nodes)))
        fileID.write('*Elset, elset=OutputElements_%d, instance=%s-%d, generate\n' % (i, partName, i))
        fileID.write('	1,	%d,	1\n' % (len(Elements)))
        fileID.write('*Transform, nset=FixedSet_%d, type=C\n' % (i))
        fileID.write('          0.,           0.,           0.,           0.,           0.,           1.\n')
    fileID.write('*End Assembly\n')

    fileID.write('*Amplitude, name=Amp-Aortic\n')
    fileID.write('    0.000000,        0.011114,        0.227544,        0.009892,        0.232141,        0.010503,        0.234439,        0.011236\n')
    fileID.write('    0.239036,        0.012091,        0.250528,        0.013434,        0.262020,        0.014533,        0.282706,        0.015632\n')
    fileID.write('    0.303392,        0.015865,        0.319481,        0.015999,        0.351659,        0.015865,        0.374643,        0.015754\n')
    fileID.write('    0.399926,        0.015144,        0.420612,        0.014777,        0.441298,        0.014900,        0.468879,        0.014777\n')
    fileID.write('    0.519444,        0.014265,        0.574606,        0.013800,        0.857313,        0.011114,        1.084857,        0.009892\n')
    fileID.write('    1.089454,        0.010503,        1.091752,        0.011236,        1.096349,        0.012091,        1.107841,        0.013434\n')
    fileID.write('    1.119333,        0.014533,        1.140019,        0.015632,        1.160705,        0.015865,        1.176794,        0.015999\n')
    fileID.write('    1.208972,        0.015865,        1.231956,        0.015754,        1.257239,        0.015144,        1.277924,        0.014777\n')
    fileID.write('    1.298610,        0.014900,        1.326191,        0.014777,        1.376757,        0.014265,        1.431919,        0.013800\n')
    fileID.write('    1.714625,        0.011114\n')

    fileID.write('*Amplitude, name=Amp-Ventricular\n')
    fileID.write('    0.000000,        0.002933,        0.020350,        0.002933,        0.031656,        0.003000,        0.049746,        0.003066\n')
    fileID.write('    0.065574,        0.003297,        0.079141,        0.003297,        0.094969,        0.003420,        0.108536,        0.003664\n')
    fileID.write('    0.119842,        0.004152,        0.128886,        0.004519,        0.142453,        0.005007,        0.151498,        0.005129\n')
    fileID.write('    0.158282,        0.005129,        0.167326,        0.004763,        0.174110,        0.004274,        0.187677,        0.003542\n')
    fileID.write('    0.212550,        0.009282,        0.232900,        0.013732,        0.248728,        0.014799,        0.269079,        0.015599\n')
    fileID.write('    0.293952,        0.016132,        0.321086,        0.016265,        0.341436,        0.016132,        0.352742,        0.015999\n')
    fileID.write('    0.373092,        0.015732,        0.397965,        0.014777,        0.416055,        0.013312,        0.490673,        0.004274\n')
    fileID.write('    0.511024,        0.003175,        0.535896,        0.002443,        0.583381,        0.002687,        0.639910,        0.002933\n')
    fileID.write('    0.856982,        0.002933,        0.877332,        0.002933,        0.888638,        0.003000,        0.906728,        0.003066\n')
    fileID.write('    0.922556,        0.003297,        0.936123,        0.003297,        0.951951,        0.003420,        0.965518,        0.003664\n')
    fileID.write('    0.976824,        0.004152,        0.985868,        0.004519,        0.999435,        0.005007,        1.008480,        0.005129\n')
    fileID.write('    1.015264,        0.005129,        1.024308,        0.004763,        1.031092,        0.004274,        1.044659,        0.003542\n')
    fileID.write('    1.069532,        0.009282,        1.089882,        0.013732,        1.105710,        0.014799,        1.126061,        0.015599\n')
    fileID.write('    1.150933,        0.016132,        1.178067,        0.016265,        1.198418,        0.016132,        1.209724,        0.015999\n')
    fileID.write('    1.230074,        0.015732,        1.254947,        0.014777,        1.273036,        0.013312,        1.347655,        0.004274\n')
    fileID.write('    1.368005,        0.003175,        1.392878,        0.002443,        1.440363,        0.002687,        1.496892,        0.002933\n')
    fileID.write('    1.713964,        0.002933\n')

    fileID.write('*Amplitude, name=Amp-Atrium\n')
    fileID.write('    0.0000000,       0.0034664,       0.0316563,       0.0034664,       0.0542680,       0.0034664,       0.0881855,       0.0037330\n')
    fileID.write('    0.1334088,       0.0043966,       0.1492370,       0.0048851,       0.1628040,       0.0047630,       0.1808933,       0.0039081\n')
    fileID.write('    0.1899379,       0.0035417,       0.2215943,       0.0037859,       0.2464671,       0.0034196,       0.2577729,       0.0029311\n')
    fileID.write('    0.4974565,       0.0046408,       0.5268517,       0.0030532,       0.5472022,       0.0029311,       0.5901643,       0.0032974\n')
    fileID.write('    0.8411537,       0.0034664,       0.8569819,       0.0034664,       0.8886382,       0.0034664,       0.9112499,       0.0034664\n')
    fileID.write('    0.9451674,       0.0037330,       0.9903907,       0.0043966,       1.0062189,       0.0048851,       1.0197859,       0.0047630\n')
    fileID.write('    1.0378752,       0.0039081,       1.0469199,       0.0035417,       1.0785762,       0.0037859,       1.1034490,       0.0034196\n')
    fileID.write('    1.1147548,       0.0029311,       1.3544384,       0.0046408,       1.3838336,       0.0030532,       1.4041841,       0.0029311\n')
    fileID.write('    1.4471462,       0.0032974,       1.6981357,       0.0034664,       1.7139638,       0.0034664\n')

    if Emod != -1:
        fileID.write('**\n')
        fileID.write('** MATERIALS\n')
        fileID.write('**\n')
        fileID.write('*Material, name=%s\n' % (MaterialName))
        fileID.write('*Density\n')
        fileID.write(' %.2e,\n' % (Dens))
        fileID.write('*Elastic\n')
        fileID.write('%f, 0.495\n' % (Emod))
    else:
        ## material properties
        fileID.write('**\n' % ())
        fileID.write('** MATERIALS\n' % ())
        fileID.write('**\n' % ())
        fileID.write('*Material, name=%s\n' % (MaterialName))
        fileID.write('*Density\n' % ())
        fileID.write(' 1e-09,\n' % ())
        fileID.write('*Hyperelastic, n=2, reduced polynomial, test data input, poisson=0.495\n' % ())
        fileID.write('*Uniaxial Test Data\n' % ())
        # pericard
        fileID.write('    0.,   0.\n' % ())
        fileID.write('  0.01, 0.04\n' % ())
        fileID.write(' 0.095, 0.09\n' % ())
        fileID.write('   0.2, 0.13\n' % ())
        fileID.write('   0.5,  0.2\n' % ())
        fileID.write('   1.5, 0.25\n' % ())
        fileID.write('   2.5,  0.3\n' % ())
        fileID.write('   3.5, 0.35\n' % ())
        fileID.write('    5., 0.45\n' % ())

    fileID.write('** \n')
    fileID.write('** INTERACTION PROPERTIES\n')
    fileID.write('** \n')
    fileID.write('*Surface Interaction, name = IntProp-1\n')
    fileID.write('*Friction\n')
    fileID.write(' %f,\n' % (tangent_behavior))
    if str.lower(press_overclosure) == 'hard':
        fileID.write('*Surface Behavior, pressure-overclosure=HARD\n')
    elif str.lower(press_overclosure) == 'linear':
        fileID.write('*Surface Behavior, pressure-overclosure=LINEAR\n')
        fileID.write('%f,\n' % (normal_behavior))

    fileID.write('**\n')
    fileID.write('** BOUNDARY CONDITIONS\n')
    fileID.write('**\n')
    fileID.write('** Name: BC-1 Type: Symmetry/Antisymmetry/Encastre\n')
    fileID.write('*Boundary\n')
    fileID.write('FixedSet_1, ENCASTRE\n')
    fileID.write('** Name: BC-2 Type: Symmetry/Antisymmetry/Encastre\n')
    fileID.write('*Boundary\n')
    fileID.write('FixedSet_2, ENCASTRE\n')
    fileID.write('** Name: BC-3 Type: Symmetry/Antisymmetry/Encastre\n')
    fileID.write('*Boundary\n')
    fileID.write('FixedSet_3, ENCASTRE\n')

    fileID.write('** ----------------------------------------------------------------\n')
    fileID.write('**\n')
    fileID.write('** STEP: Step-1\n')
    fileID.write('**\n')
    fileID.write('*Step, name=Step-1, nlgeom=YES\n')
    fileID.write('*Dynamic, Explicit\n')
    fileID.write(', %f\n' % (Time))
    fileID.write('*Bulk Viscosity\n')
    fileID.write('0.06, 1.2\n')
    fileID.write('** Mass Scaling: Semi-Automatic\n')
    fileID.write('** Whole Model\n')
    fileID.write('*Fixed Mass Scaling, factor = %f\n' % (scaligFactor))

    fileID.write('**\n')
    fileID.write('** LOADS\n')
    fileID.write('**\n')

    for iter_Part in range(1, 4):
        if PressType == 'ao':
            fileID.write('** Name: LoadVentricular-%d   Type: Pressure\n' % (iter_Part))
            fileID.write('*Dsload, amplitude=Amp-Ventricular\n')
            fileID.write('ElSet_Inflow_%d, P, 1.\n' % (iter_Part))

            fileID.write('** Name: LoadAortic-%d   Type: Pressure\n' % (iter_Part))
            fileID.write('*Dsload, amplitude=Amp-Aortic\n')
            fileID.write('ElSet_Outflow_%d, P, 1.\n' % (iter_Part))
        else:
            fileID.write('** Name: LoadVentricular-%d   Type: Pressure\n' % (iter_Part))
            fileID.write('*Dsload, amplitude=Amp-Ventricular\n')
            fileID.write('ElSet_Outflow_%d, P, 1.\n' % (iter_Part))

            fileID.write('** Name: LoadAtrium-%d   Type: Pressure\n' % (iter_Part))
            fileID.write('*Dsload, amplitude=Amp-Atrium\n')
            fileID.write('ElSet_Inflow_%d, P, 1.\n' % (iter_Part))

        fileID.write('** Name: ViscoPress-%d   Type: Pressure\n' % (iter_Part))
        fileID.write('*Dsload\n')
        fileID.write('ElSet_Inflow_%d, VP, 1e-05\n' % (iter_Part))

    fileID.write('**\n')
    fileID.write('** INTERACTIONS\n')
    fileID.write('**\n')
    fileID.write('** Interaction: Int-1-2\n')
    fileID.write('*Contact Pair, interaction=IntProp-1, mechanical constraint=KINEMATIC, cpset=Int-1-2\n')
    fileID.write('ElSet_Inflow_1, ElSet_Inflow_2\n')
    fileID.write('** Interaction: Int-2-3\n')
    fileID.write('*Contact Pair, interaction=IntProp-1, mechanical constraint=KINEMATIC, cpset=Int-2-3\n')
    fileID.write('ElSet_Inflow_2, ElSet_Inflow_3\n')
    fileID.write('** Interaction: Int-3-1\n')
    fileID.write('*Contact Pair, interaction=IntProp-1, mechanical constraint=KINEMATIC, cpset=Int-3-1\n')
    fileID.write('ElSet_Inflow_3, ElSet_Inflow_1\n')

    fileID.write('**\n' % ())
    fileID.write('** OUTPUT REQUESTS\n' % ())
    fileID.write('**\n' % ())
    fileID.write('*Restart, write, number interval=1, time marks=NO\n' % ())
    for i in range(1, 4):
        fileID.write('**\n' % ())
        fileID.write('** FIELD OUTPUT: F-Output-1-%d\n' % (i))
        fileID.write('**\n' % ())
        fileID.write('*Output, field, frequency=%d\n' % (frequencyOutput))
        fileID.write('*Node Output, nset=OutputNodes_%d\n' % (i))
        fileID.write('RT, U\n' % ())
    for i in range(1, 4):
        fileID.write('**\n' % ())
        fileID.write('** FIELD OUTPUT: F-Output-2-%d\n' % (i))
        fileID.write('**\n' % ())
        fileID.write('*Output, field, frequency=%d\n' % (frequencyOutput))
        fileID.write('*Element Output, position=NODES, elset=OutputElements_%d, directions=YES\n' % (i))
        fileID.write('LE, S\n' % ())
    for i in range(1, 4):
        if i < 3:
            j = i + 1
        else:
            j = 1
        fileID.write('** \n' % ())
        fileID.write('** HISTORY OUTPUT: H-Output-%d\n' % (i))
        fileID.write('** \n' % ())
        fileID.write('*Output, history, frequency=%d\n' % (frequencyOutput))
        fileID.write('*Contact Output, cpset=Int-%d-%d\n' % (i, j))
        fileID.write('CAREA,\n' % ())

    fileID.write('*End Step\n' % ())
    fileID.close()
    delta = datetime.now() - tt2
    log_message('Inp file creation is > ' + str(delta.seconds) + ' seconds')