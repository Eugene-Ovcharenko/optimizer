## подгонка поверхности, создание модели, выбор поверхностей для ГУ
import numpy as np
from .logger_leaflet import log_message

def generateShell(points=None, elements=None, pointsInner=None, pointsHullLower=None, meshStep=None):

    nodes = np.array(points * 1e10, dtype='int64') / 1e10
    elements = np.array(elements, dtype='int32')
    pointsInner = np.array(pointsInner * 1e10, dtype='int64') / 1e10
    pointsInner = pointsInner.T
    pointsHullLower = np.array(pointsHullLower * 1e10, dtype='int64') / 1e10
    pointsHullLower = pointsHullLower.T
    # nodes = points
    # pointsInner = pointsInner.T
    # pointsHullLower = pointsHullLower.T
    writed = 0

    # Find all points and tri in volume == shell
    for i in range(0, len(pointsInner)):
        row = np.where((np.isclose(nodes, pointsInner[i])).all(axis=1))
        if i == 0:
            prev_len = 0
            shellPoints = pointsInner[i]
            ind = row[0]
        else:
            prev_len = len(ind)
            shellPoints = np.vstack((shellPoints, pointsInner[i, :]))
            ind = np.insert(ind, len(ind), row[0])

        if len(ind)-prev_len > 1:
            log_message(i)
        row1 = (elements[:, 0] == ind[i]).nonzero()
        row2 = (elements[:, 1] == ind[i]).nonzero()
        row3 = (elements[:, 2] == ind[i]).nonzero()
        if not writed:
            tempEle = elements[row1]
            tempEle = np.vstack((tempEle, elements[row2]))
            tempEle = np.vstack((tempEle, elements[row3]))
            writed = 1
        else:
            tempEle = np.vstack((tempEle, elements[row1]))
            tempEle = np.vstack((tempEle, elements[row2]))
            tempEle = np.vstack((tempEle, elements[row3]))
    # del row1, row2, row3, row
    # Unique
    __, idx = np.unique(tempEle, axis=0, return_index=True)
    tempEle = tempEle[idx, :]
    del idx

    # Format: N, x, y, z
    shellPoints = np.hstack((np.transpose([ind]), shellPoints))

    # Save only shell elements, else delete
    tempEle2 = tempEle
    rem = np.array([], dtype='int32')
    for i in np.arange(0, 3 * len(tempEle)):
        if np.argwhere(tempEle[int(np.floor(i / 3)), int(i % 3)] == shellPoints[:, 0]).shape[0] == 0:
            rem = np.append(rem, tempEle[int(np.floor(i / 3)), int(i % 3)])

    rem = np.unique(rem)

    for val in rem:
        row = (tempEle2 == val).nonzero()[0]
        tempEle2 = np.delete(tempEle2, row, axis=0)

    del tempEle, rem, row

    # Delete elements if tri square is too low or dist between nodes is too high
    tempEle = tempEle2
    indToRem = []
    for i in range(0, len(tempEle2)):
        n1 = np.reshape(shellPoints[np.argwhere(tempEle2[i, 0] == shellPoints[:, 0])], -1)
        n2 = np.reshape(shellPoints[np.argwhere(tempEle2[i, 1] == shellPoints[:, 0])], -1)
        n3 = np.reshape(shellPoints[np.argwhere(tempEle2[i, 2] == shellPoints[:, 0])], -1)
        dist12 = np.sqrt((n1[1] - n2[1]) ** 2 + (n1[2] - n2[2]) ** 2 + (n1[3] - n2[3]) ** 2)
        dist23 = np.sqrt((n2[1] - n3[1]) ** 2 + (n2[2] - n3[2]) ** 2 + (n2[3] - n3[3]) ** 2)
        dist31 = np.sqrt((n3[1] - n1[1]) ** 2 + (n3[2] - n1[2]) ** 2 + (n3[3] - n1[3]) ** 2)
        if dist12 > meshStep * 2.5 or dist23 > meshStep * 2.5 or dist31 > meshStep * 2.5:
            indToRem = np.append(indToRem, i)
        x12 = n1[1] - n2[1]
        y12 = n1[2] - n2[2]
        z12 = n1[3] - n2[3]
        x13 = n1[1] - n3[1]
        y13 = n1[2] - n3[2]
        z13 = n1[3] - n3[3]

        square = 0.5 * np.sqrt((y12 * z13 - z12 * y13) ** 2 + (z12 * x13 * z13) ** 2 + (x12 * y13 - y12 * x13) ** 2)
        if square < meshStep * 1e-6:
            log_message(f"generateShell: square is: {square}")
            indToRem = np.append(indToRem, i)
    indToRem = np.array(indToRem, dtype='int32')

    tempEle = np.delete(tempEle, indToRem, axis=0)

    # Sort nodes by index
    shellPoints = shellPoints[shellPoints[:, 0].argsort(), :]
    # Renumirate
    shellEle = tempEle
    shellNode = shellPoints[:, 1:]
    for i in range(0, len(shellPoints)):
        idx = np.argwhere(tempEle == shellPoints[i, 0])
        for ki in idx:
            shellEle[ki[0], ki[1]] = i
        shellPoints[i, 0] = i

    # Choose BC nodes
    writedBC = 0
    for i in range(0, len(pointsHullLower)):
        logic = (shellPoints[:, 1:] == pointsHullLower[i])
        for i2 in range(0, len(logic)):
            if sum(logic[i2]) == 3:
                if not writedBC:
                    BCfix = shellPoints[i2, 0]
                    writedBC = 1
                    break
                else:
                    BCfix = np.append(BCfix, shellPoints[i2, 0])
                    break

    del tempEle, tempEle2, idx, i, shellPoints
    return shellNode, shellEle, BCfix