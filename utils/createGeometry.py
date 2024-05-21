import numpy as np
from scipy import linalg, interpolate
import matplotlib.pyplot as plt
import datetime
from .logger_leaflet import log_message
import warnings

warnings.simplefilter("ignore", RuntimeWarning)

# функция получения дуг разными методами сплайнов. Вольная интепритация аналогичной функции из community Matlab
def interparc(npoints=None, x_sample=None, y_sample=None, z_sample=None, key='linear'):
    if key == 'linear':
        xd = np.diff(x_sample)
        yd = np.diff(y_sample)
        zd = np.diff(z_sample)
        dist = (xd ** 2 + yd ** 2 + zd ** 2) ** 0.5
        u = np.cumsum(dist)
        u = np.hstack([[0], u])

        t = np.linspace(0, u.max(), npoints)
        xn = np.interp(t, u, x_sample)
        yn = np.interp(t, u, y_sample)
        zn = np.interp(t, u, z_sample)
    elif key == 'spline':
        xyz = np.vstack([x_sample, y_sample, z_sample]).T
        u = np.cumsum(np.r_[[0], np.linalg.norm(np.diff(xyz, axis=0), axis=1)])

        sx = interpolate.InterpolatedUnivariateSpline(u, x_sample)  # x(u) spline
        sy = interpolate.InterpolatedUnivariateSpline(u, y_sample)  # y(u) spline
        sz = interpolate.InterpolatedUnivariateSpline(u, z_sample)  # z(u) spline
        uu = np.linspace(u[0], u[-1], npoints)
        xn = sx(uu)
        yn = sy(uu)
        zn = sz(uu)
    elif key == 'pchip':
        xyz = np.vstack([x_sample, y_sample, z_sample]).T
        u = np.cumsum(np.r_[[0], np.linalg.norm(np.array(np.diff(xyz, axis=0), dtype='float64'), axis=1)])

        sx = interpolate.PchipInterpolator(u, x_sample)  # x(u) spline
        sy = interpolate.PchipInterpolator(u, y_sample)  # y(u) spline
        sz = interpolate.PchipInterpolator(u, z_sample)  # z(u) spline
        uu = np.linspace(u[0], u[-1], npoints)
        x = sx(uu)
        y = sy(uu)
        z = sz(uu)
        (xn, yn, zn) = np.array(interparc(npoints, x, y, z, key='linear'), dtype='float64')
    elif key == 'rbf':
        xyz = np.vstack([x_sample, y_sample, z_sample]).T
        u = np.cumsum(np.r_[[0], np.linalg.norm(np.diff(xyz, axis=0), axis=1)])

        rbfi_x = interpolate.Rbf(u, x_sample, function='multiquadric')
        rbfi_y = interpolate.Rbf(u, y_sample, function='multiquadric')
        rbfi_z = interpolate.Rbf(u, z_sample, function='multiquadric')
        uu = np.linspace(u[0], u[-1], npoints)
        x = rbfi_x(uu)
        y = rbfi_y(uu)
        z = rbfi_z(uu)
        (xn, yn, zn) = np.array(interparc(npoints, x, y, z, key='linear'), dtype='float64')
    else:
        raise Exception('interparc no such method: ' + str(key))
    return xn, yn, zn


# аналогичная функция из Matlab
def cart2pol(x, y, lz):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    z = lz
    return phi, rho, z


# аналогичная функция из Matlab
def pol2cart(phi=None, rho=None, lz=None):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    z = lz
    return x, y, z


def createGeometry(HGT=10, Lstr=0, SEC=119, DIA=13, THK=0.5, ANG=0, Lift=0, CVT=0.5, LAS=1, mesh_step=0.35):

    global R_from_dot_min, HGT_hk
    # работаем с радиусом, а не с диаметров
    RAD = DIA / 2
    # нужно для выбора точек на гранях, при заполнении поверхности
    amplif_coef = 100
    # при ANG != 0 высота отклонения от плоскости oZ
    Hk = np.tan(np.deg2rad(ANG)) * RAD
    # рудимент с матлаба, лень было править в коде
    deltaX = mesh_step

    finalrad = 0

    HGT_hk = HGT + Hk
    if HGT_hk <= 0:
        message = 'HGT + Hk <= 0'
        raise Exception(message)

    # генерация fixed грани. Задаем точками контур, строим через них shape preserved spline (PCHIP - piecewise
    # cubic Hermite interpolating polynomial)
    if Lstr < 0:
        message = "Length of straight part below Zero"
        raise Exception(message)
    if Lstr > 0:
        thetaRef = [np.deg2rad(90 - SEC / 2), np.deg2rad(90 - SEC / 2), np.deg2rad(90 - SEC / 2),
                    np.deg2rad(90 - SEC / 2), np.deg2rad(90 - SEC / 2), np.deg2rad(90 - (SEC - SEC / 6) / 2),
                    np.deg2rad(80), np.deg2rad(90), np.deg2rad(100), np.deg2rad(90 + (SEC - SEC / 6) / 2),
                    np.deg2rad(90 + SEC / 2), np.deg2rad(90 + SEC / 2), np.deg2rad(90 + SEC / 2),
                    np.deg2rad(90 + SEC / 2), np.deg2rad(90 + SEC / 2)]
        heightRef = [HGT, HGT - Lstr * 0.25, HGT - Lstr * 0.5, HGT - Lstr * 0.75, HGT - Lstr,
                     HGT/3, Lift+0.06, Lift, Lift+0.06, HGT/3,
                     HGT - Lstr, HGT - Lstr * 0.75, HGT - Lstr * 0.5, HGT - Lstr * 0.25, HGT]
        fitRho = np.linspace(RAD, RAD, len(thetaRef))
        (tX, tY, tZ) = pol2cart(thetaRef, fitRho, heightRef)
        arc = np.array(interparc(60, tX, tY, tZ, key='pchip'), dtype='float64')
    else:
        thetaRef = np.array([np.deg2rad(90 - SEC / 2), np.deg2rad(80), np.deg2rad(90), np.deg2rad(100), np.deg2rad(90 + SEC / 2)])
        heightRef = np.array([HGT, Lift+0.06, Lift, Lift+0.06, HGT])
        fitRho = np.linspace(RAD, RAD, len(thetaRef))
        (tX, tY, tZ) = pol2cart(thetaRef, fitRho, heightRef)
        arc = np.array(interparc(60, tX, tY, tZ, key='pchip'), dtype='float64')

    (tX, tY, tZ) = (arc[0], arc[1], arc[2])
    # нам нужно чтобы между точками сплайна было равное расстояние, поэтоу считаем какая длина у полученного отрезка
    # и делим на mesh_step. полученное интовое число - количество точек
    dx = np.abs(np.array(np.diff(tX)), dtype='float64')
    dy = np.abs(np.array(np.diff(tY)), dtype='float64')
    dz = np.abs(np.array(np.diff(tZ)), dtype='float64')
    distancesBetweenVertices = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
    perimeter = np.sum(distancesBetweenVertices)
    perim = perimeter
    points_in_side = int(np.ceil(perimeter / deltaX)) + 3

    del dx, dy, dz, fitRho, arc, thetaRef, heightRef

    contour_leaf = np.array(interparc(points_in_side, tX, tY, tZ, key='spline'), dtype='float64')
    tempContour_leaf = np.array(interparc(int(amplif_coef * points_in_side), tX, tY, tZ, key='spline'), dtype='float64')

    del tX, tY, tZ
    # contour_leaf = contour_leaf[:, 1:-1]
    pointsHullLower = contour_leaf

    # генерация свободного края. генерацию набора точек вынес в отдельную функцию configureTopSide

    upsideTop = np.array(configureTopSide_new(HGT, Hk, RAD, ANG, LAS, SEC), dtype='float64')

    tX = upsideTop[:, 0]
    tY = upsideTop[:, 1]
    tZ = upsideTop[:, 2]
    dx = np.diff(tX)
    dy = np.diff(tY)
    dz = np.diff(tZ)
    distancesBetweenVertices = (dx ** 2 + dy ** 2 + dz ** 2) ** 0.5
    perimeter = sum(distancesBetweenVertices)
    points_in_top = int(np.ceil(perimeter / deltaX))
    if LAS < 0.5:
        method = 'linear'
    else:
        method = 'pchip'
    topSpline = np.array(interparc(points_in_top, upsideTop[:, 0], upsideTop[:, 1], upsideTop[:, 2], key=method),
                         dtype='float64')
    topSpline = topSpline[:, 1:-1]
    pointsHullUpper = topSpline
    points = np.append(pointsHullLower, pointsHullUpper, axis=1)
    hull = points
    del dx, dy, dz, distancesBetweenVertices, perimeter

    tempTopSpline = np.array(interparc(amplif_coef * points_in_top, upsideTop[:, 0], upsideTop[:, 1], upsideTop[:, 2],
                                       key=method), dtype='float64')

    # plt.plot(topSpline[0], topSpline[1], '-o')
    # plt.plot(upsideTop[:, 0], upsideTop[:, 1], '-o')
    # plt.xlim(-15, 15)
    # plt.ylim(0, 10)
    # plt.show()
    # считаем кривизну CVT. Находим максимальный (плоская створка) и минимальный радиус окружности
    # через 3 точки (центр фиксированной грани, свободного края и центра отрезка через эти точки).
    # Затем считаем расстояние между ними и итеративно подбираем координаты центра отрезка,
    # чтобы получить заданный % (CVT) между максимально плоской и максимально выпуклой створкой
    rad = [0 for i in range(3)]
    leaf_spline = np.array([[float for x in range(3)] for y in range(3)])
    leaf_spline[0] = contour_leaf[:, int(np.ceil(contour_leaf.shape[1] / 2))]
    pl = leaf_spline[0]
    pu = topSpline[:, int(np.ceil(topSpline.shape[1] / 2))]
    # pu[0] = 0
    pc = (pl + pu) / 2
    R_from_dot_min = ((0 - pc[0]) ** 2 + (RAD - pc[1]) ** 2 + (max(HGT_hk, HGT) - pc[2]) ** 2) ** 0.5
    leaf_spline[1] = pc
    leaf_spline[2] = pu

    Rmin = ((pu[0] - pc[0]) ** 2 + (pu[1] - pc[1]) ** 2 + (pu[2] - pc[2]) ** 2) ** 0.5

    t1 = datetime.datetime.now()

    vect_coord = [0 for i in range(3)]
    timeDelta = datetime.datetime.now() - t1

    pBegin = [pc[0], pc[1], pc[2]]
    iter = 0
    delta = 1e-3
    liter = 0
    # пока не достигнем Rmin - двигай по Y и Z точку на delta
    while not (Rmin * 0.99 < rad[1] < Rmin * 1.01):

        iter += 1
        leaf_spline[1] = pc
        rad = curvature(leaf_spline)

        if Rmin * 0.99 < rad[1] < Rmin * 1.01:
            break
        if np.mod(iter, 500) == 0:
            liter += 1
            if np.mod(liter, 2) == 0:
                delta /= 2
            else:
                delta *= 3
        # log_message("pc before > (" + str(pc[0]) + " ," + str(pc[1]) + " ," + str(pc[2]) +")")
        if rad[1] < Rmin:
            pc[1] += delta
            pc[2] += delta
        else:
            pc[1] -= delta
            pc[2] -= delta
        # del y, z, X
        timeDelta = datetime.datetime.now() - t1

    pEnd = pc
    # log_message("pc after >  (" + str(pc[0]) + " ," + str(pc[1]) + " ," + str(pc[2]) + ")")

    if CVT < 0 or CVT > 1:  # заданый радиус меньше геометрически возможного - ошибка-рыбка
        message = "Error in CVT"
        raise Exception(message)

    distanceCVT = ((pEnd[0] - pBegin[0]) ** 2 + (pEnd[1] - pBegin[1]) ** 2 + (pEnd[2] - pBegin[2]) ** 2) ** 0.5
    pc = [(pl[0] + pu[0]) / 2, (pl[1] + pu[1]) / 2, (pl[2] + pu[2]) / 2]

    currRad = CVT * distanceCVT

    iter = 0
    delta = 1e-3
    t1 = datetime.datetime.now()
    liter = 0

    distanceCurr = ((pc[0] - pBegin[0]) ** 2 + (pc[1] - pBegin[1]) ** 2 + (pc[2] - pBegin[2]) ** 2) ** 0.5
    # пока не достигнем currRad - двигай по Y и Z точку на delta
    while not (currRad * 0.99 < distanceCurr < currRad * 1.01) and timeDelta.seconds < 60:

        iter += 1
        leaf_spline[1] = pc
        distanceCurr = ((pc[0] - pBegin[0]) ** 2 + (pc[1] - pBegin[1]) ** 2 + (pc[2] - pBegin[2]) ** 2) ** 0.5

        if currRad * 0.99 <= distanceCurr <= currRad * 1.01:
            break
        if np.mod(iter, 250) == 0:
            liter += 1
            deltaTime = datetime.datetime.now() - t1
            log_message(f'R = {str(distanceCurr)} | target > {str(currRad)} | time(s) = {str(deltaTime.seconds)}'
                              f' | delta = {str(delta)}')
            del deltaTime
            if np.mod(liter, 4) == 0:
                delta /= 2
            elif np.mod(liter, 4) == 2:
                delta *= 3

        if distanceCurr >= currRad:
            pc[1] += delta
            pc[2] += delta
        else:
            pc[1] -= delta
            pc[2] -= delta

        timeDelta = datetime.datetime.now() - t1

    leaf_spline[1] = pc
    # запрашивали CVT, получили CVT_real, столько времени потратили
    log_message('CVT = ' + str(CVT) + ' | CVT_real =  ' + str(distanceCurr / distanceCVT) + ' | time(s) = ' + str(
        timeDelta.seconds))
    currRad = rad[1]

    finalR = np.sqrt((0 - leaf_spline[1][0]) ** 2 + (RAD - leaf_spline[1][1]) ** 2 + (
            max(HGT_hk, HGT) - leaf_spline[1][2]) ** 2)
    del delta, iter, pc

    # Сплайн через центр отрезка (см блок выше) и конца прямой части у вершик коммесур
    if Lstr > 0:
        tempDist = np.abs(contour_leaf[2] - (HGT - Lstr))
        ind = np.argmin(tempDist)

        if ind - len(tempDist) / 2 > 0:
            ind = ind - 3
        else:
            ind = ind + 3
        del tempDist
    else:
        ind = 0

    tSplin = np.transpose(
        np.append(
            np.append(
                (contour_leaf[0, ind], contour_leaf[1, ind], contour_leaf[2, ind]), leaf_spline[1])
            , (-contour_leaf[0, ind], contour_leaf[1, ind], contour_leaf[2, ind])
        ).reshape(3, 3)
    )
    horSplineMid = np.array(interparc(30, tSplin[0], tSplin[1], tSplin[2], key='pchip'), dtype='float64')
    dx = np.abs(np.array(np.diff(horSplineMid[0])))
    dy = np.abs(np.array(np.diff(horSplineMid[1])))
    dz = np.abs(np.array(np.diff(horSplineMid[2])))
    distancesBetweenVertices = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
    distance = sum(distancesBetweenVertices)
    points_in_hor_line = int(np.ceil(distance / deltaX)) + 1
    del dx, dy, dz, distance, distancesBetweenVertices, horSplineMid
    horSplineMid = np.array(interparc(points_in_hor_line, tSplin[0], tSplin[1], tSplin[2], key='pchip'),
                            dtype='float64')
    del tSplin, leaf_spline

    # Придание толщины поверхности. Нахождение нормалей.
    # сложно описывать как именно вычитывается точка, относительно которой строится карта нормалей поверхности створки.
    # Если нужно - вставь блок, который нарисует все прямые. названия pl1,pus и другое читать как point low 1,
    # point low 2, point low center, point upper spline 1, point upper spline 2, ..., point upper z
    pl1 = upsideTop[int(np.floor(len(upsideTop) / 2)), :]
    pl2 = np.array(pol2cart(np.deg2rad(90), RAD * 1.0, Lift), dtype='float64')
    plc = 0.5 * (pl2 + pl1)

    pus1 = np.array(pol2cart(np.deg2rad(90 - SEC / 2), RAD * 1.0, HGT), dtype='float64')
    pus2 = np.array(pol2cart(np.deg2rad(90 + SEC / 2), RAD * 1.0, HGT), dtype='float64')
    pu1 = 0.5 * (pus2 + pus1)

    pu2 = np.array(pol2cart(np.deg2rad(90), 100.0 * RAD, max(HGT, HGT + Hk)), dtype='float64')
    pu2[0] = 0
    puz = np.array(pol2cart(np.deg2rad(90), -100.0 * RAD, HGT), dtype='float64')
    puz[0] = 0

    # точки пересечения прямых
    A = pl2 - pl1
    B = (pl2 + [-100, 0, 0]) - pl1
    Nx = A[1] * B[2] - A[2] * B[1]
    Ny = A[2] * B[0] - A[0] * B[2]
    Nz = A[0] * B[1] - A[1] * B[0]

    n1 = [Nx, Ny, Nz]

    nline1 = np.array(([[plc[0], n1[0]], [plc[1], n1[1]], [plc[2], n1[2]]]))
    dP1 = np.diff(nline1)
    P21 = np.transpose(1000 * dP1) + n1
    P21 = P21[0]
    nline = np.transpose(np.array([[plc[0], n1[0], P21[0]], [plc[1], n1[1], P21[1]], [plc[2], n1[2], P21[2]]]))
    uline = np.transpose(np.array([[pu2[0], pu1[0], puz[0]], [pu2[1], pu1[1], puz[1]], [pu2[2], pu1[2], puz[2]]]))

    y1, y2, z1, z2 = nline[0, 1], nline[2, 1], nline[0, 2], nline[2, 2]
    y3, y4, z3, z4 = uline[0, 1], uline[2, 1], uline[0, 2], uline[2, 2]
    u = ((y4 - y3) * (z1 - z3) - (z4 - z3) * (y1 - y3)) / ((z4 - z3) * (y2 - y1) - (y4 - y3) * (z2 - z1))
    yinter = y1 + u * (y2 - y1)
    zinter = z1 + u * (z2 - z1)
    del y1, y2, z1, z2, y3, y4, z3, z4

    # точки нормали
    try:
        normalPoint = np.empty(3)
        normalPoint[0] = 0
        normalPoint[1] = yinter*1
        normalPoint[2] = zinter
    except Exception:
        normalPoint = pol2cart(np.deg2rad(90), RAD * 3, max(HGT, HGT_hk))

    pl2 = None
    pus1 = None
    pus2 = None
    pu2 = None
    puz = None
    rx = None
    ry = None
    rz = None
    ind = None
    p1 = None
    p2 = None
    layers_count = None
    phu = None
    phl = None

    del pl1, pl2, plc, pus1, pus2, pu2, puz, nline, P21, uline, yinter, zinter
    del rx, ry, rz, ind, p1, p2, layers_count, phu, phl

    points3 = points
    hull2 = normalShift(hull, normalPoint, THK)
    points3 = np.append(points3, hull2, axis=1)

    # заполнение поверхности. если толщина больше шага сетки - добавления layers_count слоев точек на гранях
    if THK > mesh_step:
        #     disp('debug');
        layers_count = np.floor(THK / mesh_step)
        for i in range(1, int(layers_count)+1):
            tpHullLower2normal = normalShift(pointsHullLower, normalPoint, THK*i/(layers_count+1))
            tpHullUpper2normal = normalShift(pointsHullUpper, normalPoint, THK*i/(layers_count+1))
            if i > 1:
                pHullLower2normal = np.append(pHullLower2normal, tpHullLower2normal, axis=1)
                pHullUpper2normal = np.append(pHullUpper2normal, tpHullUpper2normal, axis=1)
            else:
                pHullLower2normal = tpHullLower2normal
                pHullUpper2normal = tpHullUpper2normal
        del layers_count, tpHullLower2normal, tpHullUpper2normal
    else:
        pHullLower2normal = normalShift(pointsHullLower, normalPoint, THK)
        pHullUpper2normal = normalShift(pointsHullUpper, normalPoint, THK)
    # заполенение поверхности. Подкидываем нужные массивы точек, двигаемся по хорде через коммисуры
    # и центр отрезка (см блоки выше), находим точки с одинаковой координатой X, строим сплайн.
    for iterThickness in range(1, 3):
        if iterThickness == 1:
            ThorSplineMid = horSplineMid
            tCl = tempContour_leaf
            tpoints = points
            tTS = tempTopSpline
        else:
            ThorSplineMid = normalShift(horSplineMid, normalPoint, THK)
            tCl = normalShift(tempContour_leaf, normalPoint, THK)
            tpoints = points
            tTS = normalShift(tempTopSpline, normalPoint, THK)
        max_iter = ThorSplineMid.shape[1]
        for i in range(1, max_iter - 1):
            iterX = ThorSplineMid[0, i]
            indLow = np.argmin(np.abs(iterX - np.transpose(tCl[0, :])))
            indTop = np.argmin(np.abs(iterX - np.transpose(tTS[0, :])))
            tSplineBot1 = tCl[:, indLow]
            tSplineBot2 = ThorSplineMid[:, i]
            tSplineBot3 = tTS[:, indTop]
            tSplineBot = np.array([tSplineBot1, tSplineBot2, tSplineBot3])
            tSpline2 = np.array(interparc(20, tSplineBot.T[0], tSplineBot.T[1], tSplineBot.T[2], key='pchip'),
                                dtype='float64')
            dx = np.array(np.diff(np.transpose(tSpline2[0, :])), dtype='float64')
            dy = np.array(np.diff(np.transpose(tSpline2[1, :])), dtype='float64')
            dz = np.array(np.diff(np.transpose(tSpline2[2, :])), dtype='float64')
            distancesBetweenVertices = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
            distance = sum(distancesBetweenVertices)
            if distance < deltaX:
                del indLow, indTop
                del dx, dy, dz, distancesBetweenVertices, distance
                continue
            points_in_surface = int(np.floor(distance / deltaX))
            currSplineBot = np.array(interparc(points_in_surface, tSplineBot.T[0], tSplineBot.T[1], tSplineBot.T[2],
                                               key='pchip'), dtype='float64')
            del tSpline2, tSplineBot
            del tSplineBot1, tSplineBot3, tSplineBot2
            currSpline = currSplineBot[:, range(1, currSplineBot.shape[1] - 1)]
            vec_delInd = np.array(range(0, len(currSpline[0, :])))
            for iii1 in range(0, len(vec_delInd)):
                vec_delInd[iii1] = -1
            for indP in range(0,len(currSpline[0, :])):
                ldist = np.sqrt((contour_leaf[0, :] - currSpline[0, indP]) ** 2 +
                                (contour_leaf[1, :] - currSpline[1, indP]) ** 2 +
                                (contour_leaf[2, :] - currSpline[2, indP]) ** 2)
                # log_message("min ", indP, " = ", min(ldist))
                # delInd = []
                delInd = np.argwhere(ldist < mesh_step*0.5)
                if len(delInd) > 0:
                    vec_delInd[indP] = indP
                del delInd
            try:
                currSpline = np.delete(currSpline, vec_delInd[vec_delInd != -1], axis=1)
            except:
                currSpline = currSpline
            tpoints = np.append(tpoints, currSpline, axis=1)

            del indLow, vec_delInd, indTop, currSpline, dx, dy, dz, distancesBetweenVertices, distance

        if iterThickness == 1:
            pointsInner = tpoints
        else:
            pointsOuter = tpoints
        points3 = np.append(points3, tpoints, axis=1)
    if THK > mesh_step:
        points3 = np.append(points3, pHullLower2normal, axis=1)
        points3 = np.append(points3, pHullUpper2normal, axis=1)

    points = points3
    del points3
    # избавляемся от дублей
    points = np.unique(points, axis=1)
    message = 'PC is constructed'
    del hull, hull2
    del vect_coord, topSpline, points_in_side, i, contour_leaf
    # на всякий случай округляем до decimal знака
    decimal = 14
    return (np.unique(np.around(pointsInner, decimal), axis=1),
            np.unique(np.around(pointsOuter, decimal), axis=1),
            np.unique(np.around(pHullLower2normal, decimal), axis=1),
            np.unique(np.around(pHullUpper2normal, decimal), axis=1),
            np.unique(np.around(pointsHullLower, decimal), axis=1),
            np.unique(np.around(pointsHullUpper, decimal), axis=1),
            np.unique(np.around(points, decimal), axis=1), normalPoint, finalR, currRad, message)
    # return (pointsInner, pointsOuter, pHullLower2normal, pHullUpper2normal,pointsHullLower, pointsHullUpper,
    #         points, normalPoint, finalR, currRad)

# функция генерации точек свободного края.
def configureTopSide(HGT=None, Hk=None, DIA=None, ANG=None, LAS=None, SA=None):
    maxPhi = np.deg2rad(90-SA/2)
    s105s15 = np.sin(np.deg2rad(105)) / np.sin(np.deg2rad(15))
    san = np.sin(np.deg2rad(-ANG))
    H_loc = HGT + Hk

    lDia = DIA

    upside = np.array([[float for x in range(3)] for y in range(11)])

    cylR_next = lDia
    H_next = cylR_next * san
    upside[0] = pol2cart(np.deg2rad(90 - SA / 2), cylR_next, H_next)

    cylR_next = 0.66 * lDia + 0.33 * LAS * s105s15
    H_next = cylR_next * san
    upside[1] = pol2cart(np.deg2rad(90 - SA / 2), cylR_next, H_next)

    if upside[0, 0] - upside[0, 1] < 0:
        raise Exception('LAS is too large')

    cylR_next = 0.33 * lDia + 0.66 * LAS * s105s15
    H_next = cylR_next * san
    upside[2] = pol2cart(np.deg2rad(90 - SA / 2), cylR_next, H_next)

    cylR_next = 0.15 * lDia + 0.75 * LAS * s105s15
    H_next = cylR_next * san
    upside[3] = pol2cart(np.deg2rad(90 - SA / 2), cylR_next, H_next)

    h1 = upside[2][2]
    l1 = upside[2][1]

    cylR_next = LAS * s105s15
    H_next = cylR_next * san
    upside[4] = pol2cart(np.deg2rad(90 - SA / 2), cylR_next, H_next)

    h1 = h1 - upside[3][2]
    l1 = l1 - upside[3][1]

    l2 = upside[3][1] - LAS

    cylR_next = LAS
    H_next = h1 * l2 / l1
    upside[5] = pol2cart(np.deg2rad(90), cylR_next, H_next)

    del h1, l1, l2
    cylR_next = LAS * s105s15
    H_next = cylR_next * san
    upside[6] = pol2cart(np.deg2rad(90 + SA / 2), cylR_next, H_next)

    cylR_next = 0.15 * lDia + 0.75 * LAS * s105s15
    H_next = cylR_next * san
    upside[7] = pol2cart(np.deg2rad(90 + SA / 2), cylR_next, H_next)

    cylR_next = 0.33 * lDia + 0.66 * LAS * s105s15
    H_next = cylR_next * san
    upside[8] = pol2cart(np.deg2rad(90 + SA / 2), cylR_next, H_next)

    cylR_next = 0.66 * lDia + 0.33 * LAS * s105s15
    H_next = cylR_next * san
    upside[9] = pol2cart(np.deg2rad(90 + SA / 2), cylR_next, H_next)

    cylR_next = lDia
    H_next = cylR_next * san
    upside[10] = pol2cart(np.deg2rad(90 + SA / 2), cylR_next, H_next)

    # поднимаем на высоту коммисуры, и костылем подправляем чтоб было ровно.
    upside[:, 2] += H_loc

    if upside[0, 2] != HGT:
        # log_message("added > %f" % (HGT - upside[0, 2]))
        upside[:, 2] += HGT - upside[0, 2]

    plt.plot(upside[0], upside[1], '-o')
    plt.xlim(-15, 15)
    plt.ylim(0, 10)
    plt.show()

    return upside


def configureTopSide_new(HGT=None, Hk=None, DIA=None, ANG=None, LAS=None, SA=None):
    maxPhi = np.deg2rad(90-SA/2)
    s105s15 = np.sin(np.deg2rad(105)) / np.sin(np.deg2rad(15))
    san = np.sin(np.deg2rad(-ANG))
    H_loc = HGT + Hk

    lDia = DIA

    upside = np.array([[float for x in range(3)] for y in range(11)])

    ref120 = pol2cart(np.deg2rad(90 - SA / 2), lDia, lDia*san)

    cylR_next = lDia
    H_next = cylR_next * san
    upside[0] = pol2cart(np.deg2rad(90 - SA / 2), cylR_next, H_next)

    shift = upside[0]-ref120

    cylR_next = 0.66 * lDia + 0.33 * LAS * s105s15
    H_next = cylR_next * san
    upside[1] = pol2cart(np.deg2rad(90 - SA / 2), cylR_next, H_next)

    if upside[0, 0] - upside[1, 0] < 0:
        raise Exception('LAS is too large')

    cylR_next = 0.33 * lDia + 0.66 * LAS * s105s15
    H_next = cylR_next * san
    upside[2] = pol2cart(np.deg2rad(90 - SA / 2), cylR_next, H_next)

    cylR_next = 0.15 * lDia + 0.75 * LAS * s105s15
    H_next = cylR_next * san
    upside[3] = pol2cart(np.deg2rad(90 - SA / 2), cylR_next, H_next)

    h1 = upside[3][2]
    l1 = upside[3][1]

    cylR_next = LAS * s105s15
    H_next = cylR_next * san
    upside[4] = pol2cart(np.deg2rad((90 - SA / 2)), cylR_next, H_next)

    h1 = h1 - upside[4][2]
    l1 = l1 - upside[4][1]

    l2 = upside[4][1] - LAS

    cylR_next = LAS
    H_next = h1 * l2 / l1
    upside[5] = pol2cart(np.deg2rad(90), cylR_next, H_next)

    del h1, l1, l2

    # log_message('upside before:')
    # log_message(upside)
    #
    # log_message('shifts')
    # log_message(shift)

    upside[1:5, :] += shift

    upside[[6, 7, 8, 9, 10], :] = upside[[4, 3, 2, 1, 0], :]
    upside[[6, 7, 8, 9, 10], 0] *= -1

    # log_message('upside after:')
    # log_message(upside)

    # поднимаем на высоту коммисуры, и костылем подправляем чтоб было ровно.
    upside[:, 2] += H_loc

    if upside[0, 2] != HGT:
        # log_message("added > %f" % (HGT - upside[0, 2]))
        upside[:, 2] += HGT - upside[0, 2]

    # plt.plot(upside[:, 0], upside[:, 1], '-o')
    # plt.show()

    return upside

# фукция сдвига матрицы pt по нормали к точке pn на толщину THK
def normalShift(pt=None, pn=None, THK=None):
    pn = [[pn[0]], [pn[1]], [pn[2]]]
    pt = np.array(pt)
    tn = pt - pn
    lengts_n = np.sqrt(tn[0, :] ** 2 + tn[1, :] ** 2 + tn[2, :] ** 2)
    n = np.zeros(tn.shape)
    for i in range(0, tn.shape[1]):
        n[0, i] = tn[0, i] / lengts_n[i]
        n[1, i] = tn[1, i] / lengts_n[i]
        n[2, i] = tn[2, i] / lengts_n[i]

    points = pt - 1.0 * n * THK
    return points

# 3 функции далее - выдрано и адаптировано из community Matlab для нахождения радиуса описанной окружности.
# не разбирался сильно в математике, работает и ладно
def circumcenter(A=None, B=None, C=None):
    # Center and radius of the circumscribed circle for the triangle ABC
    #  A,B,C  3D coordinate vectors for the triangle corners
    #  R      radius
    #  M      3D coordinate vector for the center
    #  k      Vector of length 1/R in the direction from A towards M
    #         (Curvature vector)
    A = np.array(A, dtype='float64')
    B = np.array(B, dtype='float64')
    C = np.array(C, dtype='float64')
    D = np.cross(np.transpose(B - A), np.transpose(C - A))
    b = np.linalg.norm(A - C)
    c = np.linalg.norm(A - B)

    E = np.cross(D, np.transpose(B - A))
    F = np.cross(D, np.transpose(C - A))
    G = (b ** 2 * E - c ** 2 * F) / np.linalg.norm(D) ** 2 / 2
    M = A + G
    R = np.linalg.norm(G)

    if R == 0:
        k = G
    else:
        k = np.transpose(G) / R ** 2

    return R, M, k


def find_intersections(line1, line2):
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
        raise Exception('lines do not intersect')

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return x, y


def curvature(X=None):
    # radius of curvature and curvature vector for 2D or 3D curve
    #  [L,R,Kappa] = curvature(X)
    #   X:   2 or 3 column array of x, y (and possibly z) coordiates
    #   L:   Cumulative arc length
    #   R:   radius of curvature
    #   k:   Curvature vector
    N = X.shape[0]
    dims = X.shape[1]
    if dims == 2:
        X = [[np.append(X[0], 0)], [np.append(X[1], 0)], [np.append(X[2], 0)]]

    L = np.zeros((N, 1))
    R = np.array(np.zeros(N), dtype='float64')
    k = np.array(np.zeros([N, 3]), dtype='float64')
    for i in np.arange(1, N - 1):
        (out1, __, __) = circumcenter(np.transpose(X[i][:]), np.transpose(X[i - 1][:]), np.transpose(X[i + 1][:]))
        R[i] = out1

    return R