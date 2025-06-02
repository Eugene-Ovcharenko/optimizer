import os
from typing import List, Tuple, Optional, Literal, Union
import numpy as np
from scipy import spatial
from alphashape import alphashape
from glob2 import glob
from utils.global_variable import *
from utils.create_geometry_utils import cart2pol, pol2cart


class ReadMesh:
    """
    Container class for storing and modifying mesh data (nodes and elements) parsed from an Abaqus `.odb` file.

    This class provides a lightweight object for managing structured mesh data retrieved during
    post-processing, enabling easy access, replacement, and augmentation of nodal and element arrays.

    Attributes:
        nodes (np.ndarray): Array of nodal coordinates (typically shape: [N, 3]).
        elements (np.ndarray): Array of element connectivity (typically shape: [M, K], where K is the number of nodes per element).

    Methods:
        repNodes(lnodes): Replace the internal node array with a new one.
        repElements(lelements): Replace the internal element array with a new one.
        addNodes(lnodes): Append new nodes to the existing node array.
        addElements(lelements): Append new elements to the existing element array.
        getNodes(): Return the stored node array.
        getElements(): Return the stored element array.

    """

    def __init__(self, lnodes, lelements):
        """
        Initialize the readedMesh object with given node and element arrays.

        Args:
            lnodes (np.ndarray): Initial node array.
            lelements (np.ndarray): Initial element connectivity array.
        """
        self.nodes = lnodes
        self.elements = lelements

    def repNodes(self, lnodes):
        """
        Replace the current node array.

        Args:
            lnodes (np.ndarray): New node array to replace the current one.
        """
        self.nodes = lnodes

    def repElements(self, lelements):
        """
        Replace the current element array.

        Args:
            lelements (np.ndarray): New element array to replace the current one.
        """
        self.elements = lelements

    def addNodes(self, lnodes):
        """
        Append new nodes to the current node array.

        Args:
            lnodes (np.ndarray): Array of new nodes to append.
        """
        self.nodes = np.append(self.nodes, lnodes)

    def addElements(self, lelements):
        """
        Append new elements to the current element array.

        Args:
            lelements (np.ndarray): Array of new elements to append.
        """
        self.elements = np.append(self.elements, lelements)

    def getNodes(self):
        """
        Get the current node array.

        Returns:
            np.ndarray: Stored node array.
        """
        return self.nodes

    def getElements(self):
        """
        Get the current element array.

        Returns:
            np.ndarray: Stored element array.
        """
        return self.elements


def read_data(
        pathToAbaqus: str = '',
        endPath: str = '',
        partName: str = ''
) -> Tuple[float, float, float, float, float, float]:
    """
    Parse simulation results and compute leaflet performance metrics from Abaqus `.odb`-derived text exports.

    This function reads mechanical response data (stress, strain, displacement) from structured output files,
    computes valve leaflet performance indicators, and optionally extracts coaptation areas and geometric scores.
    The function adapts automatically to single-component or contact-based multicomponent leaflet models.

    Args:
        pathToAbaqus (str): Path to the directory containing Abaqus job folders and results.
        endPath (str): Directory where parsed simulation output (e.g., text files) is stored.
        partName (str): Base name of the simulated model (used to locate result folders).

    Returns:
        Tuple[float, float, float, float, float, float]:
            - LMN_op: Leakage metric in open configuration.
            - LMN_cl: Leakage metric in closed configuration.
            - Smax: Maximum principal stress across all frames.
            - VMS: Maximum von Mises stress.
            - perf_index: Composite performance score [0, 1] based on stress and geometric integrity.
            - isHelicopter: Indicator of "helicopter" artifact in leaflet deformation (maximum distance of leaflet twisting).

    Raises:
        RuntimeError: If result files (e.g., stress, LE, displacement) are missing or incomplete.

    Notes:
        - The function internally switches between `read_data_single()` and `read_data_contact()` based on folder structure.
        - Output assumes post-processed `.txt` files are pre-exported from the Abaqus `.odb` file.
        - The "helicopter" indicator detects non-physiological leaflet deformations by symmetry analysis.

    Example:
        >>> read_data(pathToAbaqus='./jobs/', endPath='./results/', partName='leaflet_model.inp')
        (0.93, 0.82, 5.6, 4.8, 0.91, 0)
    """

    def read_data_contact(pathToAbaqus=None, endPath=None, partName=None):
        mesh_step = get_mesh_step()
        foldPath = endPath + partName[0:-5].upper() + '/' + partName.upper()

        mesh1 = ReadMesh(lnodes=np.loadtxt(foldPath + '-1/Nodes.txt'),
                         lelements=np.loadtxt(foldPath + '-1/Elements.txt'))
        mesh2 = ReadMesh(lnodes=np.loadtxt(foldPath + '-2/Nodes.txt'),
                         lelements=np.loadtxt(foldPath + '-2/Elements.txt'))
        mesh3 = ReadMesh(lnodes=np.loadtxt(foldPath + '-3/Nodes.txt'),
                         lelements=np.loadtxt(foldPath + '-3/Elements.txt'))
        stressFiles = glob(foldPath + '-1/' + 'Stress_*.txt')

        newstr = ''.join((ch if ch in '0123456789' else ' ') for ch in stressFiles[0])
        one_step = ([int(i) for i in newstr.split()])[-1]

        newstr = ''.join((ch if ch in '0123456789' else ' ') for ch in stressFiles[1])
        two_step = ([int(i) for i in newstr.split()])[-1]

        if get_valve_position().lower() == 'ao':
            closed_step = max(one_step, two_step)
            opened_step = min(one_step, two_step)
        else:
            closed_step = min(one_step, two_step)
            opened_step = max(one_step, two_step)

        del newstr, one_step, two_step

        s_max1_1step = np.loadtxt(foldPath + '-1/' + 'Stress_' + str(closed_step) + '.txt')
        s_max1_2step = np.loadtxt(foldPath + '-1/' + 'Stress_' + str(opened_step) + '.txt')
        le_max1_1step = np.loadtxt(foldPath + '-1/' + 'LE_' + str(closed_step) + '.txt')
        le_max1_2step = np.loadtxt(foldPath + '-1/' + 'LE_' + str(opened_step) + '.txt')

        s_max2_1step = np.loadtxt(foldPath + '-2/' + 'Stress_' + str(closed_step) + '.txt')
        s_max2_2step = np.loadtxt(foldPath + '-2/' + 'Stress_' + str(opened_step) + '.txt')
        le_max2_1step = np.loadtxt(foldPath + '-2/' + 'LE_' + str(closed_step) + '.txt')
        le_max2_2step = np.loadtxt(foldPath + '-2/' + 'LE_' + str(opened_step) + '.txt')

        s_max3_1step = np.loadtxt(foldPath + '-3/' + 'Stress_' + str(closed_step) + '.txt')
        s_max3_2step = np.loadtxt(foldPath + '-3/' + 'Stress_' + str(opened_step) + '.txt')
        le_max3_1step = np.loadtxt(foldPath + '-3/' + 'LE_' + str(closed_step) + '.txt')
        le_max3_2step = np.loadtxt(foldPath + '-3/' + 'LE_' + str(opened_step) + '.txt')

        eleCount = len(mesh1.getElements())

        disp1 = np.loadtxt(foldPath + '-1/' + 'U_' + str(opened_step) + '.txt')
        disp2 = np.loadtxt(foldPath + '-2/' + 'U_' + str(opened_step) + '.txt')
        disp3 = np.loadtxt(foldPath + '-3/' + 'U_' + str(opened_step) + '.txt')
        try:
            outGap1 = find_lumen_opened(disp1[:, 1:], mesh1.nodes[:, 1:], mesh_step=mesh_step)
        except:
            disp1 = disp1[:len(mesh1.nodes[:, 1:]), :]
            outGap1 = find_lumen_opened(disp1[:, 1:], mesh1.nodes[:, 1:], mesh_step=mesh_step)
            disp2 = disp2[:len(mesh2.nodes[:, 1:]), :]
            disp3 = disp3[:len(mesh3.nodes[:, 1:]), :]

        outGap2 = find_lumen_opened(disp2[:len(mesh2.nodes[:, 1:]), 1:], mesh2.nodes[:, 1:], mesh_step=mesh_step)
        outGap3 = find_lumen_opened(disp3[:len(mesh3.nodes[:, 1:]), 1:], mesh3.nodes[:, 1:], mesh_step=mesh_step)

        maxORFC = np.pi * (get_DIA() / 2) * (get_DIA() / 2)
        opened_area = maxORFC - (outGap1 + outGap2 + outGap3)
        LMN_op = opened_area / maxORFC
        disp1 = np.loadtxt(foldPath + '-1/' + 'U_' + str(closed_step) + '.txt')
        disp2 = np.loadtxt(foldPath + '-2/' + 'U_' + str(closed_step) + '.txt')
        disp3 = np.loadtxt(foldPath + '-3/' + 'U_' + str(closed_step) + '.txt')
        closed_area = find_lumen_closed(disp1=disp1[:, 1:], disp2=disp2[:, 1:], disp3=disp3[:, 1:],
                                        nodes1=mesh1.nodes[:, 1:],
                                        nodes2=mesh2.nodes[:, 1:], nodes3=mesh3.nodes[:, 1:])
        LMN_cl = (maxORFC - closed_area) / maxORFC
        cos120pl, cos120min = np.cos(np.deg2rad(120)), np.cos(np.deg2rad(-120))
        sin120pl, sin120min = np.sin(np.deg2rad(120)), np.sin(np.deg2rad(-120))
        rotMatrixPlus = [[cos120pl, -sin120pl, 0], [sin120pl, cos120pl, 0], [0, 0, 1]]
        rotMatrixMinus = [[cos120min, -sin120min, 0], [sin120min, cos120min, 0], [0, 0, 1]]
        isHelicopter = np.max(
            (
                is_it_twisted(nodes=(mesh1.nodes[:, 1:] + disp1[:len(mesh1.nodes[:, 1:]), 1:])),
                is_it_twisted(nodes=np.dot(mesh2.nodes[:, 1:] + disp2[:len(mesh2.nodes[:, 1:]), 1:], rotMatrixMinus)),
                is_it_twisted(nodes=np.dot(mesh3.nodes[:, 1:] + disp3[:len(mesh3.nodes[:, 1:]), 1:], rotMatrixPlus))
            )
        )
        del outGap1, outGap2, outGap3, mesh1, mesh2, mesh3, disp1, disp2, disp3, closed_step, opened_step
        del cos120pl, cos120min, sin120min, sin120pl, rotMatrixMinus, rotMatrixPlus
        try:
            carea = 0
            for file in glob(endPath + partName[0:-5].upper() + '/carea_*.txt'):
                t_carea = np.loadtxt(file)
                carea += max(t_carea.T[1])
            del t_carea
        except Exception as ex:
            raise 'Uncomplited odb. Frames < 21'
        try:
            with open(pathToAbaqus + '/results/' + partName[0:-5].upper() + '/FramesCount.txt', 'r') as DataFile:
                frames = int(DataFile.read())
        except:
            frames = 0

        return (s_max1_1step, s_max1_2step, le_max1_1step, le_max1_2step, s_max2_1step, s_max2_2step, le_max2_1step,
                le_max2_2step, s_max3_1step, s_max3_2step, le_max3_1step, le_max3_2step, eleCount, LMN_op, LMN_cl,
                isHelicopter, frames, opened_area, closed_area, maxORFC, carea)

    def read_data_single(pathToAbaqus=None, endPath=None, partName=None):
        mesh_step = get_mesh_step()
        foldPath = endPath + partName[0:-5].upper() + '/' + partName.upper()

        mesh1 = ReadMesh(lnodes=np.loadtxt(foldPath + '-1/Nodes.txt'),
                         lelements=np.loadtxt(foldPath + '-1/Elements.txt'))
        stressFiles = glob(foldPath + '-1/' + 'Stress_*.txt')

        newstr = ''.join((ch if ch in '0123456789' else ' ') for ch in stressFiles[0])
        one_step = ([int(i) for i in newstr.split()])[-1]

        newstr = ''.join((ch if ch in '0123456789' else ' ') for ch in stressFiles[1])
        two_step = ([int(i) for i in newstr.split()])[-1]

        closed_step = max(one_step, two_step)
        opened_step = min(one_step, two_step)

        del newstr, one_step, two_step

        s_max1_1step = np.loadtxt(foldPath + '-1/' + 'Stress_' + str(closed_step) + '.txt')
        s_max1_2step = np.loadtxt(foldPath + '-1/' + 'Stress_' + str(opened_step) + '.txt')
        le_max1_1step = np.loadtxt(foldPath + '-1/' + 'LE_' + str(closed_step) + '.txt')
        le_max1_2step = np.loadtxt(foldPath + '-1/' + 'LE_' + str(opened_step) + '.txt')

        eleCount = len(mesh1.getElements())

        disp1 = np.loadtxt(foldPath + '-1/' + 'U_' + str(opened_step) + '.txt')

        try:
            outGap1 = find_lumen_opened(disp1[:, 1:], mesh1.nodes[:, 1:], mesh_step=mesh_step)
        except:
            disp1 = disp1[:len(mesh1.nodes[:, 1:]), :]
            outGap1 = find_lumen_opened(disp1[:, 1:], mesh1.nodes[:, 1:], mesh_step=mesh_step)

        maxORFC = np.pi * (get_DIA() / 2) * (get_DIA() / 2)
        opened_area = maxORFC - (3 * outGap1)
        LMN_op = opened_area / maxORFC

        disp1 = np.loadtxt(foldPath + '-1/' + 'U_' + str(closed_step) + '.txt')
        closed_area = find_lumen_close_single(disp1=disp1[:, 1:], nodes1=mesh1.nodes[:, 1:])
        LMN_cl = (maxORFC - closed_area) / maxORFC
        del outGap1, mesh1, disp1, closed_step, opened_step

        try:
            with open(pathToAbaqus + '/results/' + partName[0:-5].upper() + '/FramesCount.txt', 'r') as DataFile:
                frames = int(DataFile.read())
        except:
            frames = 0

        return (s_max1_1step, s_max1_2step, le_max1_1step, le_max1_2step, eleCount, LMN_op, LMN_cl, frames, opened_area,
                closed_area, maxORFC)

    subfolders = [f.path for f in os.scandir(endPath + partName[:-5].upper()) if f.is_dir()]
    if len(subfolders) > 1:
        (s_max1_1step, s_max1_2step, le_max1_1step, le_max1_2step, s_max2_1step, s_max2_2step, le_max2_1step,
         le_max2_2step, s_max3_1step, s_max3_2step, le_max3_1step, le_max3_2step, eleCount, LMN_op, LMN_cl,
         isHelicopter, frames, opened_area, closed_area, maxORFC, carea) = read_data_contact(
            pathToAbaqus=pathToAbaqus, endPath=endPath, partName=partName
        )
        VMS = (max(s_max1_1step[-1], s_max1_2step[-1]) + max(s_max2_1step[-1], s_max2_2step[-1])
               + max(s_max3_1step[-1], s_max3_2step[-1])) / 3
        S11 = (max(s_max1_1step[0], s_max1_2step[0]) + max(s_max2_1step[0], s_max2_2step[0])
               + max(s_max3_1step[0], s_max3_2step[0])) / 3
        S22 = (max(s_max1_1step[1], s_max1_2step[1]) + max(s_max2_1step[1], s_max2_2step[1])
               + max(s_max3_1step[1], s_max3_2step[1])) / 3
        S33 = (max(s_max1_1step[2], s_max1_2step[2]) + max(s_max2_1step[2], s_max2_2step[2])
               + max(s_max3_1step[2], s_max3_2step[2])) / 3
        Smax = (max(s_max1_1step[3], s_max1_2step[3]) + max(s_max2_1step[3], s_max2_2step[3])
                + max(s_max3_1step[3], s_max3_2step[3])) / 3
        Smid = (max(s_max1_1step[4], s_max1_2step[4]) + max(s_max2_1step[4], s_max2_2step[4])
                + max(s_max3_1step[4], s_max3_2step[4])) / 3
        Smin = (max(s_max1_1step[5], s_max1_2step[5]) + max(s_max2_1step[5], s_max2_2step[5])
                + max(s_max3_1step[5], s_max3_2step[5])) / 3

        LE11 = (max(le_max1_1step[0], le_max1_2step[0]) + max(le_max2_1step[0], le_max2_2step[0])
                + max(le_max3_1step[0], le_max3_2step[0])) / 3
        LE22 = (max(le_max1_1step[1], le_max1_2step[1]) + max(le_max2_1step[1], le_max2_2step[1])
                + max(le_max3_1step[1], le_max3_2step[1])) / 3
        LE33 = (max(le_max1_1step[2], le_max1_2step[2]) + max(le_max2_1step[2], le_max2_2step[2])
                + max(le_max3_1step[2], le_max3_2step[2])) / 3
        LEmax = (max(le_max1_1step[3], le_max1_2step[3]) + max(le_max2_1step[3], le_max2_2step[3])
                 + max(le_max3_1step[3], le_max3_2step[3])) / 3
        LEmid = (max(le_max1_1step[4], le_max1_2step[4]) + max(le_max2_1step[4], le_max2_2step[4])
                 + max(le_max3_1step[4], le_max3_2step[4])) / 3
        LEmin = (max(le_max1_1step[5], le_max1_2step[5]) + max(le_max2_1step[5], le_max2_2step[5])
                 + max(le_max3_1step[5], le_max3_2step[5])) / 3
    else:
        (s_max1_1step, s_max1_2step, le_max1_1step, le_max1_2step, eleCount, LMN_op, LMN_cl, frames, opened_area,
         closed_area, maxORFC) = read_data_single(
            pathToAbaqus=pathToAbaqus, endPath=endPath, partName=partName
        )
        VMS = max(s_max1_1step[-1], s_max1_2step[-1])
        S11 = max(s_max1_1step[0], s_max1_2step[0])
        S22 = max(s_max1_1step[1], s_max1_2step[1])
        S33 = max(s_max1_1step[2], s_max1_2step[2])
        Smax = max(s_max1_1step[3], s_max1_2step[3])
        Smid = max(s_max1_1step[4], s_max1_2step[4])
        Smin = max(s_max1_1step[5], s_max1_2step[5])

        LE11 = max(le_max1_1step[0], le_max1_2step[0])
        LE22 = max(le_max1_1step[1], le_max1_2step[1])
        LE33 = max(le_max1_1step[2], le_max1_2step[2])
        LEmax = max(le_max1_1step[3], le_max1_2step[3])
        LEmid = max(le_max1_1step[4], le_max1_2step[4])
        LEmin = max(le_max1_1step[5], le_max1_2step[5])

        isHelicopter = -1

    if Smax < get_s_lim():
        perf_index = np.round(np.sqrt(
            np.power((opened_area / maxORFC), 2)
            + np.power(1 - ((maxORFC - closed_area) / maxORFC), 2)
            + np.power((1 - (Smax / get_s_lim())), 2)
        ), 4)
    else:
        perf_index = 0

    return LMN_op, LMN_cl, Smax, VMS, perf_index, isHelicopter


def is_it_twisted(nodes: np.ndarray = None) -> float:
    """
    Evaluate leaflet edge deflection outside the intended sector range ("helicopter effect").

    This function estimates the maximum lateral deviation of nodal coordinates from the
    designated sector of a tri-leaflet structure (typically ±60° from vertical centerline, i.e., 90° ± SEC/2).
    The deviation is computed as the maximum perpendicular distance from outlying points
    to a reference ragdial vector, indicating whether leaflet deformation violates sector constraints.

    Args:
        nodes (np.ndarray): Nodal coordinate array of shape (N, 3) representing leaflet surface.

    Returns:
        float: Maximum signed perpendicular distance (in-plane) from leaflet nodes to the sector boundaries.
               This value quantifies the degree of out-of-sector leaflet deviation (a "helicoptering" indicator).

    Notes:
        - The leaflet domain is defined in cylindrical polar coordinates.
        - Nodes with angular positions outside [90 - SEC/2, 90 + SEC/2] degrees are assessed for deviation.
        - The algorithm calculates the perpendicular (normal) distance to sector edge vectors projected in the XY-plane.

    Example:
        >>> max_deviation = is_it_twisted(nodes=leaflet_nodes)

    Applications:
        - Quality control and validation of leaflet model symmetry.
        - Detection of pathological or unintended leaflet flaring.
        - Design loop termination criteria or geometry constraint enforcement.

    Limitations:
        - Assumes a flat base at z = 0 for projection and distance calculation.
        - Only considers 2D projection; out-of-plane (z-axis) deformation is ignored.
        - The variable `SEC` is unused inside the function and can be removed or validated for consistency.

    Suggested Improvements:
        - Vectorize the distance computation to improve performance.
        - Include angle masking using `SEC` parameter explicitly instead of hardcoded ±60°.
        - Generalize to arbitrary sector positions and orientations.

    See Also:
        - `cart2pol`, `pol2cart`: Coordinate transformations for sector boundary calculation.
        - `computeClosed`, `generateShell`: Functions that rely on spatial leaflet layout.
    """
    # 1. Переводим узлы в полярные координаты (угол theta в радианах, радиус rho)
    theta, rho, _ = cart2pol(x=nodes[:, 0], y=nodes[:, 1], lz=nodes[:, 2])
    SEC = get_SEC()
    # 2. Пороговые углы (в радианах)
    half_width = np.deg2rad(SEC / 2)
    alpha1 = np.deg2rad(90) - half_width
    alpha2 = np.deg2rad(90) + half_width

    # 3. Максимальный радиус для построения опорных точек на краях секторов
    rho_max = np.max(rho)

    # Опорная точка p1 — начало координат
    p1 = np.zeros(2, dtype=float)

    def max_normal_distance(mask: np.ndarray = None, angle: float = None) -> float:
        # Строим конечную точку p2 на линии под углом 'angle'
        p2 = pol2cart(theta=angle, rho=rho_max, lz=0)[:2]

        # Вектор вдоль линии p1→p2
        v = p2 - p1

        # Нормализованный нормальный вектор (поворот v на +90°)
        # нормаль = [-v_y, v_x] / ||v||
        normal = np.array([-v[1], v[0]]) / np.linalg.norm(v)

        # Выбираем XY-координаты точек из заданного маски
        pts = nodes[mask, :2]  # shape (M, 2)
        if pts.size == 0:
            return 0.0

        # Перпендикулярные проекции: d = |(p3 - p1) · normal|
        dists = np.abs((pts - p1) @ normal)

        # Возвращаем максимальную проекцию
        return np.max(dists)

    # 4. Маски для «левого» и «правого» боковых секторов
    mask_minus = theta < alpha1
    mask_plus = theta > alpha2

    # 5. Считаем максимальные нормальные расстояния в каждом сек­торе
    dm = max_normal_distance(mask_minus, alpha1)
    dp = max_normal_distance(mask_plus, alpha2)

    # 6. Итоговое максимальное отклонение
    return max(dm, dp)


def calculate_square(
        n1: np.ndarray = None,
        n2: np.ndarray = None,
        n3: np.ndarray = None
) -> float:
    """
    Compute the surface area of a triangle in 3D space using the vector cross product method.

    This function is a low-level replacement for MATLAB's built-in `alphaShape` object area computation.
    It calculates the area of a triangle defined by three 3D vertices using the magnitude of the
    cross product of two edges.

    Args:
        n1 (array-like): Coordinates [x, y, z] of the first triangle vertex.
        n2 (array-like): Coordinates [x, y, z] of the second triangle vertex.
        n3 (array-like): Coordinates [x, y, z] of the third triangle vertex.

    Returns:
        float: Surface area of the triangle defined by (n1, n2, n3).

    Notes:
        - The formula is based on the magnitude of the vector cross product between two edges of the triangle.
        - This function assumes that the input points are in Cartesian coordinates.
        - All inputs should be of length 3 and should represent valid, non-collinear 3D points.

    Example:
        >>> calculate_square([0, 0, 0], [1, 0, 0], [0, 1, 0])
        0.5
    """
    x12 = n1[0] - n2[0]
    y12 = n1[1] - n2[1]
    z12 = n1[2] - n2[2]
    x13 = n1[0] - n3[0]
    y13 = n1[1] - n3[1]
    z13 = n1[2] - n3[2]

    return 0.5 * np.sqrt((y12 * z13 - z12 * y13) ** 2 + (z12 * x13 * z13) ** 2 + (x12 * y13 - y12 * x13) ** 2)


def find_lumen_opened(
        disp: np.ndarray = None,
        nodes: np.ndarray = None,
        mesh_step: float = 0.35
) -> float:
    """
    Compute the projected open area of a deformed leaflet surface by triangulating its cylindrical projection.

    This function transforms deformed node positions to cylindrical coordinates, flattens the axial component (z),
    applies a 2D Delaunay triangulation, and filters triangles based on edge lengths to avoid poor-quality mesh regions.
    The valid triangles are used to estimate the cross-sectional area that remains open after deformation.

    Args:
        disp (np.ndarray): Displacement field of the leaflet nodes, shape (N, 3), corresponding to the deformed state.
        nodes (np.ndarray): Initial coordinates of the undeformed leaflet nodes, shape (N, 3).
        mesh_step (float): Mesh resolution step. Used to filter out triangles with overly long edges.

    Returns:
        float: Estimated open cross-sectional area after deformation (in projected 2D cylindrical coordinates).

    Notes:
        - The function projects the leaflet surface to the XY plane (z=0) in cylindrical coordinates before triangulation.
        - Triangles with max edge length > `2 * mesh_step` are excluded to mitigate skewed geometry effects.
        - Triangulation is performed on the flattened raget_DIA()l projection, not true surface curvature.
        - This approach mimics the concept of projected orifice area used in valve performance metrics.

    Example:
        >>> area = find_lumen_opened(disp=U, nodes=X, mesh_step=0.25)
        >>> print(f"Projected open area: {area:.3f}")
    """

    def diff(p1, p2):
        return ((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2 + (p2[2] - p1[2]) ** 2) ** 0.5

    deformed = nodes + disp
    theta, rho, z = cart2pol(x=deformed[:, 0], y=deformed[:, 1], lz=deformed[:, 2])
    theta_undeformed, rho_undeformed, z_undeformed = cart2pol(x=nodes[:, 0], y=nodes[:, 1], lz=nodes[:, 2])
    rho[np.argwhere(rho > max(rho_undeformed))] = max(rho_undeformed)
    readed_points1 = np.array(pol2cart(theta=theta, rho=rho, lz=z), dtype='float64').T
    readed_points1[:, -1] = 0

    tri = spatial.Delaunay(readed_points1[:, :-1])
    def_square = 0
    for t in tri.simplices:
        if max(np.diff(
                [
                    diff(readed_points1[t[0], :], readed_points1[t[1], :]),
                    diff(readed_points1[t[0], :], readed_points1[t[2], :]),
                    diff(readed_points1[t[1], :], readed_points1[t[2], :])
                ]
        )) < 2 * mesh_step:
            def_square += calculate_square(readed_points1[t[0], :], readed_points1[t[1], :], readed_points1[t[2], :])

    return def_square


def find_lumen_closed(
        disp1: np.ndarray = None,
        disp2: np.ndarray = None,
        disp3: np.ndarray = None,
        nodes1: np.ndarray = None,
        nodes2: np.ndarray = None,
        nodes3: np.ndarray = None
) -> float:
    """
    Compute the total closed (coaptation) area of a tri-leaflet valve configuration from deformed node positions.

    This function estimates the combined closed area of the valve by:
    1. Applying displacement fields to nodal positions of each leaflet.
    2. Projecting leaflets into cylindrical coordinates and bounding their angular spread.
    3. Applying a 2D alpha shape algorithm to compute the projected area for each leaflet.
    4. Rotating side leaflets by ±120 degrees to their anatomical positions before area aggregation.

    Args:
        disp1, disp2, disp3 (np.ndarray): Displacement arrays (shape: N x 3) for the three valve leaflets.
        nodes1, nodes2, nodes3 (np.ndarray): Original node positions (shape: N x 3) for the three leaflets.

    Returns:
        float: Total coaptation area (in mm² or same units as input) computed as the sum of alpha-shape areas of all three deformed leaflets.

    Notes:
        - Alpha shape is used to approximate the 2D footprint (closure zone) of each leaflet.
        - Side leaflets are rotated ±120° about the Z-axis to match anatomical configuration before projection.
        - Angular clipping is applied to constrain leaflets within the functional sector (e.g., 120° range).
        - This area represents the **projected geometric coaptation** and may overestimate true contact in presence of gaps.

    Example:
        >>> area = find_lumen_closed(disp1, disp2, disp3, nodes1, nodes2, nodes3, mesh_step=0.25)
        >>> print(f"Closed leaflet area: {area:.2f} mm²")
    """

    def fit_leaf(deformed):
        theta, rho, z = cart2pol(x=deformed[:, 0], y=deformed[:, 1], lz=deformed[:, 2])
        thera_deg = np.rad2deg(theta)
        thera_deg[np.argwhere(thera_deg < (90 - 120 / 2))] = (90 - 120 / 2)
        thera_deg[np.argwhere(thera_deg > (90 + 120 / 2))] = (90 + 120 / 2)
        deformed_new = deformed
        deformed_new[:, 0], deformed_new[:, 1], deformed_new[:, 2] = pol2cart(theta=np.deg2rad(thera_deg), rho=rho,
                                                                              lz=z)

        # import matplotlib.pyplot as plt
        # # ashp = alphashape(deformed[:, :-1], alpha=1)
        # # x, y = ashp.boundary.xy
        # # plt.plot(x, y, color='#6699cc', alpha=0.7, linewidth=3, solid_capstyle='round', zorder=2)
        # plt.plot(deformed_new[:, 0], deformed_new[:, 1], 'x')
        # plt.show()

        return deformed_new

    cos120pl, cos120min = np.cos(np.deg2rad(120)), np.cos(np.deg2rad(-120))
    sin120pl, sin120min = np.sin(np.deg2rad(120)), np.sin(np.deg2rad(-120))
    rot_matrix_plus = [[cos120pl, -sin120pl, 0], [sin120pl, cos120pl, 0], [0, 0, 1]]
    rot_matrix_minus = [[cos120min, -sin120min, 0], [sin120min, cos120min, 0], [0, 0, 1]]

    deformed_1 = nodes1 + disp1
    alpha_shape_1 = alphashape(deformed_1[:, :-1], alpha=1)
    deformed_2 = fit_leaf(np.dot(nodes2 + disp2, rot_matrix_minus))
    alpha_shape_2 = alphashape(deformed_2[:, :-1], alpha=1)
    deformed_3 = fit_leaf(np.dot(nodes3 + disp3, rot_matrix_plus))
    alpha_shape_3 = alphashape(deformed_3[:, :-1], alpha=1)

    return alpha_shape_1.area + alpha_shape_2.area + alpha_shape_3.area


def find_lumen_close_single(
        disp1: np.ndarray = None,
        nodes1: np.ndarray = None
) -> float:
    """
    Estimate the total coaptation (closed) area for a single-leaflet valve model by projecting the deformed shape.

    This function:
    1. Applies the displacement field `disp1` to the original leaflet geometry `nodes1`.
    2. Projects the deformed shape into 2D space (X, Y) and constructs an alpha shape boundary.
    3. Computes the area enclosed by this alpha shape and scales it by a factor of 3, assuming
       symmetry (e.g., three identical leaflets forming a complete tri-leaflet structure).

    Args:
        disp1 (np.ndarray): Displacement array (shape: N x 3) for the single leaflet.
        nodes1 (np.ndarray): Original node coordinates (shape: N x 3) of the leaflet mesh.

    Returns:
        float: Estimated coaptation area for a full tri-leaflet geometry, derived from the single leaflet geometry.

    Notes:
        - Uses `alphashape` with `alpha=1` to estimate the boundary of the projected deformed leaflet.
        - Assumes that the leaflet geometry is symmetric and evenly spaced (e.g., aortic valve).
        - Designed for simplified simulations with a single representative leaflet.
        - Does not perform angular correction or alignment as in the three-leaflet version.

    Example:
        >>> A_closed = find_lumen_close_single(disp1, nodes1, mesh_step=0.25)
        >>> print(f"Estimated total closed area (tri-leaflet): {A_closed:.2f} mm²")
    """

    deformed = nodes1 + disp1
    alpha_shape = alphashape(deformed[:, :-1], alpha=1)

    return 3 * alpha_shape.area
