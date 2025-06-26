import os.path
import sys
import math
from typing import List, Tuple, Optional, Literal, Union
import numpy as np
from numpy import ndarray
from scipy import interpolate
from utils.global_variable import get_global_path
import warnings

warnings.simplefilter("ignore", RuntimeWarning)

##=====================================================================================================================
## Utils
##=====================================================================================================================

# interpolation function.
def perform_interpolation(
    npoints: Optional[int] = 50,
    x_sample: Union[List[float], np.array] = None,
    y_sample: Union[List[float], np.array] = None,
    z_sample: Union[List[float], np.array] = None,
    key: Literal['linear', 'spline', 'pchip', 'rbf'] = 'linear'
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Interpolate a 3D curve defined by sample points using specified method,
    and return uniformly spaced points along the curve.

    Args:
        npoints (Oprional[int]): Number of interpolated points to generate along the curve.
        x_sample (Union[List[float], np.array]): X-coordinates of the original sample points.
        y_sample (Union[List[float], np.array]): Y-coordinates of the original sample points.
        z_sample (Union[List[float], np.array]): Z-coordinates of the original sample points.
        key (Literal['linear', 'spline', 'pchip', 'rbf']): Interpolation method.
            - 'linear': Piecewise linear interpolation.
            - 'spline': Cubic spline interpolation.
            - 'pchip': Shape-preserving piecewise cubic interpolation.
            - 'rbf': Radial basis function interpolation.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: Interpolated coordinates (xn, yn, zn),
        uniformly spaced along the curve.
    """
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
        (xn, yn, zn) = np.array(perform_interpolation(npoints, x, y, z, key='linear'), dtype='float64')
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
        (xn, yn, zn) = np.array(perform_interpolation(npoints, x, y, z, key='linear'), dtype='float64')
    else:
        raise Exception('interparc no such method: ' + str(key))
    return xn, yn, zn

def cart2pol(
    x: Union[List[float], np.ndarray],
    y: Union[List[float], np.ndarray],
    lz: Union[List[float], np.ndarray]
) -> Tuple[np.ndarray, np.ndarray, Union[float, np.ndarray]]:
    """
    Convert Cartesian coordinates (x, y, lz) to cylindrical polar coordinates (theta, rho, z).

    Args:
        x (Union[float, np.ndarray]): X-coordinate(s) in Cartesian space.
        y (Union[float, np.ndarray]): Y-coordinate(s) in Cartesian space.
        lz (Union[float, np.ndarray]): Z-coordinate(s), passed through unchanged.

    Returns:
        Tuple[np.ndarray, np.ndarray, Union[float, np.ndarray]]:
            - theta: Angular coordinate(s) in radians.
            - rho: Radial distance(s) from origin in xy-plane.
            - z: Height coordinate(s), same as input `lz`.
    """
    rho = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    z = lz
    return theta, rho, z


def pol2cart(
    theta: Union[List[float], np.ndarray],
    rho: Union[List[float], np.ndarray],
    lz: Union[List[float], np.ndarray]
) -> Tuple[np.ndarray, np.ndarray, Union[float, np.ndarray]]:
    """
    Convert cylindrical polar coordinates (theta, rho, lz) to Cartesian coordinates (x, y, z).

    Args:
        theta (Union[float, np.ndarray]): Angular coordinate(s) in radians.
        rho (Union[float, np.ndarray]): Radial distance(s) from the origin in the xy-plane.
        lz (Union[float, np.ndarray]): Z-coordinate(s), passed through unchanged.

    Returns:
        Tuple[np.ndarray, np.ndarray, Union[float, np.ndarray]]:
            - x: X-coordinate(s) in Cartesian space.
            - y: Y-coordinate(s) in Cartesian space.
            - z: Z-coordinate(s), same as input `lz`.
    """
    x = rho * np.cos(theta)
    y = rho * np.sin(theta)
    z = lz
    return x, y, z


def make_top_side(
    HGT: float,
    Hk: float,
    DIA: float,
    ANG: float,
    LAS: float,
    SA: float
) -> np.ndarray:
    """
    Generate a symmetrical free-edge geometry defined in Cartesian coordinates, based on input parameters
    describing height, diameter, angle, and spacing.

    This function computes 11 control points for a free-edge shape in 3D space, applying geometric
    transformations and symmetry. It is applicable in DevOps and infrastructure-as-code contexts involving:
    - 3D deployment layouts (e.g., robotics, sensor arrays)
    - simulation preprocessing pipelines
    - cloud-native geometric configurators for structural systems.

    Args:
        HGT (float): Base height of the geometry (e.g., commissure level).
        Hk (float): Additional vertical offset (e.g., tissue or shell thickness).
        DIA (float): Diameter at the base.
        ANG (float): Tilt or inclination angle (in degrees).
        LAS (float): Arc length parameter influencing edge curvature.
        SA  (float): Sector angle of the model (in degrees), e.g., leaflet sector.

    Returns:
        np.ndarray: An (11, 3) array of points defining the edge shape in 3D Cartesian coordinates.
    """
    maxPhi = np.deg2rad(90-SA/2)
    s105s15 = np.sin(np.deg2rad(105)) / np.sin(np.deg2rad(15))
    san = np.sin(np.deg2rad(-ANG))
    H_loc = HGT + Hk

    lDia = DIA

    upside = np.array([[float for x in range(3)] for y in range(11)])

    ref120 = pol2cart(np.deg2rad(90 - SA / 2), lDia, lDia * san)

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

    upside[1:5, :] += shift

    upside[[6, 7, 8, 9, 10], :] = upside[[4, 3, 2, 1, 0], :]
    upside[[6, 7, 8, 9, 10], 0] *= -1

    upside[:, 2] += H_loc

    if upside[0, 2] != HGT:
        upside[:, 2] += HGT - upside[0, 2]


    return upside

def shift_by_normal(
    pt: Union[np.ndarray, List[List[float]]],
    pn: Union[List[float], np.ndarray],
    THK: float
) -> np.ndarray:
    """
    Apply a normal shift to a set of 3D points based on a reference point (normal base) and a thickness value.

    This function computes the direction vectors from a fixed reference point `pn` to each point in `pt`,
    normalizes those vectors, and then shifts each point along its local normal by a distance `THK`.

    Args:
        pt (Union[np.ndarray, List[List[float]]]): An array of shape (3, N) representing N points in 3D space.
        pn (Union[List[float], np.ndarray]): A single 3D point (length-3) used as the reference for the normal.
        THK (float): The thickness or distance to shift each point along its normal vector.

    Returns:
        np.ndarray: An array of shape (3, N) containing the shifted 3D points.
    """
    pn = np.array([[pn[0]], [pn[1]], [pn[2]]])   # Reshape to column vector
    pt = np.array(pt)                            # Ensure input is an ndarray of shape (3, N)

    # Compute normal vectors (from normal base to each point)
    tn = pt - pn
    lengts_n = np.sqrt(tn[0, :] ** 2 + tn[1, :] ** 2 + tn[2, :] ** 2)

    # Normalize the vectors
    n = np.zeros(tn.shape)
    for i in range(tn.shape[1]):
        n[0, i] = tn[0, i] / lengts_n[i]
        n[1, i] = tn[1, i] / lengts_n[i]
        n[2, i] = tn[2, i] / lengts_n[i]

    # Apply the shift
    points = pt - n * THK
    return points

def circumcenter(
    A: Union[List[float], np.ndarray],
    B: Union[List[float], np.ndarray],
    C: Union[List[float], np.ndarray]
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Compute the circumcenter, radius, and curvature vector of the circumscribed circle
    for a triangle defined by three 3D points A, B, and C.

    This function is adapted from a MATLAB community implementation. It is based on
    vector geometry and returns key properties of the circumcircle used in computational
    mesh processing, surface reconstruction, or robotic planning involving triangular primitives.

    Args:
        A (Union[List[float], np.ndarray]): 3D coordinates of point A.
        B (Union[List[float], np.ndarray]): 3D coordinates of point B.
        C (Union[List[float], np.ndarray]): 3D coordinates of point C.

    Returns:
        Tuple[float, np.ndarray, np.ndarray]:
            - R (float): Radius of the circumscribed circle.
            - M (np.ndarray): 3D coordinates of the circle center (circumcenter).
            - k (np.ndarray): Curvature vector (unit vector pointing from A to center, scaled by 1/R).
    """
    A = np.array(A, dtype='float64')
    B = np.array(B, dtype='float64')
    C = np.array(C, dtype='float64')

    # Normal vector of the triangle's plane
    D = np.cross(B - A, C - A)

    # Lengths of triangle sides opposite points A–C
    b = np.linalg.norm(A - C)
    c = np.linalg.norm(A - B)

    # Vectors orthogonal to triangle edges
    E = np.cross(D, B - A)
    F = np.cross(D, C - A)

    # Vector from A to circumcenter
    G = (b ** 2 * E - c ** 2 * F) / (2 * np.linalg.norm(D) ** 2)

    # Circumcenter position
    M = A + G

    # Radius is length of vector G
    R = np.linalg.norm(G)

    # Curvature vector: direction from A to M, scaled by 1/R^2
    k = G if R == 0 else G / R**2

    return R, M, k


def find_intersections(
    line1: Tuple[Tuple[float, float], Tuple[float, float]],
    line2: Tuple[Tuple[float, float], Tuple[float, float]]
) -> Tuple[float, float]:
    """
    Compute the intersection point of two lines in 2D.

    Each line is defined by two endpoints. This function uses determinant-based
    linear algebra to solve for the intersection. It raises an exception if the lines are parallel.

    Args:
        line1 (Tuple[Tuple[float, float], Tuple[float, float]]): Two points defining the first line.
        line2 (Tuple[Tuple[float, float], Tuple[float, float]]): Two points defining the second line.

    Returns:
        Tuple[float, float]: The (x, y) coordinates of the intersection point.

    Raises:
        Exception: If the lines are parallel and do not intersect (i.e., determinant is zero).
    """
    xdiff = (
        line1[0][0] - line1[1][0],
        line2[0][0] - line2[1][0]
    )
    ydiff = (
        line1[0][1] - line1[1][1],
        line2[0][1] - line2[1][1]
    )

    def det(a: Tuple[float, float], b: Tuple[float, float]) -> float:
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
        raise Exception("Lines do not intersect — they are parallel or coincident.")

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div

    return x, y


def curvature(
    X: Union[np.ndarray, list]
) -> np.ndarray:
    """
    Estimate the local radius of curvature at each point in a 2D or 3D curve
    using the circumcircle method for discrete points.

    Args:
        X (Union[np.ndarray, list]): An (N, 2) or (N, 3) array representing a sequence
            of 2D or 3D coordinates along a curve.

    Returns:
        np.ndarray: A (N,) array of curvature radii, where R[i] is the radius
            of curvature at point i. The first and last entries are zero (no curvature estimated).
    """
    X = np.array(X, dtype='float64')
    N, dims = X.shape

    # Promote 2D input to 3D by padding z = 0
    if dims == 2:
        X = np.hstack([X, np.zeros((N, 1))])  # shape becomes (N, 3)

    R = np.zeros(N, dtype='float64')  # Radius of curvature
    for i in range(1, N - 1):
        try:
            # Extract three points and compute circumradius
            A, B, C = X[i], X[i - 1], X[i + 1]
            Ri, _, _ = circumcenter(A, B, C)
            R[i] = Ri
        except Exception:
            # Handle degenerate triangle (collinear points) by assigning NaN or zero
            R[i] = np.nan

    return R

def generateShell(
    points: Union[np.ndarray, List[List[float]]],
    elements: Union[np.ndarray, List[List[int]]],
    pointsInner: Union[np.ndarray, List[List[float]]],
    pointsHullLower: Union[np.ndarray, List[List[float]]],
    meshStep: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract and clean a surface shell from a volumetric mesh, identifying valid surface elements
    and fixing boundary condition (BC) nodes based on geometric properties.

    This function performs:
    - Matching of inner shell nodes to volumetric nodes
    - Extraction of connected surface elements
    - Deduplication and filtering of degenerate or oversized triangles
    - Renumbering of nodes and elements
    - Selection of nodes for boundary conditions based on a lower hull geometry

    Args:
        points (np.ndarray): (N, 3) array of node coordinates.
        elements (np.ndarray): (M, 3) array of triangular element node indices.
        pointsInner (np.ndarray): Array of internal surface points.
        pointsHullLower (np.ndarray): Array of points forming the lower hull (e.g., boundary base).
        meshStep (float): Reference mesh resolution, used for filtering elements.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]:
            - shellNode: Filtered and sorted array of surface node coordinates.
            - shellEle: Array of filtered and renumbered triangular elements.
            - BCfix: Array of node indices suitable for fixed boundary condition enforcement.
    """

    # Normalize precision for robust matching
    nodes = np.array(points * 1e10, dtype='int64') / 1e10
    elements = np.array(elements, dtype='int32')
    pointsInner = np.array(pointsInner * 1e10, dtype='int64') / 1e10
    pointsHullLower = np.array(pointsHullLower * 1e10, dtype='int64') / 1e10

    pointsInner = pointsInner.T
    pointsHullLower = pointsHullLower.T

    writed = 0

    # Extract shell elements that touch internal surface points
    for i in range(pointsInner.shape[0]):
        row = np.where((np.isclose(nodes, pointsInner[i])).all(axis=1))
        if i == 0:
            ind = row[0]
            shellPoints = pointsInner[i]
        else:
            ind = np.insert(ind, len(ind), row[0])
            shellPoints = np.vstack((shellPoints, pointsInner[i]))

        row1 = np.where(elements[:, 0] == ind[i])[0]
        row2 = np.where(elements[:, 1] == ind[i])[0]
        row3 = np.where(elements[:, 2] == ind[i])[0]

        if not writed:
            tempEle = elements[row1]
            tempEle = np.vstack((tempEle, elements[row2], elements[row3]))
            writed = 1
        else:
            tempEle = np.vstack((tempEle, elements[row1], elements[row2], elements[row3]))

    # Remove duplicate elements
    _, idx = np.unique(tempEle, axis=0, return_index=True)
    tempEle = tempEle[idx, :]

    # Annotate shell nodes with indices
    shellPoints = np.hstack((np.transpose([ind]), shellPoints))

    # Remove elements that contain nodes outside the shell point list
    tempEle2 = tempEle.copy()
    rem = []
    for i in range(tempEle.shape[0]):
        for j in range(3):
            if tempEle[i, j] not in shellPoints[:, 0]:
                rem.append(i)
                break

    tempEle2 = np.delete(tempEle2, rem, axis=0)

    # Remove bad elements (low area or too large edges)
    indToRem = []
    for i in range(tempEle2.shape[0]):
        n1 = shellPoints[shellPoints[:, 0] == tempEle2[i, 0]][0]
        n2 = shellPoints[shellPoints[:, 0] == tempEle2[i, 1]][0]
        n3 = shellPoints[shellPoints[:, 0] == tempEle2[i, 2]][0]

        dist12 = np.linalg.norm(n1[1:] - n2[1:])
        dist23 = np.linalg.norm(n2[1:] - n3[1:])
        dist31 = np.linalg.norm(n3[1:] - n1[1:])

        if any(d > 2.5 * meshStep for d in [dist12, dist23, dist31]):
            indToRem.append(i)
            continue

        # Triangle area via cross product
        v1 = n2[1:] - n1[1:]
        v2 = n3[1:] - n1[1:]
        area = 0.5 * np.linalg.norm(np.cross(v1, v2))

        if area < meshStep * 1e-6:
            indToRem.append(i)

    tempEle = np.delete(tempEle2, indToRem, axis=0)

    # Sort and renumber nodes
    shellPoints = shellPoints[np.argsort(shellPoints[:, 0])]
    shellNode = shellPoints[:, 1:]
    shellEle = tempEle

    for i, original_idx in enumerate(shellPoints[:, 0]):
        shellEle[shellEle == original_idx] = i
        shellPoints[i, 0] = i

    # Identify nodes for boundary conditions (from lower hull)
    BCfix = []
    for i in range(pointsHullLower.shape[0]):
        for j, pt in enumerate(shellPoints[:, 1:]):
            if np.allclose(pt, pointsHullLower[i]):
                BCfix.append(shellPoints[j, 0])
                break

    return shellNode, shellEle, np.array(BCfix, dtype='int32')

##=====================================================================================================================
## Main part
##=====================================================================================================================

def generate_leaflet_pointcloud(
    HGT: float = 10,
    Lstr: float = 0,
    SEC: float = 119,
    DIA: float = 13,
    THK: float = 0.5,
    ANG: float = 0,
    Lift: float = 0,
    CVT: float = 0.5,
    LAS: float = 1,
    FCVT: float = 0.5,
    mesh_step: float = 0.35
) -> Tuple[
    np.ndarray,  # points_inflow
    np.ndarray,  # points_outer
    np.ndarray,  # out_points_hull_lower_shifted
    np.ndarray,  # out_points_hull_upper_shifted
    np.ndarray,  # points_hull_lower
    np.ndarray,  # points_hull_upper
    np.ndarray,  # points (all)
    np.ndarray,  # normal_point
    float,       # final_r
    float,       # radius_current
    str          # message
]:
    """
        Construct a parameterised 3D pointcloud of geometry for simulation, based on anatomical or engineering design criteria.

        The function generates a closed surface shell (inner and outer layers) by computing a lower contour
        and a free upper edge, connecting them via interpolated splines. Curvature at the midline is tuned
        iteratively to match a prescribed curvature variation target (CVT). Normal vectors are computed to
        create shell thickness, and surface filling is applied with a mesh resolution specified by `mesh_step`.

        Args:
            HGT (float): Base height of the structure (e.g., commissure plane).
            Lstr (float): Length of the straight portion from the base; 0 for a purely curved profile.
            SEC (float): Sector angle (in degrees) defining angular extent of the shell.
            DIA (float): Diameter at the base, used to compute radius.
            THK (float): Target thickness of the shell, normal to the mid-surface.
            ANG (float): Inclination angle of the shell axis with respect to vertical (in degrees).
            Lift (float): Local vertical elevation for curvature shaping at the free edge.
            CVT (float): Curvature variation target (0 = flat, 1 = maximally convex).
            LAS (float): Arc length parameter that defines the free edge geometry.
            mesh_step (float): Spatial resolution for point spacing on all surfaces.

        Returns:
            Tuple[
                np.ndarray,  # points_inflow: mid-surface inner layer points
                np.ndarray,  # points_outer: mid-surface outer layer points
                np.ndarray,  # out_points_hull_lower_shifted: lower contour shifted along normal
                np.ndarray,  # out_points_hull_upper_shifted: upper free edge shifted along normal
                np.ndarray,  # points_hull_lower: initial lower boundary curve
                np.ndarray,  # points_hull_upper: initial upper boundary curve
                np.ndarray,  # points: aggregated full shell point cloud
                np.ndarray,  # normal_point: 3D reference point used for normal vector projection
                float,       # final_r: computed final radius of control curvature
                float,       # radius_current: achieved curvature after CVT adjustment
                str          # message: status string, typically 'PC is constructed'
            ]

        Raises:
            Exception: If geometric or physical constraints are violated (e.g., negative height, invalid CVT).

        Notes:
            - Spatial coordinates are returned as arrays of shape (3, N), with rows representing x, y, z axes.
            - Output points are deduplicated and precision-trimmed to 14 decimal places.
            - Designed for use in mesh generation, simulation pre-processing, or digital twin geometry modeling.
        """
    radius = DIA / 2
    amplification_coefficient = 100
    top_side_shifted_by = np.tan(np.deg2rad(ANG)) * radius

    top_side_sift = HGT + top_side_shifted_by
    if top_side_sift <= 0:
        message = 'HGT + top_side_shifted_by <= 0'
        raise Exception(message)

    if CVT < 0 or CVT > 1:
        message = "Error in CVT"
        raise Exception(message)

    if Lstr < 0:
        message = "Length of straight part below Zero"
        raise Exception(message)

    def point_at_distance_cyl(
            A_cyl: Union[list[float, float, float],ndarray],
            B_cyl: Union[list[float, float, float],ndarray],
            t: float
    ):
        """
        Точка на расстоянии X от A по направлению к B
        в цилиндрической системе координат.

        Parameters
        ----------
        A_cyl, B_cyl : (r, φ, z) — углы в радианах.
        X            : отрезок, X ≥ 0 (можно > AB для экстраполяции).
        """
        # 1. В декартову систему
        Ax, Ay, Az = pol2cart(*A_cyl)
        Bx, By, Bz = pol2cart(*B_cyl)

        # 2. Длина AB
        D = math.dist((Ax, Ay, Az), (Bx, By, Bz))
        if D == 0:
            raise ValueError("A и B совпадают (нуль-вектор).")

        # 3. Параметр t и точка P
        Px = Ax + t * (Bx - Ax)
        Py = Ay + t * (By - Ay)
        Pz = Az + t * (Bz - Az)

        # 4. Обратно в цилиндрическую систему
        return cart2pol(Px, Py, Pz)

    upper_pos = 0.5*np.sum(([np.deg2rad(100), radius, Lift+0.06],[np.deg2rad(90 + SEC / 2), radius, HGT - Lstr]), axis =0)
    upper_neg = 0.5*np.sum(([np.deg2rad(80), radius, Lift+0.06],[np.deg2rad(90 - SEC / 2), radius, HGT - Lstr]), axis =0)
    lower_pos = [np.deg2rad(90 + SEC / 2), radius, Lift+0.06] # [z, phi]
    lower_neg = [np.deg2rad(90 - SEC / 2), radius, Lift+0.06] # [z, phi]

    point_pos = point_at_distance_cyl(upper_pos, lower_pos, FCVT)
    point_neg = point_at_distance_cyl(upper_neg, lower_neg, FCVT)

    if Lstr > 0:
        theta_ref = [np.deg2rad(90 - SEC / 2), np.deg2rad(90 - SEC / 2), np.deg2rad(90 - SEC / 2),
                    np.deg2rad(90 - SEC / 2), np.deg2rad(90 - SEC / 2),# np.deg2rad(90 - (SEC - SEC / 6) / 2),
                    point_neg[0], np.deg2rad(80), np.deg2rad(90), np.deg2rad(100), point_pos[0],# np.deg2rad(90 + (SEC - SEC / 6) / 2),
                    np.deg2rad(90 + SEC / 2), np.deg2rad(90 + SEC / 2), np.deg2rad(90 + SEC / 2),
                    np.deg2rad(90 + SEC / 2), np.deg2rad(90 + SEC / 2)]
        height_ref = [HGT, HGT - Lstr * 0.25, HGT - Lstr * 0.5, HGT - Lstr * 0.75, HGT - Lstr,
                     #HGT/3,
                      point_neg[2], Lift+0.06, Lift, Lift+0.06,  point_pos[2],
                     # HGT/3,
                     HGT - Lstr, HGT - Lstr * 0.75, HGT - Lstr * 0.5, HGT - Lstr * 0.25, HGT]
        fit_rho = np.linspace(radius, radius, len(theta_ref))
        (temp_x, temp_y, temp_z) = pol2cart(theta_ref, fit_rho, height_ref)
        arc = np.array(perform_interpolation(60, temp_x, temp_y, temp_z, key='pchip'), dtype='float64')
    else:
        theta_ref = np.array([np.deg2rad(90 - SEC / 2), np.deg2rad(80), np.deg2rad(90), np.deg2rad(100), np.deg2rad(90 + SEC / 2)])
        height_ref = np.array([HGT, Lift+0.06, Lift, Lift+0.06, HGT])
        fit_rho = np.linspace(radius, radius, len(theta_ref))
        (temp_x, temp_y, temp_z) = pol2cart(theta_ref, fit_rho, height_ref)
        arc = np.array(perform_interpolation(60, temp_x, temp_y, temp_z, key='pchip'), dtype='float64')

    t_ang, t_rad, t_z = cart2pol(arc[0], arc[1], arc[2])
    for l_iter in np.arange(0, len(t_rad)):
        if t_rad[l_iter] != radius: t_rad[l_iter] = radius
    arc[0], arc[1], arc[2] = pol2cart(t_ang, t_rad, t_z)
    del t_ang, t_rad, t_z

    (temp_x, temp_y, temp_z) = (arc[0], arc[1], arc[2])
    # define points count
    dx = np.abs(np.array(np.diff(temp_x)), dtype='float64')
    dy = np.abs(np.array(np.diff(temp_y)), dtype='float64')
    dz = np.abs(np.array(np.diff(temp_z)), dtype='float64')
    distances_between_vertices = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
    points_in_side = int(np.ceil(np.sum(distances_between_vertices) / mesh_step)) + 3


    basepoints = pol2cart(theta_ref, fit_rho, height_ref)
    del dx, dy, dz, fit_rho, arc, theta_ref, height_ref

    contour_leaf = np.array(perform_interpolation(points_in_side, temp_x, temp_y, temp_z, key='spline'), dtype='float64')
    temp_contour_leaf = np.array(perform_interpolation(int(amplification_coefficient * points_in_side), temp_x, temp_y, temp_z, key='spline'), dtype='float64')
    del temp_x, temp_y, temp_z

    # import matplotlib.pyplot as plt
    # from mpl_toolkits.mplot3d import Axes3D
    #
    # cart_new_point_pos = pol2cart(point_pos[0],point_pos[1],point_pos[2])
    # cart_new_point_neg = pol2cart(point_neg[0],point_neg[1],point_neg[2])
    # cart_upper_pos = pol2cart(upper_pos[0], radius, upper_pos[2])
    # cart_lower_pos = pol2cart(lower_pos[0], radius, lower_pos[2])
    # cart_upper_neg = pol2cart(upper_neg[0], radius, upper_neg[2])
    # cart_lower_neg = pol2cart(lower_neg[0], radius, lower_neg[2])
    #
    # fig = plt.figure(figsize=(10, 10))
    # ax1 = fig.add_subplot(111, projection='3d')
    # # ax1.get_proj = lambda: np.dot(Axes3D.get_proj(ax1), np.diag([3, 3, 3, 1]))
    # ax1.scatter(contour_leaf[0], contour_leaf[1], contour_leaf[2], c='k', s=10)
    # ax1.scatter(cart_upper_pos[0], cart_upper_pos[1], cart_upper_pos[2], c='red', s=20)
    # ax1.scatter(cart_lower_pos[0], cart_lower_pos[1], cart_lower_pos[2], c='c', s=20)
    # ax1.scatter(cart_upper_neg[0], cart_upper_neg[1], cart_upper_neg[2], c='red', s=20)
    # ax1.scatter(cart_lower_neg[0], cart_lower_neg[1], cart_lower_neg[2], c='c', s=20)
    # ax1.scatter(cart_new_point_pos[0], cart_new_point_pos[1], cart_new_point_pos[2], c='purple', s=20)
    # ax1.scatter(cart_new_point_neg[0], cart_new_point_neg[1], cart_new_point_neg[2], c='purple', s=20)
    # ax1.scatter(basepoints[0], basepoints[1], basepoints[2], c='green', s=10)
    # ax1.set_xlim((-radius+2,radius+2))
    # ax1.set_ylim((-radius+2,radius+2))
    # ax1.set_zlim((0,HGT+2))
    # ax1.set_xlabel('X')
    # ax1.set_ylabel('Y')
    # ax1.set_zlabel('Z')
    # ax1.set_title(f'diff: {FCVT:0.2f}')
    # plt.savefig(f'{os.path.join(get_global_path(),"inps/pics/")}pic_diff_{FCVT:0.2f}.jpg')

    # sys.exit()
    points_hull_lower = contour_leaf

    # top free edge
    upside_top = np.array(make_top_side(HGT, top_side_shifted_by, radius, ANG, LAS, SEC), dtype='float64')

    dx = np.diff(upside_top[:, 0])
    dy = np.diff(upside_top[:, 1])
    dz = np.diff(upside_top[:, 2])
    distances_between_vertices = (dx ** 2 + dy ** 2 + dz ** 2) ** 0.5
    points_in_top = int(np.ceil(sum(distances_between_vertices) / mesh_step))
    if LAS < 0.5:
        method = 'linear'
    else:
        method = 'pchip'
    top_spline = np.array(perform_interpolation(points_in_top, upside_top[:, 0], upside_top[:, 1], upside_top[:, 2], key=method),
                         dtype='float64')
    points_hull_upper = top_spline[:, 1:-1]
    points = np.append(points_hull_lower, points_hull_upper, axis=1)
    hull = points
    del dx, dy, dz, distances_between_vertices

    tempTopSpline = np.array(perform_interpolation(amplification_coefficient * points_in_top, upside_top[:, 0], upside_top[:, 1], upside_top[:, 2],
                                                   key=method), dtype='float64')

    # calculate CVT
    # find min curvature radius
    rad = [0 for i in range(3)]
    leaf_spline = np.array([[float for x in range(3)] for y in range(3)])
    leaf_spline[0] = contour_leaf[:, int(np.ceil(contour_leaf.shape[1] / 2))]
    pl = leaf_spline[0]
    pu = top_spline[:, int(np.ceil(top_spline.shape[1] / 2))]
    pc = (pl + pu) / 2
    leaf_spline[1] = pc
    leaf_spline[2] = pu

    r_min = ((pu[0] - pc[0]) ** 2 + (pu[1] - pc[1]) ** 2 + (pu[2] - pc[2]) ** 2) ** 0.5

    p_begin = [pc[0], pc[1], pc[2]]
    iter = 0
    delta = 1e-3
    local_iter = 0

    # find max curvature radius. iterate while not reach r_min - shift point by delta along Y and Z
    while not (r_min * 0.99 < rad[1] < r_min * 1.01):
        iter += 1
        leaf_spline[1] = pc
        rad = curvature(leaf_spline)
        if r_min * 0.99 < rad[1] < r_min * 1.01:
            break
        if np.mod(iter, 500) == 0:
            local_iter += 1
            if np.mod(local_iter, 2) == 0:
                delta /= 2
            else:
                delta *= 3
        if rad[1] < r_min:
            pc[1] += delta
            pc[2] += delta
        else:
            pc[1] -= delta
            pc[2] -= delta

    # find actual curvature radius. same with previous
    r_max = ((pc[0] - p_begin[0]) ** 2 + (pc[1] - p_begin[1]) ** 2 + (pc[2] - p_begin[2]) ** 2) ** 0.5
    pc = [(pl[0] + pu[0]) / 2, (pl[1] + pu[1]) / 2, (pl[2] + pu[2]) / 2] #middle point
    radius_current = CVT * r_max
    iter = 0
    delta = 1e-3
    local_iter = 0
    distance_current = ((pc[0] - p_begin[0]) ** 2 + (pc[1] - p_begin[1]) ** 2 + (pc[2] - p_begin[2]) ** 2) ** 0.5
    while not (radius_current * 0.99 < distance_current < radius_current * 1.01):
        iter += 1
        leaf_spline[1] = pc
        distance_current = ((pc[0] - p_begin[0]) ** 2 + (pc[1] - p_begin[1]) ** 2 + (pc[2] - p_begin[2]) ** 2) ** 0.5
        if radius_current * 0.99 <= distance_current <= radius_current * 1.01:
            break
        if np.mod(iter, 250) == 0:
            local_iter += 1
            if np.mod(local_iter, 4) == 0:
                delta /= 2
            elif np.mod(local_iter, 4) == 2:
                delta *= 3
        if distance_current >= radius_current:
            pc[1] += delta
            pc[2] += delta
        else:
            pc[1] -= delta
            pc[2] -= delta
    leaf_spline[1] = pc
    radius_current = rad[1]
    final_curvature_radius = np.sqrt((0 - leaf_spline[1][0]) ** 2 + (radius - leaf_spline[1][1]) ** 2 + (
            max(top_side_sift, HGT) - leaf_spline[1][2]) ** 2)
    del delta, iter, pc

    # spline through center of belly line and commisures top points
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
    middle_spline = np.array(perform_interpolation(30, tSplin[0], tSplin[1], tSplin[2], key='pchip'), dtype='float64')
    dx = np.abs(np.array(np.diff(middle_spline[0])))
    dy = np.abs(np.array(np.diff(middle_spline[1])))
    dz = np.abs(np.array(np.diff(middle_spline[2])))
    distances_between_vertices = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
    distance = sum(distances_between_vertices)
    points_in_hor_line = int(np.ceil(distance / mesh_step)) + 1
    del dx, dy, dz, distance, distances_between_vertices, middle_spline
    middle_spline = np.array(perform_interpolation(points_in_hor_line, tSplin[0], tSplin[1], tSplin[2], key='pchip'),
                            dtype='float64')
    del tSplin, leaf_spline

    # adds "thickness". shift contour by THK along normal
    pl1 = upside_top[int(np.floor(len(upside_top) / 2)), :]
    pl2 = np.array(pol2cart(np.deg2rad(90), radius * 1.0, Lift), dtype='float64')
    plc = 0.5 * (pl2 + pl1)

    pus1 = np.array(pol2cart(np.deg2rad(90 - SEC / 2), radius * 1.0, HGT), dtype='float64')
    pus2 = np.array(pol2cart(np.deg2rad(90 + SEC / 2), radius * 1.0, HGT), dtype='float64')
    pu1 = 0.5 * (pus2 + pus1)

    pu2 = np.array(pol2cart(np.deg2rad(90), 100.0 * radius, max(HGT, HGT + top_side_shifted_by)), dtype='float64')
    pu2[0] = 0
    puz = np.array(pol2cart(np.deg2rad(90), -100.0 * radius, HGT), dtype='float64')
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

    # normal points
    try:
        normal_point = np.empty(3)
        normal_point[0] = 0
        normal_point[1] = yinter*1
        normal_point[2] = zinter
    except Exception:
        normal_point = pol2cart(np.deg2rad(90), radius * 3, max(HGT, top_side_sift))

    del pl1, pl2, plc, pus1, pus2, pu2, puz, nline, P21, uline, yinter, zinter, ind

    leaflet_shifted = points
    hull_shifted = shift_by_normal(hull, normal_point, THK)
    leaflet_shifted = np.append(leaflet_shifted, hull_shifted, axis=1)

    # заполнение поверхности. если толщина больше шага сетки - добавления layers_count слоев точек на гранях
    if THK > mesh_step:
        #     disp('debug');
        layers_count = np.floor(THK / mesh_step)
        for i in range(1, int(layers_count)+1):
            temp_points_hull_lower_shifted = shift_by_normal(points_hull_lower, normal_point, THK * i / (layers_count + 1))
            temp_ppoints_hull_upper_shifted = shift_by_normal(points_hull_upper, normal_point, THK * i / (layers_count + 1))
            if i > 1:
                out_points_hull_lower_shifted = np.append(out_points_hull_lower_shifted, temp_points_hull_lower_shifted, axis=1)
                out_points_hull_upper_shifted = np.append(out_points_hull_upper_shifted, temp_ppoints_hull_upper_shifted, axis=1)
            else:
                out_points_hull_lower_shifted = temp_points_hull_lower_shifted
                out_points_hull_upper_shifted = temp_ppoints_hull_upper_shifted
        del layers_count, temp_points_hull_lower_shifted, temp_ppoints_hull_upper_shifted
    else:
        out_points_hull_lower_shifted = shift_by_normal(points_hull_lower, normal_point, THK)
        out_points_hull_upper_shifted = shift_by_normal(points_hull_upper, normal_point, THK)

    # filling the leaflet with points.
    for iterThickness in range(1, 3):
        if iterThickness == 1:
            temp_spline_mid = middle_spline
            t_cl = temp_contour_leaf
            temporal_points = points
            t_ts = tempTopSpline
        else:
            temp_spline_mid = shift_by_normal(middle_spline, normal_point, THK)
            t_cl = shift_by_normal(temp_contour_leaf, normal_point, THK)
            temporal_points = points
            t_ts = shift_by_normal(tempTopSpline, normal_point, THK)
        max_iter = temp_spline_mid.shape[1]
        for i in range(1, max_iter - 1):
            iter_x = temp_spline_mid[0, i]
            ind_low = np.argmin(np.abs(iter_x - np.transpose(t_cl[0, :])))
            ind_top = np.argmin(np.abs(iter_x - np.transpose(t_ts[0, :])))
            t_spline_bot = np.array([t_cl[:, ind_low], temp_spline_mid[:, i], t_ts[:, ind_top]])
            t_spline2 = np.array(perform_interpolation(20, t_spline_bot.T[0], t_spline_bot.T[1], t_spline_bot.T[2], key='pchip'), dtype='float64')
            dx = np.array(np.diff(np.transpose(t_spline2[0, :])), dtype='float64')
            dy = np.array(np.diff(np.transpose(t_spline2[1, :])), dtype='float64')
            dz = np.array(np.diff(np.transpose(t_spline2[2, :])), dtype='float64')
            distances_between_vertices = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
            distance = sum(distances_between_vertices)
            if distance < mesh_step:
                del ind_low, ind_top
                del dx, dy, dz, distances_between_vertices, distance
                continue
            points_in_surface = int(np.floor(distance / mesh_step))
            curr_spline_bot = np.array(perform_interpolation(points_in_surface, t_spline_bot.T[0], t_spline_bot.T[1], t_spline_bot.T[2],
                                                           key='pchip'), dtype='float64')
            del t_spline2, t_spline_bot
            curr_spline = curr_spline_bot[:, range(1, curr_spline_bot.shape[1] - 1)]
            vec_del_ind = np.array(range(0, len(curr_spline[0, :])))
            for absolutely_temporal_iteration_index in range(0, len(vec_del_ind)):
                vec_del_ind[absolutely_temporal_iteration_index] = -1
            for ind_point in range(0,len(curr_spline[0, :])):
                local_dist = np.sqrt((contour_leaf[0, :] - curr_spline[0, ind_point]) ** 2 +
                                (contour_leaf[1, :] - curr_spline[1, ind_point]) ** 2 +
                                (contour_leaf[2, :] - curr_spline[2, ind_point]) ** 2)
                ind_to_del_if_lower_mesh_step = np.argwhere(local_dist < mesh_step*0.5)
                if len(ind_to_del_if_lower_mesh_step) > 0:
                    vec_del_ind[ind_point] = ind_point
                del ind_to_del_if_lower_mesh_step
            try:
                curr_spline = np.delete(curr_spline, vec_del_ind[vec_del_ind != -1], axis=1)
            except:
                curr_spline = curr_spline
            temporal_points = np.append(temporal_points, curr_spline, axis=1)

            del ind_low, vec_del_ind, ind_top, curr_spline, dx, dy, dz, distances_between_vertices, distance

        if iterThickness == 1:
            points_inflow = temporal_points
        else:
            points_outflow = temporal_points
        leaflet_shifted = np.append(leaflet_shifted, temporal_points, axis=1)
    if THK > mesh_step:
        leaflet_shifted = np.append(leaflet_shifted, out_points_hull_lower_shifted, axis=1)
        leaflet_shifted = np.append(leaflet_shifted, out_points_hull_upper_shifted, axis=1)

    points = leaflet_shifted
    del leaflet_shifted
    # remove duplicates
    points = np.unique(points, axis=1)
    message = 'PC is constructed'
    del hull, hull_shifted
    del top_spline, points_in_side, i, contour_leaf
    # round to "decimals"
    decimal = 8
    return (np.unique(np.around(points_inflow, decimal), axis=1),
            np.unique(np.around(points_outflow, decimal), axis=1),
            np.unique(np.around(out_points_hull_lower_shifted, decimal), axis=1),
            np.unique(np.around(out_points_hull_upper_shifted, decimal), axis=1),
            np.unique(np.around(points_hull_lower, decimal), axis=1),
            np.unique(np.around(points_hull_upper, decimal), axis=1),
            np.unique(np.around(points, decimal), axis=1), normal_point, final_curvature_radius, radius_current, message)