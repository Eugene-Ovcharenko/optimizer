import numpy as np
from scipy.spatial import Delaunay
import trimesh

def triangulate_points_pca(points, mesh_step, filter_options=None):
    """
    Выполняет триангуляцию 3D облака точек через проекцию на главную плоскость (PCA).
    
    Заменяет Open3D для построения сетки. Включает адаптивную фильтрацию артефактов.

    Args:
        points (np.ndarray): Массив точек (N, 3).
        mesh_step (float): Базовый шаг сетки для фильтрации длинных ребер.
        filter_options (list, optional): Опции фильтрации элементов:
            - 'edge': удаляет треугольники с ребрами > 2.0 * mesh_step.
            - 'area': удаляет треугольники с площадью < 25% от средней.
            - 'curvature': удаляет элементы с экстремальной гауссовой кривизной.
            По умолчанию ['edge'].

    Returns:
        np.ndarray: Массив индексов треугольников (M, 3).
    """
    if filter_options is None:
        filter_options = ['edge']

    # 1. PCA Projection to 2D
    centroid = np.mean(points, axis=0)
    centered = points - centroid
    cov = np.cov(centered.T)
    val, vec = np.linalg.eigh(cov)
    idx = np.argsort(val)[::-1]
    vec = vec[:, idx]
    projected_2d = centered @ vec[:, :2]
    
    # 2. Delaunay Triangulation in 2D
    tri = Delaunay(projected_2d)
    elements = tri.simplices
    initial_count = len(elements)
    
    # 3. Filtering
    max_edge_len = mesh_step * 5.0
    
    # Calculate areas for area-based filtering
    areas = []
    for face in elements:
        v0, v1, v2 = points[face[0]], points[face[1]], points[face[2]]
        area = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0))
        areas.append(area)
    areas = np.array(areas)
    avg_area = np.mean(areas)
    
    # Calculate curvature for curvature-based filtering
    # We use a simplified local curvature: angle between triangle normal and its vertices' neighbors?
    # Or just use the already implemented compute_gaussian_curvature
    curvatures = None
    if 'curvature' in filter_options:
        curvatures = compute_gaussian_curvature(points, elements)

    filtered_elements = []
    report = {
        'initial_count': initial_count,
        'removed_edge': 0,
        'removed_area': 0,
        'removed_curvature': 0,
        'final_count': 0
    }

    for i, face in enumerate(elements):
        v0, v1, v2 = points[face[0]], points[face[1]], points[face[2]]
        
        # Edge filter
        d01 = np.linalg.norm(v0 - v1)
        d12 = np.linalg.norm(v1 - v2)
        d20 = np.linalg.norm(v2 - v0)
        if 'edge' in filter_options and (d01 > max_edge_len or d12 > max_edge_len or d20 > max_edge_len):
            report['removed_edge'] += 1
            continue
            
        # Area filter (e.g., < 25% of average)
        if 'area' in filter_options and areas[i] < 0.05 * avg_area:
            report['removed_area'] += 1
            continue
            
        # Curvature filter (remove extreme curvature artifacts)
        if 'curvature' in filter_options and curvatures is not None:
            # Element curvature as average of its vertices
            elem_curv = np.mean(np.abs(curvatures[face]))
            if elem_curv >  1.0/(1.5*mesh_step**2) : # Threshold for "extreme" curvature
                report['removed_curvature'] += 1
                continue
                
        filtered_elements.append(face)
    
    report['final_count'] = len(filtered_elements)
    
    # print("\n--- Triangulation Report ---")
    # print(f"Initial elements: {report['initial_count']}")
    # if 'edge' in filter_options:
    #     print(f"Removed by edge length (> {max_edge_len:.2f}): {report['removed_edge']}")
    # if 'area' in filter_options:
    #     print(f"Removed by small area (< {0.05*avg_area:.4f}) | {avg_area:.4f}: {report['removed_area']}")
    # if 'curvature' in filter_options:
    #     print(f"Removed by extreme curvature: {report['removed_curvature']}")
    # print(f"Final elements: {report['final_count']}")
    # print("----------------------------\n")
            
    return np.array(filtered_elements)

def create_volumetric_mesh(pts_in, pts_out, elements):
    """
    Создает закрытый (watertight) объемный 3D-объект из двух поверхностей.

    Автоматически генерирует боковые грани (стенки), соединяя соответствующие 
    узлы внутренней и внешней поверхностей.

    Args:
        pts_in (np.ndarray): Узлы внутренней поверхности (N, 3).
        pts_out (np.ndarray): Узлы внешней поверхности (N, 3). Должны иметь 1-к-1 соответствие с pts_in.
        elements (np.ndarray): Индексы треугольников базовой поверхности (M, 3).

    Returns:
        tuple: (vertices, all_faces)
            - vertices (np.ndarray): Объединенный массив узлов (2*N, 3).
            - all_faces (np.ndarray): Список всех граней (включая стенки) для закрытого объема.
    """
    n_pts = len(pts_in)
    
    # Vertices: Inflow then Outflow
    vertices = np.vstack([pts_in, pts_out])
    
    # Inflow faces (original)
    faces_in = elements
    
    # Outflow faces (offset by n_pts and reversed winding for outward normals)
    faces_out = elements + n_pts
    faces_out = faces_out[:, [0, 2, 1]] # reverse winding
    
    # Side faces: find boundary edges and connect them
    edge_count = {}
    for face in elements:
        for i in range(3):
            edge = tuple(sorted((face[i], face[(i+1)%3])))
            # Store original order to keep track of winding
            orig_edge = (face[i], face[(i+1)%3])
            if edge not in edge_count:
                edge_count[edge] = [0, orig_edge]
            edge_count[edge][0] += 1
            
    boundary_edges = []
    for edge, info in edge_count.items():
        if info[0] == 1:
            boundary_edges.append(info[1])
            
    side_faces = []
    for v1, v2 in boundary_edges:
        # Create a quad (v1_in, v2_in, v2_out, v1_out)
        # Tri 1: (v1_in, v2_in, v2_out)
        side_faces.append([v1, v2, v2 + n_pts])
        # Tri 2: (v1_in, v2_out, v1_out)
        side_faces.append([v1, v2 + n_pts, v1 + n_pts])
        
    all_faces = np.vstack([faces_in, faces_out, side_faces])
    
    return vertices, all_faces

def compute_gaussian_curvature(points, elements):
    """
    Рассчитывает гауссову кривизну в каждом узле методом дефицита углов.
    
    K(v) = (2*pi - sum(angles)) / Area(v). 
    Для граничных узлов используется целевой угол pi.

    Args:
        points (np.ndarray): Координаты узлов (N, 3).
        elements (np.ndarray): Связность элементов (M, 3).

    Returns:
        np.ndarray: Массив значений гауссовой кривизны для каждого узла.
    """
    n_points = points.shape[0]
    angle_sums = np.zeros(n_points)
    vertex_areas = np.zeros(n_points)
    
    # Find boundary vertices
    edge_count = {}
    for face in elements:
        for i in range(3):
            edge = tuple(sorted((face[i], face[(i+1)%3])))
            edge_count[edge] = edge_count.get(edge, 0) + 1
    
    boundary_vertices = set()
    for edge, count in edge_count.items():
        if count == 1:
            boundary_vertices.add(edge[0])
            boundary_vertices.add(edge[1])

    for face in elements:
        v_indices = face
        pts = points[v_indices]
        
        # Edges
        e0 = pts[1] - pts[0]
        e1 = pts[2] - pts[1]
        e2 = pts[0] - pts[2]
        
        # Lengths
        l0 = np.linalg.norm(e0)
        l1 = np.linalg.norm(e1)
        l2 = np.linalg.norm(e2)
        
        if l0 < 1e-10 or l1 < 1e-10 or l2 < 1e-10:
            continue
            
        # Angles using dot product
        # Angle at v0
        cos0 = np.dot(e0, -e2) / (l0 * l2)
        # Angle at v1
        cos1 = np.dot(e1, -e0) / (l1 * l0)
        # Angle at v2
        cos2 = np.dot(e2, -e1) / (l2 * l1)
        
        angles = np.arccos(np.clip([cos0, cos1, cos2], -1.0, 1.0))
        
        # Triangle Area
        area = 0.5 * np.linalg.norm(np.cross(e0, -e2))
        
        for i in range(3):
            angle_sums[v_indices[i]] += angles[i]
            vertex_areas[v_indices[i]] += area / 3.0

    gaussian_curv = np.zeros(n_points)
    for i in range(n_points):
        if vertex_areas[i] > 1e-10:
            target_angle = np.pi if i in boundary_vertices else 2 * np.pi
            gaussian_curv[i] = (target_angle - angle_sums[i]) / vertex_areas[i]
            
    return gaussian_curv

def evaluate_leaflet_developability(points, elements, DIA, points_to_exclude=None, mesh_step=0.45, epsilon=0.015, D_ref=25.0):
    """
    Автоматизированная оценка развертываемости створки методом Weighted Integral Deviation (WID).
    
    Вместо точечного максимума использует площадь-взвешенное RMS гауссовой кривизны (K_RMS) 
    и адаптивный порог, зависящий от диаметра клапана.

    Args:
        points (np.ndarray): Узлы сетки.
        elements (np.ndarray): Элементы сетки.
        DIA (float): Диаметр клапана для расчета адаптивного порога T_adj.
        points_to_exclude (np.ndarray, optional): Точки (линия подшива), исключаемые из анализа.
        mesh_step (float): Шаг сетки для поиска исключаемых узлов.
        epsilon (float): Коэффициент базовой чувствительности (0.015).
        D_ref (float): Референсный диаметр (25.0 мм).

    Returns:
        tuple: (developability_index, is_developable, results)
            - developability_index (float): Отношение K_RMS / T_adj (дизайн хорош при < 1.0).
            - is_developable (bool): Флаг прохождения теста.
            - results (dict): Словарь с детальной статистикой (K_RMS, T_adj, K_90 и др.).
    """
    n_points = points.shape[0]
    n_elements = elements.shape[0]
    
    # 1. Compute vertex Gaussian curvatures
    k_vertex = compute_gaussian_curvature(points, elements)
    
    # Identify indices to exclude
    exclude_indices = set()
    if points_to_exclude is not None:
        # points_to_exclude is expected to be (3, M) or (M, 3)
        pts_ex = points_to_exclude.T if points_to_exclude.shape[0] == 3 else points_to_exclude
        for p in pts_ex:
            dists = np.linalg.norm(points - p, axis=1)
            idx = np.argmin(dists)
            if dists[idx] < mesh_step * 0.5:
                exclude_indices.add(idx)

    # 2. Compute element areas and element-wise Gaussian curvature
    element_areas = []
    k_element = []
    active_elements_mask = []
    
    for i, face in enumerate(elements):
        pts = points[face]
        # Area
        v1 = pts[1] - pts[0]
        v2 = pts[2] - pts[0]
        area = 0.5 * np.linalg.norm(np.cross(v1, v2))
        
        # Check if element should be excluded (if any vertex is in exclude_indices)
        if any(v in exclude_indices for v in face):
            active_elements_mask.append(False)
        else:
            active_elements_mask.append(True)
            
        element_areas.append(area)
        k_element.append(np.mean(k_vertex[face]))
        
    element_areas = np.array(element_areas)
    k_element = np.array(k_element)
    active_elements_mask = np.array(active_elements_mask)
    
    # Use only active elements for RMS calculation
    active_areas = element_areas[active_elements_mask]
    active_k = k_element[active_elements_mask]
    
    total_area = np.sum(active_areas)
    if total_area < 1e-10:
        return 0.0, True, {"error": "Total active area is zero"}

    # 3. Handle Boundaries (Singularity Penalty) for remaining active elements
    # Identify boundary vertices
    edge_count = {}
    for face in elements[active_elements_mask]:
        for j in range(3):
            edge = tuple(sorted((face[j], face[(j+1)%3])))
            edge_count[edge] = edge_count.get(edge, 0) + 1
    
    boundary_vertices = set()
    for edge, count in edge_count.items():
        if count == 1:
            boundary_vertices.update(edge)
            
    # Calculate mean absolute curvature for singularity detection
    mean_abs_k = np.sum(np.abs(active_k) * active_areas) / total_area
    singularity_limit = 5.0 * mean_abs_k
    
    # Apply weights for K_RMS calculation
    weights = active_areas.copy()
    for i, face in enumerate(elements[active_elements_mask]):
        if any(v in boundary_vertices for v in face):
            if np.abs(active_k[i]) > singularity_limit:
                weights[i] *= (singularity_limit / np.abs(active_k[i]))

    # 4. Primary Metric: RMS of Gaussian Curvature (K_RMS)
    k_rms = np.sqrt(np.sum((active_k**2) * weights) / np.sum(weights))
    
    # 5. Adaptive Threshold
    t_adj = epsilon * (D_ref / DIA)
    
    # 6. Decision Logic
    is_developable = k_rms < t_adj
    
    # 7. Refinement: borderline check with 90th percentile
    k_90 = np.percentile(np.abs(active_k), 90) if len(active_k) > 0 else 0

    results = {
        "k_rms": k_rms,
        "t_adj": t_adj,
        "mean_abs_k": mean_abs_k,
        "k_90": k_90,
        "total_area": total_area,
        "is_developable": is_developable,
        "developability_index": k_rms / t_adj,
        "k_vertex": k_vertex,
        "excluded_elements_count": np.sum(~active_elements_mask)
    }
    
    return k_rms / t_adj, is_developable, results
