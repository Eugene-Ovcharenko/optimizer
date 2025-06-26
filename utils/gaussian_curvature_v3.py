import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def unfold_surface(points_inner, shell_elements, method='conformal'):
    points_array = np.array(points_inner)
    if points_array.shape[0] == 3 and points_array.shape[1] > 3:
        points = points_array.T
    elif points_array.shape[1] == 3:
        points = points_array
    else:
        raise ValueError(f"Неподдерживаемый формат: {points_array.shape}")

    elements = np.array(shell_elements, dtype=int)
    n_points = points.shape[0]

    if np.max(elements) >= n_points or np.min(elements) < 0:
        raise ValueError("Некорректные индексы в элементах")

    if method == 'conformal':
        flat_points = _conformal_mapping(points, elements)
    elif method == 'angle_preserving':
        flat_points = _angle_preserving(points, elements)
    elif method == 'pca':
        flat_points = _pca_projection(points)
    else:
        raise ValueError(f"Неизвестный метод: {method}")

    quality_metrics = _analyze_quality(points, elements, flat_points)

    return {
        'points_3d': points,
        'points_2d': flat_points,
        'elements': elements,
        'quality_metrics': quality_metrics,
        'method': method
    }


def evaluate_developability(points_inner, shell_elements, tolerance=1e-3, visualize=True, method='conformal'):
    points_array = np.array(points_inner)
    if points_array.shape[0] == 3 and points_array.shape[1] > 3:
        points = points_array.T
    elif points_array.shape[1] == 3:
        points = points_array
    else:
        raise ValueError(f"Неподдерживаемый формат: {points_array.shape}")

    elements = np.array(shell_elements, dtype=int)
    n_points = points.shape[0]

    boundary_vertices = _find_boundary_vertices(elements, n_points)
    gaussian_curvatures = _compute_gaussian_curvature_boundary_aware(points, elements, boundary_vertices)

    max_curvature = np.max(np.abs(gaussian_curvatures))
    mean_curvature = np.mean(np.abs(gaussian_curvatures))
    is_developable = max_curvature < tolerance

    curvature_stats = {
        'max_abs_curvature': max_curvature,
        'mean_abs_curvature': mean_curvature,
        'std_curvature': np.std(gaussian_curvatures),
        'percentile_95': np.percentile(np.abs(gaussian_curvatures), 95),
        'high_curvature_points': np.sum(np.abs(gaussian_curvatures) > tolerance),
        'boundary_points': len(boundary_vertices)
    }

    unfold_result = unfold_surface(points_inner, shell_elements, method=method)

    results = {
        'is_developable': is_developable,
        'gaussian_curvatures': gaussian_curvatures,
        'curvature_stats': curvature_stats,
        'tolerance': tolerance,
        'unfold_quality': unfold_result['quality_metrics'],
        'points_3d': points,
        'points_2d': unfold_result['points_2d'],
        'elements': elements,
        'boundary_vertices': boundary_vertices
    }

    if visualize:
        _visualize_results(results)

    return results


def _sort_boundary_vertices(points, elements, boundary_vertices):
    adjacency = {}
    for v in boundary_vertices:
        adjacency[v] = []

    for face in elements:
        for i in range(3):
            v1, v2 = face[i], face[(i + 1) % 3]
            if v1 in boundary_vertices and v2 in boundary_vertices:
                adjacency[v1].append(v2)
                adjacency[v2].append(v1)

    for v in boundary_vertices:
        adjacency[v] = list(set(adjacency[v]))

    start_vertex = boundary_vertices[0]
    for v in boundary_vertices:
        if len(adjacency[v]) <= 2:
            start_vertex = v
            break

    sorted_boundary = [start_vertex]
    current = start_vertex
    prev = None

    while len(sorted_boundary) < len(boundary_vertices):
        neighbors = [v for v in adjacency[current] if v != prev]
        if not neighbors:
            break

        next_vertex = neighbors[0]
        sorted_boundary.append(next_vertex)
        prev = current
        current = next_vertex

        if current == start_vertex:
            break

    return sorted_boundary

def _find_boundary_vertices(elements, n_points):
    edge_count = {}

    for face in elements:
        edges = [
            tuple(sorted([face[0], face[1]])),
            tuple(sorted([face[1], face[2]])),
            tuple(sorted([face[2], face[0]]))
        ]
        for edge in edges:
            edge_count[edge] = edge_count.get(edge, 0) + 1

    boundary_edges = [edge for edge, count in edge_count.items() if count == 1]
    boundary_vertices = set()
    for edge in boundary_edges:
        boundary_vertices.update(edge)

    return list(boundary_vertices)


def _compute_gaussian_curvature_boundary_aware(points, elements, boundary_vertices):
    n_points = points.shape[0]
    gaussian_curvs = np.zeros(n_points)
    vertex_areas = np.zeros(n_points)
    angle_sums = np.zeros(n_points)

    boundary_set = set(boundary_vertices)

    for face in elements:
        idx0, idx1, idx2 = face
        v0, v1, v2 = points[idx0], points[idx1], points[idx2]

        a = np.linalg.norm(v2 - v1)
        b = np.linalg.norm(v2 - v0)
        c = np.linalg.norm(v1 - v0)

        if a < 1e-12 or b < 1e-12 or c < 1e-12:
            continue

        edge1 = v1 - v0
        edge2 = v2 - v0
        cross_product = np.cross(edge1, edge2)
        area = 0.5 * np.linalg.norm(cross_product)

        if area < 1e-12:
            continue

        cos_A = np.clip((b * b + c * c - a * a) / (2 * b * c), -1, 1)
        cos_B = np.clip((a * a + c * c - b * b) / (2 * a * c), -1, 1)
        cos_C = np.clip((a * a + b * b - c * c) / (2 * a * b), -1, 1)

        angle_A = np.arccos(cos_A)
        angle_B = np.arccos(cos_B)
        angle_C = np.arccos(cos_C)

        angle_sums[idx0] += angle_A
        angle_sums[idx1] += angle_B
        angle_sums[idx2] += angle_C

        vertex_areas[idx0] += area / 3
        vertex_areas[idx1] += area / 3
        vertex_areas[idx2] += area / 3

    boundary_angle_defects = _compute_boundary_angle_defects(points, elements, boundary_vertices)

    for i in range(n_points):
        if vertex_areas[i] > 1e-12:
            if i in boundary_set:
                expected_angle = boundary_angle_defects.get(i, np.pi)
                gaussian_curvs[i] = (expected_angle - angle_sums[i]) / vertex_areas[i]
            else:
                gaussian_curvs[i] = (2 * np.pi - angle_sums[i]) / vertex_areas[i]

            gaussian_curvs[i] = np.clip(gaussian_curvs[i], -10, 10)

    return gaussian_curvs


def _compute_boundary_angle_defects(points, elements, boundary_vertices):
    boundary_set = set(boundary_vertices)
    boundary_angles = {}

    # Строим граф связности граничных вершин
    boundary_graph = {}
    for v in boundary_vertices:
        boundary_graph[v] = []

    for face in elements:
        for i in range(3):
            v1, v2 = face[i], face[(i + 1) % 3]
            if v1 in boundary_set and v2 in boundary_set:
                boundary_graph[v1].append(v2)
                boundary_graph[v2].append(v1)

    # Удаляем дубликаты
    for v in boundary_graph:
        boundary_graph[v] = list(set(boundary_graph[v]))

    # Вычисляем углы в исходной 3D поверхности для граничных вершин
    for v in boundary_vertices:
        neighbors = boundary_graph[v]

        if len(neighbors) == 2:
            # Обычная граничная точка
            p0 = points[v]
            p1 = points[neighbors[0]]
            p2 = points[neighbors[1]]

            vec1 = p1 - p0
            vec2 = p2 - p0

            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)

            if norm1 > 1e-12 and norm2 > 1e-12:
                cos_angle = np.clip(np.dot(vec1, vec2) / (norm1 * norm2), -1, 1)
                exterior_angle = np.arccos(cos_angle)
                # Для граничной вершины ожидаемая сумма углов равна π минус внешний угол
                boundary_angles[v] = np.pi - exterior_angle
            else:
                boundary_angles[v] = np.pi

        elif len(neighbors) == 1:
            # Конечная точка границы
            boundary_angles[v] = np.pi / 2

        elif len(neighbors) > 2:
            # Точка пересечения нескольких граничных сегментов
            # Вычисляем сумму внешних углов
            total_exterior_angle = 0
            p0 = points[v]

            # Сортируем соседей по углу
            angles_with_neighbors = []
            for neighbor in neighbors:
                vec = points[neighbor] - p0
                angle = np.arctan2(vec[1], vec[0])  # Проекция на плоскость XY
                angles_with_neighbors.append((angle, neighbor))

            angles_with_neighbors.sort()
            sorted_neighbors = [neighbor for _, neighbor in angles_with_neighbors]

            for i in range(len(sorted_neighbors)):
                n1 = sorted_neighbors[i]
                n2 = sorted_neighbors[(i + 1) % len(sorted_neighbors)]

                vec1 = points[n1] - p0
                vec2 = points[n2] - p0

                norm1 = np.linalg.norm(vec1)
                norm2 = np.linalg.norm(vec2)

                if norm1 > 1e-12 and norm2 > 1e-12:
                    cos_angle = np.clip(np.dot(vec1, vec2) / (norm1 * norm2), -1, 1)
                    angle = np.arccos(cos_angle)
                    total_exterior_angle += angle

            boundary_angles[v] = 2 * np.pi - total_exterior_angle

        else:
            # Изолированная вершина (не должно происходить на правильной границе)
            boundary_angles[v] = np.pi

    return boundary_angles


def _detect_corner_vertices(points, elements, boundary_vertices):
    boundary_set = set(boundary_vertices)
    corner_vertices = {}

    adjacency = {}
    for v in boundary_vertices:
        adjacency[v] = []

    for face in elements:
        for i in range(3):
            v1, v2 = face[i], face[(i + 1) % 3]
            if v1 in boundary_set and v2 in boundary_set:
                adjacency[v1].append(v2)
                adjacency[v2].append(v1)

    for v in boundary_vertices:
        neighbors = list(set(adjacency[v]))
        if len(neighbors) == 2:
            p0 = points[v]
            p1 = points[neighbors[0]]
            p2 = points[neighbors[1]]

            vec1 = p1 - p0
            vec2 = p2 - p0

            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)

            if norm1 > 1e-12 and norm2 > 1e-12:
                cos_angle = np.clip(np.dot(vec1, vec2) / (norm1 * norm2), -1, 1)
                angle = np.arccos(cos_angle)

                if angle < np.pi * 0.6:
                    corner_vertices[v] = np.pi / 2

    return corner_vertices


def _conformal_mapping(points, elements):
    n = points.shape[0]
    boundary_vertices = _find_boundary_vertices(elements, n)

    if len(boundary_vertices) > 8:
        return _boundary_constrained_mapping(points, elements, boundary_vertices)
    else:
        return _laplacian_mapping(points, elements)


def _boundary_constrained_mapping(points, elements, boundary_vertices):
    n = points.shape[0]
    sorted_boundary = _sort_boundary_vertices(points, elements, boundary_vertices)

    boundary_coords_3d = []
    for v in sorted_boundary:
        boundary_coords_3d.append(points[v])

    perimeter_lengths = []
    total_perimeter = 0
    for i in range(len(sorted_boundary)):
        v1 = sorted_boundary[i]
        v2 = sorted_boundary[(i + 1) % len(sorted_boundary)]
        length = np.linalg.norm(points[v1] - points[v2])
        perimeter_lengths.append(length)
        total_perimeter += length

    segments = _detect_boundary_segments(points, sorted_boundary)

    if len(segments) >= 4:
        flat_boundary = _map_rectangular_boundary(perimeter_lengths, segments)
    else:
        flat_boundary = _map_circular_boundary(perimeter_lengths)

    W = _build_laplacian_matrix(points, elements)

    flat_points = np.zeros((n, 2))
    for i, v in enumerate(sorted_boundary):
        flat_points[v] = flat_boundary[i]

    interior_vertices = [v for v in range(n) if v not in set(sorted_boundary)]

    if interior_vertices:
        flat_points = _solve_interior_vertices(W, flat_points, sorted_boundary, interior_vertices)

    return flat_points


def _detect_boundary_segments(points, sorted_boundary):
    if len(sorted_boundary) < 4:
        return []

    angles = []
    n = len(sorted_boundary)

    for i in range(n):
        prev_v = sorted_boundary[(i - 1) % n]
        curr_v = sorted_boundary[i]
        next_v = sorted_boundary[(i + 1) % n]

        v1 = points[prev_v] - points[curr_v]
        v2 = points[next_v] - points[curr_v]

        v1_norm = np.linalg.norm(v1)
        v2_norm = np.linalg.norm(v2)

        if v1_norm > 1e-12 and v2_norm > 1e-12:
            cos_angle = np.clip(np.dot(v1, v2) / (v1_norm * v2_norm), -1, 1)
            angle = np.arccos(cos_angle)
            angles.append(angle)
        else:
            angles.append(np.pi)

    corner_threshold = np.pi * 0.4
    corners = []

    for i, angle in enumerate(angles):
        if angle < corner_threshold:
            corners.append(i)

    if len(corners) < 3:
        return []

    segments = []
    for i in range(len(corners)):
        start = corners[i]
        end = corners[(i + 1) % len(corners)]

        if end > start:
            segment = list(range(start, end + 1))
        else:
            segment = list(range(start, len(sorted_boundary))) + list(range(0, end + 1))

        segments.append(segment)

    return segments


def _map_rectangular_boundary(perimeter_lengths, segments):
    if len(segments) != 4:
        return _map_circular_boundary(perimeter_lengths)

    side_lengths = []
    for segment in segments:
        length = sum(perimeter_lengths[i] for i in range(len(segment) - 1))
        side_lengths.append(length)

    width = (side_lengths[0] + side_lengths[2]) / 2
    height = (side_lengths[1] + side_lengths[3]) / 2

    corners = [(0, 0), (width, 0), (width, height), (0, height)]

    flat_boundary = []
    corner_idx = 0

    for i, segment in enumerate(segments):
        start_corner = corners[i]
        end_corner = corners[(i + 1) % 4]

        segment_length = side_lengths[i]
        current_length = 0

        for j, vertex_idx in enumerate(segment):
            if j == 0:
                flat_boundary.append(start_corner)
            elif j == len(segment) - 1:
                flat_boundary.append(end_corner)
            else:
                if vertex_idx < len(perimeter_lengths):
                    current_length += perimeter_lengths[vertex_idx]

                t = current_length / segment_length if segment_length > 0 else 0
                t = np.clip(t, 0, 1)

                x = start_corner[0] + t * (end_corner[0] - start_corner[0])
                y = start_corner[1] + t * (end_corner[1] - start_corner[1])
                flat_boundary.append((x, y))

    return np.array(flat_boundary)


def _map_circular_boundary(perimeter_lengths):
    total_perimeter = sum(perimeter_lengths)
    flat_boundary = []
    current_length = 0

    for length in perimeter_lengths:
        angle = 2 * np.pi * current_length / total_perimeter
        x = np.cos(angle)
        y = np.sin(angle)
        flat_boundary.append((x, y))
        current_length += length

    return np.array(flat_boundary)


def _build_laplacian_matrix(points, elements):
    n = points.shape[0]
    W = np.zeros((n, n))

    for face in elements:
        for i in range(3):
            v1, v2, v3 = face[i], face[(i + 1) % 3], face[(i + 2) % 3]
            p1, p2, p3 = points[v1], points[v2], points[v3]

            e1, e2 = p3 - p2, p1 - p2
            norm1, norm2 = np.linalg.norm(e1), np.linalg.norm(e2)

            if norm1 > 1e-12 and norm2 > 1e-12:
                cos_angle = np.clip(np.dot(e1, e2) / (norm1 * norm2), -0.999, 0.999)
                cot_weight = cos_angle / np.sqrt(1 - cos_angle ** 2 + 1e-12)
                cot_weight = np.clip(cot_weight, -5, 5)

                W[v1, v3] += cot_weight
                W[v3, v1] += cot_weight

    L = np.diag(np.sum(W, axis=1)) - W
    return L


def _solve_interior_vertices(L, flat_points, boundary_vertices, interior_vertices):
    n = len(interior_vertices)
    boundary_set = set(boundary_vertices)

    if n == 0:
        return flat_points

    A = np.zeros((n, n))
    b_u = np.zeros(n)
    b_v = np.zeros(n)

    for i, v_int in enumerate(interior_vertices):
        row_sum = 0
        for j in range(flat_points.shape[0]):
            if j in boundary_set:
                coeff = L[v_int, j]
                b_u[i] -= coeff * flat_points[j, 0]
                b_v[i] -= coeff * flat_points[j, 1]
            elif j in interior_vertices:
                j_idx = interior_vertices.index(j)
                A[i, j_idx] = L[v_int, j]

    for i in range(n):
        if abs(A[i, i]) < 1e-12:
            A[i, i] = 1.0

    try:
        u_coords = np.linalg.solve(A, b_u)
        v_coords = np.linalg.solve(A, b_v)

        for i, v_int in enumerate(interior_vertices):
            flat_points[v_int, 0] = u_coords[i]
            flat_points[v_int, 1] = v_coords[i]
    except:
        pass

    return flat_points


def _laplacian_mapping(points, elements):
    n = points.shape[0]
    L = _build_laplacian_matrix(points, elements)
    L += 1e-8 * np.eye(n)

    try:
        eigenvals, eigenvecs = eigsh(L, k=min(5, n - 1), which='SM', maxiter=2000)
        if len(eigenvals) >= 3:
            flat_points = np.column_stack([eigenvecs[:, 1], eigenvecs[:, 2]])
        else:
            flat_points = _pca_projection(points)
    except:
        flat_points = _pca_projection(points)

    return flat_points


def _angle_preserving(points, elements):
    initial_flat = _pca_projection(points)

    if points.shape[0] > 3000:
        return initial_flat

    def angle_energy(flat_coords):
        flat_2d = flat_coords.reshape(-1, 2)
        energy = 0.0

        step = max(1, len(elements) // 500)
        for i in range(0, len(elements), step):
            face = elements[i]

            v0, v1, v2 = points[face]
            e1_3d, e2_3d = v1 - v0, v2 - v0
            norm1_3d, norm2_3d = np.linalg.norm(e1_3d), np.linalg.norm(e2_3d)

            if norm1_3d > 1e-12 and norm2_3d > 1e-12:
                cos_3d = np.clip(np.dot(e1_3d, e2_3d) / (norm1_3d * norm2_3d), -1, 1)

                v0_2d, v1_2d, v2_2d = flat_2d[face]
                e1_2d, e2_2d = v1_2d - v0_2d, v2_2d - v0_2d
                norm1_2d, norm2_2d = np.linalg.norm(e1_2d), np.linalg.norm(e2_2d)

                if norm1_2d > 1e-12 and norm2_2d > 1e-12:
                    cos_2d = np.clip(np.dot(e1_2d, e2_2d) / (norm1_2d * norm2_2d), -1, 1)
                    energy += (cos_3d - cos_2d) ** 2

        return energy

    try:
        result = minimize(angle_energy, initial_flat.flatten(), method='L-BFGS-B',
                          options={'maxiter': 50, 'maxfun': 200})
        return result.x.reshape(-1, 2)
    except:
        return initial_flat


def _pca_projection(points):
    centered = points - np.mean(points, axis=0)
    U, s, Vt = np.linalg.svd(centered.T, full_matrices=False)
    return centered @ U[:, :2]


def _analyze_quality(points_3d, elements, points_2d):
    distance_errors = []
    area_errors = []

    for face in elements:
        for i in range(3):
            for j in range(i + 1, 3):
                v1, v2 = face[i], face[j]

                dist_3d = np.linalg.norm(points_3d[v1] - points_3d[v2])
                dist_2d = np.linalg.norm(points_2d[v1] - points_2d[v2])

                if dist_3d > 1e-12:
                    error = abs(dist_2d - dist_3d) / dist_3d
                    distance_errors.append(error)

    for face in elements:
        v0, v1, v2 = points_3d[face]
        area_3d = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0))

        v0_2d, v1_2d, v2_2d = points_2d[face]
        area_2d = 0.5 * abs((v1_2d[0] - v0_2d[0]) * (v2_2d[1] - v0_2d[1]) -
                            (v2_2d[0] - v0_2d[0]) * (v1_2d[1] - v0_2d[1]))

        if area_3d > 1e-12:
            error = abs(area_2d - area_3d) / area_3d
            area_errors.append(error)

    return {
        'distance_error_mean': np.mean(distance_errors) if distance_errors else 0,
        'distance_error_max': np.max(distance_errors) if distance_errors else 0,
        'area_error_mean': np.mean(area_errors) if area_errors else 0,
        'area_error_max': np.max(area_errors) if area_errors else 0
    }


def _visualize_results(results):
    points_3d = results['points_3d']
    points_2d = results['points_2d']
    elements = results['elements']
    curvatures = results['gaussian_curvatures']
    boundary_vertices = results['boundary_vertices']

    fig = plt.figure(figsize=(15, 5))

    ax1 = fig.add_subplot(131, projection='3d')
    ax1.plot_trisurf(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2],
                     triangles=elements, alpha=0.7, cmap='viridis')
    ax1.scatter(points_3d[boundary_vertices, 0], points_3d[boundary_vertices, 1],
                points_3d[boundary_vertices, 2], c='red', s=30)
    ax1.set_title('3D поверхность (красные - граница)')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')

    ax2 = fig.add_subplot(132)
    ax2.triplot(points_2d[:, 0], points_2d[:, 1], elements, 'b-', alpha=0.3)
    ax2.scatter(points_2d[:, 0], points_2d[:, 1], c='blue', s=1)
    boundary_curvature = np.clip(curvatures[boundary_vertices], -5, 5)
    ax2.scatter(points_2d[boundary_vertices, 0], points_2d[boundary_vertices, 1], c=boundary_curvature, cmap='RdBu_r', s=20)
    ax2.set_title('Развертка (красные - граница)')
    ax2.set_xlabel('U')
    ax2.set_ylabel('V')
    ax2.axis('equal')

    ax3 = fig.add_subplot(133, projection='3d')
    curvatures_clipped = np.clip(curvatures, -5, 5)
    scatter = ax3.scatter(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2],
                          c=curvatures_clipped, cmap='RdBu_r', s=20)
    plt.colorbar(scatter, ax=ax3, shrink=0.8)
    ax3.set_title('Гауссова кривизна')
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_zlabel('Z')

    plt.tight_layout()
    plt.show()