import numpy as np
from utils.gaussian_curvature_v2 import evaluate_developability

def gaussian_tolerance_from_area_strain(diameter_mm: float,
                                        max_area_strain: float = 0.03) -> float:
    """
    Расчёт предельной |K| (мм⁻²) по допустимому относительному
    изменению площади δA/A.

    Parameters
    ----------
    diameter_mm : float
        Наибольший линейный размер исследуемого фрагмента (мм).
    max_area_strain : float, optional
        Допустимое δA/A (по умолчанию 0.03 → 3 %).

    Returns
    -------
    float
        Порог |K|_max, который можно передать в evaluate_developability.
    """
    radius = 0.5 * diameter_mm
    return 5*24.0 * max_area_strain / (2*diameter_mm ** 2)


def adaptive_tolerance(points_3d: np.ndarray,
                       elements: np.ndarray,
                       area_strain_grid=(0.01, 0.03, 0.06, 0.10)):
    """
    Подбор минимального порога |K|, удовлетворяющего критерию разворачиваемости.

    Parameters
    ----------
    points_3d : (N,3) ndarray
        Узлы поверхностной сетки.
    elements : (M,3) ndarray
        Треугольные элементы.
    area_strain_grid : iterable of float
        Список допустимых δA/A, перебираемых по возрастанию.

    Returns
    -------
    float | None
        Найденный порог |K|_max (мм⁻²) или None, если ни один не подошёл.
    """
    # Геометрический размер (диаметр) поверхности
    bbox = np.ptp(points_3d, axis=0)  # размах по осям
    diameter = float(np.linalg.norm(bbox))

    for strain in area_strain_grid:
        tol = gaussian_tolerance_from_area_strain(diameter, strain)
        res = evaluate_developability(points_3d, elements,
                                      tolerance=tol, visualize=False)
        if res["is_developable"]:
            return tol  # найден минимально‑достаточный порог

    return None  # поверхность остаётся неразворачиваемой даже при max strain
