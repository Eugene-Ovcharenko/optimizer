import numpy as np
from .createGeometry import cart2pol, pol2cart
from .logger_leaflet import log_message

# def helicopter(disp=None, nodes=None):
#     deformed = nodes + disp
#     theta, rho, _ = cart2pol(x=deformed[:, 0], y=deformed[:, 1], lz=deformed[:, 2])
#     from matplotlib import pyplot as plt
#     plt.plot(deformed[:, 0], deformed[:, 1], 'ok')
#     plt.show()
#     del deformed
#     result1 = len(np.argwhere(np.rad2deg(theta) < (90-120/2)))
#     result2 = len(np.argwhere(np.rad2deg(theta) > (90+120/2)))
#     if result1+result2 > 12:
#         return 1
#     else:
#         return 0

def helicopter(nodes=None, SEC=None):
    theta, rho, _ = cart2pol(x=nodes[:, 0], y=nodes[:, 1], lz=nodes[:, 2])
    # result1 = len(np.argwhere(np.rad2deg(theta) < (90-120/2)))
    # result2 = len(np.argwhere(np.rad2deg(theta) > (90+120/2)))
    p1 = np.array((0, 0),dtype='float64')
    p2 = np.array(pol2cart(theta=np.deg2rad(90-120/2), rho=np.max(rho), lz=0)[:-1])
    points3_minus = (nodes[np.argwhere(np.rad2deg(theta) < (90 - 120 / 2))][:, 0])[:, :-1]
    max_normal_distance_minus = 0
    max_normal_distance_plus = 0
    for p3 in points3_minus:
        max_normal_distance_minus = np.abs(np.cross(p2 - p1, p3 - p1) / np.linalg.norm(p2 - p1))
    p2 = np.array(pol2cart(theta=np.deg2rad(90 + 120 / 2), rho=np.max(rho), lz=0)[:-1])
    points3_plus = (nodes[np.argwhere(np.rad2deg(theta) > (90 + 120 / 2))][:, 0])[:, :-1]
    for p3 in points3_plus:
        max_normal_distance_plus = np.abs(np.cross(p2 - p1, p3 - p1) / np.linalg.norm(p2 - p1))

    max_normal_distance = np.max((max_normal_distance_plus, max_normal_distance_minus))
    return max_normal_distance
#    log_message('res1 > %d | res2 > %d' % (result1, result2))
#     if result1+result2 > 12:
#         return 1
#     else:
#         return 0