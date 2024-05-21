import numpy as np
from .createGeometry import cart2pol
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
    result1 = len(np.argwhere(np.rad2deg(theta) < (90-120/2)))
    result2 = len(np.argwhere(np.rad2deg(theta) > (90+120/2)))
#    log_message('res1 > %d | res2 > %d' % (result1, result2))
    if result1+result2 > 12:
        return 1
    else:
        return 0