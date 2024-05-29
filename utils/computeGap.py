import numpy as np
from .createGeometry import cart2pol
from .createGeometry import pol2cart
from .logger_leaflet import log_message
from scipy import spatial
from alphashape import alphashape

# в матлабе было проще... накинул встроенный alphaShape, вызвал встроенное поле .square и радуешься...
# здесь - считаем площадь каждого треугольника и суммируем
def compute_square(n1=None, n2=None, n3=None):
    x12 = n1[0] - n2[0]
    y12 = n1[1] - n2[1]
    z12 = n1[2] - n2[2]
    x13 = n1[0] - n3[0]
    y13 = n1[1] - n3[1]
    z13 = n1[2] - n3[2]

    return 0.5 * np.sqrt((y12 * z13 - z12 * y13) ** 2 + (z12 * x13 * z13) ** 2 + (x12 * y13 - y12 * x13) ** 2)


# не помню для чего и где используется, пусть побудет.
def unique_rows(A, atol=10e-5):
    #Get unique (within atol) rows of a 2D np.array A.
    remove = np.zeros(A.shape[0], dtype=bool)  # Row indexes to be removed.
    for i in range(A.shape[0]):  # Not very optimized, but simple.
        equals = np.all(np.isclose(A[i, :], A[(i + 1):, :], atol=atol), axis=1)
        remove[(i + 1):] = np.logical_or(remove[(i + 1):], equals)
    return A[np.logical_not(remove)]


# вычисление площади просвета
def computeOpened(disp=None, nodes=None, mesh_step=0.35):
    # получаем матрицу нодов деформированной створки, в цилиндрических координатах откидываем ноды > RAD,
    # накидываем триангуляцию, считаем площадь

    deformed = nodes + disp
    theta, rho, z = cart2pol(x=deformed[:, 0], y=deformed[:, 1], lz=deformed[:, 2])
    thetaN, rhoN, zN = cart2pol(x=nodes[:, 0], y=nodes[:, 1], lz=nodes[:, 2])
    rho[np.argwhere(rho > max(rhoN))] = max(rhoN)
    # rhoN[np.argwhere(rhoN > max(nodes[:, 1]))] = max(nodes[:, 1])
    readedPoints1 = np.array(pol2cart(phi=theta, rho=rho, lz=z), dtype='float64').T
    # nodes = np.array(pol2cart(phi=thetaN, rho=rhoN, lz=zN), dtype='float64').T
    # readedPoints1 = deformed
    # nodes[:, -1] = 0
    readedPoints1[:, -1] = 0

    # del theta, rho, z

    tri = spatial.Delaunay(nodes[:, :-1])
    init_square = 0
    for t in tri.simplices:
        init_square += compute_square(nodes[t[0], :], nodes[t[1], :], nodes[t[2], :])
    del t

    import matplotlib.pyplot as plt
    plt.triplot(nodes[:, 0], nodes[:, 1], tri.simplices)
    plt.plot(nodes[:, 0], nodes[:, 1], 'o')
    #
    shape = alphashape(readedPoints1[:, :-1], alpha=1)
    def_square = shape.area

    # x, y = shape.boundary.xy
    # plt.plot(x, y, color='#6699cc', alpha=0.7,linewidth=3, solid_capstyle='round', zorder=2)
    # plt.plot(readedPoints1[:, 0], readedPoints1[:, 1], 'x')
    # plt.show()
    # del x, y

    # tri_def = spatial.Delaunay(readedPoints1[:, :-1])
    # def_square = 0
    # for t in tri_def.simplices:
    #     def_square += compute_square(readedPoints1[t[0], :], readedPoints1[t[1], :], readedPoints1[t[2], :])
    # plt.triplot(readedPoints1[:, 0], readedPoints1[:, 1], tri_def.simplices)
    # plt.plot(readedPoints1[:, 0], readedPoints1[:, 1], 'x')
    # plt.show()
    # log_message("init_square %.6f - def_square %.6f = %.6f" % (init_square, def_square, init_square - def_square))
    del shape, readedPoints1, nodes
    return def_square


def computeClosed(disp1=None, disp2=None, disp3=None, nodes1=None, nodes2=None, nodes3=None, mesh_step=None):
    # получаем матрицу нодов деформированной створки, в цилиндрических координатах откидываем ноды > RAD,
    # накидываем триангуляцию, считаем площадь
    deformed1 = nodes1 + disp1
    ashp1 = alphashape(deformed1[:, :-1], alpha=1)
    deformed2 = nodes2 + disp2
    ashp2 = alphashape(deformed2[:, :-1], alpha=1)
    deformed3 = nodes3 + disp3
    ashp3 = alphashape(deformed3[:, :-1], alpha=1)

    # import matplotlib.pyplot as plt
    # x, y = ashp.boundary.xy
    # plt.plot(x, y, color='#6699cc', alpha=0.7, linewidth=3, solid_capstyle='round', zorder=2)
    # plt.plot(deformed[:, 0], deformed[:, 1], 'x')
    # plt.show()

    return (ashp1.area + ashp2.area + ashp3.area)


def computeClosed_single(disp1=None, nodes1=None, mesh_step=None):
    # получаем матрицу нодов деформированной створки, в цилиндрических координатах откидываем ноды > RAD,
    # накидываем триангуляцию, считаем площадь
    deformed = nodes1 + disp1
    ashp = alphashape(deformed[:, :-1], alpha=1)

    # import matplotlib.pyplot as plt
    # x, y = ashp.boundary.xy
    # plt.plot(x, y, color='#6699cc', alpha=0.7, linewidth=3, solid_capstyle='round', zorder=2)
    # plt.plot(deformed[:, 0], deformed[:, 1], 'x')
    # plt.show()

    return (3 * ashp.area )