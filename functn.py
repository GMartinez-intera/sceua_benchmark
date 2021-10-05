import numpy as np
import math


def functn(ifunc, nopt, x):
    if ifunc == 1:
        return goldstein_price(nopt, x)

    if ifunc == 2:
        return rosenbrock(nopt, x)

    if ifunc == 3:
        return six_hump_camelback(nopt, x)

    if ifunc == 4:
        return rastrigin(nopt, x)

    if ifunc == 5:
        return griewank(nopt, x)

    if ifunc == 6:
        return shekel(nopt, x)

    if ifunc == 7:
        return hartman(nopt, x)

    if ifunc == 8:
        return hosaki(nopt, x)


def goldstein_price(nopt, x):
    #
    # This is the Goldstein-Price Function
    # Bound X1=[-2,2], X2=[-2,2]
    # Global Optimum: 3.0,(0.0,-1.0)
    x1 = x[0]
    x2 = x[1]
    u1 = (x1 + x2 + 1.0) ** 2
    u2 = 19. - 14. * x1 + 3. * x1 ** 2 - 14. * x2 + 6. * x1 * x2 + 3. * x2 ** 2
    u3 = (2. * x1 - 3. * x2) ** 2
    u4 = 18. - 32. * x1 + 12. * x1 ** 2 + 48. * x2 - 36. * x1 * x2 + 27. * x2 ** 2
    u5 = u1 * u2
    u6 = u3 * u4
    f = (1. + u5) * (30. + u6)
    return f


def rosenbrock(nopt, x):
    #
    # This is the Rosenbrock Function
    # Bound: X1=[-5,5], X2=[-2,8]
    # Global Optimum: 0,(1,1)
    x1 = x[0]
    x2 = x[1]
    a = 100
    f = a * (x2 - x1 ** 2) ** 2 + (1 - x1) ** 2

    return f


def six_hump_camelback(nopt, x):
    #
    #  This is the Six-hump Camelback Function.
    #  Bound: X1=[-5,5], X2=[-5,5]
    #  True Optima: -1.031628453489877, (-0.08983,0.7126), (0.08983,-0.7126)
    #
    x1 = x[0]
    x2 = x[1]
    f = (4 - 2.1 * x1 ** 2 + (x1 ** 4) / 3) * x1 ** 2 + x1 * x2 + (-4 + 4 * x2 ** 2) * x2 ** 2

    return f


def rastrigin(nopt, x):
    #
    #  This is the Rastrigin Function
    #  Bound: X1=[-1,1], X2=[-1,1]
    #  Global Optimum: -2, (0,0)

    x1 = x[0]
    x2 = x[1]
    f = x1 ** 2 + x2 ** 2 - math.cos(18.0 * x1) - math.cos(18.0 * x2)

    return f


def griewank(nopt, x):
    #
    #  This is the Griewank Function (2-D or 10-D)
    #  Bound: X(i)=[-600,600], for i=1,2,...,10
    #  Global Optimum: 0, at origin
    if nopt == 2:
        d = 200
    else:
        d = 4000

    u1 = 0.0
    u2 = 1.0
    for j in range(nopt):
        u1 = u1 + x[j] ** 2 / d
        u2 = u2 * math.cos(x[j] / math.sqrt(j + 1))
    f = u1 - u2 + 1

    return f


def shekel(nopt, x):
    #
    #  This is the Shekel Function
    #  Bound: X(i)=[0,10], j=1,2,3,4
    #  Global Optimum:-10.5364098252,(4,4,4,4)
    #
    #  Data for Skekel function coefficients (n=4, m=10)

    a1 = np.array([[4., 1., 8., 6., 3., 2., 5., 8., 6., 7.],
                   [4., 1., 8., 6., 7., 9., 5., 1., 2., 3.6],
                   [4., 1., 8., 6., 3., 2., 3., 8., 6., 7.],
                   [4., 1., 8., 6., 7., 9., 3., 1., 2., 3.6]])
    c1 = np.array([.1, .2, .2, .4, .4, .6, .3, .7, .5, .5])

    f = 0.0
    for i in range(10):
        u = 0.0
        for j in range(nopt):
            u = u + (x[j] - a1[j, i]) ** 2

        u = 1.0 / (u + c1[i])
        f = f - u

    return f


def hartman(nopt, x):
    #
    #   This is the Hartman Function
    #   Bound: X(j)=[0,1], j=1,2,...,6
    #   Global Optimum:-3.322368011415515,
    #   (0.201,0.150,0.477,0.275,0.311,0.657)
    #
    #   Data for Hartman function coefficients (6-D)
    a2 = np.array([[10., 0.05, 3., 17.],
                   [3., 10., 3.5, 8.],
                   [17., 17., 1.7, .05],
                   [3.5, 0.1, 10., 10.],
                   [1.7, 8., 17., .1],
                   [8., 14., 8., 14.]])
    c2 = np.array([1., 1.2, 3., 3.2])
    p2 = np.array([[.1312, .2329, .2348, .4047],
                   [.1696, .4135, .1451, .8828],
                   [.5569, .8307, .3522, .8732],
                   [.0124, .3736, .2883, .5743],
                   [.8283, .1004, .3047, .1091],
                   [.5886, .9991, .6650, .0381]])

    #  Data for Hartman function coefficient (3-D)
    a3 = np.array([[3., .1, 3., .1],
                   [10., 10., 10., 10.],
                   [30., 35., 30., 35.]])
    c3 = np.array([1., 1.2, 3., 3.2])
    p3 = np.array([[.3689, .4699, .1091, .03815],
                   [.1170, .4387, .8732, .5743],
                   [.2673, .7470, .5547, .8828]])

    f = 0.0
    for i in range(4):
        u = 0.0
        for j in range(nopt):
            if nopt == 3:
                a2[j, i] = a3[j, i]
                p2[j, i] = p3[j, i]

            u = u + a2[j, i] * (x[j] - p2[j, i]) ** 2

        if nopt == 3:
            c2[i] = c3[i]
        f = f - c2[i] * math.exp(-u)

    return f


def hosaki(nopt, x):
    #
    # This is the Hosaki Function
    #  Bound: X1=[0, 5], X2=[0, 6]
    #  Global Optimum: -2.3458, (4,2))
    x1 = x[0]
    x2 = x[1]
    f = (1 - 8 * x1 + 7 * pow(x1, 2) - 7 * pow(x1, 3) / 3 + pow(x1, 4) / 4) * pow(x2, 2) * math.exp(-x2)

    return f
