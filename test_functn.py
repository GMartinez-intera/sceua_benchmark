import numpy as np

from sceua import sceua

if __name__ == "__main__":
    results = []
    for ifunc in range(1, 8+1):
        print("Enter the function number is " + str(ifunc))
        bl = bu = x0 = f0 = namef = None
        if ifunc == 1:
            #  This is the Goldstein-Price Function
            #  Bound X1=[-2,2], X2=[-2,2]; Global Optimum: 3.0,(0.0,-1.0)
            bl = np.array([-2, -2])
            bu = np.array([2, 2])
            x0 = np.array([0, -1])
            f0 = 3.0
            namef = 'goldstein_price'

        if ifunc == 2:
            # This is the Rosenbrock Function
            # Bound: X1=[-5,5], X2=[-2,8]; Global Optimum: 0,(1,1)
            bl = np.array([-5, -5])
            bu = np.array([5, 5])
            x0 = np.array([1, 1])
            f0 = 0.0
            namef = 'rosenbrock'

        if ifunc == 3:
            # This is the Six-hump Camelback Function.
            # Bound: X1=[-5,5], X2=[-5,5]
            # True Optima: -1.03628453489877, (-0.08983,0.7126), (0.08983,-0.7126)
            bl = np.array([-5, -2])
            bu = np.array([5, 2])
            x0 = np.array([[-0.08983, 0.7126], [0.08983, 0.7126]])
            f0 = -1.03628453489877
            namef = 'six_hump_camelback'

        if ifunc == 4:
            # This is the Rastrigin Function
            # Bound: X1=[-1,1], X2=[-1,1]; Global Optimum: -2, (0,0)
            bl = -1*np.ones(2)
            bu = 1*np.ones(2)
            x0 = np.zeros(2)
            f0 = -2.0
            namef = 'rastrigin'

        if ifunc == 5:
            # This is the Griewank Function (2-D or 10-D)
            # Bound: X(i)=[-600,600], for i=1,2,...,10
            # Global Optimum: 0, at origin
            bl = -600*np.ones(10)
            bu = 600*np.ones(10)
            x0 = np.zeros(10)
            f0 = 0.0
            namef = 'griewank'

        if ifunc == 6:
            # This is the Shekel Function; Bound: X(i)=[0,10], j=1,2,3,4
            # Global Optimum:-10.5364098252,(4,4,4,4)
            bl = np.zeros(4)
            bu = np.ones(4)*100
            x0 = np.ones(4)*4
            f0 = -10.5364098252
            namef = 'shekel'

        if ifunc == 7:
            # This is the Hartman Function
            # Bound: X(j)=[0,1], j=1,2,...,6;
            # Global Optimum:-3.322368011415515, (0.201,0.150,0.477,0.275,0.311,0.657)
            bl = np.zeros(6)
            bu = np.ones(6)
            x0 = np.array([0.201, 0.150, 0.477, 0.275, 0.311, 0.657])
            f0 = -3.322368011415515
            namef = 'hartman'

        if ifunc == 8:
            # This is the Hosaki Function
            #  Bound: X1=[0, 5], X2=[0, 6]
            #  Global Optimum: -2.3458, (4,2)
            bl = np.array([0, 0])
            bu = np.array([5, 6])
            x0 = np.array([4, 2])
            f0 = -2.3458
            namef = 'hosaki'

        Smaxn = 100000
        Skstop = 10
        Spcento = 0.001
        Speps = 0.00001
        import random
        Siseed = random.randrange(1, 100)
        Siniflg = 0

        ngs = len(x0)+5
        print("The number of complexes is " + str(ngs))

        scebestx, scebestf, icall = sceua(x0, bl, bu, Smaxn, Skstop, Spcento, Speps, ngs, Siseed, Siniflg, ifunc)

        results.append(["-->{0} {1} with {2} in {3} trials".format(ifunc, namef, len(x0), icall),
                        np.round(np.array(scebestx), decimals=3),
                        np.round(np.array(x0), decimals=3),
                        [round(scebestf, 3),  round(f0, 3)]])

    for elem in results:
        for val in elem:
            print(val)
