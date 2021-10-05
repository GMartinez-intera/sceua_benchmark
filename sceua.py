import numpy as np
import functn


def cceua(s, sf, fbl, fbu, icall, fmaxn, ifunct):
    #  This is the subroutine for generating a new point in a simplex
    #
    #   s(.,.) = the sorted simplex in order of increasing function values
    #   s(.) = function values in increasing order
    #
    # LIST OF LOCAL VARIABLES
    #   sb(.) = the best point of the simplex
    #   sw(.) = the worst point of the simplex
    #   w2(.) = the second worst point of the simplex
    #   fw = function value of the worst point
    #   ce(.) = the centroid of the simplex excluding wo
    #   snew(.) = new point generated from the simplex
    #   iviol = flag indicating if constraints are violated
    #         = 1 , yes
    #         = 0 , no

    (nps, nopt) = s.shape
    alpha = 1.0
    beta = 0.5

    #  Assign the worst points:
    sw = s[-1, :]
    fw = sf[-1]

    #  Compute the centroid of the simplex excluding the worst point:
    ce = s[0:-1, :].mean(axis=0)

    #  Attempt a reflection point
    snew = ce + alpha * (ce - sw)

    #  Check if is outside the bounds:
    ibound = 0

    s1 = snew - fbl
    idx = np.where(s1 < 0)[0]
    if idx.size > 0:
        ibound = 1

    s1 = fbu - snew
    idx = np.where(s1 < 0)[0]
    if idx.size > 0:
        ibound = 2

    if ibound >= 1:
        snew = fbl + np.random.rand(1, nopt)[0] * (fbu - fbl)

    fnew = functn.functn(ifunct, nopt, snew)
    icall = icall + 1

    #  Reflection failed; now attempt a contraction point:
    if fnew > fw:
        snew = sw + beta * (ce - sw)
        fnew = functn.functn(ifunct, nopt, snew)
        icall = icall + 1

        #  Both reflection and contraction have failed, attempt a random point;
        if fnew > fw:
            snew = fbl + np.random.rand(1, nopt)[0] * (fbu - fbl)
            fnew = functn.functn(ifunct, nopt, snew)
            icall = icall + 1

    # END OF CCE
    return snew, fnew, icall


def sceua(x0, bl, bu, maxn, kstop, pcento, peps, ngs, iseed, iniflg, ifunct):
    # This is the subroutine implementing the SCE algorithm,
    # written in Matlab by Q.Duan, 9/2004
    # translated to Python by G.Martinez, 9/2021
    #
    # Definition:
    #  x0 = the initial parameter array at the start;
    #     = the optimized parameter array at the end;
    #  f0 = the objective function value corresponding to the initial parameters
    #     = the objective function value corresponding to the optimized parameters
    #  bl = the lower bound of the parameters
    #  bu = the upper bound of the parameters
    #  iseed = the random seed number (for repetetive testing purpose)
    #  iniflg = flag for initial parameter array (=1, included it in initial
    #           population; otherwise, not included)
    #  ngs = number of complexes (sub-populations)
    #  npg = number of members in a complex
    #  nps = number of members in a simplex
    #  nspl = number of evolution steps for each complex before shuffling
    #  mings = minimum number of complexes required during the optimization process
    #  maxn = maximum number of function evaluations allowed during optimization
    #  kstop = maximum number of evolution loops before convergency
    #  percento = the percentage change allowed in kstop loops before convergency

    # LIST OF LOCAL VARIABLES
    #    x(.,.) = coordinates of points in the population
    #    xf(.) = function values of x(.,.)
    #    xx(.) = coordinates of a single point in x
    #    cx(.,.) = coordinates of points in a complex
    #    cf(.) = function values of cx(.,.)
    #    s(.,.) = coordinates of points in the current simplex
    #    sf(.) = function values of s(.,.)
    #    bestx(.) = best point at current shuffling loop
    #    bestf = function value of bestx(.)
    #    worstx(.) = worst point at current shuffling loop
    #    worstf = function value of worstx(.)
    #    xnstd(.) = standard deviation of parameters in the population
    #    gnrng = normalized geometri%mean of parameter ranges
    #    lcs(.) = indices locating position of s(.,.) in x(.,.)
    #    bound(.) = bound on ith variable being optimized
    #    ngs1 = number of complexes in current population
    #    ngs2 = number of complexes in last population
    #    iseed1 = current random seed
    #    criter(.) = vector containing the best criterion values of the last
    #                10 shuffling loops

    # global BESTX BESTF ICALL PX PF

    # Initialize SCE parameters:
    nopt = len(x0)
    npg = 2 * nopt + 1
    nps = nopt + 1
    nspl = npg
    mings = ngs
    npt = npg * ngs

    bound = bu - bl

    # Create an initial population to fill array x(npt,nopt):
    np.random.seed(iseed)

    x = np.zeros([npt, nopt])
    xf = np.empty([npt])

    for i in range(npt):
        x[i, :] = bl + np.random.rand(1, nopt) * bound

    if iniflg == 1:
        x[0, :] = x0

    nloop = 0
    icall = 0
    for i in range(npt):
        xf[i] = functn.functn(ifunct, nopt, x[i, :])
        icall = icall + 1

    f0 = xf[0]
    # Sort the population in order of increasing function values;
    idx = xf.argsort()
    xf = xf[idx]
    x = x[idx]

    # Record the best and worst points;
    bestx = x[0, :]
    bestf = xf[0]
    worstx = x[-1, :]
    worstf = xf[-1]

    # Global variables
    BESTF = [bestf]
    BESTX = [bestx]
    ICALL = [icall]

    # Compute the standard deviation for each parameter
    xnstd = x.std(axis=0)

    # Computes the normalized geometric range of the parameters
    gnrng = np.exp(np.mean(np.log((x.max(axis=0) - x.min(axis=0)) / bound)))

    print('The Initial Loop: 0')
    print('BESTF  : ' + str(bestf))
    print('BESTX  : ' + str(bestx))
    print('WORSTF : ' + str(worstf))
    print('WORSTX : ' + str(worstx))
    print(' ')

    # Check for convergency;
    if icall >= maxn:
        print('*** OPTIMIZATION SEARCH TERMINATED BECAUSE THE LIMIT')
        print('ON THE MAXIMUM NUMBER OF TRIALS ')
        print(maxn)
        print('HAS BEEN EXCEEDED.  SEARCH WAS STOPPED AT TRIAL NUMBER:')
        print(icall)
        print('OF THE INITIAL LOOP!')

    if gnrng < peps:
        print('THE POPULATION HAS CONVERGED TO A PRESPECIFIED SMALL PARAMETER SPACE')

    # Begin evolution loops:
    nloop = 0
    criter = []
    criter_change = 1e+5

    while (icall < maxn) & (gnrng > peps) & (criter_change > pcento):
        nloop = nloop + 1

        # Loop on complexes (sub-populations);
        for igs in np.arange(1, ngs + 1):
            # Partition the population into complexes (sub-populations);
            k1 = np.arange(1, npg + 1)
            k2 = (k1 - 1) * ngs + igs

            cx = x[k2 - 1, :]
            cf = xf[k2 - 1]

            # Evolve sub-population igs for nspl steps:
            for loop in np.arange(1, nspl + 1):
                # Select simplex by sampling the complex according to a linear
                # probability distribution
                lcs = np.zeros(nps)
                lcs[0] = 1
                for k3 in np.arange(2, nps + 1):
                    for iters in np.arange(1, 1000 + 1):
                        lpos = 1 + np.floor(
                            npg + 0.5 - ((npg + 0.5) ** 2 - npg * (npg + 1) * np.random.rand()) ** (1 / 2))
                        idx = np.where(lcs[0:k3 - 1] == lpos)[0]
                        if idx.size == 0:
                            break
                    lcs[k3 - 1] = lpos
                lcs = np.sort(lcs.astype(int))

                # Construct the simplex:
                s = cx[lcs - 1, :]
                sf = cf[lcs - 1]
                snew, fnew, icall = cceua(s, sf, bl, bu, icall, maxn, ifunct)
                # Replace the worst point in Simplex with the new point:
                s[-1, :] = snew
                sf[-1] = fnew

                # Replace the simplex into the complex;
                cx[lcs - 1, :] = s
                cf[lcs - 1] = sf

                # Sort the complex;
                idx = cf.argsort()
                cf = cf[idx]
                cx = cx[idx, :]
                # End of Inner Loop for Competitive Evolution of Simplexes

            # Replace the complex back into the population
            x[k2 - 1, :] = cx[k1 - 1, :]
            xf[k2 - 1] = cf[k1 - 1]
            # End of Loop on Complex Evolution;

        # Shuffled the complexes;
        idx = xf.argsort()
        xf = xf[idx]
        x = x[idx, :]

        PX = x
        PF = xf

        # Record the best and worst points;
        bestx = x[0, :]
        bestf = xf[0]
        worstx = x[-1, :]
        worstf = xf[-1]
        BESTX.append(bestx)
        BESTF.append(bestf)
        ICALL.append(icall)

        # Compute the standard deviation for each parameter
        xnstd = x.std(axis=0)

        # Computes the normalized geometric range of the parameters
        gnrng = np.exp(np.mean(np.log((x.max(axis=0) - x.min(axis=0)) / bound)))

        print('Evolution Loop: ' + str(nloop) + '  - Trial - ' + str(icall))
        print('BESTF  : ' + str(bestf))
        print('BESTX  : ' + str(bestx))
        print('WORSTF : ' + str(worstf))
        print('WORSTX : ' + str(worstx))
        print(' ')

        # Check for convergency;
        if icall >= maxn:
            print('*** OPTIMIZATION SEARCH TERMINATED BECAUSE THE LIMIT')
            print('ON THE MAXIMUM NUMBER OF TRIALS ' + str(maxn) + ' HAS BEEN EXCEEDED!')

        if gnrng < peps:
            print('THE POPULATION HAS CONVERGED TO A PRESPECIFIED SMALL PARAMETER SPACE')

        criter.append(bestf)
        if nloop >= kstop:
            criter_change = abs(criter[nloop - 1] - criter[nloop - kstop + 1 - 1]) * 100  # check
            criter_change = criter_change / np.mean(np.abs(np.array(criter[nloop - kstop + 1 - 1:nloop - 1 + 1])))
            if criter_change < pcento:
                print('THE BEST POINT HAS IMPROVED IN LAST ' + str(kstop) + ' LOOPS BY \n'
                      + 'LESS THAN THE THRESHOLD ' + str(pcento) + '%')
                print('CONVERGENCY HAS ACHIEVED BASED ON OBJECTIVE FUNCTION CRITERIA!!!')
                # break

        # End of the Outer Loops

    print('SEARCH WAS STOPPED AT TRIAL NUMBER: ' + str(icall))
    print('NORMALIZED GEOMETRIC RANGE = ' + str(gnrng))
    print('THE BEST POINT HAS IMPROVED IN LAST ' + str(kstop) + ' LOOPS BY ' + str(criter_change) + '%')

    # END of Subroutine sceua
    return bestx, bestf, icall
