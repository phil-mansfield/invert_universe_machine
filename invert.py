import symlib
import numpy as np
import scipy.optimize
import matplotlib.pyplot as plt

def dlny_dlnx(f, x, dlnx=0.001):
    """ dlny_dlnx computes the log-slope of f evlauated at x.
    """
    lny_hi = np.log(f(np.exp(np.log(x) + dlnx)))
    lny_lo = np.log(f(np.exp(np.log(x) - dlnx)))
    return (lny_hi - lny_lo)/(2*dlnx)

def invert_smhm(mhalo, alpha, beta, sigma_ms):
    """ invert_smhm inverts the stellar-mass-halo-mass relation, accounting for
    intrinsic scatter. The user must first compute the halo mass corresponding
    to a direct inversion of the no-scatter smhmr (mhalo), the slope of the
    mass function (alpha), the slope of the log-slope of the smhmr wrt stellar
    mass, and the scatter M* at a fixed Mhalo.
    """
    inv_sigma = sigma_ms/beta
    inv_mu = 10**(np.log(10)*(sigma_ms/beta)**2 * (1+alpha) + np.log10(mhalo))
    return inv_mu, inv_sigma

def highest_mvir(mstar_func, slope_limit, guess=1e12, infinity=1e20):
    """ highest_mvir computes the halo mass where the slope gets shalloe than
    slope_limit or infinity if this never happens.
    """
    if dlny_dlnx(mstar_func, infinity) > slope_limit:
        return np.inf
    return invert_func(lambda mvir: dlny_dlnx(mstar_func, mvir),
                       slope_limit, guess=guess)

def invert_func(func, y0, guess=1e12):
    """ invert_func finds x0 st func(x0) = y0. is guess is a tuple, it is
    treated as bounds.
    """
    if type(guess) == tuple:
        lo, hi = guess
        return scipy.optimize.brentq(lambda x: func(x) - y0, lo, hi)
    else:
        res = scipy.optimize.fsolve(lambda x: func(x) - y0, guess)
        return res[0]

def vir_from_mstar(log10_mstar, z, target=None, scatter=lambda mvir: 0.15,
                   slope_limit=0.4):
    """ vir_from_mstar computes virial quantities from stellar masses and
    redshifts. target is the virial quantitiy and must be "mvir" or "rvir".

    Returns a pair of arrays, the first if the output and the second is
    a boolean array which is true if the returned value is outside the valid
    range and false otherwise.
    """
    
    um_model = symlib.UniverseMachineMStarFit()

    alpha = -1.92

    n = len(log10_mstar)
    mvir = np.zeros(n)
    out_of_range = np.zeros(n, dtype=bool)
    for i in range(n):
        mstar = 10**log10_mstar[i]
        
        mstar_func = lambda mvir: um_model.m_star(mvir, z, scatter=0.0)
        mvir_max = highest_mvir(mstar_func, slope_limit)

        if np.isfinite(mvir_max) and mstar_func(mvir_max) < mstar:
            mvir[i], _ =  invert_smhm(mvir_max, alpha,
                                      slope_limit, scatter(mvir_max))
            out_of_range[i] = True
        else:
            mhalo = invert_func(mstar_func, mstar,
                                guess=(1e5, min(mvir_max, 1e17)))
            beta = dlny_dlnx(mstar_func, mhalo)
            mvir[i], _ = invert_smhm(mhalo, alpha, beta, scatter(mhalo))

    h100 = 0.7
    omega_M = 0.286

    if target == "mvir":
        return mvir, out_of_range
    elif target == "rvir":
        a_host = 1/(z + 1)
        rvir_cmov = symlib.lib.mvir_to_rvir(mvir*h100, a_host, omega_M)
        rvir_Mpc = rvir_cmov/h100 * a_host
        return rvir_Mpc*1e3, out_of_range
    else:
        assert(0)
