import scipy
import numpy as np


def mvstdnormcdf(lower, upper, corrcoef, **kwds):
    n = len(lower)

    lower = np.array(lower)
    upper = np.array(upper)
    corrcoef = np.array(corrcoef)

    correl = np.zeros(int(n * (n - 1) / 2.0))

    if (lower.ndim != 1) or (upper.ndim != 1):
        raise ValueError('can handle only 1D bounds')
    if len(upper) != n:
        raise ValueError('bounds have different lengths')
    if n == 2 and corrcoef.size == 1:
        correl = corrcoef

    elif corrcoef.ndim == 1 and len(corrcoef) == n * (n - 1) / 2.0:
        correl = corrcoef

    elif corrcoef.shape == (n, n):
        correl = corrcoef[np.tril_indices(n, -1)]

    else:
        raise ValueError('corrcoef has incorrect dimension')

    if 'maxpts' not in kwds:
        if n > 2:
            kwds['maxpts'] = 10000 * n

    lowinf = np.isneginf(lower)
    uppinf = np.isposinf(upper)
    infin = 2.0 * np.ones(n)

    np.putmask(infin, lowinf, 0)  # infin.putmask(0,lowinf)
    np.putmask(infin, uppinf, 1)  # infin.putmask(1,uppinf)
    # this has to be last
    np.putmask(infin, lowinf * uppinf, -1)

    error, cdfvalue, inform = scipy.stats.kde.mvn.mvndst(lower, upper, infin,
                                                         correl, **kwds)
    return cdfvalue


def mvnormcdf(upper, mu, cov, lower=None, **kwds):
    '''multivariate normal cumulative distribution function

    This is a wrapper for scipy.stats.kde.mvn.mvndst which calculates
    a rectangular integral over a multivariate normal distribution.

    Parameters
    ----------
    lower, upper : array_like, 1d
       lower and upper integration limits with length equal to the number
       of dimensions of the multivariate normal distribution. It can contain
       -np.inf or np.inf for open integration intervals
    mu : array_lik, 1d
       list or array of means
    cov : array_like, 2d
       specifies covariance matrix
    optional keyword parameters to influence integration
        * maxpts : int, maximum number of function values allowed. This
             parameter can be used to limit the time. A sensible
             strategy is to start with `maxpts` = 1000*N, and then
             increase `maxpts` if ERROR is too large.
        * abseps : float absolute error tolerance.
        * releps : float relative error tolerance.

    Returns
    -------
    cdfvalue : float
        value of the integral


    Notes
    -----
    This function normalizes the location and scale of the multivariate
    normal distribution and then uses `mvstdnormcdf` to call the integration.

    See Also
    --------
    mvstdnormcdf : location and scale standardized multivariate normal cdf
    '''
    upper = np.array(upper)

    if upper.shape[0] < 2:
        return scipy.stats.norm.cdf(upper, loc=mu, scale=np.sqrt(cov))

    if lower is None:
        lower = -np.ones(upper.shape) * np.inf

    else:
        lower = np.array(lower)

    cov = np.array(cov)
    stdev = np.sqrt(np.diag(cov))  # standard deviation vector

    lower = (lower - mu) / stdev
    upper = (upper - mu) / stdev
    divrow = np.atleast_2d(stdev)
    corr = cov / divrow / divrow.T

    return mvstdnormcdf(lower, upper, corr, **kwds)
