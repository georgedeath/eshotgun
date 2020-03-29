import cma
import numpy as np


from .cy_qei import cy_qEI


def qEIMCMC(m, K, incumbent,
            n_samples=10**4, obj_sense=-1):
    '''
    q-point expected improvement: Monte Carlo estimate.

    Parameters.
    ------------
    m (ndarray) : mean vector. shape = (D, 1)
    K (ndarray): covariance. shape = (D, D)
    incumbent (float): best seen solution so far
    n_samples (int): number of samples.
    obj_sense (int): whether to (-1) minimise or (1) maximise.

    returns the estimated q-point expected improvement
    '''

    L = np.linalg.cholesky(K)

    fbest = obj_sense * incumbent

    N = np.random.randn(n_samples, m.shape[0])

    Mi = m.T + (L @ N.T).T  # shape = (n_samples, q)
    Mi = np.max(obj_sense * Mi, axis=1)

    return np.mean(np.maximum(0, obj_sense * (fbest - Mi)))


# some cma wrapper function to evaluate x
def qei_feval(X, model, incumbent, n_dim, func, cf):
    # if we're given a list of solutions
    if isinstance(X, list):
        n_sol = len(X)

    # given an array of solutions
    elif isinstance(X, np.ndarray) and len(X.shape) == 2:
        n_sol = X.shape[0]

    # given a vector (i.e. 1 solution)
    else:
        X = np.reshape(X, (1, -1))
        n_sol = 1

    ei = [0] * n_sol

    got_cf = cf is not None

    for i in range(n_sol):
        # reshape each solution from (x0, ..., xD, x0, ...., xD, etc)
        # to ((x0, ..., xD), (x0, ..., xD), ...)
        x = X[i].reshape(-1, n_dim)

        # check if each of them are valid
        not_valid = False
        for xx in x:
            if got_cf and (not cf(xx)):
                not_valid = True
                break

        if not_valid:
            ei[i] = 1000
        else:
            m, K = model.predict(x, full_cov=True)

            # note we want the negative qEI here as CMA-ES minimises
            ei[i] = -func(m, K, incumbent)

    # if we've been given one solution, CMA-ES expects a float back
    if n_sol == 1:
        return ei[0]

    # else it expects a list of fevals back
    return ei


def batch_qEI(model, lb, ub, maxeval, q, cf):
    # n_samples only used if method is 'qEIMCMC'
    n_dim = lb.size

    # problem bounds
    lbq = np.tile(lb, q)
    ubq = np.tile(ub, q)

    # use the (more) analytical for a batch size of two (as it is fast enough)
    if q == 2:
        func = cy_qEI

    # else it is impractically slow so switch to the MCMC qEI version
    else:
        def func(m, K, incumbent, obj_sense=-1):
            return qEIMCMC(m, K, incumbent, 10**4, obj_sense)

    # best seen solution
    incumbent = model.Y.min()

    # cma-es options setup
    cma_options = {'bounds': [list(lbq), list(ubq)],
                   'tolfun': 1e-15,
                   'maxfevals': maxeval,
                   'verb_log': 0,
                   'verbose': 1,
                   'verb_disp': 0,
                   'CMA_stds': np.abs(ubq - lbq)}

    if cf is None:
        x0 = lambda: np.random.uniform(lbq, ubq)

    else:
        def inital_point_generator(cf, lb, ub, n_dim, q):
            def wrapper():
                x = np.zeros(n_dim * q, dtype='float')
                for i in range(q):
                    while True:
                        v = np.random.uniform(lb, ub)
                        if np.all(v >= lb) and np.all(v <= ub) and cf(v):
                            x[i * n_dim:(i + 1) * n_dim] = v
                            break
                return x
            return wrapper

        x0 = inital_point_generator(cf, lb, ub, n_dim, q)

    # run CMA-ES with bipop (small and large population sizes) for up
    # to 9 restarts (or until it runs out of budget)
    xopt, es = cma.fmin2(qei_feval, x0=x0, sigma0=0.25,
                         options=cma_options,
                         args=(model, incumbent, n_dim, func, cf),
                         bipop=True, restarts=9)

    return np.reshape(xopt, (q, n_dim))
