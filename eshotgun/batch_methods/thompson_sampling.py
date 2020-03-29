import numpy as np
from numpy.random import normal as numpy_normal
from GPy.util.linalg import jitchol


def gp_sample(model, x, n_samples):
    if len(x.shape) == 1:
        x = np.reshape(x, (1, -1))
        n_points = 1
    else:
        n_points = x.shape[0]

    # special case if we're only have 1 realisation of 1 point
    if n_points == 1 and n_samples == 1:
        m, cov = model.predict(x, full_cov=False)
        L = np.sqrt(cov)
        U = numpy_normal()
        return m + L * U

    # else general case, do things properly
    m, cov = model.predict(x, full_cov=True)
    L = jitchol(cov)
    U = numpy_normal(size=(n_points, n_samples))
    return m + L @ U


def TS_RANDOM(model, lb, ub, maxeval, cf):
    # uniformly generate 'maxeval' points across lb, ub
    X = np.random.uniform(low=lb, high=ub, size=(maxeval, lb.size))
    res = np.zeros(maxeval)

    # draw realisations of the modelled function and evaluate its mean
    # value at the uniformly generated point
    got_cf = cf is not None
    for i, x in enumerate(X):
        if got_cf and (not cf(x)):
            res[i] = np.inf
        else:
            res[i] = gp_sample(model, x, n_samples=1)

    # select the point with the best (lowest) mean value to next evaluate
    best_idx = np.argmin(res)
    return X[best_idx, :]


def batch_TS(model, lb, ub, maxeval, q, cf):
    # model - gpy regression model
    # lb, ub - lower, upper bounds of modelled function
    # total number of gp locations to (cheaply) evaluate
    # q - number of batch points
    Xnew = np.zeros((q, lb.size))

    # split the evaluation budget equally between TS "runs", with any
    # remaining evaluation budget given randomly across the runs
    gp_budget = np.full(q, fill_value=maxeval // q, dtype='int')
    gp_budget[np.random.choice(q, size=maxeval % q, replace=False)] += 1

    for i, budget in enumerate(gp_budget):
        Xnew[i] = TS_RANDOM(model, lb, ub, budget, cf)

    return Xnew
