import cma
import nlopt
import scipy
import warnings
import numpy as np

from pyDOE2.doe_lhs import lhs


def aquisition_DIRECT(model, aq_func, cf, aq_kwargs={}):
    dim = model.X.shape[1]
    cf_info = CF_INFO(cf)

    def f(X):
        # if there's a constraint function, evaluate it
        if cf_info.not_valid(X):
            return cf_info.bad_value

        X = X.reshape(-1, dim)

        pred_mean, pred_var = model.predict(X, full_cov=False)
        pred_std = np.sqrt(pred_var)

        # negative as optimisers are minimising
        return -np.squeeze(aq_func(pred_mean, pred_std, **aq_kwargs))

    return f


def aquisition_LBFGSB(model, aq_func, cf, aq_kwargs={}):
    dim = model.X.shape[1]
    cf_info = CF_INFO(cf)

    def f(X):
        # if there's a constraint function, evaluate it
        if cf_info.not_valid(X):
            return cf_info.bad_value

        X = X.reshape(-1, dim)

        pred_mean, pred_var = model.predict(X, full_cov=False)
        pred_std = np.sqrt(pred_var)

        # negative as optimisers are minimising
        return -np.squeeze(aq_func(pred_mean, pred_std, **aq_kwargs))

    return f


def aquisition_CMAES(model, aq_func, cf=None, aq_kwargs={}):
    dim = model.X.shape[1]
    cf_info = CF_INFO(cf)

    def f(X):
        # if X is a list, assume that each list element is either
        # a float or a numpy array -> convert to (N, ndim) numpy array
        if isinstance(X, list):
            X = np.reshape(np.array(X), (len(X), dim))

        # else must be a numpy array so we have to assume that it is (N, ndim)
        elif len(X.shape) != 2:
            X = np.atleast_2d(X)

        pred_mean, pred_var = model.predict(X, full_cov=False)
        pred_std = np.sqrt(pred_var)

        # negative as optimisers are minimising
        aq_res = -aq_func(pred_mean, pred_std, **aq_kwargs)
        aq_res = aq_res.ravel().tolist()

        # evaluate constraint function for each decision vector
        for i in range(len(aq_res)):
            if cf_info.not_valid(X[i, :]):
                aq_res[i] = cf_info.bad_value.flat[0]

        if len(aq_res) == 1:
            return aq_res[0]

        return aq_res

    return f


class CF_INFO:
    def __init__(self, cf):
        self.cf = cf
        self.got_cf = cf is not None
        self.bad_value = np.array(np.inf)

    def not_valid(self, X):
        return self.got_cf and (not self.cf(X))


def minimise_with_DIRECT(f, lb, ub, maxeval=5000, cf=None, ftol_abs=1e-15):
    dim = lb.size

    # define a direct optimisation instance
    opt = nlopt.opt(nlopt.GN_DIRECT_L_RAND, dim)
    opt.set_min_objective(f)

    # set the lower and upper bounds
    opt.set_lower_bounds(lb)
    opt.set_upper_bounds(ub)

    # set max evaluations and function tolerance - this latter option
    # has a big performance influence when optimising in a small region
    opt.set_maxeval(maxeval)
    opt.set_ftol_abs(ftol_abs)

    # perform the optimisation
    xopt = opt.optimize(np.random.uniform(lb, ub))

    return xopt


def minimise_with_CMAES(f, lb, ub, maxeval=5000, cf=None, ftol_abs=1e-15):
    # set the options
    cma_options = {'bounds': [list(lb), list(ub)],
                   'tolfun': ftol_abs,
                   'maxfevals': maxeval,
                   'verb_disp': 0,
                   'verb_log': 0,
                   'verbose': -1,
                   'CMA_stds': np.abs(ub - lb),
                   }

    if cf is None:
        x0 = lambda: np.random.uniform(lb, ub)

    else:
        def inital_point_generator(cf, lb, ub):
            def wrapper():
                while True:
                    x = np.random.uniform(lb, ub)
                    if np.all(x >= lb) and np.all(x <= ub) and cf(x):
                        return x
            return wrapper

        class feas_func:
            def __init__(self, cf):
                self.cf = cf
                self.c = 0

            def __call__(self, x, f):
                if self.c > 10000:
                    return True

                is_feas = self.cf(x)

                if not is_feas:
                    self.c += 1

                return is_feas

        is_feasible = feas_func(cf)
        cma_options['is_feasible'] = is_feasible
        x0 = inital_point_generator(cf, lb, ub)

    # ignore warnings about flat fitness (i.e starting in a flat EI location)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')

        # run CMA-ES with bipop (small and large population sizes) for up to
        # 9 restarts (or until it runs out of budget)
        xopt, _ = cma.fmin2(f, x0=x0, sigma0=0.25, options=cma_options,
                            bipop=True, restarts=9)
        warnings.resetwarnings()

    return xopt


def minimise_with_LBFGSB(f, lb, ub, maxeval=5000, cf=None, ftol_abs=1e-15):
    dim = lb.size

    # number of optimisation runs and *estimated* number of L-BFGS-B
    # function evaluations per run; note this was calculate empirically and
    # may not be true for all functions.
    N_opt_runs = 10
    fevals_assumed_per_run = 100

    N_LHS_samples = maxeval-(N_opt_runs*fevals_assumed_per_run)
    if N_LHS_samples <= N_opt_runs:
        N_LHS_samples = N_opt_runs

    # initially perform a grid search using LHS (maximin) for N_LHS_samples
    x0_points = lhs(dim, samples=N_LHS_samples, criterion='m')
    x0_points = x0_points * (ub - lb)[np.newaxis, :] + lb[np.newaxis, :]

    fx = f(x0_points).ravel()

    # select the top N_opt_runs to evaluate with L-BFGS-B
    x0_points = x0_points[np.argsort(fx)[:N_opt_runs], :]

    # Find the best optimum by starting from n_restart different random points.
    bounds = [(l, b) for (l, b) in zip(lb, ub)]

    # storage for the best found location (xb) and its function value (fx)
    xb = np.zeros((N_opt_runs, dim))
    fx = np.zeros((N_opt_runs, 1))

    # ensure we're using a good stopping criterion
    # ftol = factr * numpy.finfo(float).eps
    factr = ftol_abs / np.finfo(float).eps

    for i, x0 in enumerate(x0_points):
        xb[i, :], fx[i, :], d = scipy.optimize.fmin_l_bfgs_b(f,
                                                             x0=x0,
                                                             bounds=bounds,
                                                             approx_grad=True,
                                                             factr=factr)

    best_idx = np.argmin(fx.flat)
    return xb[best_idx, :]


def minimise_with_LBFGSB_once(f, lb, ub, x0, maxeval=5000, ftol_abs=1e-15):
    bounds = [(l, b) for (l, b) in zip(lb, ub)]

    # ensure we're using a good stopping criterion
    # ftol = factr * numpy.finfo(float).eps
    factr = ftol_abs / np.finfo(float).eps

    xb, fx, d = scipy.optimize.fmin_l_bfgs_b(f, x0=x0, bounds=bounds,
                                             approx_grad=True, factr=factr)

    return xb

