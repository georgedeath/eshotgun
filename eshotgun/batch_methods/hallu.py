import numpy as np

from .sequential_functions import UCB, EI
from .acquisition_optimisers import (aquisition_CMAES, minimise_with_CMAES,
                                     aquisition_LBFGSB, minimise_with_LBFGSB)


def hallu(model, lb, ub, maxeval, q, cf, method='EI'):
    # model - gpy regression model
    # lb, ub - lower, upper bounds of modelled function
    # total number of gp locations to (cheaply) evaluate
    # q - number of batch points
    # method - either 'EI' or 'UCB'
    n_dim = lb.size
    Xnew = np.zeros((q, n_dim))

    # split the evaluation budget equally between each of the "runs", with any
    # remaining evaluation budget given randomly across the runs
    gp_budget = np.full(q, fill_value=maxeval // q)
    gp_budget[np.random.choice(q, size=maxeval % q, replace=False)] += 1

    # we can only use CMA-ES on 2 or more dimensional functions
    if n_dim > 1:
        opt_acq_func, opt_caller = aquisition_CMAES, minimise_with_CMAES
    else:
        opt_acq_func, opt_caller = aquisition_LBFGSB, minimise_with_LBFGSB

    for i, budget in enumerate(gp_budget):
        # create function to optimize
        if method == 'EI':
            aq_kwargs = {'y_best': np.min(model.Y.ravel())}
            fopt = opt_acq_func(model, EI, cf, aq_kwargs=aq_kwargs)
        else:
            fopt = opt_acq_func(model, UCB, cf)

        # run optimiser a max of budget/q evaluations of the gp to
        # select a new point to expensively evaluate
        Xnew[i] = opt_caller(fopt, lb, ub, int(budget), cf=cf)

        # predict the value of the selected location
        mpred, _ = model.predict(np.atleast_2d(Xnew[i]), full_cov=False)

        # add it to the model with a predicted value of the GP's prediction.
        # this is the "hallucination" method and only changes the variance
        # of the model (as the prediction is just the mean prediction)
        XX = np.concatenate((model.X, np.atleast_2d(Xnew[i])))
        YY = np.concatenate((model.Y, mpred))

        # update the model to optimize, taking into account new location
        model.set_XY(XX, YY)

    return Xnew
