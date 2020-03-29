"""
Code adapted from: https://github.com/a5a/asynchronous-BO
Full paper: http://proceedings.mlr.press/v97/alvi19a.html

Asynchronous Batch Bayesian Optimisation with Improved Local Penalisation:
@InProceedings{pmlr-v97-alvi19a,
  title = 	  {Asynchronous Batch {B}ayesian Optimisation with
               Improved Local Penalisation},
  author = 	  {Alvi, Ahsan and Ru, Binxin and Calliess, Jan-Peter and
               Roberts, Stephen and Osborne, Michael A.},
  booktitle = {Proceedings of the 36th International Conference on
               Machine Learning},
  pages = 	  {253--262},
  year = 	  {2019},
  volume = 	  {97},
  month = 	  {09--15 Jun},
  publisher = {PMLR}
}
"""
from typing import Optional

import numpy as np
from scipy.stats import norm

from .acquisition_optimisers import minimise_with_LBFGSB_once
from .egreedy_shotgun import estimate_L


def PLAyBOOK(model, f_lb, f_ub, feval_budget, q, cf, acq_func='EI'):
    n_dim = f_lb.size

    # storage for the new points to expensively evaluate and
    # the estimated Lipschitz constant centred on them
    Xnew = np.zeros((q, n_dim))
    L = np.zeros(q)

    # best seen function evaluate
    M = np.min(model.Y.flat)

    # GP length scale
    ls = model.kern.lengthscale[0]

    # create an acquisition function
    if acq_func == 'EI':
        acqf = EI(model, M)

    # amount of points to sample for each location as defined by PLAyBOOK
    N_POINTS = 3000

    for i in range(q):
        # sample N_POINTS uniformly across space as in
        # the PLAyBOOK paper, Section 6.1
        Xn = np.random.uniform(f_lb, f_ub, size=(N_POINTS, n_dim))

        # if we're in the first iteration, there's nothing to penalize and
        # so optimize over the normal acquisition function
        if i == 0:
            func = acqf.evaluate
        else:
            # build HLP and optimise over the penalized landscape
            HLP = HardMinAwareConeAcquisition(model,
                                              acqf,
                                              x_batch=Xnew[:i - 1, :],
                                              L=L[:i - 1],
                                              best=M)
            func = HLP.evaluate

        # wrap the constraint function around the penalizer
        func = CF_WRAPPER(func, cf, n_dim)

        # evaluate the samples
        af = func(Xn)

        # select the best 5 to optimize with L-BFGS-B
        best_inds = np.argsort(af)[::-1][:5]  # largest first

        best_acq = af[best_inds[0]]
        best_x = Xn[best_inds[0], :]

        for x0 in Xn[best_inds, :]:
            # minimise the negative acq (i.e. maximise the acq)
            xe = minimise_with_LBFGSB_once(lambda x: -func(x),
                                           f_lb, f_ub,
                                           x0, maxeval=1000)
            fe = func(xe)

            if fe > best_acq:
                best_acq = fe
                best_x = xe

        Xnew[i] = best_x
        L[i] = estimate_L(model, best_x, ls, lb=f_lb, ub=f_ub)

    return Xnew


def LP(model, f_lb, f_ub, feval_budget, q, cf, acq_func='EI'):
    n_dim = f_lb.size

    # storage for the new points to expensively evaluate
    Xnew = np.zeros((q, n_dim))

    # estimate the global Lipschitz constant - using np.inf as the
    # length-scale pushes the boundaries of the region to evaluate L
    # in to be the entire space (as it clips to [f_lb, f_ub])
    L = estimate_L(model, np.random.uniform(f_lb, f_ub),
                   np.inf, lb=f_lb, ub=f_ub)
    # best seen function evaluate
    M = np.min(model.Y.flat)

    # create an acquisition function
    if acq_func == 'EI':
        acqf = EI(model, M)

    # amount of points to sample for each location as defined by PLAyBOOK
    N_POINTS = 3000

    for i in range(q):
        # sample N_POINTS uniformly across space as in
        # the PLAyBOOK paper, Section 6.1
        Xn = np.random.uniform(f_lb, f_ub, size=(N_POINTS, n_dim))

        # if we're in the first iteration, there's nothing to penalize and
        # so optimize over the normal acquisition function
        if i == 0:
            func = acqf.evaluate

        else:
            # build LP and optimise over the penalized landscape
            LPA = LocallyPenalisedAcquisition(model,
                                              acqf,
                                              x_batch=Xnew[:i - 1, :],
                                              L=L,
                                              best=M)
            func = LPA.evaluate

        # wrap the constraint function around the penalizer
        func = CF_WRAPPER(func, cf, n_dim)

        # evaluate the samples
        af = func(Xn)

        # select the best 5 to optimize with L-BFGS-B
        best_inds = np.argsort(af)[::-1][:5]  # largest first

        best_acq = af[best_inds[0]]
        best_x = Xn[best_inds[0], :]

        for x0 in Xn[best_inds, :]:
            # minimise the negative acq (i.e. maximise the acq)
            xe = minimise_with_LBFGSB_once(lambda x: -func(x),
                                           f_lb, f_ub,
                                           x0, maxeval=1000)
            fe = func(xe)

            if fe > best_acq:
                best_acq = fe
                best_x = xe

        Xnew[i] = best_x

    return Xnew


class CF_WRAPPER:
    def __init__(self, func, cf, n_dim):
        self.n_dim = n_dim
        self.func = func
        self.cf = cf
        self.got_cf = cf is not None

    def __call__(self, X):
        if not self.got_cf:
            return self.func(X)

        X = np.reshape(X, (-1, self.n_dim))
        N = X.shape[0]

        valid_mask = np.ones((N), dtype='bool')
        fx = np.zeros((N))

        # find the decision vectors that fail the constraint function
        for i in range(N):
            if not self.cf(X[i, :]):
                valid_mask[i] = False
                fx[i] = -np.inf

        # evaluate those that pass with the penalizer
        fx[valid_mask] = self.func(X[valid_mask, :])

        # deal with single/multiple decision vectors
        if N == 1:
            return fx[0]
        else:
            return fx


# ---- Functions below from https://github.com/a5a/asynchronous-BO
# MIT License
#
# Copyright (c) 2019 Ahsan Alvi
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

class AcquisitionFunction(object):
    """
    Base class for acquisition functions. Used to define the interface
    """

    def __init__(self, surrogate=None, verbose=False):
        self.surrogate = surrogate
        self.verbose = verbose

    def evaluate(self, x: np.ndarray, **kwargs) -> np.ndarray:
        raise NotImplementedError


class EI(AcquisitionFunction):
    """
    Expected improvement acquisition function for a Gaussian model

    Model should return (mu, var)
    """

    def __init__(self, surrogate, best: np.ndarray, verbose=False):
        self.best = best
        super().__init__(surrogate, verbose)

    def __str__(self) -> str:
        return "EI"

    def evaluate(self, x: np.ndarray, **kwargs) -> np.ndarray:
        """
        Evaluates the EI acquisition function.

        Parameters
        ----------
        x
            Input to evaluate the acquisition function at

        """
        if self.verbose:
            print("Evaluating EI at", x)
        mu, var = self.surrogate.predict(np.atleast_2d(x))
        var = np.clip(var, 1e-8, np.inf)

        s = np.sqrt(var)
        gamma = (self.best - mu) / s
        return (s * gamma * norm.cdf(gamma) + s * norm.pdf(gamma)).flatten()


class PenalisedAcquisition(AcquisitionFunction):
    """Penalised acquisition function parent class

    Parameters
    ----------
    surrogate

    acq
        An instance of AcquisitionFunction, e.g. EI, UCB, etc

    x_batch
        Locations already in the batch

    best
        Best value so far of the function

    verbose
    """

    def __init__(self, surrogate,
                 acq: AcquisitionFunction,
                 x_batch: np.ndarray,
                 best: Optional[np.ndarray] = None,
                 verbose=False):
        super().__init__(surrogate, verbose)

        if best is None:
            try:
                self.best = acq.best
            except NameError:
                self.best = None
            except AttributeError:
                self.best = None
        else:
            self.best = best

        # shape is (1 x n_samples), or float
        if isinstance(best, np.ndarray):
            self.best = self.best.reshape(1, -1)

        self.acq = acq
        self.x_batch = x_batch

    def __str__(self) -> str:
        return f"{self.acq.__str__}-LP{len(self.x_batch)}"

    def evaluate(self, x, **kwargs) -> np.ndarray:
        """Evaluate the penalised acquisition function.

        Note that the result is log(acq), as this makes adding in the
        penalizers easier and numerically more stable. The resulting
        location of the optimum remains the same

        Parameters
        ----------
        x
            Location(s) to evaluate the acquisition function at

        Returns
        -------
        np.ndarray
            Value(s) of the acquisition function at x
        """
        out = self._penalized_acquisition(x)
        return out

    def _penalized_acquisition(self, x):
        raise NotImplementedError


class LocallyPenalisedAcquisition(PenalisedAcquisition):
    """LP Acquisition function for use in Batch BO via Local Penalization
    Parameters
    ----------
    surrogate
    acq
        An instance of AcquisitionFunction, e.g. EI, UCB, etc
    L
        Estimate of the Lipschitz constant
    x_batch
        Locations already in the batch
    best
        Best value so far of the function
    transform
        None or softplus
    verbose
    """

    def __init__(self, surrogate,
                 acq: AcquisitionFunction,
                 x_batch: np.ndarray,
                 L,
                 best: Optional[np.ndarray] = None,
                 transform='softplus',
                 verbose=False):

        super().__init__(surrogate,
                         acq,
                         x_batch,
                         best=best,
                         verbose=verbose)

        self.L = L

        if transform is None:
            self.transform = 'none'
        else:
            self.transform = transform

        self.r_x0, self.s_x0 = self._hammer_function_precompute()

    def _hammer_function_precompute(self):
        """
        Pre-computes the parameters of a penalizer centered at x0.
        """
        x0 = self.x_batch
        best = self.best
        surrogate = self.surrogate
        L = self.L

        assert x0 is not None

        if len(x0.shape) == 1:
            x0 = x0[None, :]
        m = surrogate.predict(x0)[0]
        pred = surrogate.predict(x0)[1].copy()
        pred[pred < 1e-16] = 1e-16
        s = np.sqrt(pred)
        r_x0 = np.abs(m - best) / L
        s_x0 = s / L
        r_x0 = r_x0.flatten()
        s_x0 = s_x0.flatten()
        return r_x0, s_x0

    def _hammer_function(self, x, x0, r, s):
        '''
        Creates the function to define the exclusion zones
        '''
        return norm.logcdf((np.sqrt(
            (np.square(
                np.atleast_2d(x)[:, None, :]
                - np.atleast_2d(x0)[None, :, :])).sum(-1)) - r) / s)

    def _penalized_acquisition(self, x):
        '''
        Creates a penalized acquisition function using 'hammer' functions
        around the points collected in the batch
        .. Note:: the penalized acquisition is always mapped to the log
        space. This way gradients can be computed additively and are more
        stable.
        '''
        fval = self.acq.evaluate(x)
        x_batch = self.x_batch
        r_x0 = self.r_x0
        s_x0 = self.s_x0

        if self.transform == 'softplus':
            fval_org = fval.copy()
            fval = np.log1p(np.exp(fval_org))
        elif self.transform == 'none':
            fval = fval + 1e-50

        if x_batch is not None:
            log_fval = np.log(fval)
            h_vals = self._hammer_function(x, x_batch, r_x0, s_x0)
            log_fval += h_vals.sum(axis=-1)
            fval = np.exp(log_fval)
        return fval


class HardMinAwareConeAcquisition(PenalisedAcquisition):
    """HLP Acquisition function for use in Batch BO

    Cone with information on y_min

    Parameters
    ----------
    surrogate

    acq
        An instance of AcquisitionFunction, e.g. EI, UCB, etc

    L
        Estimate of the Lipschitz constant

    x_batch
        Locations already in the batch

    best
        Best value so far of the function

    transform
        None or softplus

    verbose
    """

    def __init__(self, surrogate,
                 acq: AcquisitionFunction,
                 x_batch: np.ndarray,
                 L,
                 best: Optional[np.ndarray] = None,
                 transform='softplus',
                 verbose=False,
                 **kwargs):

        super().__init__(surrogate,
                         acq,
                         x_batch,
                         best=best,
                         verbose=verbose)

        self.L = L

        if transform is None:
            self.transform = 'none'
        else:
            self.transform = transform

        self.r_mu, self.r_std = self._cone_function_precompute()

    def _cone_function_precompute(self):
        x0 = self.x_batch
        L = self.L
        M = self.best
        mu, var = self.surrogate.predict(x0)
        r_mu = (mu.flatten() - M) / L
        r_std = np.sqrt(var.flatten()) / L

        r_mu = r_mu.flatten()
        r_std = r_std.flatten()
        return r_mu, r_std

    def _cone_function(self, x, x0):
        """
        Creates the function to define the exclusion zones

        Using half the Lipschitz constant as the gradient of the penalizer.

        We use the log of the penalizer so that we can sum instead of multiply
        at a later stage.
        """
        r_mu = self.r_mu
        r_std = self.r_std

        x_norm = np.sqrt(np.square(
            np.atleast_2d(x)[:, None, :] - np.atleast_2d(x0)[None, :, :]).sum(
            -1))
        norm_jitter = 0
        return 1 / (r_mu + r_std) * (x_norm + norm_jitter)

    def _penalized_acquisition(self, x):
        '''
        Creates a penalized acquisition function using the 4th norm between
        the acquisition function and the cone
        '''
        fval = self.acq.evaluate(x)
        x_batch = self.x_batch

        if self.transform == 'softplus':
            fval_org = fval.copy()
            fval = np.log1p(np.exp(fval_org))
        elif self.transform == 'none':
            fval = fval + 1e-50

        if x_batch is not None:
            h_vals = self._cone_function(x, x_batch).prod(-1)
            h_vals = h_vals.reshape([1, -1])
            clipped_h_vals = np.linalg.norm(
                np.concatenate((h_vals,
                                np.ones(h_vals.shape)), axis=0), -5,
                axis=0)

            fval *= clipped_h_vals

        return fval
