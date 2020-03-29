import numpy as np


# simple wrapper around functions to act as though they reside in [0, 1]^d
class uniform_problem_wrapper():
    def __init__(self, problem):
        self.problem = problem
        self.dim = problem.dim

        self.real_lb = problem.lb
        self.real_ub = problem.ub

        self.lb = np.zeros(self.dim)
        self.ub = np.ones(self.dim)

        self.real_cf = problem.cf
        self.set_cf()

    def __call__(self, x):
        x = np.atleast_2d(x)

        # map x back to original space
        x = x * (self.real_ub - self.real_lb)[np.newaxis, :] + self.real_lb[np.newaxis, :]

        return self.problem(x)

    def set_cf(self):
        if self.real_cf is None:
            self.cf = None
            return

        def cf_wrapper(x):
            x = np.atleast_2d(x)

            # map x back to original space
            x = x * (self.real_ub - self.real_lb)[np.newaxis, :] + self.real_lb[np.newaxis, :]

            return self.real_cf(x)

        self.cf = cf_wrapper
