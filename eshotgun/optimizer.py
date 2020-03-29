import os

import numpy as np
import GPy as gp

from . import test_problems
from . import batch_methods


# force numpy, blas, and mkl to use a max number of threads
n_threads = 4
os.environ["MKL_NUM_THREADS"] = "{:d}" .format(n_threads)
os.environ["NUMEXPR_NUM_THREADS"] = "{:d}" .format(n_threads)
os.environ["OMP_NUM_THREADS"] = "{:d}" .format(n_threads)
os.environ["OPENBLAS_NUM_THREADS"] = "{:d}" .format(n_threads)
os.environ["VECLIB_MAXIMUM_THREADS"] = "{:d}" .format(n_threads)


def build_and_fit_GP(Xtr, Ytr):
    # create a gp model with the training data and fit it
    kernel = gp.kern.Matern52(input_dim=Xtr.shape[1], ARD=False)
    model = gp.models.GPRegression(Xtr, Ytr, kernel, normalizer=True)

    model.constrain_positive('')
    (kern_variance, kern_lengthscale,
     gaussian_noise) = model.parameter_names()

    model[kern_variance].constrain_bounded(1e-6, 1e6, warning=False)
    model[kern_lengthscale].constrain_bounded(1e-6, 1e6, warning=False)
    model[gaussian_noise].constrain_fixed(1e-6, warning=False)

    model.optimize_restarts(optimizer='lbfgs',
                            num_restarts=10,
                            num_processes=1,
                            verbose=False)

    return model


def optimize(problem_name, run_no, batch_method, batch_method_args,
             batch_size, budget, overwrite_existing=False):

    # filename to save to and path to training data
    save_file = f'results/{problem_name:}_{run_no:}'
    save_file += f'_{batch_size:d}_{budget:}_{batch_method:}'
    for _, val in batch_method_args.items():
        save_file += f'_{val:}'
    save_file += '.npz'

    data_file = f'training_data/{problem_name:}_{run_no:}.npz'

    # check to see if the save file exists
    if os.path.exists(save_file):
        if not overwrite_existing:
            print('Save file already exists:', save_file)
            print('Set overwrite_existing to True to overwrite the run.')
            return

    # load the function's additional arguments, if there are any
    with np.load(data_file, allow_pickle=True) as data:
        if 'arr_2' in data:
            f_arguments = data['arr_2'].item()
        else:
            f_arguments = {}
    # get the test problem class and instantiate it
    f_class = getattr(test_problems, problem_name)
    f = f_class(**f_arguments)

    # map it to reside in [0, 1]^d
    f = test_problems.util.uniform_problem_wrapper(f)

    # problem characteristics
    f_lb, f_ub, f_dim, f_cf = f.lb, f.ub, f.dim, f.cf

    # load the training data - it resides in full (not unit) space
    with np.load(data_file, allow_pickle=True) as data:
        Xtr = data['arr_0']
        Ytr = data['arr_1']

        # map it down to unit space
        Xtr = (Xtr - f.real_lb) / (f.real_ub - f.real_lb)

    n_train = Ytr.size

    # try to resume the run
    if os.path.exists(save_file):
        with np.load(save_file, allow_pickle=True) as data:
            Xcontinue = data['Xtr']
            Ycontinue = data['Ytr']

        # check if it has finished
        if Ycontinue.size >= n_train + budget:
            print('Run already finished:', save_file)
            return

        # if not, do some sanity checking
        n_already_completed = Ycontinue.size - n_train

        if n_already_completed % batch_size != 0:
            print('Completed batches do not match batch size:', save_file)
            print('Number of training data:', n_train)
            print('Saved evaluations size:', Ycontinue.size)
            print(n_already_completed, 'is not completely divisible by',
                  batch_size)
            return

        # we're safe to resume the run
        Xtr = Xcontinue
        Ytr = Ycontinue
        print('Resuming the run from:', save_file)
        print('Xtr shape:', Xtr.shape)

    # GP budget
    feval_budget = 10000 * f_dim

    # get the batch method class
    batch_f = getattr(batch_methods, batch_method)

    while Xtr.shape[0] < budget + n_train:
        model = build_and_fit_GP(Xtr, Ytr)

        Xnew = batch_f(model, f_lb, f_ub, feval_budget,
                       batch_size, f_cf, **batch_method_args)

        Ynew = np.zeros((batch_size, 1))
        for i in range(batch_size):
            # try to evaluate the solutions
            while True:
                try:
                    Ynew[i] = f(Xnew[i, :])
                    break

                # if this fails, try to generate a new one - note that this
                # will only occur on the PitzDaily test problem as there are
                # times when the CFD mesh fails to converge.
                except:
                    while True:
                        Xnew[i, :] = np.random.uniform(f_lb, f_ub)
                        if (f_cf is None) or f_cf(Xnew[i, :]):
                            break

        Xtr = np.concatenate((Xtr, np.atleast_2d(Xnew)))
        Ytr = np.concatenate((Ytr, Ynew))

        batch_no = int((Xtr.shape[0] - n_train) / batch_size)

        s = 'Batch {: >3d}: fmin -> {:g}'.format(batch_no, np.min(Ytr))
        print(s)

        # save results
        np.savez(save_file, Xtr=Xtr, Ytr=Ytr, budget=budget,
                 batch_method=batch_method,
                 batch_method_args=batch_method_args,
                 batch_size=batch_size)


if __name__ == "__main__":
    problem_name = 'Branin'
    run_no = 1
    batch_method = 'eShotgun'
    batch_method_args = {'epsilon': 0.1, 'pf': False}
    batch_size = 2
    budget = 10  # exclusive of training data
    overwrite_existing = True

    optimize(problem_name, run_no, batch_method, batch_method_args,
             batch_size, budget, overwrite_existing=overwrite_existing)
