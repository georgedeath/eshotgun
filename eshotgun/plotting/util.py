import os
import numpy as np
import tqdm.notebook

from .. import test_problems


def load_results(results_dir, problem_names, method_names, batch_sizes,
                 budget, exp_no_start, exp_no_end):
    results = {}

    for batch_size in tqdm.notebook.tqdm(batch_sizes, desc='Batch sizes'):
        results[batch_size] = {}

        for problem_name in tqdm.notebook.tqdm(problem_names, leave=False,
                                               desc='Problems'):
            results[batch_size][problem_name] = {}

            # load the problem
            f_class = getattr(test_problems, problem_name)
            f = f_class()
            f_yopt = f.yopt

            expected_points = budget + 2 * f.dim

            for method_name in tqdm.notebook.tqdm(method_names, leave=False,
                                                  desc='Methods'):

                D = np.zeros((exp_no_end - exp_no_start + 1, expected_points))

                # get the raw results for each problem instance
                for i, run_no in enumerate(range(exp_no_start,
                                                 exp_no_end + 1)):
                    fn = f'{problem_name:}_{run_no:}'
                    fn += f'_{batch_size:}_{budget:}_{method_name:}.npz'
                    filepath = os.path.join(results_dir, fn)

                    with np.load(filepath, allow_pickle=True) as data:
                        Ytr = np.squeeze(data['Ytr'])

                    if Ytr.size > expected_points:
                        print('{:s} has too many'.format(fn)
                              + ' function evaluations: '
                              + '{:d} (budget = {:d})'.format(Ytr.size,
                                                              expected_points))

                    elif Ytr.size < expected_points:
                        print('{:s} has too few'.format(fn)
                              + ' function evaluations: '
                              + '{:d} (budget = {:d})'.format(Ytr.size,
                                                              expected_points))

                    elif np.any(np.isnan(Ytr)):
                        print('Nans found: {:s}'.format(fn))

                    else:
                        D[i, :] = Ytr

                # calculate the absolute distance to the minima
                D = np.abs(D - f_yopt)

                # calculate the best (lowest) value seen at each iteration
                D = np.minimum.accumulate(D, axis=1)

                results[batch_size][problem_name][method_name] = D

    return results
