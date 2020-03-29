import numpy as np
from scipy.stats import median_absolute_deviation, wilcoxon
from statsmodels.stats.multitest import multipletests


def create_table_data(results, problem_names, method_names, n_exps,
                      use_correction=True):
    """

    """
    method_names = np.array(method_names)
    n_methods = len(method_names)

    # table_data[problem_name] = {'median', 'MAD', 'stats_equal_to_best_mask'}
    table_data = {}

    for problem_name in problem_names:
        best_seen_values = np.zeros((n_methods, n_exps))

        for i, method_name in enumerate(method_names):
            # best seen evaluate at the end of the optimisation run
            best_seen_values[i, :] = results[problem_name][method_name][:, -1]

        medians = np.median(best_seen_values, axis=1)
        MADS = median_absolute_deviation(best_seen_values, axis=1)

        # best method -> lowest median value
        best_method_idx = np.argmin(medians)

        # mask of methods equivlent to the best
        stats_equal_to_best_mask = np.zeros(n_methods, dtype='bool')
        stats_equal_to_best_mask[best_method_idx] = True

        # perform wilcoxon signed rank test between best and all other methods
        p_values = []
        for i, method_name in enumerate(method_names):
            if i == best_method_idx:
                continue
            # a ValueError will be thrown if the runs are all identical,
            # therefore we can assign a p-value of 0 as they are identical
            try:
                _, p_value = wilcoxon(best_seen_values[best_method_idx, :],
                                      best_seen_values[i, :])
                p_values.append(p_value)

            except ValueError:
                p_values.append(0)

        if use_correction:
            # calculate the Holm-Bonferroni correction
            reject_hyp, _, _, _ = multipletests(p_values, alpha=0.05,
                                                method='holm')

        else:
            reject_hyp = [p < 0.05 for p in p_values]

        for reject, method_name in zip(reject_hyp, [m for m in method_names
                                       if m != method_names[best_method_idx]]):
            # if we can't reject the hypothesis that a technique is
            # statistically equivilent to the best method
            if not reject:
                idx = np.where(np.array(method_names) == method_name)[0][0]
                stats_equal_to_best_mask[idx] = True

        # store the data
        table_data[problem_name] = {'medians': medians,
                                    'MADS': MADS,
                                    'stats_equal_to_best_mask': stats_equal_to_best_mask}

    return table_data


def create_table(table_data, problem_rows, problem_paper_rows,
                 problem_dim_rows, method_names, method_names_for_table,
                 caption=''):
    """

    """

    head = r"""
  \begin{table*}[t]
  \setlength{\tabcolsep}{2pt}
  \sisetup{table-format=1.2e-1,table-number-alignment=center}
  \resizebox{1\textwidth}{!}{%
  \begin{tabular}{l | SS| SS| SS| SS| SS}"""

    foot = r"""  \end{tabular}
  }
  \vspace*{0.1mm}
  \caption{""" + caption

    foot += r"""}
  \label{tbl:synthetic_results}
  \end{table*}"""

    print(head)
    for probs, probs_paper, probs_dim in zip(problem_rows, problem_paper_rows,
                                             problem_dim_rows):

        print(r'    \toprule')
        print(r'    \bfseries Method')

        # column titles: Problem name (dim).
        print_string = ''
        for prob, dim in zip(probs_paper, probs_dim):
            print_string += r'    & \multicolumn{2}{c'
            # last column does not have a vertical dividing line
            if prob != probs_paper[-1]:
                print_string += r'|'
            print_string += r'}{\bfseries '
            print_string += r'{:s} ({:d})'.format(prob, dim)
            print_string += '} \n'

        print_string = print_string[:-2] + ' \\\\ \n'

        # column titles: Median MAD
        for prob in probs:
            print_string += r'    & \multicolumn{1}{c}{Median}'
            print_string += r' & \multicolumn{1}{c'
            # last column does not have a vertical dividing line
            if prob != probs[-1]:
                print_string += r'|'
            print_string += '}{MAD}\n'
        print_string = print_string[:-1] + '  \\\\ \\midrule'
        print(print_string)

        # results printing
        for i, (method_name, method_name_table) in enumerate(zip(method_names,
                                                                 method_names_for_table)):
            print_string = '    '
            print_string += method_name_table + ' & '

            # table_data[problem_name] = {'median', 'MAD', 'stats_equal_to_best_mask'}
            for prob in probs:
                med = '{:4.2e}'.format(table_data[prob]['medians'][i])
                mad = '{:4.2e}'.format(table_data[prob]['MADS'][i])

                best_methods = table_data[prob]['stats_equal_to_best_mask']
                best_idx = np.argmin(table_data[prob]['medians'])

                if i == best_idx:
                    med = r'\best ' + med
                    mad = r'\best ' + mad

                elif best_methods[i]:
                    med = r'\statsimilar ' + med
                    mad = r'\statsimilar ' + mad

                print_string += med + ' & ' + mad + ' & '

            print_string = print_string[:-2] + '\\\\'
            print(print_string)

        print('\\bottomrule')

    print(foot)
    print()
