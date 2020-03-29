"""
Command-line parser for running experiments.

usage: batch_simulation_script.py [-h] -p PROBLEM_NAME -r RUN_NO -s BATCH_SIZE
                                  -b BUDGET -a BATCH_METHOD
                                  [-aa BATCH_METHOD_ARGS [BATCH_METHOD_ARGS ...]]

epslion-shotgun optimisation experimental evaluation
--------------------------------------------
Example:
    Running the eshotgun method with pareto front selection with a value of
    epsilon = 0.1 and using Pareto front selection on the WangFreitas
    test problem with initial poitns set #1, a budget of 200 expensive
    function evaluations and a batch size of 10:
    > python batch_simulation_script.py -p WangFreitas -r 1 -s 10 -b 200 -a eShotgun -aa epsilon=0.1 pf=True

    Running TS on push4 method (note the lack of -aa argument):
    > python batch_simulation_script.py -p push4 -r 1 -s 10 -b 200 -a TS

optional arguments:
  -h, --help            show this help message and exit
  -p PROBLEM_NAME       Test problem name. e.g. Branin, logGSobol
  -r RUN_NO             Run number Note that the corresponding npz file
                        containing the initial training locations must exist
                        in the "training_data" directory.
  -s BATCH_SIZE         Batch size. e.g. 2, 5, 10, 20
  -b BUDGET             Budget (including training points).
  -a BATCH_METHOD       Acquisition function name. e.g: TS, qEI hallu
                        (hallucination), eShotgun, PLAyBOOK or LP.
  -aa BATCH_METHOD_ARGS [BATCH_METHOD_ARGS ...]
                        Acquisition function parameters, must be in pairs of
                        parameter=values, e.g. for the e-shotgun methods:
                        epsilon=0.1 pf=False 
                        [Note: not needed for methods without arguments]
"""
import argparse as ap
from eshotgun import optimize

if __name__ == "__main__":
    parser = ap.ArgumentParser(formatter_class=ap.RawDescriptionHelpFormatter,
                               description='''
epsilon-shotgun optimisation experimental evaluation
--------------------------------------------
Example:
    Running the eshotgun method with Pareto front selection with a value of
    epsilon = 0.1 and using Pareto front selection on the WangFreitas
    test problem with initial points set #1, a budget of 200 expensive
    function evaluations and a batch size of 10:
    > python batch_simulation_script.py -p WangFreitas -r 1 -s 10 -b 200 -a eShotgun -aa epsilon=0.1 pf=True

    Running the hallucinate method with EI on push4 method
    > python batch_simulation_script.py -p push4 -r 1 -s 10 -b 200 -a hallu -aa method=EI

    Running TS on push4 method (note the lack of -aa argument):
    > python batch_simulation_script.py -p push4 -r 1 -s 10 -b 200 -a TS
''')

    parser.add_argument('-p',
                        dest='problem_name',
                        type=str,
                        help='Test problem name. e.g. Branin, logGSobol',
                        required=True)

    parser.add_argument('-r',
                        dest='run_no',
                        type=int,
                        help='Run number'
                             + ' Note that the corresponding npz file'
                             + ' containing the initial training locations'
                             + ' must exist in the "training_data" directory.',
                        required=True)

    parser.add_argument('-s',
                        dest='batch_size',
                        type=int,
                        help='Batch size. e.g. 2, 5, 10, 20',
                        required=True)

    parser.add_argument('-b',
                        dest='budget',
                        type=int,
                        help='Budget (including training points).',
                        required=True)

    parser.add_argument('-a',
                        dest='batch_method',
                        type=str,
                        help='Acquisition function name. e.g: TS, qEI'
                             + ' hallu (hallucination), eShotgun,'
                             + ' PLAyBOOK or LP.',
                        required=True)

    parser.add_argument('-aa',
                        dest='batch_method_args',
                        nargs='+',
                        help='Acquisition function parameters, must be in'
                             + ' pairs of parameter=values,'
                             + ' e.g. for the e-shotgun methods:'
                             + ' epsilon=0.1 pf=False '
                             + '[Note: not needed for methods without arguments]',
                        required=False)

    # parse the args so they appear as a.argname, eg: a.budget
    a = parser.parse_args()

    # convert the batch method args into a dict
    batch_method_args = {}
    if a.batch_method_args is not None:
        for kv in a.batch_method_args:
            k, v = kv.split('=')

            # float parsing
            try:
                batch_method_args[k] = float(v)
                continue
            except ValueError:
                pass

            # boolean parsing
            if v.lower() == 'true':
                v = True
            elif v.lower() == 'false':
                v = False

            batch_method_args[k] = v

    # perform the experiment
    optimize(a.problem_name,
             a.run_no,
             a.batch_method,
             batch_method_args,
             a.batch_size,
             a.budget,
             overwrite_existing=True)
