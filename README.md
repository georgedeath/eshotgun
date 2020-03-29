
## ε-shotgun: ε-greedy Batch Bayesian Optimisation

This repository contains the Python3 code for the ε-shotgun method presented in:
> George De Ath, Richard M. Everson, Jonathan E. Fieldsend, and Alma A. M. Rahat. 2020. ε-shotgun : ε-greedy Batch Bayesian Optimisation. In Genetic and Evolutionary Computation Conference (GECCO ’20), July 8–12, 2020, Cancún, Mexico. ACM, New York, NY, USA, 9 pages. 
> **Paper:** https://doi.org/10.1145/3377930.3390154
> **Preprint:** https://arxiv.org/abs/2002.01873

The repository also contains all training data used for the initialisation of each of the 51 optimisation runs carried to evaluate each method, the optimisation results of each of the runs on each of the methods evaluated, and a jupyter notbook to generate the figures and tables in the paper.

The remainder of this document details:
- The steps needed to install the package and related python modules on your system: [docker](#installation-docker) / [manual](#installation-manual)
- The format of the [training data](#training-data) and [saved runs](#optimisation-results).
- How to [repeat the experiments](#reproduction-of-experiments).
- How to [reproduce the figures in the paper](#reproduction-of-figures-and-tables-in-the-paper).

### Citation
If you use any part of this code in your work, please cite:
```bibtex
@inproceedings{death:eshotgun,
	author = {George {De Ath} and Richard M. Everson and Jonathan E. Fieldsend and Alma A. M. Rahat},
	title = {ε-shotgun : ε-greedy Batch Bayesian Optimisation},
	year = {2020},
	publisher = {Association for Computing Machinery},
	address = {New York, NY, USA},
	url = {https://doi.org/10.1145/3377930.3390154},
	doi = {10.1145/3377930.3390154},
	booktitle = {Proceedings of the Genetic and Evolutionary Computation Conference},
}
```

### Installation (docker)
The easiest method to automatically set up the environment needed for the optimisation library to run and to repeat the experiments carried out in this work is to use [docker](http://www.docker.com). Install instructions for docker for many popular operating systems are can be found [here](https://docs.docker.com/install/). Once docker has been installed, the docker container can be download and ran as follows:
```bash
> # download the docker container
> docker pull georgedeath/eshotgun
> # run the container
> docker run -it georgedeath/eshotgun
Welcome to the OpenFOAM v5 Docker Image
..
```
Once the above commands have been ran you will be in the command prompt of the container, run the following commands to test the optimzer by running the ε-shotgun with ε = 0.1 with a batch size of 2
and a budget of 10 function evaluations (CTRL+C to prematurely halt the run):
```bash
> python -m eshotgun.optimizer
..
```

### Installation (manual)
Manual installation is straight-forward for the optimisation library apart from the configuration of the PitzDaily test problem due to the installation and compilation of [OpenFOAM®](http://www.openfoam.com). Note that if you do not wish to use the PitzDaily test problem then the library will work fine without the optional instructions included at the end of this section. The following instructions will assume that [Anaconda3](https://docs.anaconda.com/anaconda/install/) has been installed and that you are running the following commands from the command prompt/console:

```bash
> # clone git repository
> git clone https://bitbucket.org/georgedeath/eshotgun /es
> cd /es
> # install python packages
> conda create -n eshotgun scipy numpy matplotlib statsmodels swig jupyter 
> conda activate eshotgun
> conda install pygmo --channel conda-forge
> pip install -r requirements.txt
> # compile qEI cython instructions
> cd /es/eshotgun/batch_methods/
> python setup.py build_ext --inplace
> cd /es
```
Note that, on windows, to install `swig` and `pygame` it may be necessary to also install [Visual C++ build tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/).

Now follow the linked instructions to [install OpenFOAM5](https://openfoamwiki.net/index.php/Installation/Linux/OpenFOAM-5.x/Ubuntu) (this will take 30min - 3hours to install). Note that this has only been tested with the Ubuntu 12.04 and 18.04 instructions. Once this has been successfully installed, the command `of5x` has to be ran before the PitzDaily test problem can be evaluated.

Finally, compile the pressure calculation function and check that the test problem works correctly:
```bash
> of5x
> cd /es/eshotgun/test_problems/Exeter_CFD_Problems/data/PitzDaily/solvers/
> wmake calcPressureDifference
> # test the PitzDaily solver
> cd /es
> python -m eshotgun.test_problems.pitzdaily
PitzDaily successfully instantiated..
Generated valid solution, evaluating..
Fitness value: [0.24748876]
```
Please ignore errors like `Getting LinuxMem: [Errno 2] No such file or directory: '/proc/621/status` as these are from OpenFOAM and do not impact the optimisation process.

### Training data
The initial training locations for each of the 51 sets of [Latin hypercube](https://www.jstor.org/stable/1268522) samples are located in the `training_data` directory in this repository with the filename structure `ProblemName_number`, e.g. the first set of training locations for the Branin problem is stored in `Branin_1.npz`. Each of these files is a compressed numpy file created with [numpy.savez](https://docs.scipy.org/doc/numpy/reference/generated/numpy.savez.html). It has two [numpy.ndarrays](https://docs.scipy.org/doc/numpy/reference/generated/numpy.ndarray.html) containing the 2*D initial locations and their corresponding fitness values. To load and inspect these values use the following instructions:
```python
> cd /es
> python
>>> import numpy as np
>>> with np.load('training_data/Branin_1.npz') as data:
	Xtr = data['arr_0']
	Ytr = data['arr_1']
>>> Xtr.shape, Ytr.shape
((4, 2), (4, 1))
```
The robot pushing test problems (push4 and push8) have a third array `'arr_2'`  that contains their instance-specific parameters:
```python
> cd /es
> python
>>> import numpy as np
>>> with np.load('training_data/push4_1.npz', allow_pickle=True) as data:
	Xtr = data['arr_0']
	Ytr = data['arr_1']
	instance_params = data['arr_2']
>>> instance_params
array({'t1_x': -4.268447250704135, 't1_y': -0.6937799887556437}, dtype=object)
```
these are automatically passed to the problem function when it is instantiated to create a specific problem instance.

### Optimisation results
The results of all optimisation runs can be found in the `results` directory. The filenames have the following structure: `ProblemName_Run_TotalBudget_Method_OPTIONS.npz`, where 'OPTIONS' takes the form of '0.1' for ε-shotgun without Pareto front selection and with a value of ε = 0.1, '0.1_True' for the same but with Pareto front selection, and 'EI' for the hallucinate method. Similar to the training data, these are also [numpy.ndarrays](https://docs.scipy.org/doc/numpy/reference/generated/numpy.ndarray.html)  and contain two items, `Xtr` and `Ytr`, corresponding to the evaluated locations in the optimisation run and their function evaluations. Note that the evaluations and their function values will also include the initial 2*D training locations at the beginning of the arrays.

The following example loads the first optimisation run on the Branin test problem with the ε-eshotgun method using ε = 0.1:
```python
> cd /es
> python
>>> import numpy as np
>>> # load the 
>>> with np.load('results/Branin_1_2_200_eShotgun_0.1.npz', allow_pickle=True) as data:
	Xtr = data['Xtr']
	Ytr = data['Ytr']
>>> Xtr.shape, Ytr.shape
((250, 2), (250, 1))
```

### Reproduction of experiments
The python file `batch_simulation_script.py` provides a convenient way to reproduce an individual experimental evaluation carried out the paper. It has the following syntax:
```bash
> python batch_simulation_script.py -h
usage: batch_simulation_script.py [-h] -p PROBLEM_NAME -r RUN_NO -s BATCH_SIZE
                                  -b BUDGET -a BATCH_METHOD
                                  [-aa BATCH_METHOD_ARGS [BATCH_METHOD_ARGS ...]]

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

optional arguments:
  -h, --help            show this help message and exit
  -p PROBLEM_NAME       Test problem name. e.g. Branin, logGSobol
  -r RUN_NO             Run number Note that the corresponding npz file
                        containing the initial training locations must exist
                        in the "training_data" directory.
  -s BATCH_SIZE         Batch size. e.g. 2, 5, 10, 20
  -b BUDGET             Budget (including training points).
  -a BATCH_METHOD       Acquisition function name. e.g: TS, qEI, hallu
                        (hallucination), eShotgun, PLAyBOOK or LP.
  -aa BATCH_METHOD_ARGS [BATCH_METHOD_ARGS ...]
                        Acquisition function parameters, must be in pairs of
                        parameter=values, e.g. for the e-shotgun methods:
                        epsilon=0.1 pf=False 
                        [Note: not needed for methods without arguments]
```

### Reproduction of figures and tables in the paper
The jupyter notebook [eshotgun_results_plots.ipynb](eshotgun_results_plots.ipynb) contains the code to load and process the optimisation results (stored in the `results` directory) as well as the code to produce all results figures and tables used in the paper and supplementary material.
