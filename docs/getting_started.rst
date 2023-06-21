Installation
^^^^^^^^^^^^

Install the latest release from PyPi:

.. code:: shell

   pip install causaltune

.. note::
    
    Mac/ OS users: For some machines, it can happen that the package LightGBM which is a dependency of AutoML / Flaml will not automatically be installed properly. In that case, a workaround is to set up a conda environment and install LightGBM through the conda-forge channel:

    .. code-block:: shell

        conda create -n <my_env> python=3.9.16 
        conda activate <my_env> 
        pip install causaltune
        conda install -c conda-forge lightgbm



Quick Start
--------------

The CausalTune package can be used like a scikit-style estimator:

.. code-block:: python
    
    from causaltune import CausalTune
    from causaltune.datasets import synth_ihdp

    # prepare dataset
    data = synth_ihdp()
    data.preprocess_dataset()


    # init CausalTune object with chosen metric to optimise
    ct = CausalTune(time_budget=600, metric="energy_distance")

    # run CausalTune
    ct.fit(data)

    # return best estimator
    print(f"Best estimator: {ct.best_estimator}")


For Developers
----------------

Clone this repository and run the following command from the top-most folder of the repository.

.. code:: shell

    pip install -r requirements-dev.txt

This project uses pytest for testing. To run tests locally after installing the package, you can run

 .. code:: shell
    
    python setup.py pytest

Requirements
---------------

CausalTune requires the following packages:

* numpy
* pandas
* econml
* dowhy
* flaml
* scikit-learn
* matplotlib
* dcor
* wise-pizza
* seaborn
  
If you cloned the repository, they can be installed via

.. code:: shell

    pip install -r requirements.txt
