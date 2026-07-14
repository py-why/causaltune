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


By default ``fit()`` optimises with the Optuna backend (default sampler: TPE).
Pass ``framework="hyperopt"`` or ``framework="flaml"`` to switch, and ``algo=``
to select a specific sampler / search algorithm. ``framework``, ``algo`` and
``framework_params`` can also be set on the ``CausalTune`` constructor; a value
passed to ``fit()`` overrides the constructor default. Hyperopt requires the
optional extra (``pip install causaltune[hyperopt]``).

Backend interface parity
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The three backends aim for the same option surface (best-effort, with graceful
degradation where a backend cannot support a feature):

* **Estimator list, num_samples / time_budget, algo** — supported identically on
  all three.
* **Warm start** (``try_init_configs``) — supported on all three. flaml consumes
  the init configs natively; optuna enqueues them (``enqueue_trial``) and
  hyperopt seeds them as the first trials.
* **Resume** (``resume=True``) — supported on all three. flaml uses cost-aware
  warm rebuild; optuna/hyperopt route through hiertunehub's
  ``resume_from_results``. Resume continues the *same* in-memory ``CausalTune``
  instance (same ``estimator_list`` / data / backend) and runs an *additional*
  ``num_samples`` trials; the recommended mode is time-bounded
  (``num_samples=-1`` with a ``time_budget``).
* **verbose** — honoured on all three (optuna via ``optuna.logging``).
* **Parallelism** — flaml via ``use_ray``; optuna exposes ``n_jobs`` but it is
  clamped to 1 by default because CausalTune's objective is not thread-safe.
  Opting into ``n_jobs>1`` (via ``framework_params``) warns and is at your own
  risk; prefer ``use_ray`` for real parallelism.
* **Cost-aware search** (``cost_attr`` / ``low_cost_partial_config``) — a
  FLAML-only concept with no optuna/hyperopt equivalent; remains flaml-only.

The ``framework_params`` dict is an advanced escape hatch: its entries are merged
into the backend call (you win on conflicts). Overriding a CausalTune-managed key
(e.g. ``n_trials``) warns; a reserved key that the wrapper passes explicitly
(``config``/``mode``/``metric``/``trials``/``objective``/``search_space``) raises.


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
* optuna
* scikit-learn
* matplotlib
* dcor
* wise-pizza
* seaborn
  
If you cloned the repository, they can be installed via

.. code:: shell

    pip install -r requirements.txt
