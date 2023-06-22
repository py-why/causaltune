.. CausalTune documentation master file, created by
   sphinx-quickstart on Tue Jun 20 11:42:02 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to CausalTune's documentation!
======================================

*Related resources:* pyhwy_ , pypi_causaltune_

.. toctree::
   :maxdepth: 1
   :caption: Contents:
   :glob:

   getting_started
   whatcanthisdo
   howitworks
   notebook_examples
   causaltune

Installation guidelines can be found at :doc:`getting_started`

**CausalTune** is a library for automated tuning and selection for causal estimators.

Its estimators are taken from EconML_ augmented by a couple of extra models
(currently Transformed Outcome and a dummy model to be used as a baseline), all called in a uniform fashion via a
DoWhy_ wrapper.

Our contribution is enabling automatic estimator tuning and selection by out-of-sample scoring of causal estimators, notably using the energy_score_.
We use FLAML_ for hyperparameter optimisation.

We perform automated hyperparameter tuning of first stage models (for the treatment and outcome models)
as well as hyperparameter tuning and model selection for the second stage model (causal estimator).

The estimators provide not only per-row treatment impact estimates, but also confidence intervals for these,
using builtin EconML functionality for that where it is available and bootstrapping where it is not (see `example notebook <https://github.com/py-why/causaltune/blob/main/notebooks/Standard%20errors.ipynb>`_).

Just like DoWhy and EconML, we assume that the causal graph provided by the user accurately describes the data-generating process.
So for example, we assume that for CATE estimation, the list of backdoor variables under the graph/confounding variables
provided by the user do reflect all sources of confounding between the treatment and the outcome. See `here <https://github.com/py-why/causaltune/blob/main/notebooks/CausalityDataset%20setup.ipynb>`_ for a detailed explanation of causal graphs that are supported by CausalTune.

The validation methods in CausalTune cannot catch such violations and therefore this is an important assumption.

We also implement the ERUPT_
calculation_ (also known as policy value),
allowing after an (even partially) randomized test to estimate what the impact
of other treatment assignment policies would have been. This can also be used as an alternative out-of-sample score,
though energy score performed better in our synthetic data experiments.

.. _pywhy: https://www.pywhy.org/
.. _pypi_causaltune: https://pypi.org/project/causaltune/

.. _EconML: https://github.com/microsoft/EconML/
.. _FLAML: https://github.com/microsoft/FLAML
.. _DoWhy: https://github.com/microsoft/DoWhy/
.. _calculation: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3111957
.. _ERUPT: https://medium.com/building-ibotta/erupt-expected-response-under-proposed-treatments-ff7dd45c84b4
.. _energy_score: https://arxiv.org/abs/2212.10076

* :ref:`modindex`
* :ref:`search`

Date: |today|

