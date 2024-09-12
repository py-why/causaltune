# CausalTune: A library for automated Causal Inference model estimation and selection

[CausalTune Docs](https://causaltune.readthedocs.io/en/latest/)

**CausalTune** is a library for automated tuning and selection for causal estimators.

Its estimators are taken from [EconML](https://github.com/microsoft/EconML/) augmented by a couple of extra models
(currently Transformed Outcome and a dummy model to be used as a baseline), all called in a uniform fashion via a
[DoWhy](https://github.com/microsoft/DoWhy/) wrapper.

Our contribution is enabling automatic estimator tuning and selection by out-of-sample scoring of causal estimators, notably using the [energy score](https://arxiv.org/abs/2212.10076).
We use [FLAML](https://github.com/microsoft/FLAML) for hyperparameter optimisation.

We perform automated hyperparameter tuning of first stage models (for the treatment and outcome models)
as well as hyperparameter tuning and model selection for the second stage model (causal estimator).

The estimators provide not only per-row treatment impact estimates, but also confidence intervals for these,
using builtin EconML functionality for that where it is available and bootstrapping where it is not (see [example notebook](https://github.com/py-why/causaltune/blob/main/notebooks/Standard%20errors.ipynb)).

Just like DoWhy and EconML, we assume that the causal graph provided by the user accurately describes the data-generating process.
So for example, we assume that for CATE estimation, the list of backdoor variables under the graph/confounding variables
provided by the user do reflect all sources of confounding between the treatment and the outcome. See [here](https://github.com/py-why/causaltune/blob/main/notebooks/CausalityDataset%20setup.ipynb) for a detailed explanation of causal graphs that are supported by CausalTune.

The validation methods in CausalTune cannot catch such violations and therefore this is an important assumption.

We also implement the [ERUPT](https://medium.com/building-ibotta/erupt-expected-response-under-proposed-treatments-ff7dd45c84b4)
[calculation](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3111957) (also known as policy value),
allowing after an (even partially) randomized test to estimate what the impact
of other treatment assignment policies would have been. This can also be used as an alternative out-of-sample score,
though energy score performed better in our synthetic data experiments.

<summary><strong><em>Table of Contents</em></strong></summary>

- [CausalTune: A library for automated Causal Inference model estimation and selection](#causaltune-a-library-for-automated-causal-inference-model-estimation-and-selection)
  - [What can this do for you?](#what-can-this-do-for-you)
    - [1. Supercharge A/B tests by getting impact by customer, instead of just an average](#1-supercharge-ab-tests-by-getting-impact-by-customer-instead-of-just-an-average)
    - [2. Continuous testing combined with exploitation: (Dynamic) uplift modelling](#2-continuous-testing-combined-with-exploitation-dynamic-uplift-modelling)
    - [3. Estimate the benefit of smarter (but still partially random) assignment compared to fully random without the need for an actual fully random test group](#3-estimate-the-benefit-of-smarter-but-still-partially-random-assignment-compared-to-fully-random-without-the-need-for-an-actual-fully-random-test-group)
    - [4. Observational inference](#4-observational-inference)
    - [5. IV models: Impact of customer choosing to use a feature](#5-iv-models-impact-of-customer-choosing-to-use-a-feature)
  - [Installation](#installation)
    - [Requirements](#requirements)
    - [Installation Guide for Mac (for installation from source):](#installation-guide-for-mac-for-installation-from-source)
  - [Quick Start](#quick-start)
  - [Supported Models](#supported-models)
  - [Supported Metrics](#supported-metrics)
  - [Citation](#citation)
  - [For Developers](#for-developers)
    - [Installation from source](#installation-from-source)
    - [Testing](#testing)
  - [Contribution](#contribution)


## What can this do for you?

The automated search over the many powerful models from EconML and elsewhere allows you to easily do the following

### 1. Supercharge A/B tests by getting impact by customer, instead of just an average

By enriching the results of a regular A/B/N test with customer features, and running CausalTune on the
resulting dataset, you can get impact estimates as a function of customer features, allowing precise targeting by
impact in the next iteration. CausalTune also serves as a variance reduction method leveraging the availability of any additional features. [Example notebook](https://github.com/py-why/causaltune/blob/main/notebooks/AB%20testing.ipynb)

### 2. Continuous testing combined with exploitation: (Dynamic) uplift modelling

The per-customer impact estimates, even if noisy, can be used to implement per-customer Thompson sampling for new customers, biasing random treatment assignment towards ones we think are most likely to work. As we still control the per-customer propensity to treat, same methods as above can be applied to keep refining our impact estimates.

Thus, there is no need to either wait for the test to gather enough data for significance, nor to ever end the
test, before using its results to assign the most impactful treatment (based on our knowledge so far) to each customer.

As in this case the propensity to treat is known for each customer, we [allow to explicitly supply it](https://github.com/py-why/causaltune/blob/main/notebooks/Propensity%20Model%20Selection.ipynb)
as a column to the estimators, instead of estimating it from the data like in other cases.

### 3. Estimate the benefit of smarter (but still partially random) assignment compared to fully random without the need for an actual fully random test group

The previous section described using causal estimators to bias treatment assignment towards the choice we think is
most likely to work best for a given customer.

However, after the fact we would like to know the extra benefit of that compared to a fully random assignment.
The ERUPT technique [sample notebook](https://github.com/py-why/causaltune/blob/main/notebooks/ERUPT%20under%20simulated%20random%20assignment.ipynb) re-weights the actual outcomes to produce an unbiased estimate of the average outcome that a fully random assignment would have yielded, with no actual additional group needed.


### 4. Observational inference

The traditional application of causal inference. For example, estimating the impact on
volumes and churn likelihood of the time it takes us to answer a customer query. As the set of customers
who have support queries is most likely not randomly sampled, confounding corrections are needed.

As with other usecases, the advanced causal inference models allow impact estimation as a function
of customer features, rather than just averages, **under the assumption that all relevant confounders are observed**.

To use this, just set `propensity_model` to an instance of the desired classifier when instantiating `CausalTune`, or to `"auto"` if you want to use the FLAML classifier (the default setting is `"dummy"` which assumes random assigment and infers
the assignment probability from the data). [Example notebook](https://github.com/py-why/causaltune/blob/main/notebooks/Propensity%20Model%20Selection.ipynb)

If you have reason to suppose unobserved confounders, such as customer intent (did the customer do a lot of volume
because of the promotion, or did they sign up for the promotion because they intended to do lots of volume anyway?)
consider looking for an instrumental variable instead.
<!--
Can we reformulate to (did the customer do a lot of volume because of the promotion or is the customer more likely
to sign up for the promotion because they have a higher volume per se?)
-->

Note that our derivation of energy score as a valid out-of-sample score for causal models is strictly speaking not
applicable for this usecase, but still appears to work reasonably well in practice.

### 5. IV models: Impact of customer choosing to use a feature

Instrumental variable (IV) estimation to avoid an estimation bias from unobserved confounders.

A natural use case for IV models is making a feature or a promotion available to a customer, and trying to
measure the impact of the customer actually choosing to use the feature (the impact of making the feature
available can be solved with 1. and 2. above).

Here we use feature availability as an instrumental variable (assuming its assignment to be strictly randomized),
and search over IV models in EconML to estimate the impact of the customer choosing to use it. To score IV model fits out of sample, we again use the [energy score](https://arxiv.org/abs/2212.10076). [Example notebook](https://github.com/py-why/causaltune/blob/main/notebooks/Comparing%20IV%20Estimators.ipynb)

Please be aware we have not yet extensively used the IV model fitting functionality internally, so if you run into any issues, please report them!

## Installation
To install from source, see [For Developers](#for-developers) section below.


### Requirements

CausalTune works with Python 3.8 and 3.9.

It requires the following libraries to work:
- NumPy
- Pandas
- EconML
- DoWhy
- FLAML
- Scikit-Learn
- Dcor

The easiest way to install the dependencies is via
```
pip install -r requirements.txt
```
into the virtual environment of your choice.

### Installation Guide for Mac (for installation from source):

Mac/ OS users: For some machines, it can happen that the package LightGBM which is a dependency of AutoML / Flaml will not automatically be installed properly. In that case, a workaround is to set up a conda environment and install LightGBM through the conda-forge channel

1. Clone the Repository and navigate to the repository

2. Set Up a Conda Environment using an appropriate Python Version
	- Ensure Anaconda or Miniconda is installed.
	- Create a new Conda environment: `conda create -n causaltune-env python=3.9.x`
	- Activate the environment: `conda activate causaltune-env`.

3. Install the dependency lightgbm seperatly before attempting to install other dependencies
	- `conda install -c conda-forge lightgbm`

4. Install Dependencies
	- Navigate to the directory containing 'requirements.txt'.
	- Install dependencies: `pip3 install -r requirements.txt`.

5. Load CausalTune from the Local Repository
   	- Adjust Python script paths as needed before running import causaltune, e.g.:
    ```
    import sys
    sys.path.append('path/to/cloned/repository')
    import causaltune
    ```


## Quick Start

The CausalTune package can be used like a scikit-style estimator:

```Python
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

```

Now if ***outcome_model="auto"*** in the CausalTune constructor, we search over a simultaneous search space for the EconML estimators and for FLAML wrappers for common regressors. The old behavior is now achieved by ***outcome_model="nested"*** (Refitting AutoML for each estimator).

You can also preprocess the data in the CausalityDataset using one of the popular category encoders: ***OneHot, WoE, Label, Target***.

## Supported Models
The package supports the following causal estimators:
* Meta Learners:
    * S-Learner
    * T-Learner
    * X-Learner
    * Domain Adaptation Learner
* DR Learners:
    * Forest DR Learner
    * Linear DR Learner
    * Sparse Linear DR Learner
* DML Learners:
    * Linear DML
    * Sparse Linear DML
    * Causal Forest DML
* Ortho Forests:
    * DR Ortho Forest
    * DML Ortho Forest
* Transformed Outcome

## Supported Metrics
We support a variety of different metrics that quantify the performance of a causal model:
* Energy distance
* ERUPT (Expected Response Under Proposed Treatments)
* Qini coefficient and AUC (area under curve)
* ATE (average treatment effect)

## Citation
If you use CausalTune in your research, please cite us as follows:

Timo Debono, Julian Teichgräber, Timo Flesch, Edward Zhang, Guy Durant, Wen Hao Kho, Mark Harley, Egor Kraev. **CausalTune: A Python package for Automated Causal Inference model estimation and selection.** https://github.com/py-why/causaltune. 2022. Version 0.x
You can use the following BibTex entry:
```
@misc{CausalTune,
  author={Timo Debono, Julian Teichgr\"aber, Timo Flesch, Edward Zhang, Guy Durant, Wen Hao Kho, Mark Harley, Egor Kraev},
  title={{CausalTune}: {A Python package for Automated Causal Inference model estimation and selection}},
  howpublished={https://github.com/py-why/causaltune},
  note={Version 0.x},
  year={2022}
}
```
## For Developers
### Installation from source
We use [Setuptools](https://setuptools.readthedocs.io/en/latest/index.html) for building and distributing our package. To install the latest version from source, clone this repository and run the following command from the top-most folder of the repository
```
pip install -e .
```
### Testing
We use [PyTest](https://docs.pytest.org/) for testing. If you want to contribute code, make sure that the tests run without errors.

## Contribution
See the [Contribution file](https://github.com/py-why/causaltune/blob/main/CONTRIBUTING.md) for contribution licensing and code guidelines.
