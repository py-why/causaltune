# Auto-Causality: A library for automated Causal Inference model estimation and selection


**AutoCausality** is a library for automated Causal Inference,
building on the [FLAML](https://github.com/microsoft/FLAML) package for hyperparameter optimisation
and the [EconML](https://github.com/microsoft/EconML/) and [DoWhy](https://github.com/microsoft/DoWhy/)
packages for ML-based Causal Inference, with a couple of extra models (currently Transformed Outcome and a
dummy model to be used as a baseline).

It performs automated hyperparameter tuning of first stage models (for the treatment and outcome models) as well as hyperparameter tuning and model selection for the second stage model (causal estimator).

<summary><strong><em>Table of Contents</em></strong></summary>

- [What can this do for you?](#what-can-this-do-for-you)
  - [Segment A/B tests by per-customer impact](#1-supercharge-ab-tests-by-getting-impact-by-customer-instead-of-just-an-average)
  - [Continuous testing](#2-continuous-testing-combined-with-exploitation)
  - [Observational inference](#3-observational-inference)
  - [Impact of customer choosing to use a feature (IV models)](#4-impact-of-customer-choosing-to-use-a-feature-iv-models)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Supported Models](#supported-models)
- [Supported Metrics](#supported-metrics)
- [Citation](#citation)
- [For Developers](#for-developers)
  - [Installation from source](#installation-from-source)
  - [Tests](#testing)


## What can this do for you?

The automated search over the many powerful models from EconML and elsewhere allows you to easily do the following

### 1. Supercharge A/B tests by getting impact by customer, instead of just an average
By enriching the results of a regular A/B/N test with customer features, and running auto-causality on the
resulting dataset, you can get impact estimates as a function of customer features, allowing precise targeting by
impact in the next iteration.

**Currently tested and used: binary treatment, strictly random assignment**.

**Freshly merged: Multiple (categorical) treatments** Just merged into main, tests pass but hasn't been tested by extensive usage yet


### 2. Continuous testing combined with exploitation
The per-customer impact estimates from the previous section, even if noisy, can be used to implement
per-customer Thompson sampling for new customers, biasing random treatment assignment towards ones we think
are most likely to work. As we still control the per-customer propensity to treat, same methods as above can be
applied to keep refining our impact estimates.

Thus, there is no need to either wait for the test to gather enough data for significance, nor to ever end the
test, before using its results to assign the most impactful treatment to each customer.

**Freshly merged: Taking propensity to treat from a column in the supplied dataset** Just merged into main, tests pass but hasn't been tested by extensive usage yet

### 3. Observational inference
The traditional application of causal inference. For example, estimating the impact on
volumes and churn likelihood of the time it takes us to answer a customer query. As the set of customers
who have support queries is most likely not randomly sampled, confounding corrections are needed.

As with other usecases, the advanced causal inference models allow impact estimation as a function
of customer features, rather than just averages.

**Ready to use** Use of user-supplied propensity function. Just set
`propensity_model` to an instance of the desired classifier when instantiating `AutoCausality`, or to `"auto"`
if you want to use the FLAML classifier.


### 4. Impact of customer choosing to use a feature (IV models)
The case we're focussing on is making a feature or a promotion available to a customer, and trying to
measure the impact of the customer actually choosing to use the feature (the impact of making the feature
available can be solved with 1. and 2. above).

Here we use feature availability as an instrumental variable, and search over IV models in EconML to estimate the
impact of the customer choosing to use it. To compare IV models, we use what appears to be a novel approach (described
[here](https://github.com/transferwise/auto-causality/blob/main/docs/Comparing_IV_models.pdf),
publication is in the works)

**Available, hasn't been extensively used yet** No known issues, but please be aware we haven't yet
extensively used this internally, so if you run into any issues, please report them!

## Installation
To install from source, see [For Developers](#for-developers) section below.
TODO: make available as package on pypi


**Requirements**
AutoCausality requires the following libraries to work:
- NumPy
- Pandas
- EconML
- DoWhy
- Scikit-Learn

If you run into any problems, try installing the dependencies manually:
```
pip install -r requirements.txt
```

## Quick Start
The autocausality package can be used like a scikit-style estimator:

```Python
from auto_causality import AutoCausality
from auto_causality.datasets import synth_ihdp

# prepare dataset
data = synth_ihdp()
data.preprocess_dataset()

# init autocausality object with chosen metric to optimise
ac = AutoCausality(time_budget=10, metric='erupt')

# run autocausality
myresults = ac.fit(data)

# return best estimator
print(f"Best estimator: {ac.best_estimator}")

```

## Supported Models
The package supports the following causal models:
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
* Qini coefficient
* AUC (area under curve)
* R-Scorer
* ATE (average treatment effect)

## Citation
If you use AutoCausality in your research, please cite us as follows:
Timo Flesch, Edward Zhang, Guy Durant, Wen Hao Kho, Mark Harley, Egor Kraev. **Auto-Causality: A Python package for Automated Causal Inference model estimation and selection.** https://github.com/transferwise/auto-causality. 2022. Version 0.x
You can use the following BibTex entry:
```
@misc{autocausality,
  author={Timo Flesch, Edward Zhang, Guy Durant, Wen Hao Kho, Mark Harley, Egor Kraev},
  title={{Auto-Causality}: {A Python package for Automated Causal Inference model estimation and selection}},
  howpublished={https://github.com/transferwise/auto-causality},
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
We use [PyTest](https://docs.pytest.org/) for testing. If you want to contribute code, make sure that the tests in tests/autocausality/test_endtoend.py run without errors.

