# Auto-Causality: A library for automated Causal Inference model estimation and selection                                                                                                                                                                                                                                                                                                                   


**AutoCausality** is a library for automated Causal Inference, building on the [FLAML](https://github.com/microsoft/FLAML) package for hyperparameter optimisation and the [EconML](https://github.com/microsoft/EconML/) and [DoWhy](https://github.com/microsoft/DoWhy/) packages for ML-based Causal Inference. It performs automated hyperparameter tuning of first stage models (for the treatment and outcome models) as well as hyperparameter tuning and model selection for the second stage model (causal estimator).  

For now, the package only supports CATE models, instrumental variable models are coming next!


<summary><strong><em>Table of Contents</em></strong></summary>

- [Installation](#installation)  
- [Quick Start](#quick-start)
- [Supported Models](#supported-models)
- [Supported Metrics](#supported-metrics)
- [Citation](#citation)
- [For Developers](#for-developers)  
    -[Installation from source](#installation-from-source)  
    -[Tests](#testing)  


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
from auto_causality.datasets import synth_ihdp, preprocess_dataset
# prepare dataset
data_df = synth_ihdp()
data_df, features_X, features_W, targets, treatment = preprocess_dataset(data_df)

# init autocausality object with chosen metric to optimise
ac = AutoCausality(time_budget=10,metric='erupt')

# run autocausality
myresults = ac.fit(data_df, treatment, targets[0],
 features_W, features_X)

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

## Supported Metrics
We support a variety of different metrics that quantify the performance of a causal model:
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

