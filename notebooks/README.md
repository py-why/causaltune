# CausalTune Notebooks

### 1. [Random assignment, binary CATE example](https://github.com/transferwise/auto-causality/blob/pywhy-integration/notebooks/Random%20assignment%2C%20binary%20CATE%20example.ipynb)
This notebook serves as a good starting when exploring CausalTune. It encompasses an introduction to 
- estimator & metric selection, 
- choosing training time budgets,
- finding the best estimator and configuration, scoring and visualisation.


### 2. [CausalityDataset setup](https://github.com/transferwise/auto-causality/blob/pywhy-integration/notebooks/CausalityDataset%20setup.ipynb)
Causaltune models require work based on a CausalityDataset instance which includes all relevant data and information to build the causal graph. This notebook briefly outlines how to define a CausalityDataset.

### 3. [AB Testing](https://github.com/transferwise/auto-causality/blob/pywhy-integration/notebooks/AB_testing.ipynb)

A guide on how to use CausalTune for AB test evaluation.

It explores both variance reduction techniques leveraging additional features and segmentation analysis by feeding conditional average treatment effects (CATEs) into the automated segmentation analytics tool [wise-pizza](https://github.com/transferwise/wise-pizza/).


### 4. [ERUPT under simulated random assignment](https://github.com/transferwise/auto-causality/blob/pywhy-integration/notebooks/ERUPT%20under%20simulated%20random%20assignment.ipynb)

This analysis walks the user through evaluating and conducting uplift modelling with the ERUPT metric. 

### 5. [Propensity Model Selection](https://github.com/transferwise/auto-causality/blob/pywhy-integration/notebooks/Propensity%20Model%20Selection.ipynb)

The estimators that CausalTune is based on (e.g. doubly robust learners, metalearners etc.) mostly require propensity score weighting. CausalTune therefore requires propensity score weights. This notebook displays the different methods that can be used to bild the propensity score weights.

Those include 
   - [Default:] use a dummy estimator.
   - Letting AutoML fit the propoensity model,
   - supply a custom sklearn-compatible prediction model,
   - supply an array of custom propensities to treat.

### 6. [Standard errors](https://github.com/transferwise/auto-causality/blob/pywhy-integration/notebooks/Standard%20errors.ipynb)
Standard errors are important for statistical inference. CausalTune allows for computation of standard errors of the best estimator as identified by CausalTune. The standard errors are computed with econml methods which includes analytical estimates for some estimators and bootstraps for others. 

### 7. [Comparing IV estimators](https://github.com/transferwise/auto-causality/blob/pywhy-integration/notebooks/Comparing%20IV%20Estimators.ipynb)
CausalTune can compare instrumental variable (IV) estimators based on the energy distance. This notebooks shows how to run the estimator selection and how to compare IV estimators. 

It also walks through how to interpret the energy distance and additional result visualisation.



