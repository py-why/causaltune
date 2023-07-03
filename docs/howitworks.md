# Model selection

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