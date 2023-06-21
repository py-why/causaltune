## What can this do for you?

The automated search over the many powerful models from EconML and elsewhere allows you to easily do the following

### 1. Supercharge A/B tests by getting impact by customer, instead of just an average

By enriching the results of a regular A/B/N test with customer features, and running CausalTune on the
resulting dataset, you can get impact estimates as a function of customer features, allowing precise targeting by
impact in the next iteration. CausalTune also serves as a variance reduction method leveraging the availability of any additional features. [Example notebook](https://github.com/py-why/causaltune/blob/main/notebooks/AB%20testing.ipynb)

### 2. Continuous testing combined with exploitation: (Dynamic) uplift modelling

The per-customer impact estimates, even if noisy, can be used to implement per-customer Thompson sampling for new customers, biasing random treatment assignment towards ones we think are most likely to work. As we still control the per-customer propensity to treat, same methods as above can be applied to keep refining our impact estimates.

Thus, there is no need to either wait for the test to gather enough data for significance, nor to ever end the
test, before using its results to assign the most impactful treatment (based on our knowlede so far) to each customer.

As in this case the propensity to treat is known for each customer, we [allow to explicitly supply it](https://github.com/py-why/causaltune/blob/main/notebooks/Propensity%20Model%20Selection.ipynb)
as a column to the estimators, instead of estimating it from the data like in other cases.

### 3. Estimate the benefit of smarter (but still partially random) assignment compared to fully random without the need for an actual fully random test group

Previous section described using causal estimators to bias treatment assignment towards the choice we think is
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

