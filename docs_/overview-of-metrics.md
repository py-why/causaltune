## Overview of different metrics

#### Value metrics

Our estimators, which are optimized using Flaml, returns CATE, or conditional average treatment effect. Using CATE, we implement a policy depending on whether the treatment has positive or negative effect on the outcome (policy parameter in ERUPT.score()).

We can use the ERUPT metric to test this policy by taking the average outcome of the samples in which the treatment equals the policy. (It is weighted by the inverse probability of the treatment using a dummy classifier to even out the treatment probabilities of each class so that the final weights match the policies).

(This is done by taking the mean of outcome*weight over all the samples, including those which the weight is 0 because the policy does not match the treatment. It might appear to mean that in order to maximize the ERUPT metric, you would want the policy to match the treatment rather than match the treatment only when the outcome is good. However, the ERUPT metric is actually an unbiased estimator of the expected profit. (See: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3111957 page 10) Furthermore, we implement normalization just in case, so that the average weight over all the samples is 1.)

The ERUPT metric therefore should a reasonable proxy for measuring the value of a policy, should the goal be to maximise it.

The Qini and AUC metrics are similar to the ERUPT metric in this regard, except they do not simply consider whether CATE > 0, but the ranking of CATE values across samples. They very roughly consider the surplus value when the top k% of individuals are treated according to the policy, as compared to the control group. These metrics may be more useful if there are limits on the size of the population that can be treated, or if there is an inherent cost to treatment.

#### Using τ -risk

This class of metrics fundamentally try to estimate CATE using a different method (eg. taking the example with the closest Mahalanobis distance and opposite treatment as a counterfactual) and measuring the effectiveness of our estimator by comparing its CATE estimate with the one given with the other method. 

(It’s somewhat analogous to measuring the effectiveness of a standard supervised learning model by comparing its predictions to the predictions of another machine learning model on a validation set because we don’t have the labels, y. This does sound a bit stupid but I suppose the idea is that if significantly different models/estimators agree on the same “solution” it is more likely to be correct.)

The overview at https://arxiv.org/pdf/1804.05146.pdf has 3 different formulations of the above, match (which is using the counterfactual with the closest Mahalanobis distance), IPTW (which uses the IPTW-weighted (transformed) outcome as an estimator), and R.

τ -risk(R) basically measures the l2 distance between (the difference of the actual outcome and the outcome predicted using a separate regression model without the treatment as a feature) and (the difference between actual treatment and predicted treatment multiplied by the CATE). This is similar to double machine learning in the sense that we are comparing the residues of both predictions).

τ -risk(R) is implemented in our package as "r_score"


Some other variations of τ -risk can be found at
https://arxiv.org/pdf/2008.12892.pdf
http://proceedings.mlr.press/v119/saito20a/saito20a.pdf
https://usaito.github.io/files/cfcv_ws.pdf

------------



Some additional references:

https://arxiv.org/pdf/2002.05897.pdf

http://proceedings.mlr.press/v67/gutierrez17a/gutierrez17a.pdf

https://www.research.manchester.ct.uk/portal/files/205626621/FULL_TEXT.PDF

https://arxiv.org/pdf/2104.04103.pdf

https://arxiv.org/pdf/1702.02896.pdf

https://www.sciencedirect.com/science/article/abs/pii/S0169716120300432 

http://proceedings.mlr.press/v70/shalit17a/shalit17a.pdf


