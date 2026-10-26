# Minibatch likelihood and prior weighting

The current SVI model constructs the observation plate with the minibatch size
B and supplies no full-dataset size N. The default alpha is one, so each step
uses a B-observation likelihood and one global variational KL term. Preserving
optimizer state across epochs does not change that objective.

For uniform minibatches, unit likelihood/KL weights and a conventional full-data
ELBO, an unbiased stochastic likelihood estimate would instead scale the
minibatch log likelihood by N/B. Relative to that convention, the current
unscaled objective has approximately N/B greater prior-to-likelihood weight
when expressed on the same full-data scale. This statement compares objectives;
it is not a measured universal factor in prediction error or uncertainty.

KL warmup does not supply dataset-size normalization. The positive-epoch beta
schedule changes KL weighting during warmup; the simple N/B comparison above
is for beta=1 and alpha=1. The current epoch-zero special branch is not changed
by this work. Short final batches can have a different B, and nonuniform
balancing changes the sampled distribution as well as the weighting question.

The accounting regressions execute the production observation model and SVI
factory. They check unscaled observation sums and beta-weighted analytic KL on
a controlled Normal latent example. They are not training convergence,
calibration, coverage or scientific-objective validation.

## Acceptance decision

Choose an ordinary full-dataset posterior objective or explicitly accept the
current tempered/regularized objective for the intended deployment distribution.
For ordinary uniform sampling, dataset-aware plate/likelihood scaling can supply
N/B. With balancing or other nonuniform sampling, first define the target data
distribution and whether importance corrections are required; multiplying by
N/B alone does not restore the original empirical-data objective.

This follow-up preserves the current objective. If its implementation or
weighting changes, publish a separately versioned, validated training run and
reassess convergence, NLL and calibration/coverage. Documenting an unchanged
objective does not by itself require retraining. Preserve historical source,
data, configuration, dependencies and calibration artifacts for old checkpoints.
