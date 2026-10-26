# Compatibility and numerical changes

Full-covariance head parameters now define a Cholesky factor. Retrain and
revalidate full-covariance models: old raw head parameters and RHO_MAX do not
retain their previous meaning. RHO_MAX does not bound Pearson correlation;
target order affects the marginal scale limits.

The diagonal layout is unchanged, but consistent 1e-6 variance stabilization
can raise low-end std by about 1.1% under default bounds. Recheck calibration
factors and coverage. This inference API exposes marginal uncertainty and
does not preserve cross-target aleatoric covariance in physical sampling.

Preserve historical checkpoints with their original source, data,
configuration, dependencies and calibration artifacts. Changes to warmup or
optimizer persistence require separately versioned corrected training runs;
inference-only corrections do not automatically require new training.
