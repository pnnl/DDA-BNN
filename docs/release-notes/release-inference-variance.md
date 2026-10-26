# Release Inference uncertainty compatibility

The nested Monte Carlo correction removes expected finite-inner-sample noise
from epistemic variance, uses an unbiased within-group aleatoric estimate and
clips negative estimated epistemic variance to zero. Independent inner draws
are assumed; finite-sample and clipping bias remain.

Total variance is the sum of the corrected components. It can differ from the
previous pooled-cloud variance. For groups [0,2] and [4,6], total variance is
5.5 rather than the pooled value 5. Means and pooled quantiles are unchanged
for the same supplied cloud. Revalidate fitted calibration and empirical
coverage after changing these statistics.

Inference helpers return mean, aleatoric std, epistemic std, total std and
quantiles. The calibration helper instead returns mean, total std, aleatoric
std, epistemic std and quantiles. Preserve caller unpacking and test parity
when integrating their updates.

Automatic tau loading is distinct from the variance correction; explicitly
request taus=None for raw inference. Inference-only changes do not by
themselves require retraining. Preserve the source/configuration provenance
of historical checkpoints, and separately version corrected training runs.
