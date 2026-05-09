from abc import abstractmethod
from typing import Optional

import jax
import jax.numpy as jnp

AGGREGATION_METHODS = {
    "none": lambda x, axis: x,
    "mean": lambda x, axis: jnp.mean(x, axis=axis),
    "sum": lambda x, axis: jnp.sum(x, axis=axis),
    "median": lambda x, axis: jnp.median(x, axis=axis),
    "max": lambda x, axis: jnp.max(x, axis=axis),
    "min": lambda x, axis: jnp.min(x, axis=axis),
}


class ProbabilityMetric:
    """Compare a predicted ensemble against a true ensemble.

    Subclasses implement ``compute`` for a single time step on flattened state
    vectors; ``__call__`` handles flattening, vmapping over time, and aggregation.
    """

    def __init__(self, name: str, time_aggregation: Optional[str] = None) -> None:
        self.name = name
        self.time_aggregation = (
            time_aggregation if time_aggregation is not None else "none"
        )

    @abstractmethod
    def compute(
        self, pred_ensemble: jnp.ndarray, true_ensemble: jnp.ndarray
    ) -> jnp.ndarray:
        """Compute metric for a single time step.

        Args:
            pred_ensemble: shape (n_pred_ensemble, n_state) — flattened state
            true_ensemble: shape (n_true_ensemble, n_state) — flattened state

        Returns:
            Scalar metric value for this time step.
        """
        raise NotImplementedError

    def __call__(
        self, pred_ensemble: jnp.ndarray, true_ensemble: jnp.ndarray
    ) -> jnp.ndarray:
        """Compute metric over all time steps, then aggregate.

        Args:
            pred_ensemble: shape (n_pred_ensemble, n_time, n_states, n_dim)
            true_ensemble: shape (n_true_ensemble, n_time, n_states, n_dim)

        Returns:
            Aggregated metric. Shape depends on time_aggregation:
            - None:      (n_time,)
            - otherwise: scalar
        """
        n_pred, n_time, n_states, n_dim = pred_ensemble.shape
        n_true = true_ensemble.shape[0]

        pred_flat = pred_ensemble.reshape(n_pred, n_time, n_states * n_dim)
        true_flat = true_ensemble.reshape(n_true, n_time, n_states * n_dim)

        # vmap over time axis (axis=1 for both)
        compute_over_time = jax.vmap(self.compute, in_axes=(1, 1))
        result = compute_over_time(pred_flat, true_flat)  # (n_time,)

        return AGGREGATION_METHODS[self.time_aggregation](result, axis=0)


def _gaussian_kl(
    mu_p: jnp.ndarray,
    sigma_p: jnp.ndarray,
    mu_q: jnp.ndarray,
    sigma_q: jnp.ndarray,
) -> jnp.ndarray:
    """KL(P || Q) between two multivariate Gaussians.

    KL = 0.5 * [tr(Σ_q^{-1} Σ_p) + (μ_q - μ_p)^T Σ_q^{-1} (μ_q - μ_p) - d + ln(det Σ_q / det Σ_p)]
    """
    d = mu_p.shape[0]
    sigma_q_inv = jnp.linalg.inv(sigma_q)
    trace_term = jnp.trace(sigma_q_inv @ sigma_p)
    diff = mu_q - mu_p
    quad_term = diff @ sigma_q_inv @ diff
    _, logdet_p = jnp.linalg.slogdet(sigma_p)
    _, logdet_q = jnp.linalg.slogdet(sigma_q)
    return 0.5 * (trace_term + quad_term - d + logdet_q - logdet_p)


class KLDivergence(ProbabilityMetric):
    """Non-parametric KL(pred || true) estimated via per-dimension histograms.

    For each state dimension, shared bin edges are derived from the combined
    pred+true samples. Laplace smoothing is applied before calling
    ``jax.scipy.special.kl_div`` element-wise to avoid inf values in empty
    bins. The per-dimension KL values are then averaged over state dimensions.
    """

    def __init__(
        self,
        time_aggregation: Optional[str] = None,
        n_bins: int = 50,
    ) -> None:
        super().__init__("kl_divergence", time_aggregation)
        self.n_bins = n_bins

    def compute(
        self, pred_ensemble: jnp.ndarray, true_ensemble: jnp.ndarray
    ) -> jnp.ndarray:
        n_bins = self.n_bins
        n_pred = pred_ensemble.shape[0]
        n_true = true_ensemble.shape[0]

        def kl_per_dim(pred_vals: jnp.ndarray, true_vals: jnp.ndarray) -> jnp.ndarray:
            # Shared bin edges over the combined range of both ensembles
            all_vals = jnp.concatenate([pred_vals, true_vals])
            lo = jnp.min(all_vals)
            hi = jnp.max(all_vals)
            margin = 1e-6 * (jnp.abs(hi - lo) + 1.0)
            lo, hi = lo - margin, hi + margin

            bin_width = (hi - lo) / n_bins
            p_idx = jnp.clip(
                jnp.floor((pred_vals - lo) / bin_width).astype(jnp.int32), 0, n_bins - 1
            )
            q_idx = jnp.clip(
                jnp.floor((true_vals - lo) / bin_width).astype(jnp.int32), 0, n_bins - 1
            )

            p_counts = jnp.zeros(n_bins).at[p_idx].add(1.0)
            q_counts = jnp.zeros(n_bins).at[q_idx].add(1.0)

            # Laplace smoothing: avoids zero bins that would cause inf in kl_div
            p = (p_counts + 1.0) / (n_pred + n_bins)
            q = (q_counts + 1.0) / (n_true + n_bins)

            return jnp.sum(jax.scipy.special.kl_div(p, q))

        kl_per_dim_vmap = jax.vmap(kl_per_dim, in_axes=(1, 1))
        return jnp.mean(kl_per_dim_vmap(pred_ensemble, true_ensemble))


class GaussianKLDivergence(ProbabilityMetric):
    """KL(pred || true) assuming both ensembles are Gaussian.

    Fits a mean and sample covariance to each ensemble, then evaluates the
    analytical Gaussian KL divergence. A small diagonal regularization is
    added to the covariances to handle rank-deficient ensembles.
    """

    def __init__(
        self,
        time_aggregation: Optional[str] = None,
        regularization: float = 1e-6,
    ) -> None:
        super().__init__("gaussian_kl", time_aggregation)
        self.regularization = regularization

    def compute(
        self, pred_ensemble: jnp.ndarray, true_ensemble: jnp.ndarray
    ) -> jnp.ndarray:
        # pred_ensemble: (n_pred, n_state), true_ensemble: (n_true, n_state)
        n_pred, n_state = pred_ensemble.shape
        n_true = true_ensemble.shape[0]

        mu_p = jnp.mean(pred_ensemble, axis=0)
        mu_q = jnp.mean(true_ensemble, axis=0)

        pred_c = pred_ensemble - mu_p
        true_c = true_ensemble - mu_q

        sigma_p = (pred_c.T @ pred_c) / (n_pred - 1) + self.regularization * jnp.eye(
            n_state
        )
        sigma_q = (true_c.T @ true_c) / (n_true - 1) + self.regularization * jnp.eye(
            n_state
        )

        return _gaussian_kl(mu_p, sigma_p, mu_q, sigma_q)
