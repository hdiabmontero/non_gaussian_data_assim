"""Sequential-importance-resampling (SIR / bootstrap) particle filter."""

from typing import Any

import jax
import jax.numpy as jnp

from non_gaussian_data_assim.da_methods.base import BaseDataAssimilationMethod
from non_gaussian_data_assim.forward_models.base import BaseForwardModel
from non_gaussian_data_assim.observations.observation_operator import (
    ObservationOperator,
)


def _systematic_resample(weights: jnp.ndarray, rng_key: jax.Array) -> jnp.ndarray:
    """Systematic resampling. Returns ``ensemble_size`` indices drawn from ``weights``."""
    n = weights.shape[0]
    cumulative = jnp.cumsum(weights)
    u0 = jax.random.uniform(rng_key, ()) / n
    u = u0 + jnp.arange(n) / n
    indices = jnp.searchsorted(cumulative, u)
    return jnp.clip(indices, 0, n - 1)


class ParticleFilter(BaseDataAssimilationMethod):
    """Bootstrap (SIR) particle filter.

    The forecast step is the deterministic propagation of each particle
    through the forward model (handled by the base class). The analysis step:

      1. Computes log-likelihoods of each particle under
         :math:`y \\sim \\mathcal{N}(H(x_i), R)`.
      2. Normalizes weights with the log-sum-exp trick.
      3. Computes the effective sample size
         :math:`N_\\text{eff} = 1 / \\sum_i w_i^2`.
      4. Systematically resamples when
         ``N_eff < resample_threshold * ensemble_size``.
      5. Optionally jitters resampled particles by
         ``jitter_scale * N(0, I)`` to combat sample impoverishment.

    The bootstrap PF is asymptotically exact but degenerates rapidly in
    high dimensions; for L96 / Kuramoto it is included mainly as a baseline.
    """

    def __init__(
        self,
        ensemble_size: int,
        R: jnp.ndarray,
        obs_operator: ObservationOperator,
        forward_operator: BaseForwardModel,
        name: str = "particle_filter",
        resample_threshold: float = 0.5,
        jitter_scale: float = 0.0,
    ) -> None:
        super().__init__(name, obs_operator, forward_operator)
        self.ensemble_size = ensemble_size
        self.R = R
        self.R_inv = jnp.linalg.inv(R)
        self.num_states = forward_operator.num_states
        self.state_dim = forward_operator.state_dim
        self.resample_threshold = resample_threshold
        self.jitter_scale = jitter_scale

    def _analysis_step(
        self,
        prior_ensemble: jnp.ndarray,
        obs_vect: jnp.ndarray,
        rng_key: jax.random.PRNGKey,
        **kwargs: Any,
    ) -> jnp.ndarray:
        # Predicted observations for each particle: [ensemble, num_obs].
        predicted_obs = self.obs_operator(prior_ensemble)

        # Log-likelihood under N(., R).
        residual = predicted_obs - obs_vect[None, :]
        log_w = -0.5 * jnp.einsum("ei,ij,ej->e", residual, self.R_inv, residual)

        # Numerically stable normalization.
        log_w = log_w - jnp.max(log_w)
        weights = jnp.exp(log_w)
        weights = weights / jnp.sum(weights)

        # Effective sample size.
        n_eff = 1.0 / jnp.sum(weights**2)
        threshold = self.resample_threshold * self.ensemble_size

        resample_key, jitter_key = jax.random.split(rng_key)

        def _resample(_: Any) -> jnp.ndarray:
            indices = _systematic_resample(weights, resample_key)
            resampled = prior_ensemble[indices]
            if self.jitter_scale > 0.0:
                resampled = resampled + self.jitter_scale * jax.random.normal(
                    jitter_key, resampled.shape
                )
            return resampled

        def _no_resample(_: Any) -> jnp.ndarray:
            return prior_ensemble

        return jax.lax.cond(n_eff < threshold, _resample, _no_resample, operand=None)
