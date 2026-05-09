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


class EnsembleMetric:
    """Score an ensemble forecast against a truth trajectory.

    Subclasses implement ``compute`` for a single time step; ``__call__``
    vmaps that over the time axis and applies an optional time aggregation.
    """

    def __init__(self, name: str, time_aggregation: Optional[str] = None) -> None:
        self.name = name
        self.time_aggregation = (
            time_aggregation if time_aggregation is not None else "none"
        )

    @abstractmethod
    def compute(self, ensemble: jnp.ndarray, truth: jnp.ndarray) -> jnp.ndarray:
        """Compute metric for a single time step.

        Args:
            ensemble: shape (n_ensemble, *state_shape)
            truth:    shape (*state_shape,)

        Returns:
            Scalar score for this time step.
        """
        raise NotImplementedError

    def __call__(self, ensemble: jnp.ndarray, truth: jnp.ndarray) -> jnp.ndarray:
        """Compute metric over all time steps, then aggregate.

        Args:
            ensemble: shape (n_ensemble, n_time, *state_shape)
            truth:    shape (n_time, *state_shape)

        Returns:
            Aggregated metric. Shape depends on time_aggregation:
            - None:      (n_time,)
            - otherwise: scalar
        """
        compute_over_time = jax.vmap(self.compute, in_axes=(1, 0))
        result = compute_over_time(ensemble, truth)  # (n_time,)
        return AGGREGATION_METHODS[self.time_aggregation](result, axis=0)


def _crps_pointwise(ensemble_vals: jnp.ndarray, truth_val: jnp.ndarray) -> jnp.ndarray:
    """Fair CRPS for one scalar truth and its E ensemble samples.

    Uses the sort identity
        sum_{i,j} |x_i - x_j| = 2 * sum_k (2k - E - 1) * x_(k)
    (1-indexed, sorted ascending) so the spread term is O(E log E) instead of
    the O(E^2) pairwise form.
    """
    E = ensemble_vals.shape[0]
    mae_term = jnp.mean(jnp.abs(ensemble_vals - truth_val))
    sorted_ens = jnp.sort(ensemble_vals)
    i = jnp.arange(E)
    spread_term = jnp.sum((2 * i + 1 - E) * sorted_ens) / (E * (E - 1))
    return mae_term - spread_term


class CRPS(EnsembleMetric):
    """Continuous Ranked Probability Score, averaged over all state dimensions.

    For each entry of ``truth`` at a given time step this evaluates
        CRPS(F, y) = E|X - y| - 0.5 * E|X - X'|
    using the sort-based fair estimator, then averages over the flattened
    state to return a scalar per time step.
    """

    def __init__(self, time_aggregation: Optional[str] = None) -> None:
        super().__init__("crps", time_aggregation)

    def compute(self, ensemble: jnp.ndarray, truth: jnp.ndarray) -> jnp.ndarray:
        truth_flat = truth.reshape(-1)
        ensemble_flat = ensemble.reshape(ensemble.shape[0], -1)
        pointwise = jax.vmap(_crps_pointwise, in_axes=(1, 0))(ensemble_flat, truth_flat)
        return jnp.mean(pointwise)
