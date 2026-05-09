from abc import abstractmethod
from typing import Optional

import jax
import jax.numpy as jnp

AGGREGATION_METHODS = {
    "none": lambda x, axis: x,
    "mean": lambda x, axis: jnp.mean(x, axis=axis),
    "median": lambda x, axis: jnp.median(x, axis=axis),
    "max": lambda x, axis: jnp.max(x, axis=axis),
    "min": lambda x, axis: jnp.min(x, axis=axis),
    "std": lambda x, axis: jnp.std(x, axis=axis),
    "var": lambda x, axis: jnp.var(x, axis=axis),
}


class Metric:
    def __init__(
        self,
        name: str,
        ensemble_aggregation: Optional[str] = None,
        time_aggregation: Optional[str] = None,
    ) -> None:
        self.name = name
        self.ensemble_aggregation = (
            ensemble_aggregation if ensemble_aggregation is not None else "none"
        )
        self.time_aggregation = (
            time_aggregation if time_aggregation is not None else "none"
        )

    @abstractmethod
    def compute(self, pred: jnp.ndarray, truth: jnp.ndarray) -> jnp.ndarray:
        """Compute metric for a single ensemble member at a single time step.

        Args:
            pred: state vector for one member at one time, shape (n_state,)
            truth: ground truth state vector at that time, shape (n_state,)
        """
        raise NotImplementedError

    def __call__(self, preds: jnp.ndarray, truth: jnp.ndarray) -> jnp.ndarray:
        """Compute metric over all ensemble members and time steps, then aggregate.

        Args:
            preds: shape (n_ensemble, n_time, n_states, n_dim)
            truth: shape (n_time, n_states, n_dim)

        Returns:
            Aggregated metric. Shape depends on aggregation settings:
            - both None:              (n_ensemble, n_time)
            - ensemble only:          (n_time,)
            - time only:              (n_ensemble,)
            - both:                   scalar
        """
        compute_over_time = jax.vmap(self.compute, in_axes=(0, 0))
        compute_over_ensemble_and_time = jax.vmap(compute_over_time, in_axes=(0, None))

        # result: (n_ensemble, n_time)
        result = compute_over_ensemble_and_time(preds, truth)

        # aggregate time first (axis=1) so ensemble stays at axis=0
        result = AGGREGATION_METHODS[self.time_aggregation](result, axis=1)
        result = AGGREGATION_METHODS[self.ensemble_aggregation](result, axis=0)

        return result


class RMSE(Metric):
    def __init__(
        self,
        ensemble_aggregation: Optional[str] = None,
        time_aggregation: Optional[str] = None,
    ) -> None:
        super().__init__("rmse", ensemble_aggregation, time_aggregation)

    def compute(self, pred: jnp.ndarray, truth: jnp.ndarray) -> jnp.ndarray:
        return jnp.sqrt(jnp.mean((pred - truth) ** 2))


class MAE(Metric):
    def __init__(
        self,
        ensemble_aggregation: Optional[str] = None,
        time_aggregation: Optional[str] = None,
    ) -> None:
        super().__init__("mae", ensemble_aggregation, time_aggregation)

    def compute(self, pred: jnp.ndarray, truth: jnp.ndarray) -> jnp.ndarray:
        return jnp.mean(jnp.abs(pred - truth))


class MAPE(Metric):
    def __init__(
        self,
        ensemble_aggregation: Optional[str] = None,
        time_aggregation: Optional[str] = None,
    ) -> None:
        super().__init__("mape", ensemble_aggregation, time_aggregation)

    def compute(self, pred: jnp.ndarray, truth: jnp.ndarray) -> jnp.ndarray:
        return jnp.mean(jnp.abs(pred - truth) / (jnp.abs(truth) + 1e-6))


def print_metrics_table(
    reference_errors: dict[str, float],
    posterior_errors: dict[str, float],
    title: str = "",
) -> None:
    col_w = 14
    header = f"{'Metric':<10} | {'Reference':>{col_w}} | {'Posterior':>{col_w}}"
    sep = "-" * len(header)
    if title:
        print(f"\n{title}")
    print(sep)
    print(header)
    print(sep)
    for metric in reference_errors:
        reference_val = float(reference_errors[metric])
        posterior_val = float(posterior_errors[metric])
        print(
            f"{metric.upper():<10} | {reference_val:{col_w}.6f} | {posterior_val:{col_w}.6f}"
        )
    print(sep)
