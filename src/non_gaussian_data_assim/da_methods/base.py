import functools
from abc import abstractmethod
from typing import Any, Callable, Optional

import jax
import jax.numpy as jnp
import numpy as np

from non_gaussian_data_assim.forward_models.base import BaseForwardModel
from non_gaussian_data_assim.observations.observation_operator import (
    ObservationOperator,
)


def da_rollout(
    da_model: Callable,
    observations: jnp.ndarray,
    rng_key: jax.random.PRNGKey,
    include_initial_state: bool = True,
) -> Callable:
    """Rollout the data assimilation system using the given DA model.

    Args:
        da_model: The data assimilation model callable that takes
            (prior_ensemble, obs_vect, rng_key) and returns the posterior ensemble.
        observations: Array of observations with shape (num_steps, obs_dim).
            The first observation (index 0) is the initial observation at time 0.
        rng_key: JAX random key for random operations.
        include_initial_state: Whether to include the initial state in the output.
            If True, output has shape (ensemble_size, num_steps, ...).
            If False, output has shape (ensemble_size, num_steps - 1, ...).

    Returns:
        A function that takes the initial ensemble and returns the trajectory.
    """
    initial_rng_key = rng_key

    def scan_fn(
        state: tuple[jnp.ndarray, jax.random.PRNGKey], obs_vect: jnp.ndarray
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Scan function for data assimilation."""
        posterior_state, rng_key = state
        rng_key, key = jax.random.split(rng_key)
        next_state = da_model(
            prior_ensemble=posterior_state, obs_vect=obs_vect, rng_key=key
        )
        return (next_state, rng_key), next_state

    def da_rollout_fn(init_ensemble: jnp.ndarray) -> jnp.ndarray:
        """Data assimilation rollout function."""
        initial_state = (init_ensemble, initial_rng_key)
        _, posterior_trajectory = jax.lax.scan(scan_fn, initial_state, xs=observations)

        # posterior_trajectory has shape (num_steps - 1, ensemble_size, ...)
        # Transpose to (ensemble_size, num_steps - 1, ...)
        posterior_trajectory = jnp.transpose(
            posterior_trajectory, (1, 0) + tuple(range(2, posterior_trajectory.ndim))
        )

        if include_initial_state:
            # Add initial state at the beginning
            return jnp.concatenate(
                [init_ensemble[:, None, ...], posterior_trajectory], axis=1
            )

        return posterior_trajectory

    return da_rollout_fn


class BaseDataAssimilationMethod:
    """Base class for data assimilation methods."""

    def __init__(
        self,
        name: str,
        obs_operator: ObservationOperator,
        forward_operator: BaseForwardModel,
    ):
        """Initialize the data assimilation method."""
        self.name = name
        self.obs_operator = obs_operator
        self.forward_operator = forward_operator

    def _assimilate_data(
        self,
        prior_ensemble: np.ndarray,
        obs_vect: np.ndarray,
        rng_key: Optional[jax.random.PRNGKey] = None,
        return_model_integration_steps: bool = False,
        **kwargs: Any,
    ) -> np.ndarray:
        """Assimilate the data."""

        forecast_ensemble = self._forecast_step(
            prior_ensemble,
            return_model_integration_steps=return_model_integration_steps,
        )
        analysis_ensemble = self._analysis_step(
            forecast_ensemble[:, -1], obs_vect, rng_key=rng_key, **kwargs
        )
        if return_model_integration_steps:
            analysis_ensemble = jnp.concatenate(
                [forecast_ensemble[:, :-1, :, :], analysis_ensemble[:, None, :, :]],
                axis=1,
            )

        return analysis_ensemble

    @abstractmethod
    def _analysis_step(
        self,
        prior_ensemble: np.ndarray,
        obs_vect: np.ndarray,
        rng_key: jax.random.PRNGKey,
        **kwargs: Any,
    ) -> np.ndarray:
        """
        Assimilate the data.

        Args:
            prior_ensemble (np.ndarray): Prior ensemble [Ensemble size, State dimension].
            obs_vect (np.ndarray): Observation vector.
            rng_key (Optional[jax.random.PRNGKey]): Optional RNG key for random operations.

        Returns:
            np.ndarray: Analysis ensemble [Ensemble size, State dimension].
        """
        raise NotImplementedError

    def _forecast_step(
        self, ensemble: np.ndarray, return_model_integration_steps: bool = False
    ) -> np.ndarray:
        """Forecast the ensemble."""
        return self.forward_operator(
            ensemble,
            return_model_integration_steps=return_model_integration_steps,
            is_ensemble=True,
        )

    def __call__(
        self,
        prior_ensemble: np.ndarray,
        obs_vect: np.ndarray,
        rng_key: Optional[jax.random.PRNGKey] = None,
        return_model_integration_steps: bool = False,
        **kwargs: Any,
    ) -> np.ndarray:
        """Run the data assimilation method."""
        return self._assimilate_data(
            prior_ensemble,
            obs_vect,
            rng_key=rng_key,
            return_model_integration_steps=return_model_integration_steps,
            **kwargs,
        )

    def rollout(
        self,
        prior_ensemble: np.ndarray,
        observations: jnp.ndarray,
        rng_key: Optional[jax.random.PRNGKey] = None,
        return_model_integration_steps: bool = False,
        **kwargs: Any,
    ) -> jnp.ndarray:
        """Rollout the data assimilation method."""

        if return_model_integration_steps:
            da_rollout_fn = da_rollout(
                self._assimilate_data,
                observations,
                rng_key,
                include_initial_state=True,
                **kwargs,
            )
        else:
            da_rollout_fn = da_rollout(
                self._assimilate_data,
                observations,
                rng_key,
                include_initial_state=True,
                **kwargs,
            )

        da_rollout_fn = jax.jit(da_rollout_fn)
        return da_rollout_fn(prior_ensemble)
