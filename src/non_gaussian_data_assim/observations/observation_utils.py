"""Utilities for generating observations from a true trajectory."""

import jax
import jax.numpy as jnp

from non_gaussian_data_assim.observations.observation_operator import (
    ObservationOperator,
)


def generate_observations(
    rng_key: jax.Array,
    true_sol: jnp.ndarray,
    obs_operator: ObservationOperator,
    R: jnp.ndarray,
    data_assimilation_steps: int,
    model_integration_steps: int,
) -> jnp.ndarray:
    """Sample noisy observations of the true trajectory at each outer step.

    Args:
        rng_key: PRNG key used for the observation noise.
        true_sol: Truth trajectory with shape [1, total_steps, num_states, state_dim].
        obs_operator: Observation operator applied to each truth snapshot.
        R: Observation-noise covariance matrix.
        data_assimilation_steps: Number of assimilation steps.
        model_integration_steps: Number of inner integration steps between assimilation steps.

    Returns:
        Observations array of shape [data_assimilation_steps, obs_operator.num_obs].
    """
    observations = jnp.zeros((data_assimilation_steps, obs_operator.num_obs))  # type: ignore[attr-defined]
    for i in range(data_assimilation_steps):
        obs_at_t = obs_operator(true_sol[:, 1 + model_integration_steps * (i + 1)])
        rng_key, key = jax.random.split(rng_key)
        obs_at_t = obs_at_t + jax.random.multivariate_normal(
            key, jnp.zeros(obs_operator.num_obs), R  # type: ignore[attr-defined]
        )
        observations = observations.at[i].set(obs_at_t.flatten())
    return observations
