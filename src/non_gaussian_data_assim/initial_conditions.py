"""Helpers for unified DA experiments: initial states and prior ensembles."""

import jax
import jax.numpy as jnp


def random_normal_initial_state(
    rng_key: jax.Array,
    num_states: int,
    state_dim: int,
    scale: float = 1.0,
    periodic_boundary: bool = False,
) -> jnp.ndarray:
    """Sample a single random-normal initial state of shape [1, num_states, state_dim]."""
    X_0 = jax.random.normal(rng_key, (1, num_states, state_dim)) * scale
    if periodic_boundary:
        X_0 = X_0.at[0, 0, -1].set(X_0[0, 0, 0])
    return X_0


def random_normal_prior_ensemble(
    rng_key: jax.Array,
    ensemble_size: int,
    num_states: int,
    state_dim: int,
    scale: float = 1.0,
    periodic_boundary: bool = False,
) -> jnp.ndarray:
    """Sample a random-normal prior ensemble of shape [ensemble_size, num_states, state_dim]."""
    ensemble = (
        jax.random.normal(rng_key, (ensemble_size, num_states, state_dim)) * scale
    )
    if periodic_boundary:
        ensemble = ensemble.at[:, :, -1].set(ensemble[:, :, 0])
    return ensemble


def kuramoto_cosine_initial_state(
    rng_key: jax.Array,
    state_dim: int,
    domain_length: float,
    magnitude: float,
) -> jnp.ndarray:
    """Deterministic two-mode cosine initial state for Kuramoto-Sivashinsky."""
    del rng_key
    x = jnp.linspace(0.0, domain_length, state_dim)
    profile = magnitude * jnp.cos(2 * jnp.pi * x / domain_length) + magnitude * jnp.cos(
        4 * jnp.pi * x / domain_length
    )
    return profile.reshape(1, 1, state_dim)


def kuramoto_cosine_prior_ensemble(
    rng_key: jax.Array,
    ensemble_size: int,
    state_dim: int,
    domain_length: float,
    magnitude_min: float,
    magnitude_max: float,
) -> jnp.ndarray:
    """Prior ensemble for Kuramoto-Sivashinsky with magnitudes drawn uniformly."""
    x = jnp.linspace(0.0, domain_length, state_dim)
    magnitudes = jax.random.uniform(
        rng_key, (ensemble_size,), minval=magnitude_min, maxval=magnitude_max
    )

    def profile(m: jnp.ndarray) -> jnp.ndarray:
        return m * jnp.cos(2 * jnp.pi * x / domain_length) + m * jnp.cos(
            4 * jnp.pi * x / domain_length
        )

    profiles = jax.vmap(profile)(magnitudes)
    return profiles.reshape(ensemble_size, 1, state_dim)
