import pdb
import time
from dataclasses import dataclass
from typing import Callable

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from jax.scipy.stats import gaussian_kde
from scipy.io import loadmat
from tqdm import tqdm

from non_gaussian_data_assim.da_methods.agmf import AdaptiveGaussianMixtureFilter
from non_gaussian_data_assim.da_methods.enkf import EnsembleKalmanFilter
from non_gaussian_data_assim.da_methods.pff import ParticleFlowFilter
from non_gaussian_data_assim.forward_models.identity import IdentityModel
from non_gaussian_data_assim.forward_models.sine import SineModel
from non_gaussian_data_assim.metrics.probability_metrics import KLDivergence
from non_gaussian_data_assim.observation_operator import (
    LinearObservationOperator,
    SineObservationOperator,
    SineObservationOperatorNoError,
)

jax.config.update("jax_disable_jit", False)


def get_2d_kde(
    samples: jnp.ndarray,
    nbins: int,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
) -> jnp.ndarray:
    """Get the 2D KDE."""
    k = gaussian_kde(samples.T)
    xi, yi = np.mgrid[
        x_range[0] : x_range[1] : nbins * 1j,  # type: ignore[misc]
        y_range[0] : y_range[1] : nbins * 1j,  # type: ignore[misc]
    ]
    zi = k(np.vstack([xi.flatten(), yi.flatten()]))
    zi = zi.reshape(xi.shape)
    return xi, yi, zi


def prepare_samples_raw(
    samples: jnp.ndarray,
    nbins: int,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray, np.ndarray], np.ndarray]:
    """Prepare samples for KL-div and plotting."""

    xi, yi, zi = get_2d_kde(samples, nbins, x_range, y_range)

    diag_samples = np.diag(zi)
    return samples, (xi, yi, zi), diag_samples


@dataclass
class PreparedSamples:
    """Holds samples and KDE grid data for plotting and KL divergence."""

    samples: np.ndarray
    xi: np.ndarray
    yi: np.ndarray
    zi: np.ndarray
    diag: np.ndarray


def prepare_samples(
    samples: jnp.ndarray,
    nbins: int,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
) -> PreparedSamples:
    """Prepare samples for KL-div and plotting; returns PreparedSamples dataclass."""

    raw_samples, (xi, yi, zi), diag = prepare_samples_raw(
        samples, nbins, x_range, y_range
    )
    return PreparedSamples(samples=raw_samples, xi=xi, yi=yi, zi=zi, diag=diag)


def kalman_gain(
    x0: jnp.ndarray,
    target_cov: jnp.ndarray,
    obs_matrix: jnp.ndarray,
    obs_cov: jnp.ndarray,
    observations: jnp.ndarray,
) -> jnp.ndarray:
    """Kalman gain."""
    K = target_cov @ obs_matrix.T
    return K @ jnp.linalg.inv(obs_matrix @ target_cov @ obs_matrix.T + obs_cov)


def get_true_posterior(
    x0: jnp.ndarray,
    target_mean: jnp.ndarray,
    target_cov: jnp.ndarray,
    obs_matrix: jnp.ndarray,
    obs_cov: jnp.ndarray,
    observations: jnp.ndarray,
    rng_key: jnp.ndarray,
    num_samples: int = 100,
) -> jnp.ndarray:
    """Get the true posterior."""
    K = kalman_gain(x0, target_cov, obs_matrix, obs_cov, observations)

    posterior_mean = target_mean + K @ (observations - obs_matrix @ target_mean)
    posterior_cov = (jnp.eye(x0.shape[1]) - K @ obs_matrix) @ target_cov

    posterior_samples = jax.random.multivariate_normal(
        rng_key, posterior_mean, posterior_cov, num_samples
    )

    return posterior_samples


SEED = 42

ENSEMBLE_SIZE = 500

DA_METHOD = "enkf"
DA_METHODS = {
    "enkf": EnsembleKalmanFilter,
    "agmf": AdaptiveGaussianMixtureFilter,
    "pff": ParticleFlowFilter,
}
SPECIFIC_DA_ARGS = {
    "enkf": {
        "inflation_factor": 1.0,
    },
    "agmf": {
        "inflation_factor": 1.0,
        "nc_threshold": 0.5,
        "w_prev": np.ones(ENSEMBLE_SIZE) / ENSEMBLE_SIZE,
    },
    "pff": {
        # "return_pff_trajectory": True,
        "num_pseudo_time_steps": 200,
        "step_size": 1 / 10,
        "alpha": 1 / 10,
        # "stepper": "runge_kutta_4",
        "stepper": "forward_euler",
        # "stepper": "backward_euler",
    },
}

NUM_STATES = 1
STATE_DIM = 2

# Domain
X_RANGE = (-1, 7)
Y_RANGE = (-1, 7)

# Observation ids
OBS_IDS = np.arange(0, STATE_DIM)
OBS_STATES = (0,)

# Observation error covariance matrix
R = jnp.eye(len(OBS_IDS))
B = jnp.eye(STATE_DIM)


def main() -> None:
    """Main function."""
    rng_key = jax.random.PRNGKey(SEED)

    # Define the forward model
    forward_model = IdentityModel(state_dim=STATE_DIM)

    # Define the observation operator
    rng_key, key = jax.random.split(rng_key)
    obs_operator = LinearObservationOperator(
        obs_states=OBS_STATES,
        obs_indices=OBS_IDS,
        state_dim=STATE_DIM,
    )

    # Generate observations
    observations = jnp.array([1.0, 1.0])

    da_model = DA_METHODS[DA_METHOD](
        ensemble_size=ENSEMBLE_SIZE,
        R=R,
        obs_operator=obs_operator,
        forward_operator=forward_model,
        **SPECIFIC_DA_ARGS[DA_METHOD],
    )

    # Initialize the prior ensemble
    rng_key, key = jax.random.split(rng_key)
    prior_ensemble = (
        jax.random.normal(key, (ENSEMBLE_SIZE, NUM_STATES, STATE_DIM)) + 5.0
    )

    prior_samples = prepare_samples(prior_ensemble[:, 0], 100, X_RANGE, Y_RANGE)

    # Initialize the posterior ensemble
    posterior_ensemble = prior_ensemble.copy()

    # Perform the data assimilation
    posterior_ensemble = posterior_ensemble.reshape(
        ENSEMBLE_SIZE, 1, NUM_STATES, STATE_DIM
    )
    rng_key, key = jax.random.split(rng_key)
    t0 = time.time()
    posterior_ensemble = da_model(
        prior_ensemble=posterior_ensemble[:, -1],
        obs_vect=observations,
        rng_key=key,
        return_model_integration_steps=False,
    )
    t1 = time.time()
    print(f"Time taken: {t1 - t0} seconds")

    posterior_ensemble = posterior_ensemble[:, 0]
    posterior_samples = prepare_samples(posterior_ensemble, 100, X_RANGE, Y_RANGE)

    posterior_mean = posterior_ensemble.mean(axis=0)
    posterior_cov = jnp.cov(posterior_ensemble.T)

    print(f"Posterior mean: {posterior_mean}")
    print(f"Posterior cov: \n {posterior_cov}")

    likelihood_ensemble = jax.random.normal(key, (ENSEMBLE_SIZE, STATE_DIM)) + 1.0
    likelihood_samples = prepare_samples(likelihood_ensemble, 100, X_RANGE, Y_RANGE)

    rng_key, key = jax.random.split(rng_key)
    true_posterior_samples = get_true_posterior(
        x0=prior_ensemble[:, 0],
        target_mean=jnp.array([5.0, 5.0]),
        target_cov=jnp.array([[1.0, 0.0], [0.0, 1.0]]),
        obs_matrix=jnp.array([[1.0, 0.0], [0.0, 1.0]]),
        obs_cov=R,
        observations=observations,
        rng_key=key,
    )
    true_posterior_samples = prepare_samples(
        true_posterior_samples, 100, X_RANGE, Y_RANGE
    )
    # KL divergence: compare prior and posterior against the true posterior
    kl = KLDivergence(time_aggregation="mean")

    def to_metric_shape(samples: jnp.ndarray) -> jnp.ndarray:
        """Reshape (n_ensemble, state_dim) → (n_ensemble, n_time=1, n_states=1, state_dim)."""
        return samples.reshape(samples.shape[0], 1, 1, STATE_DIM)

    true_post = to_metric_shape(jnp.array(true_posterior_samples.samples))
    kl_prior = kl(to_metric_shape(prior_ensemble[:, 0]), true_post)
    kl_posterior = kl(to_metric_shape(posterior_ensemble), true_post)

    print(f"KL(prior     || true posterior): {float(kl_prior):.4f}")
    print(f"KL(posterior || true posterior): {float(kl_posterior):.4f}")

    # plot a density
    plt.figure(figsize=(20, 10))
    plt.subplot(3, 3, 1)
    plt.pcolormesh(
        true_posterior_samples.xi,
        true_posterior_samples.yi,
        true_posterior_samples.zi,
        shading="gouraud",
    )
    plt.title("True Posterior")
    plt.subplot(3, 3, 2)
    plt.pcolormesh(
        prior_samples.xi, prior_samples.yi, prior_samples.zi, shading="gouraud"
    )
    plt.title("Prior")
    plt.subplot(3, 3, 3)
    plt.pcolormesh(
        likelihood_samples.xi,
        likelihood_samples.yi,
        likelihood_samples.zi,
        shading="gouraud",
    )
    plt.title("Likelihood")
    plt.subplot(3, 3, 4)
    plt.pcolormesh(
        posterior_samples.xi,
        posterior_samples.yi,
        posterior_samples.zi,
        shading="gouraud",
    )
    plt.title("Posterior")
    plt.subplot(3, 3, 5)
    plt.plot(posterior_samples.diag, label="Posterior")
    plt.plot(likelihood_samples.diag, label="Likelihood")
    plt.plot(prior_samples.diag, label="Prior")
    plt.plot(true_posterior_samples.diag, label="True Posterior")
    plt.legend()
    plt.title("Diagonal values")

    plt.subplot(3, 3, 6)
    plt.pcolormesh(
        true_posterior_samples.xi,
        true_posterior_samples.yi,
        jnp.abs(true_posterior_samples.zi - posterior_samples.zi),
        shading="gouraud",
    )
    plt.title(
        f"|True Posterior - Posterior|, max: {jnp.max(jnp.abs(true_posterior_samples.zi - posterior_samples.zi)):.2f}"
    )
    plt.colorbar()

    plt.show()


if __name__ == "__main__":
    main()
