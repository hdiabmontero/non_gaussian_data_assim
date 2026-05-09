import pdb

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from non_gaussian_data_assim.da_methods.agmf import AdaptiveGaussianMixtureFilter
from non_gaussian_data_assim.da_methods.base import da_rollout
from non_gaussian_data_assim.da_methods.enkf import EnsembleKalmanFilter

# from non_gaussian_data_assim.da_methods.enkf_loc import EnsembleKalmanFilterLocalization
# from non_gaussian_data_assim.da_methods.particle_filter import ParticleFilter
# from non_gaussian_data_assim.da_methods.pff import ParticleFlowFilter
# from non_gaussian_data_assim.da_methods.pff_loc import ParticleFlowFilterLocalization
from non_gaussian_data_assim.da_methods.pff import ParticleFlowFilter
from non_gaussian_data_assim.forward_models.kuramoto_sivashinsky import (
    KuramotoSivashinsky,
)
from non_gaussian_data_assim.observation_operator import ObservationOperator

np.random.seed(42)

# Constants and parameters
DT = 0.05
NU = 1.0
OUTER_STEPS = 500
NUM_STATES = 1
STATE_DIM = 512
INNER_STEPS = 4
NUM_SKIP_OBS = 4
ENSEMBLE_SIZE = 100
DOMAIN_LENGTH = 100

# Observation ids
OBS_IDS = jnp.arange(0, STATE_DIM, NUM_SKIP_OBS)
OBS_STATES = (0,)

# Observation error covariance matrix
R = jnp.eye(len(OBS_IDS)) * 0.25

x = jnp.linspace(start=0, stop=DOMAIN_LENGTH, num=STATE_DIM)

# initial condition
X_0_FN = lambda magnitude: magnitude * jnp.cos(
    (2 * jnp.pi * x) / DOMAIN_LENGTH
) + magnitude * jnp.cos((4 * jnp.pi * x) / DOMAIN_LENGTH)


def main() -> None:
    """Main function."""

    rng_key = jax.random.PRNGKey(42)

    # Define the forward model
    forward_model = KuramotoSivashinsky(
        dt=DT, inner_steps=INNER_STEPS, state_dim=STATE_DIM, domain_length=DOMAIN_LENGTH
    )

    # Rollout the true solution
    X_0 = X_0_FN(0.1).reshape(1, 1, STATE_DIM)
    true_sol = forward_model.rollout(X_0, OUTER_STEPS - 1)

    # Define the observation operator
    obs_operator = ObservationOperator(
        obs_states=OBS_STATES, obs_indices=OBS_IDS, state_dim=STATE_DIM
    )

    # Generate observations
    observations = jnp.zeros((OUTER_STEPS, len(OBS_IDS)))
    for i in range(OUTER_STEPS):
        obs_at_t = obs_operator(true_sol[:, i])  # [1, num_obs]
        rng_key, key = jax.random.split(rng_key)
        obs_at_t = obs_at_t + jax.random.multivariate_normal(
            key, jnp.zeros(len(OBS_IDS)), R
        )  # np.sqrt(R)
        observations = observations.at[i].set(obs_at_t.flatten())  # [num_obs]

    # Define the data assimilation model
    da_model = AdaptiveGaussianMixtureFilter(
        ensemble_size=ENSEMBLE_SIZE,
        R=R,
        obs_operator=obs_operator,
        forward_operator=forward_model,
        inflation_factor=8.0,
        localization_distance=10,
        w_prev=np.ones(ENSEMBLE_SIZE) / ENSEMBLE_SIZE,
        nc_threshold=0.5,
    )

    # Initialize the prior ensemble
    rng_key, key = jax.random.split(rng_key)
    magnitude_samples = jax.random.uniform(
        key, (ENSEMBLE_SIZE,), minval=0.0, maxval=1.5
    )
    prior_ensemble = jnp.array([X_0_FN(magnitude) for magnitude in magnitude_samples])
    prior_ensemble = prior_ensemble.reshape(ENSEMBLE_SIZE, NUM_STATES, STATE_DIM)

    # Initialize the posterior ensemble
    posterior_ensemble = prior_ensemble.copy()
    posterior_ensemble = posterior_ensemble.reshape(
        ENSEMBLE_SIZE, 1, NUM_STATES, STATE_DIM
    )

    # Rollout the prior ensemble
    prior_ensemble = forward_model.rollout(prior_ensemble, OUTER_STEPS - 1)

    # Perform the data assimilation
    rng_key, key = jax.random.split(rng_key)
    posterior_ensemble = da_model.rollout(
        posterior_ensemble[:, 0], observations[1:], key
    )

    # Perform the data assimilation
    # for i in tqdm(range(1, OUTER_STEPS)):
    #     rng_key, key = jax.random.split(rng_key)
    #     posterior_next = da_model(
    #         prior_ensemble=posterior_ensemble[:, i - 1], obs_vect=observations[:, i], rng_key=key
    #     )
    #     posterior_ensemble = jnp.concatenate(
    #         [posterior_ensemble, posterior_next[:, None, :, :]], axis=1
    #     )

    true_sol = true_sol.reshape(OUTER_STEPS, STATE_DIM)

    # Calculate the prior and posterior errors
    prior_error = true_sol - prior_ensemble.mean(axis=(0, 2))
    prior_error = np.sqrt(np.sum(prior_error**2))
    posterior_error = true_sol - posterior_ensemble.mean(axis=(0, 2))
    posterior_error = np.sqrt(np.sum(posterior_error**2))

    print(f"Prior error: {prior_error}")
    print(f"Posterior error: {posterior_error}")

    idx_to_plot = 17

    plt.figure()
    for i, (state, state_name) in enumerate(
        zip(
            [
                true_sol,
                prior_ensemble.mean(axis=(0, 2)),
                posterior_ensemble.mean(axis=(0, 2)),
                true_sol - prior_ensemble.mean(axis=(0, 2)),
                true_sol - posterior_ensemble.mean(axis=(0, 2)),
            ],
            [
                "True Solution",
                "Prior Ensemble Mean",
                "Posterior Ensemble Mean",
                "|True - Prior| difference",
                "|True - Posterior| difference",
            ],
        )
    ):
        plt.subplot(2, 3, i + 1)
        plt.imshow(state, origin="lower", vmin=true_sol.min(), vmax=true_sol.max())
        plt.colorbar()
        plt.title(state_name)

    plt.subplot(2, 3, 6)
    # Shade the standard deviation of the posterior on the time series plot
    mean_post = posterior_ensemble.mean(axis=(0, 2))[:, idx_to_plot]
    std_post = posterior_ensemble.std(axis=(0, 2))[:, idx_to_plot]
    time_axis = np.arange(posterior_ensemble.shape[1])
    plt.fill_between(
        time_axis,
        mean_post - std_post,
        mean_post + std_post,
        color="tab:blue",
        alpha=0.2,
        label="Posterior ± Std",
    )
    for state_at_point, state_name, color in zip(
        [
            prior_ensemble.mean(axis=(0, 2))[:, idx_to_plot],
            mean_post,
            true_sol[:, idx_to_plot],
        ],
        ["Prior Ensemble Mean", "Posterior Ensemble Mean", "True Solution"],
        ["tab:red", "tab:blue", "black"],
    ):
        plt.plot(
            state_at_point,
            label=state_name,
            color=color,
            linewidth=3,
            linestyle="--" if state_name == "True Solution" else "-",
        )
    plt.legend()
    plt.xlabel("Time")
    plt.ylabel("State 25")
    plt.ylim(-10, 10)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
