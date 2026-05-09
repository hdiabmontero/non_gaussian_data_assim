import pdb
import time

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from non_gaussian_data_assim.da_methods.agmf import AdaptiveGaussianMixtureFilter
from non_gaussian_data_assim.da_methods.base import da_rollout
from non_gaussian_data_assim.da_methods.enkf import EnsembleKalmanFilter
from non_gaussian_data_assim.da_methods.pff import ParticleFlowFilter
from non_gaussian_data_assim.forward_models.lorenz_63 import Lorenz63Model
from non_gaussian_data_assim.forward_models.lorenz_96 import Lorenz96Model
from non_gaussian_data_assim.observation_operator import ObservationOperator

SEED = 42

# Constants and parameters
DT = 0.01
SIGMA = 10.0
BETA = 2.6666666
RHO = 28.0
T0 = 0.0

OUTER_STEPS = 250
INNER_STEPS = 25


NUM_STATES = 1
STATE_DIM = 3
ENSEMBLE_SIZE = 50

# Observation ids
OBS_IDS = np.arange(0, 3)
OBS_STATES = (0,)

# Observation error covariance matrix
R = jnp.eye(len(OBS_IDS)) * 20.0


def main() -> None:
    """Main function."""
    rng_key = jax.random.PRNGKey(SEED)

    # Initial state - shape: [1, num_states, state_dim] for single ensemble member
    rng_key, key = jax.random.split(rng_key)
    X_0 = jax.random.normal(key, (1, NUM_STATES, STATE_DIM)) * 10

    # Define the forward model
    forward_model = Lorenz63Model(
        dt=DT,
        inner_steps=INNER_STEPS,
        sigma=SIGMA,
        beta=BETA,
        rho=RHO,
    )

    # Rollout the true solution
    true_sol = forward_model.rollout(X_0, OUTER_STEPS, return_inner_steps=True)

    # Define the observation operator
    obs_operator = ObservationOperator(
        obs_states=OBS_STATES, obs_indices=OBS_IDS, state_dim=STATE_DIM
    )

    # Generate observations
    observations = jnp.zeros((OUTER_STEPS, len(OBS_IDS)))
    for i in range(0, OUTER_STEPS):
        obs_at_t = obs_operator(true_sol[:, 1 + INNER_STEPS * (i + 1)])  # [1, num_obs]

        rng_key, key = jax.random.split(rng_key)
        obs_at_t = obs_at_t + jax.random.multivariate_normal(
            key, jnp.zeros(len(OBS_IDS)), R
        )  # np.sqrt(R)
        observations = observations.at[i].set(obs_at_t.flatten())  # [num_obs]

    # Instantiate the data assimilation model
    # da_model = ParticleFlowFilter(
    #     ensemble_size=ENSEMBLE_SIZE,
    #     R=R,
    #     obs_operator=obs_operator,
    #     forward_operator=forward_model,
    #     num_pseudo_time_steps=75,
    #     step_size=3.0,
    #     stepper="backward_euler",
    # )
    da_model = EnsembleKalmanFilter(
        ensemble_size=ENSEMBLE_SIZE,
        R=R,
        obs_operator=obs_operator,
        forward_operator=forward_model,
    )

    # Initialize the prior ensemble
    rng_key, key = jax.random.split(rng_key)
    prior_ensemble = jax.random.normal(key, (ENSEMBLE_SIZE, NUM_STATES, STATE_DIM)) * 10

    # Initialize the posterior ensemble
    posterior_ensemble = prior_ensemble.copy()
    posterior_ensemble = posterior_ensemble.reshape(
        ENSEMBLE_SIZE, 1, NUM_STATES, STATE_DIM
    )

    # Rollout the prior ensemble
    prior_ensemble = forward_model.rollout(
        prior_ensemble, OUTER_STEPS, return_inner_steps=True
    )

    # Perform the data assimilation
    # rng_key, key = jax.random.split(rng_key)
    # t0 = time.time()
    # posterior_ensemble = da_model.rollout(
    #     posterior_ensemble[:, 0], observations[1:], rng_key
    # )
    # t1 = time.time()
    # print(f"Time taken: {t1 - t0} seconds")

    # Perform the data assimilation
    posterior_ensemble = posterior_ensemble.reshape(
        ENSEMBLE_SIZE, 1, NUM_STATES, STATE_DIM
    )
    for i in tqdm(range(0, OUTER_STEPS)):
        rng_key, key = jax.random.split(rng_key)
        posterior_next = da_model(
            prior_ensemble=posterior_ensemble[:, -1],
            obs_vect=observations[i],
            rng_key=key,
            return_inner_steps=True,
        )
        if jnp.isnan(posterior_next).any():
            print(f"NaN in posterior_next at time {i}")
            break

        posterior_ensemble = jnp.concatenate(
            [posterior_ensemble, posterior_next], axis=1
        )

    # Calculate the prior and posterior errors
    true_sol = true_sol.reshape(OUTER_STEPS * INNER_STEPS + 1, STATE_DIM)

    # Calculate the prior and posterior errors
    prior_error = true_sol - prior_ensemble.mean(axis=(0, 2))
    prior_error = np.sqrt(np.sum(prior_error**2))
    posterior_error = true_sol - posterior_ensemble.mean(axis=(0, 2))
    posterior_error = np.sqrt(np.sum(posterior_error**2))

    print(f"Prior error: {prior_error}")
    print(f"Posterior error: {posterior_error}")

    mean_prior = prior_ensemble.mean(axis=(0, 2))
    mean_post = posterior_ensemble.mean(axis=(0, 2))
    std_post = posterior_ensemble.std(axis=(0, 2))
    time_axis = np.arange(posterior_ensemble.shape[1])

    plt.figure()
    for state_idx in range(STATE_DIM):
        plt.subplot(STATE_DIM, 1, state_idx + 1)
        for i, (state_name, state_data, color) in enumerate(
            zip(
                ["Prior Ensemble Mean", "Posterior Ensemble Mean", "True Solution"],
                [mean_prior, mean_post, true_sol],
                ["tab:red", "tab:blue", "black"],
            )
        ):
            plt.plot(
                time_axis,
                state_data[:, state_idx],
                label=state_name,
                color=color,
                linewidth=3,
                linestyle="--" if state_name == "True Solution" else "-",
            )
        plt.fill_between(
            time_axis,
            mean_post[:, state_idx] - std_post[:, state_idx],
            mean_post[:, state_idx] + std_post[:, state_idx],
            color="tab:blue",
            alpha=0.2,
            label="Posterior ± Std",
        )
        plt.legend()
        plt.xlabel("Time")
        plt.ylabel("State")
        plt.ylim(true_sol[:, state_idx].min(), true_sol[:, state_idx].max())
    plt.show()


if __name__ == "__main__":
    main()
